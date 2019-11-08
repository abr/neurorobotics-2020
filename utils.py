import numpy as np

from abr_control.controllers import OSC, Damping, RestingConfig
from abr_control.controllers import path_planners, signals
from abr_control.utils import transformations
from abr_control.utils.transformations import quaternion_multiply, quaternion_inverse
from nengolib.stats import ScatteredHypersphere

import nengo
import numpy as np
import scipy.special

from scipy.linalg import svd
from scipy.special import beta, betainc, betaincinv
from nengo.dists import Distribution, UniformHypersphere
from nengo.utils.compat import is_integer


def _get_approach(
    target_pos, approach_buffer=0.03, offset=None, z_rot=None, rot_wrist=False
):
    """
    Takes the target location, and returns an orientation to approach the target, along
    with a target position that is approach_buffer meters short of the target, with an
    xyz offset determined by offset in meters. The orientation is set to be a vector
    that would connect the robot base to the target, with the gripper parallel to the
    ground.

    Parameters
    ----------
    target_pos: list of 3 floats
        xyz cartesian loc of target of interest [meters]
    approach_buffer: float, Optional (Default: 0.03)
        we want to approach the target along the orientation that connects the base of
        the arm to the target, but we want to stop short before going for the grasp.
        This variable sets that distance to stop short of the target [meters]
    offset: float, Optional (Default: None)
        sometimes it is desirable to approach a target from above or below.  This gets
        added to the final target position
    z_rot: float, optional (Default: pi/2)
        rotates the z axis for the final approach pi/2 to be parallel with ground
        (gripper pointing outwards), pi to be perpendicular (gripper pointing down)
    rot_wrist: boolean, optional (Default: False)
        True to rotate the gripper an additional pi/2 rad along the wrist rotation axis
    """
    if z_rot is None:
        theta1 = np.pi / 2
    else:
        theta1 = z_rot

    theta2_scale = 1
    if rot_wrist:
        # rotates twice so our gripper grasping orientation is shifted by pi/2
        # along the wrist rotation axis
        theta2_scale = 2

    # save a copy of the target in case weird pointer things happen with lists
    # target_z = np.copy(target_pos[2])
    target_pos = np.asarray(target_pos)

    dist_to_target = np.linalg.norm(target_pos)
    approach_vector = np.copy(target_pos)
    approach_vector /= np.linalg.norm(approach_vector)
    # print('get_approach target_pos: ', target_pos)

    approach_pos = approach_vector * (dist_to_target - approach_buffer)
    # approach_vector[2] = 0

    # world z pointing up, rotate by pi/2 to be parallel with ground
    q1 = [np.cos(theta1 / 2), 0, np.sin(theta1 / 2), 0]
    # now we rotate about z to get x pointing up
    theta2 = theta2_scale * np.arctan2(target_pos[1], target_pos[0])
    # print('theta2: ', theta2)
    q2 = [np.cos(theta2 / 2), 0, 0, np.sin(theta2 / 2)]

    # get our combined rotation
    q3 = transformations.quaternion_multiply(q2, q1)

    approach_pos += offset

    return approach_pos, q3


def get_approach_path(
    robot_config,
    path_planner,
    q,
    target_pos,
    max_reach_dist=None,
    min_z=0,
    target_orientation=None,
    start_pos=None,
    z_rot=None,
    rot_wrist=False,
    **kwargs
):
    """
    Accepts a robot config, path planner, and target_position, returns the
    generated position and orientation paths to approach the target for grasping

    Parameters
    ----------
    robot_config: instantiated abr_control/arms/base_config.py subclass
        used to determine the current arms orientation and position
    path_planner: instantiated
        abr_control/controllers/path_planners/path_planner.py subclass
        used to filter the path to the final target, and to generate the
        orientation path with the same reaching profile
    q: list of floats
        the current joint possitions of the arm [radians]
    target_pos: list of 3 floats, Optional (Default: None)
        cartesian location of final target [meters], if None a random one will
        be generated
    max_reach_dist: float, Optional (Default: None)
        the maximum distance [meters] from origin the arm can reach. The target
        is normalized and the target is set along that vector, max_reach_dist
        from the origin. If None, the target is left as is
    target_orientation: list of three floats, Optional (Default: None)
        euler angles for target orientation, leave as None to automatically
        calculate the orientation for a grasping approach
        (see _get_approach() )
    start_pos: list of three floats, Optional (Default: None)
        if left as None will use the current EE position
    z_rot: float, optional (Default: pi/2)
        rotates the z axis for the final approach
        pi/2 to be parallel with ground (gripper pointing outwards)
        pi to be perpendicular (gripper pointing down)
    rot_wrist: boolean, optional (Default: False)
        True to rotate the gripper an additional pi/2 rad along the wrist
        rotation axis
    """

    if target_pos[2] < min_z:
        # make sure the target isn't too close to the ground
        target_pos[2] = min_z

    if max_reach_dist is not None:
        # normalize to make sure our target is within reaching distance
        target_pos = target_pos / np.linalg.norm(target_pos) * max_reach_dist

    # get our EE starting orientation and position
    starting_R = robot_config.R("EE", q)
    starting_orientation = transformations.quaternion_from_matrix(starting_R)
    if start_pos is None:
        start_pos = robot_config.Tx("EE", q)

    # calculate our target approach position and orientation
    approach_pos, approach_orient = _get_approach(
        target_pos=target_pos, z_rot=z_rot, rot_wrist=rot_wrist, **kwargs
    )

    if target_orientation is not None:
        # print('Using manual target orientation')
        approach_orient = target_orientation

    # generate our path to our approach position
    path_planner.generate_path(position=start_pos, target_pos=approach_pos)

    # generate our orientation planner
    _, orientation_planner = path_planner.generate_orientation_path(
        orientation=starting_orientation, target_orientation=approach_orient
    )
    target_data = {
        "target_pos": target_pos,
        "approach_pos": approach_pos,
        "approach_orient": approach_orient,
    }

    return path_planner, orientation_planner, target_data


def osc6dof(robot_config, rest_angles=None):
    # damp the movements of the arm
    damping = Damping(robot_config, kv=10)
    null = [damping]
    # rest_angles = robot_config.START_ANGLES
    if rest_angles is not None:
        rest = RestingConfig(robot_config, rest_angles, kp=4, kv=0)
        null.append(rest)

    # create operational space controller
    ctrlr = OSC(
        robot_config,
        kp=120,  # position gain
        kv=20,
        ko=180,  # orientation gain
        null_controllers=null,
        vmax=None,  # [m/s, rad/s]
        # control all DOF [x, y, z, alpha, beta, gamma]
        ctrlr_dof=[True, True, True, True, True, True],
    )
    return ctrlr


def osc3dof(robot_config, rest_angles=None):
    # damp the movements of the arm
    damping = Damping(robot_config, kv=10)
    null = [damping]
    # rest_angles = robot_config.START_ANGLES
    if rest_angles is not None:
        rest = RestingConfig(robot_config, rest_angles, kp=4, kv=0)
        null.append(rest)

    # create operational space controller
    ctrlr = OSC(
        robot_config,
        kp=100,  # position gain
        kv=20,
        null_controllers=null,
        vmax=None,  # [m/s, rad/s]
        # control all DOF [x, y, z, alpha, beta, gamma]
        ctrlr_dof=[True, True, True, False, False, False],
    )
    return ctrlr


def adapt(in_index, spherical):
    """
    Parameters
    ----------
    in_index: list of booleans
        which input dimensions are being adapted
    spherical: boolean
        whether or not to use spherical conversion
    """
    n_input = np.sum(in_index) * 2 + spherical
    n_neurons = 1000
    n_ensembles = 10

    # means and variances
    variances_q = np.ones(6) * 6.28
    variances_dq = np.ones(6) * 1.25
    variances = np.hstack((variances_q[in_index], variances_dq[in_index]))
    means_q = np.zeros(6)
    means_dq = np.zeros(6)
    means = np.hstack((means_q[in_index], means_dq[in_index]))

    # start with all zeros
    weights = None

    # intercepts
    intercept_bounds = [-0.3, -0.1]
    intercept_mode = -0.2
    intercepts = signals.dynamics_adaptation.AreaIntercepts(
        dimensions=n_input,
        base=signals.dynamics_adaptation.Triangular(
            intercept_bounds[0], intercept_mode, intercept_bounds[1]
        ),
    )

    intercepts = np.array(intercepts.sample(n_neurons * n_ensembles))

    intercepts = intercepts.reshape(n_ensembles, n_neurons)

    # encoders
    hypersphere = ScatteredHypersphere(surface=True)
    encoders = hypersphere.sample(n_neurons * n_ensembles, n_input)

    # not incoporated, but sometimes we zero enc dims at this step
    # if cpu['zeroed_enc_range'] is not None:
    #     cpu['encoders'] = scripts.zero_encoder_dims.run(
    #         encoders=cpu['encoders'],
    #         zeroed_dims=cpu['zeroed_enc_range'])

    encoders = encoders.reshape(n_ensembles, n_neurons, n_input)

    # we need to include the spherical dimension to get our encoders
    # to the right dimensionality, but on instantiation of the adaptive
    # controller we exclude it, as the class handles this depending on
    # spherical conversion
    adaptive = signals.dynamics_adaptation.DynamicsAdaptation(
        n_input=n_input - spherical,
        n_output=5,
        n_neurons=n_neurons,
        n_ensembles=n_ensembles,
        pes_learning_rate=3e-4,
        intercepts=intercepts,
        weights=weights,
        seed=0,
        encoders=encoders,
        means=means,
        variances=variances,
        spherical=spherical,
    )

    return adaptive


def second_order_path_planner(n_timesteps=1000, error_scale=1e-3):
    """
    Define your path planner of choice here
    """
    traj_planner = path_planners.BellShaped(
        error_scale=error_scale, n_timesteps=n_timesteps
    )
    return traj_planner


def first_order_arc(n_timesteps):
    traj_planner = path_planners.FirstOrderArc(n_timesteps)
    return traj_planner


def first_order_arc_dmp(n_timesteps, error_scale=1):
    traj_planner = path_planners.FirstOrderArcDMP(n_timesteps, error_scale)
    return traj_planner


def target_shift(interface, base_location, scale=0.01, xlim=None, ylim=None, zlim=None, rlim=None):
    """
    Gets the user input from the mujoco_viewer to shift the target and returns the
    base_location with the shift*scale. The final target is clipped along x, y, z
    depending on the xlim, ylim, and zlim values

    Parameters
    ----------
    interface: abr_control.interface.mujoco_interface class
    base_location: array of 3 floats
        the current target to be shifted [meters]
    scale: float, optional (Default: 0.01)
        the amount to move with each button press [meters]
    xlim: array of 2 floats, Optional (Default: -1, 1)
        the minimum and maximum allowable values [meters]
    ylim: array of 2 floats, Optional (Default: -1, 1)
        the minimum and maximum allowable values [meters]
    zlim: array of 2 floats, Optional (Default: 0, 1)
        the minimum and maximum allowable values [meters]
    rlim: array of 2 float, Optional (Default: [None, None])
        the minimum and maxium radius from the arm origin to allow targets
    """
    if xlim is None:
        xlim = [-1, 1]
    if ylim is None:
        ylim = [-1, 1]
    if zlim is None:
        zlim = [0, 1]
    if rlim is None:
        rlim = [None, None]

    def clip(val, minimum, maximum):
        val = max(val, minimum)
        val = min(val, maximum)
        return val

    shifted_target = base_location + scale * np.array(
        [
            interface.viewer.target_x,
            interface.viewer.target_y,
            interface.viewer.target_z,
        ]
    )

    # check that we're within radius thresholds, if set
    if rlim[0] is not None:
        if np.linalg.norm(shifted_target) < rlim[0]:
            return base_location

    if rlim[1] is not None:
        if np.linalg.norm(shifted_target) > rlim[1]:
            return base_location

    shifted_target = np.array(
        [
            clip(shifted_target[0], xlim[0], xlim[1]),
            clip(shifted_target[1], ylim[0], ylim[1]),
            clip(shifted_target[2], zlim[0], zlim[1]),
        ]
    )
    interface.viewer.target_x = 0
    interface.viewer.target_y = 0
    interface.viewer.target_z = 0

    return shifted_target


def scale_inputs(spherical, means, variances, input_signal):
    """
    Currently set to accept joint position and velocities as time
    x dimension arrays, and returns them scaled based on the means and
    variances set on instantiation

    PARAMETERS
    ----------
    input_signal : numpy.array
        the current desired input signal, typical joint positions and
        velocities in [rad] and [rad/sec] respectively

    The reason we do a shift by the means is to try and center the majority
    of our expected input values in our scaling range, so when we stretch
    them by variances they encompass more of our input range.
    """
    scaled_input = (input_signal - means) / variances

    if spherical:
        # from nengolib.stats import spherical_transform
        # put into the 0-1 range
        scaled_input = scaled_input / 2 + 0.5
        # project onto unit hypersphere in larger state space
        scaled_input = scaled_input.flatten()
        scaled_input = spherical_transform(scaled_input.reshape(1, len(scaled_input)))

    return scaled_input


def calculate_rotQ():
    theta1 = np.pi/2
    xyz1 = np.array([0, 0, 1])
    rot1_quat = transformations.quaternion_about_axis(theta1, xyz1)

    theta2 = np.pi/2
    xyz2 = np.array([0, 1, 0])
    rot2_quat = transformations.quaternion_about_axis(theta2, xyz2)

    theta3 = -np.pi/2
    xyz3 = np.array([1, 0, 0])
    rot3_quat = transformations.quaternion_about_axis(theta3, xyz3)

    rotQ = quaternion_multiply(rot3_quat, quaternion_multiply(rot2_quat, rot1_quat))

    return rotQ


def get_weights(sim, conn_learn):
    """ Save the current weights to the specified test_name folder

    Save weights for individual runs. A group of runs is
    classified as a session. Multiple sessions can then be used
    to average over a set of learned runs. If session or run are set to None
    then the test_name location will be searched for the highest numbered
    folder and file respectively
    """
    return [sim.signals[sim.model.sig[conn]["weights"]] for conn in conn_learn]


class AreaIntercepts(nengo.dists.Distribution):
    """ Generate an optimally distributed set of intercepts in
    high-dimensional space.
    """

    dimensions = nengo.params.NumberParam("dimensions")
    base = nengo.dists.DistributionParam("base")

    def __init__(self, dimensions, base=nengo.dists.Uniform(-1, 1)):
        super(AreaIntercepts, self).__init__()
        self.dimensions = dimensions
        self.base = base

    def __repr(self):
        return "AreaIntercepts(dimensions=%r, base=%r)" % (self.dimensions, self.base)

    def transform(self, x):
        sign = 1
        if x > 0:
            x = -x
            sign = -1
        return sign * np.sqrt(
            1 - scipy.special.betaincinv((self.dimensions + 1) / 2.0, 0.5, x + 1)
        )

    def sample(self, n, d=None, rng=np.random):
        s = self.base.sample(n=n, d=d, rng=rng)
        for ii, ss in enumerate(s):
            s[ii] = self.transform(ss)
        return s


class Triangular(nengo.dists.Distribution):
    """ Generate an optimally distributed set of intercepts in
    high-dimensional space using a triangular distribution.
    """

    left = nengo.params.NumberParam("dimensions")
    right = nengo.params.NumberParam("dimensions")
    mode = nengo.params.NumberParam("dimensions")

    def __init__(self, left, mode, right):
        super(Triangular, self).__init__()
        self.left = left
        self.right = right
        self.mode = mode

    def __repr__(self):
        return "Triangular(left=%r, mode=%r, right=%r)" % (
            self.left,
            self.mode,
            self.right,
        )

    def sample(self, n, d=None, rng=np.random):
        if d is None:
            return rng.triangular(self.left, self.mode, self.right, size=n)
        else:
            return rng.triangular(self.left, self.mode, self.right, size=(n, d))


def random_orthogonal(d, rng=None):
    rng = np.random if rng is None else rng
    m = UniformHypersphere(surface=True).sample(d, d, rng=rng)
    u, s, v = svd(m)
    return np.dot(u, v)


class SphericalCoords(Distribution):
    def __init__(self, m):
        super(SphericalCoords, self).__init__()
        self.m = m

    def __repr__(self):
        return "%s(%r)" % (type(self).__name__, self.m)

    def sample(self, n, d=None, rng=np.random):
        """Samples ``n`` points in ``d`` dimensions."""
        shape = self._sample_shape(n, d)
        y = rng.uniform(size=shape)
        return self.ppf(y)

    def pdf(self, x):
        """Evaluates the PDF along the values ``x``."""
        return np.pi * np.sin(np.pi * x) ** (self.m - 1) / beta(self.m / 2.0, 0.5)

    def cdf(self, x):
        """Evaluates the CDF along the values ``x``."""
        y = 0.5 * betainc(self.m / 2.0, 0.5, np.sin(np.pi * x) ** 2)
        return np.where(x < 0.5, y, 1 - y)

    def ppf(self, y):
        """Evaluates the inverse CDF along the values ``x``."""
        y_reflect = np.where(y < 0.5, y, 1 - y)
        z_sq = betaincinv(self.m / 2.0, 0.5, 2 * y_reflect)
        x = np.arcsin(np.sqrt(z_sq)) / np.pi
        return np.where(y < 0.5, x, 1 - x)


def spherical_transform(samples):
    samples = np.asarray(samples)
    samples = samples[:, None] if samples.ndim == 1 else samples
    coords = np.empty_like(samples)
    n, d = coords.shape

    # inverse transform method (section 1.5.2)
    for j in range(d):
        coords[:, j] = SphericalCoords(d - j).ppf(samples[:, j])

    # spherical coordinate transform
    mapped = np.ones((n, d + 1))
    i = np.ones(d)
    i[-1] = 2.0
    s = np.sin(i[None, :] * np.pi * coords)
    c = np.cos(i[None, :] * np.pi * coords)
    mapped[:, 1:] = np.cumprod(s, axis=1)
    mapped[:, :-1] *= c
    return mapped


def _rd_generate(n, d, seed=0.5):
    """Generates the first ``n`` points in the ``R_d`` sequence."""

    # http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
    def gamma(d, n_iter=20):
        """Newton-Raphson-Method to calculate g = phi_d."""
        x = 1.0
        for _ in range(n_iter):
            x -= (x ** (d + 1) - x - 1) / ((d + 1) * x ** d - 1)
        return x

    g = gamma(d)
    alpha = np.zeros(d)
    for j in range(d):
        alpha[j] = (1 / g) ** (j + 1) % 1

    z = np.zeros((n, d))
    z[0] = (seed + alpha) % 1
    for i in range(1, n):
        z[i] = (z[i - 1] + alpha) % 1

    return z


class Rd(Distribution):
    def __repr__(self):
        return "%s()" % (type(self).__name__)

    def sample(self, n, d=1, rng=np.random):
        """Samples ``n`` points in ``d`` dimensions."""
        if d == 1:
            # Tile the points optimally. TODO: refactor
            return np.linspace(1.0 / n, 1, n)[:, None]
        if d is None or not is_integer(d) or d < 1:
            # TODO: this should be raised when the ensemble is created
            raise ValueError("d (%d) must be positive integer" % d)
        return _rd_generate(n, d)


class ScatteredHypersphere(UniformHypersphere):
    def __init__(self, surface, base=Rd()):
        super(ScatteredHypersphere, self).__init__(surface)
        self.base = base

    def __repr__(self):
        return "%s(surface=%r, base=%r)" % (
            type(self).__name__,
            self.surface,
            self.base,
        )

    def sample(self, n, d=1, rng=np.random):
        """Samples ``n`` points in ``d`` dimensions."""
        if d == 1:
            return super(ScatteredHypersphere, self).sample(n, d, rng)

        if self.surface:
            samples = self.base.sample(n, d - 1, rng)
            radius = 1.0
        else:
            samples = self.base.sample(n, d, rng)
            samples, radius = samples[:, :-1], samples[:, -1:] ** (1.0 / d)

        mapped = spherical_transform(samples)

        # radius adjustment for ball versus sphere, and a random rotation
        rotation = random_orthogonal(d, rng=rng)
        return np.dot(mapped * radius, rotation)
