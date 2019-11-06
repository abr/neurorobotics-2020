import numpy as np

from abr_control.controllers import OSC, Damping, RestingConfig
from abr_control.controllers import path_planners, signals
from abr_control.utils import transformations
from nengolib.stats import ScatteredHypersphere


def _get_approach(
        target_pos, approach_buffer=0.03, z_offset=0, z_rot=None, rot_wrist=False):
    """
    Takes the target location, and returns an
    orientation to approach the target, along with a target position that
    is approach_buffer meters short of the target, with a z offset
    determined by z_offset in meters. The orientation is set to be a vector
    that would connect the robot base to the target, with the gripper parallel
    to the ground

    Parameters
    ----------
    target_pos: list of 3 floats
        xyz cartesian loc of target of interest [meters]
    approach_buffer: float, Optional (Default: 0.03)
        we want to approach the target along the orientation that connects
        the base of the arm to the target, but we want to stop short before
        going for the grasp. This variable sets that distance to stop short
        of the target [meters]
    z_offset: float, Optional (Default: 0.2)
        sometimes it is desirable to approach a target from above or below.
        This gets added to the final target position
    z_rot: float, optional (Default: pi/2)
        rotates the z axis for the final approach
        pi/2 to be parallel with ground (gripper pointing outwards)
        pi to be perpendicular (gripper pointing down)
    rot_wrist: boolean, optional (Default: False)
        True to rotate the gripper an additional pi/2 rad along the wrist
        rotation axis
    """
    if z_rot is None:
        theta1 = np.pi/2
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


    # get signs of target directions so our approach target stays in the same
    # octant as the provided target
    # target_sign = target_pos / abs(target_pos)

    dist_to_target = np.linalg.norm(target_pos)
    approach_vector = np.copy(target_pos)
    approach_vector /= np.linalg.norm(approach_vector)
    # print('get_approach target_pos: ', target_pos)

    approach_pos = approach_vector * (dist_to_target - approach_buffer)
    #approach_vector[2] = 0

    # world z pointing up, rotate by pi/2 to be parallel with ground
    q1 = [np.cos(theta1/2),
        0,
        np.sin(theta1/2),
        0
        ]
    # now we rotate about z to get x pointing up
    theta2 = theta2_scale * np.arctan2(target_pos[1], target_pos[0])
    # print('theta2: ', theta2)
    q2 = [np.cos(theta2/2),
        0,
        0,
        np.sin(theta2/2),
        ]


    # get our combined rotation
    q3 = transformations.quaternion_multiply(q2, q1)

    approach_pos[2] += z_offset

    return approach_pos, q3


def get_approach_path(
        robot_config, path_planner, q, target_pos, max_reach_dist=None,
        min_z=0, target_orientation=None, start_pos=None, z_rot=None,
        rot_wrist=False, **kwargs):
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
    starting_R = robot_config.R('EE', q)
    starting_orientation = transformations.quaternion_from_matrix(
        starting_R)
    if start_pos is None:
        start_pos = robot_config.Tx('EE', q)

    # calculate our target approach position and orientation
    approach_pos, approach_orient = _get_approach(
        target_pos=target_pos, z_rot=z_rot, rot_wrist=rot_wrist, **kwargs)

    if target_orientation is not None:
        # print('Using manual target orientation')
        approach_orient = target_orientation

    # generate our path to our approach position
    path_planner.generate_path(
        position=start_pos, target_pos=approach_pos)

    # generate our orientation planner
    _, orientation_planner = path_planner.generate_orientation_path(
        orientation=starting_orientation,
        target_orientation=approach_orient)
    target_data = {
                'target_pos': target_pos,
                'approach_pos': approach_pos,
                'approach_orient': approach_orient}

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
        ctrlr_dof = [True, True, True, True, True, True])
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
        kp=120,  # position gain
        kv=25,
        null_controllers=null,
        vmax=None,  # [m/s, rad/s]
        # control all DOF [x, y, z, alpha, beta, gamma]
        ctrlr_dof = [True, True, True, False, False, False])
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
    variances = np.hstack((
        variances_q[in_index], variances_dq[in_index]))
    means_q = np.zeros(6)
    means_dq = np.zeros(6)
    means = np.hstack((
        means_q[in_index], means_dq[in_index]))

    # start with all zeros
    weights = None

    # intercepts
    intercept_bounds = [-0.3, -0.1]
    intercept_mode = -0.2
    intercepts = signals.dynamics_adaptation.AreaIntercepts(
        dimensions=n_input,
        base=signals.dynamics_adaptation.Triangular(
            intercept_bounds[0],
            intercept_mode,
            intercept_bounds[1]))

    intercepts = np.array(intercepts.sample(
        n_neurons * n_ensembles))

    intercepts = intercepts.reshape(
        n_ensembles, n_neurons)

    # encoders
    hypersphere = ScatteredHypersphere(surface=True)
    encoders = hypersphere.sample(
        n_neurons * n_ensembles, n_input)

    # not incoporated, but sometimes we zero enc dims at this step
    # if cpu['zeroed_enc_range'] is not None:
    #     cpu['encoders'] = scripts.zero_encoder_dims.run(
    #         encoders=cpu['encoders'],
    #         zeroed_dims=cpu['zeroed_enc_range'])

    encoders = (encoders.reshape(
        n_ensembles, n_neurons, n_input))

    # we need to include the spherical dimension to get our encoders
    # to the right dimensionality, but on instantiation of the adaptive
    # controller we exclude it, as the class handles this depending on
    # spherical conversion
    adaptive = signals.dynamics_adaptation.DynamicsAdaptation(
        n_input=n_input-spherical,
        n_output=5,
        n_neurons=n_neurons,
        n_ensembles=n_ensembles,
        pes_learning_rate=1e-4,
        intercepts=intercepts,
        weights=weights,
        seed=0,
        encoders=encoders,
        means=means,
        variances=variances,
        spherical=spherical
        )

    return adaptive

def second_order_path_planner(n_timesteps=1000, error_scale=1e-3):
    """
    Define your path planner of choice here
    """
    traj_planner = path_planners.BellShaped(
        error_scale=error_scale, n_timesteps=n_timesteps)
    return traj_planner

def first_order_arc(n_timesteps):
    traj_planner = path_planners.FirstOrderArc(n_timesteps)
    return traj_planner


# def arc_path_planner(n_timesteps=2000, **kwargs):
#     """
#     Define your path planner of choice here
#     """
#
#     traj_planner = path_planners.FirstOrderArc(n_timesteps=n_timesteps)
#     return traj_planner


def target_shift(interface, base_location, scale=0.01, xlim=None, ylim=None, zlim=None):
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
    xlim: list of 2 floats, Optional (Default: -1, 1)
        the minimum and maximum allowable values [meters]
    ylim: list of 2 floats, Optional (Default: -1, 1)
        the minimum and maximum allowable values [meters]
    zlim: list of 2 floats, Optional (Default: 0, 1)
        the minimum and maximum allowable values [meters]
    """
    if xlim is None:
        xlim = [-1, 1]
    if ylim is None:
        ylim = [-1, 1]
    if zlim is None:
        zlim = [0, 1]

    def clip(val, minimum, maximum):
        val = max(val, minimum)
        val = min(val, maximum)
        return val

    shifted_target = base_location + scale * np.array([
        interface.viewer.target_x,
        interface.viewer.target_y,
        interface.viewer.target_z])
    shifted_target = np.array([
        clip(shifted_target[0], xlim[0], xlim[1]),
        clip(shifted_target[1], ylim[0], ylim[1]),
        clip(shifted_target[2], zlim[0], zlim[1])])
    interface.viewer.target_x = 0
    interface.viewer.target_y = 0
    interface.viewer.target_z = 0

    return shifted_target
