"""
To run this demo you will need to download the required stl and texture files
Run the following two commands from the directory this file is in:

TO DOWNLOAD FILES
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1N4RSyJeHCMFKgtjdXOD_4A7izpFcFMKw' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1N4RSyJeHCMFKgtjdXOD_4A7izpFcFMKw" -O meshes && rm -rf /tmp/cookies.txt

TO EXTRACT FILES
tar -zxvf meshes

To run the demo with Nengo running on cpu:
    python loihi_demo.py cpu

To run the demo with Nengo on loihi
    NXSDKHOST=loihighrd python loihi_demo.py

To control the demo with an xbox controller append 'gamepad' without quotes to either of the above two commands

To start the demo in demo mode, append 'demo' without quotes
"""

import glfw
import mujoco_py
import nengo
import nengo_loihi
import numpy as np
import sys
import time

from nengo_loihi import decode_neurons

from abr_control.controllers import OSC, Damping
from abr_control.arms.mujoco_config import MujocoConfig
from abr_control.interfaces.mujoco import Mujoco
from abr_control.controllers.signals.dynamics_adaptation import (
    AreaIntercepts,
    Triangular
)
from abr_control.controllers import path_planners
from abr_control._vendor.nengolib.stats.ntmdists import (
    ScatteredHypersphere,
    spherical_transform
)

backend = "cpu"
if len(sys.argv) > 1:
    for arg in sys.argv:
        arg = str(arg)
        if arg == 'loihi':
            backend = 'loihi'

print("Using %s as backend" % backend)

class ExitSim(Exception):
    print('Restarting simulation')
    pass


def initialize_mujoco(robot_config):
    # create our Mujoco interface
    interface = Mujoco(robot_config, dt=0.001, visualize=True)
    interface.connect()
    interface.send_target_angles(robot_config.START_ANGLES)
    return interface


def restart_mujoco(net, robot_config):
    net.interface.disconnect()
    glfw.destroy_window(net.interface.viewer.window)
    del net.interface
    time.sleep(.25)
    net.interface = initialize_mujoco(robot_config)
    initialize_interface(net.interface)
    net.interface.set_mocap_xyz(name="target", xyz=net.interface.viewer.target)
    net.model = net.interface.sim.model
    net.data = net.interface.sim.data


def initialize_interface(interface):
    interface.set_mocap_xyz("obstacle", [0, 0, -100])
    interface.set_mocap_xyz("path_planner", [0, 0, -100])
    interface.set_mocap_xyz("target_orientation", [0, 0, -100])
    interface.set_mocap_xyz("path_planner_orientation", [0, 0, -100])
    interface.viewer.target = np.array([-0.4, 0.5, 0.4])


def scale_inputs(input_signal, means, variances, spherical=False):
    '''
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
    '''
    scaled_input = (input_signal - means) / variances

    if spherical:
        # put into the 0-1 range
        scaled_input = scaled_input / 2 + .5
        # project onto unit hypersphere in larger state space
        scaled_input = scaled_input.flatten()
        scaled_input = spherical_transform(
            scaled_input.reshape(1, len(scaled_input)))

    return scaled_input



def demo(backend):
    rng = np.random.RandomState(9)
    # joint pos and vel as input, joint force as output
    n_input = 10
    n_output = int(n_input/2)
    n_neurons = 1000
    n_ensembles = 10
    pes_learning_rate = 1e-5 if backend == "cpu" else 1e-5
    seed = 0
    spherical = True  # project the input onto the surface of a D+1 hypersphere

    means = np.zeros(n_input)
    variances = np.hstack((np.ones(int(n_input/2)) * 6.28, np.ones(int(n_input/2)) * 1.25))

    if spherical:
        n_input += 1

    # synapse time constants
    tau_input = 0.012  # on input connection
    tau_training = 0.012  # on the training signal
    tau_output = 0.012  # on the output from the adaptive ensemble

    # set up neuron intercepts
    intercepts_bounds = [-0.4, -0.1]
    intercepts_mode = -0.3

    intercepts_dist = AreaIntercepts(
        dimensions=n_input,
        base=Triangular(intercepts_bounds[0], intercepts_mode, intercepts_bounds[1]),
    )
    intercepts = intercepts_dist.sample(n=n_neurons * n_ensembles, rng=rng)
    intercepts = intercepts.reshape(n_ensembles, n_neurons)

    np.random.seed = seed
    encoders_dist = ScatteredHypersphere(surface=True)
    encoders = encoders_dist.sample(n_neurons * n_ensembles, n_input, rng=rng)
    encoders = encoders.reshape(n_ensembles, n_neurons, n_input)

    # initialize our robot config for the jaco2
    robot_config = MujocoConfig(
        xml_file="jaco2",
        use_sim_state=True)

    net = nengo.Network(seed=seed)
    # Set the default neuron type for the network
    net.config[nengo.Ensemble].neuron_type = nengo.LIF()

    net.interface = initialize_mujoco(robot_config)
    net.model = net.interface.sim.model
    net.data = net.interface.sim.data

    # damp the movements of the arm
    damping = Damping(robot_config, kv=10)
    null = [damping]
    # rest_angles = robot_config.START_ANGLES
    rest_angles = None
    if rest_angles is not None:
        rest = RestingConfig(robot_config, rest_angles, kp=4, kv=0)
        null.append(rest)

    # create operational space controller
    net.ctrlr = OSC(
        robot_config,
        kp=100,  # position gain
        kv=20,
        null_controllers=null,
        vmax=None,  # [m/s, rad/s]
        # control all DOF [x, y, z, alpha, beta, gamma]
        ctrlr_dof=[True, True, True, False, False, False],
    )

    object_xyz = np.array([-0.5, 0.0, 0.2])

    # get target object id so we can change its colour
    target_geom_id = net.model.geom_name2id("target")
    green = [0, 0.9, 0, 0.5]
    red = [0.9, 0, 0, 0.5]

    OUTPUT_ZEROS = np.zeros(n_input + n_output)

    net.bodies = ["link1", "link2", "link3", "link4", "link5", "link6"]
    net.base_gravity = np.hstack((net.interface.model.opt.gravity, np.zeros(3)))

    initialize_interface(net.interface)

    def initialize_net(net):
        net.next_reach = True
        net.u = np.zeros(robot_config.N_JOINTS + 3)
        net.count = 0
        net.at_target = 0

    with net:
        # make the target offset from that start position
        net.interface.set_mocap_xyz(name="target", xyz=net.interface.viewer.target)
        initialize_net(net)
        # NOTE can visualize path planner by setting path_vis to True
        net.path_vis = False  # start out not displaying path planner target
        net.pos = robot_config.Tx("EE")
        net.n_timesteps = 1000
        net.trajectory_planner = path_planners.Arc(net.n_timesteps)


        def arm_func(t, u_adapt):
            interface = net.interface
            viewer = interface.viewer
            model = net.model
            data = net.data

            adapt_scale = 1
            if backend == "loihi":
                adapt_scale = 10

            ran_at_least_once = False
            while not ran_at_least_once or not viewer.adapt:

                ran_at_least_once = True

                if viewer.exit:
                    glfw.destroy_window(viewer.window)
                    raise ExitSim()

                if viewer.restart_sim:
                    initialize_net(net)
                    restart_mujoco(net, robot_config, UI)
                    interface = net.interface
                    viewer = interface.viewer
                    model = net.model
                    data = net.data

                # if the reaching mode has changed, recalculate reaching parameters ---
                if net.next_reach:
                    feedback = interface.get_feedback()

                    # generate our path to the new target
                    net.trajectory_planner.generate_path(
                        position=net.pos,
                        target_pos=viewer.target,
                    )
                    interface.set_mocap_xyz("target", viewer.target)
                    net.next_reach = False

                # get arm feedback
                feedback = interface.get_feedback()
                hand_xyz = robot_config.Tx("EE")

                # update our path planner position and orientation --------------------
                error = np.linalg.norm(
                    hand_xyz - viewer.target
                )
                if error < 0.05:  # when close enough, don't use path planner
                    net.pos = viewer.target
                    net.vel = np.zeros(3)
                else:
                    net.pos, net.vel = net.trajectory_planner.next()
                orient = np.zeros(3)

                if net.pos is None:
                    # if net.pos hasn't been set somehow make sure it's set
                    net.pos = hand_xyz
                target = np.hstack([net.pos, orient])

                # calculate our osc control signal ------------------------------------
                net.u[: robot_config.N_JOINTS] = net.ctrlr.generate(
                    q=feedback["q"], dq=feedback["dq"], target=target
                )

                if viewer.adapt:
                    # adaptive signal added (no signal for last joint)
                    net.u[:n_output] += u_adapt * adapt_scale

                # get our gripper command ---------------------------------------------
                u_gripper = np.ones(3) * -0.2
                net.u[robot_config.N_JOINTS :] = u_gripper

                # set the world gravity
                gravity = np.array([0, 0, -9.81*2, 0, 0, 0])

                # apply our gravity term
                for body in net.bodies:
                    interface.set_external_force(
                        body,
                        (
                            (gravity - net.base_gravity)
                            * net.model.body_mass[
                                net.model.body_name2id(body)
                            ]
                        ),
                    )

                # send to mujoco, stepping the sim forward
                interface.send_forces(net.u)

                # ----------------
                if error < 0.02:
                    net.model.geom_rgba[target_geom_id] = green
                else:
                    net.model.geom_rgba[target_geom_id] = red

                # TODO: if adapting add a hold timesteps
                if error < 0.02: # or net.count >= net.n_timesteps:
                    # add to our at target counter
                    #if error < 0.02:
                    net.at_target += 1
                    # if we maxed out our hold + timesteps, or we've been at target
                    #if net.count >= net.n_timesteps*1.5 or net.at_target > 250:
                    if net.at_target > 250:
                        net.next_reach = True
                        net.count = 0
                        net.at_target = 0
                        net.pos = robot_config.Tx("EE")
                        # scale random target to +/- 0.6 in xy, and 0.2-0.6 in z
                        # limitting z above 0.2 so we don't have to have a complicated
                        # check to see if x and y are inside the base near the origin
                        viewer.target = (
                            np.random.rand(3)
                            * np.array([1.2, 1.2, 0.4])
                            - np.array([0.6, 0.6, -0.2]))
                else:
                    net.at_target = 0

                net.count += 1

                # toggle the path planner visualization -------------------------------
                if net.path_vis:
                    interface.set_mocap_xyz("path_planner", target[:3])
                else:
                    interface.set_mocap_xyz(
                        "path_planner", np.array([0, 0, -100])
                    )

            # if adaptation is on, generate context signal for neural population ------
            feedback = interface.get_feedback()
            context = scale_inputs(
                spherical=spherical,
                means=means,
                variances=variances,
                input_signal=np.hstack([feedback["q"][:int((n_input-spherical)/2)], feedback["dq"][:int((n_input-spherical)/2)]]),
            )
            training_signal = -net.ctrlr.training_signal[:int((n_input-spherical)/2)]
            output_signal = np.hstack([context.flatten(), training_signal.flatten()])

            # TODO: scale the training signal here
            return output_signal

        # -----------------------------------------------------------------------------

        arm = nengo.Node(
            arm_func, size_in=n_output, size_out=n_input + n_output, label="arm"
        )
        arm_probe = nengo.Probe(arm)

        input_decodeneurons = decode_neurons.Preset5DecodeNeurons()
        onchip_input = input_decodeneurons.get_ensemble(dim=n_input)
        nengo.Connection(arm[:n_input], onchip_input, synapse=None)
        inp2ens_transform = np.hstack(
            [np.eye(n_input), -(np.eye(n_input))] * input_decodeneurons.pairs_per_dim
        )

        output_decodeneurons = decode_neurons.Preset5DecodeNeurons()
        onchip_output = output_decodeneurons.get_ensemble(dim=n_output)
        out2arm_transform = (
            np.hstack(
                [np.eye(n_output), -(np.eye(n_output))]
                * output_decodeneurons.pairs_per_dim
            )
            / 2000.0
        )  # divide by 100 (neuron firing rate) * 20 (on/off neurons per dim)
        nengo.Connection(
            onchip_output.neurons, arm, transform=out2arm_transform, synapse=tau_output
        )

        adapt_ens = []
        conn_learn = []
        for ii in range(n_ensembles):
            adapt_ens.append(
                nengo.Ensemble(
                    n_neurons=n_neurons,
                    dimensions=n_input,
                    intercepts=intercepts[ii],
                    radius=np.sqrt(n_input),
                    encoders=encoders[ii],
                    label="ens%02d" % ii,
                )
            )

            # hook up input signal to adaptive population to provide context
            inp2ens_transform_ii = np.dot(encoders[ii], inp2ens_transform)
            nengo.Connection(
                onchip_input.neurons,
                adapt_ens[ii].neurons,
                transform=inp2ens_transform_ii,
                synapse=tau_input,
            )

            conn_learn.append(
                nengo.Connection(
                    adapt_ens[ii],
                    onchip_output,
                    learning_rule_type=nengo.PES(
                        pes_learning_rate, pre_synapse=tau_training
                    ),
                    transform=rng.uniform(-0.01, 0.01, size=(n_output, n_input)),
                )
            )

            # hook up the training signal to the learning rule
            nengo.Connection(arm[n_input:], conn_learn[ii].learning_rule, synapse=None)

    return net, robot_config


if __name__ == "__main__":
    # if we're running outside of Nengo GUI
    # while 1:
    print('---------------------------------------------')
    print('--------- Applying 2 x earth gravity --------')
    print('---- Use left shift to toggle adaptation ----')
    print('---------------------------------------------')
    net, robot_config = demo(backend)
    try:
        if backend == "loihi":
            with nengo_loihi.Simulator(
                net, target="loihi", hardware_options=dict(snip_max_spikes_per_step=300)
            ) as sim:
                while 1:
                    sim.run(1e5)

        elif backend == "cpu":
            with nengo.Simulator(net) as sim:
                while 1:
                    sim.run(1e5)

    except ExitSim:
        pass

    finally:
        net.interface.disconnect()
# else:
#     # if we're running inside Nengo GUI
#     try:
#         model, robot_config = demo()
#     finally:
#         interface.disconnect()
