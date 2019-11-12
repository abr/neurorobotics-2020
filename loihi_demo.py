import nengo
import sys
import glfw

import mujoco_py
import nengo_loihi
import numpy as np
import timeit
import traceback


from nengo_loihi import decode_neurons

from abr_control.controllers import OSC, Damping
from abr_control.arms.mujoco_config import MujocoConfig
from abr_control.interfaces.mujoco import Mujoco
from abr_control.utils.transformations import (
    quaternion_multiply,
    quaternion_inverse,
    quaternion_from_euler,
)

from utils import (
    AreaIntercepts,
    Triangular,
    scale_inputs,
    ScatteredHypersphere,
    get_approach_path,
    osc6dof,
    osc3dof,
    second_order_path_planner,
    target_shift,
    adapt,
    first_order_arc,
    first_order_arc_dmp,
    calculate_rotQ,
    ExitSim,
    RestartMujoco,
)

from reach_list import gen_reach_list

if len(sys.argv) > 1:
    backend = str(sys.argv[1])
else:
    backend = "loihi"
print("Using %s as backend" % backend)


def initialize_mujoco(robot_config):
    # create our Mujoco interface
    interface = Mujoco(robot_config, dt=0.001, visualize=True)
    interface.connect()
    interface.send_target_angles(robot_config.START_ANGLES)

    return interface

def restart_mujoco(net, interface, robot_config):
    interface.disconnect()
    del interface
    interface = initialize_mujoco(robot_config)
    interface.set_mocap_xyz(name="target", xyz=net.final_xyz)
    return interface

def demo(backend):
    rng = np.random.RandomState(9)

    n_input = 10
    n_output = 5

    n_neurons = 1000
    n_ensembles = 10
    pes_learning_rate = 1e-5 if backend == 'cpu' else 1e-5
    seed = 0
    spherical = True  # project the input onto the surface of a D+1 hypersphere
    if spherical:
        n_input += 1

    means = np.zeros(10)
    variances = np.hstack((np.ones(5) * 6.28, np.ones(5) * 1.25))

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
    robot_config = MujocoConfig("jaco2_gripper", use_sim_state=True)

    interface = initialize_mujoco(robot_config)
    model = interface.sim.model
    data = interface.sim.data

    net = nengo.Network(seed=seed)
    # Set the default neuron type for the network
    net.config[nengo.Ensemble].neuron_type = nengo.LIF()

    object_xyz = np.array([-0.5, 0.0, 0.02])
    deposit_xyz = np.array([-0.4, 0.5, 0.4])

    scale = 0.05
    xlim = [-0.5, 0.5]
    ylim = [-0.5, 0.5]
    zlim = [0.0, 0.7]
    rlim = [0.4, 1.0]

    def clip(val, minimum, maximum):
        val = max(val, minimum)
        val = min(val, maximum)
        return val

    reach_list = gen_reach_list(robot_config, object_xyz, deposit_xyz)

    # max grip force
    max_grip = 8
    fkp = 144
    fkv = 15

    fingers = ["joint_thumb", "joint_index", "joint_pinky"]

    green = [0, 0.9, 0, 0.5]
    red = [0.9, 0, 0, 0.5]

    adapt_on = [0.9, 0.75, 0.1, 1]
    adapt_off = [0.5, 0.5, 0.5, 0.1]

    OUTPUT_ZEROS = np.zeros(n_input + n_output)
    adapt_geom_id = model.geom_name2id("adapt")
    target_geom_id = model.geom_name2id("target")

    net.old_reach_mode = None
    net.reach_index = -1
    net.next_reach = False
    net.reach = None
    net.u_gripper_prev = np.zeros(3)
    net.path_vis = False  # start out not displaying path planner target
    net.u = np.zeros(robot_config.N_JOINTS + 3)
    net.gravities = {
        'earth': np.array([0, 0, -9.81, 0, 0, 0]),
        'moon': np.array([0, 0, -1.62, 0, 0, 0]),
        'mars': np.array([0, 0, -3.71, 0, 0, 0]),
        'jupiter': np.array([0, 0, -24.92, 0, 0, 0]),
        'ISS': np.array([0, 0, 0, 0, 0, 0]),
        }
    net.bodies = ['link1', 'link2', 'link3', 'link4', 'link5', 'link6', 'dumbbell']
    net.base_gravity = np.hstack((interface.model.opt.gravity, np.zeros(3)))
    interface.set_mocap_xyz('moon', [0, 0, -100])
    interface.set_mocap_xyz('mars', [0, 0, -100])
    interface.set_mocap_xyz('jupiter', [0, 0, -100])
    interface.set_mocap_xyz('ISS', [0, 0, -100])
    interface.set_mocap_xyz('moon_floor', [0, 0, -100])
    interface.set_mocap_xyz('mars_floor', [0, 0, -100])
    interface.set_mocap_xyz('jupiter_floor', [0, 0, -100])
    interface.set_mocap_xyz('ISS_floor', [0, 0, -100])
 
    interface.set_mocap_xyz('earth', [1, 1, 0.5])
    interface.set_mocap_xyz('obstacle', [0, 0, -100])
    interface.set_mocap_xyz('path_planner', [0, 0, -100])
    interface.set_mocap_xyz('target_orientation', [0, 0, -100])
    interface.set_mocap_xyz('path_planner_orientation', [0, 0, -100])

    with net:

        net.old_reach_mode = None
        net.reach_index = -1
        net.next_reach = False
        net.reach = None
        net.u_gripper_prev = np.zeros(3)
        net.path_vis = False  # start out not displaying path planner target
        net.u = np.zeros(robot_config.N_JOINTS + 3)
        net.prev_planet = 'earth'

        net.final_xyz = deposit_xyz
        # make the target offset from that start position
        interface.set_mocap_xyz(name="target", xyz=net.final_xyz)

        def arm_func(t, u_adapt):
            global interface
            adapt_scale = 1
            if backend == 'loihi':
                adapt_scale = 10

            ran_at_least_once = False
            while not ran_at_least_once or not interface.viewer.adapt:
                ran_at_least_once = True
                net.reach_mode = interface.viewer.reach_mode

                if interface.viewer.exit:
                    glfw.destroy_window(interface.viewer.window)
                    raise ExitSim()

                if interface.viewer.restart_sim:
                    raise RestartMujoco()

                if net.reach_mode != net.old_reach_mode:
                    print("Reach mode changed")
                    net.next_reach = True
                    net.old_reach_mode = net.reach_mode
                    net.reach_index = -1

                # if the reaching mode has changed, recalculate reaching parameters ---
                if net.next_reach:
                    print("Generating next reach")
                    net.reach_index += 1
                    if net.reach_index >= len(reach_list[net.reach_mode]):
                        interface.viewer.reach_mode = "reach_target"
                        net.reach_mode = interface.viewer.reach_mode
                        net.reach_index = 0

                    net.reach = reach_list[net.reach_mode][net.reach_index]
                    net.u_gripper_prev = np.zeros(3)

                    feedback = interface.get_feedback()

                    # if we're reaching to target, update with user changes
                    if net.reach_mode == "reach_target":
                        net.reach["target_pos"] = net.final_xyz

                    if net.reach["target_options"] == "object":

                        net.reach["target_pos"] = interface.get_xyz(
                            "handle", object_type="geom"
                        )

                        # target orientation should be that of an object in the environment
                        objQ = interface.get_orientation("handle", object_type="geom")
                        net.rotQ = calculate_rotQ()
                        quat = quaternion_multiply(net.rotQ, objQ)
                        net.startQ = np.copy(quat)
                        net.reach["orientation"] = quat

                    elif net.reach["target_options"] == "shifted":
                        # account for the object in the hand having slipped / rotated
                        net.rotQ = calculate_rotQ()

                        # get xyz of the hand
                        hand_xyz = interface.get_xyz("EE", object_type="body")
                        # get xyz of the object
                        object_xyz = interface.get_xyz("handle", object_type="geom")

                        net.reach["target"] = object_xyz + (object_xyz - hand_xyz)

                        # get current orientation of hand
                        handQ_prime = interface.get_orientation(
                            "EE", object_type="body"
                        )
                        # get current orientation of object
                        objQ_prime = interface.get_orientation(
                            "handle", object_type="geom"
                        )

                        # get the difference between hand and object
                        rotQ_prime = quaternion_multiply(
                            handQ_prime, quaternion_inverse(objQ_prime)
                        )
                        # compare with difference at start of movement
                        dQ = quaternion_multiply(
                            rotQ_prime, quaternion_inverse(net.rotQ)
                        )
                        # transform the original target by the difference
                        net.shiftedQ = quaternion_multiply(net.startQ, dQ)

                        net.reach["orientation"] = net.shiftedQ

                    elif net.reach["target_options"] == "shifted2":
                        net.reach["orientation"] = net.shiftedQ

                    # calculate our position and orientation path planners, with their
                    # corresponding approach
                    (
                        net.trajectory_planner,
                        net.orientation_planner,
                        net.target_data,
                    ) = get_approach_path(
                        robot_config=robot_config,
                        path_planner=net.reach["traj_planner"](
                            net.reach["n_timesteps"]
                        ),
                        q=feedback["q"],
                        target_pos=net.reach["target_pos"],
                        target_orientation=net.reach["orientation"],
                        start_pos=net.reach["start_pos"],
                        max_reach_dist=None,
                        min_z=0.0,
                        approach_buffer=net.reach["approach_buffer"],
                        offset=net.reach["offset"],
                        z_rot=net.reach["z_rot"],
                        rot_wrist=net.reach["rot_wrist"],
                    )

                    net.next_reach = False
                    net.count = 0

                # get the target location from the interface
                net.old_final_xyz = net.final_xyz

                # check if the user moved the target ----------------------------------
                change = np.array(
                    [
                        interface.viewer.target_x,
                        interface.viewer.target_y,
                        interface.viewer.target_z,
                    ]
                )
                if not np.allclose(change, 0):
                    net.final_xyz = net.final_xyz + scale * change
                    # check that we're within radius thresholds, if set
                    if rlim[0] is not None:
                        if np.linalg.norm(net.final_xyz) < rlim[0]:
                            net.final_xyz = net.old_final_xyz

                    if rlim[1] is not None:
                        if np.linalg.norm(net.final_xyz) > rlim[1]:
                            net.final_xyz = net.old_final_xyz

                    net.final_xyz = np.array(
                        [
                            clip(net.final_xyz[0], xlim[0], xlim[1]),
                            clip(net.final_xyz[1], ylim[0], ylim[1]),
                            clip(net.final_xyz[2], zlim[0], zlim[1]),
                        ]
                    )
                    interface.viewer.target_x = 0
                    interface.viewer.target_y = 0
                    interface.viewer.target_z = 0
                    # update visualization of target
                    interface.set_mocap_xyz("target", net.final_xyz)

                # get arm feedback
                feedback = interface.get_feedback()
                hand_xyz = robot_config.Tx("EE", feedback["q"])

                # update our path planner position and orientation --------------------
                if net.reach_mode == "reach_target":
                    error = np.linalg.norm(
                        hand_xyz - net.final_xyz + net.reach["offset"]
                    )
                    if error < 0.05:  # when close enough, don't use path planner
                        net.pos = net.final_xyz + net.reach["offset"]
                        net.vel = np.zeros(3)
                    else:
                        if not np.allclose(net.final_xyz, net.old_final_xyz, atol=1e-5):
                            net.n_timesteps = net.reach["n_timesteps"] - net.count
                            net.trajectory_planner.generate_path(
                                position=net.pos,
                                target_pos=net.final_xyz + net.reach["offset"],
                            )
                        net.pos, net.vel = net.trajectory_planner.next()
                    orient = np.zeros(3)

                else:
                    error = np.linalg.norm((hand_xyz - net.target_data["approach_pos"]))
                    net.pos, net.vel = net.trajectory_planner.next()
                    orient = net.orientation_planner.next()

                target = np.hstack([net.pos, orient])

                # calculate our osc control signal ------------------------------------
                net.u[: robot_config.N_JOINTS] = net.reach["ctrlr"].generate(
                    q=feedback["q"], dq=feedback["dq"], target=target
                )

                if interface.viewer.adapt:
                    # adaptive signal added (no signal for last joint)
                    net.u[: robot_config.N_JOINTS - 1] += u_adapt * adapt_scale

                # if net.count % 500 == 0:
                #     print('u_adapt: ', u_adapt)
                # get our gripper command ---------------------------------------------
                finger_q = np.array(
                    [data.qpos[model.get_joint_qpos_addr(finger)] for finger in fingers]
                )
                finger_dq = np.array(
                    [data.qvel[model.get_joint_qpos_addr(finger)] for finger in fingers]
                )

                u_gripper = fkp * (net.reach["grasp_pos"] - finger_q) - fkv * finger_dq
                u_gripper = (
                    net.reach["f_alpha"] * u_gripper
                    + (1 - net.reach["f_alpha"]) * net.u_gripper_prev
                )
                u_gripper = np.clip(u_gripper, a_max=max_grip, a_min=-max_grip)
                net.u_gripper_prev[:] = np.copy(u_gripper)
                net.u[robot_config.N_JOINTS :] = u_gripper * interface.viewer.gripper

                # set the world gravity
                gravity = net.gravities[interface.viewer.planet]

                # incorporate dumbbell mass change
                if interface.viewer.additional_mass != 0:
                    interface.model.body_mass[interface.model.body_name2id('dumbbell')] += interface.viewer.additional_mass
                    interface.viewer.additional_mass = 0

                # apply our gravity term
                for body in net.bodies:
                    interface.set_external_force(
                        body,
                        ((gravity - net.base_gravity)
                        * interface.model.body_mass[interface.model.body_name2id(body)])
                        )

                # send to mujoco, stepping the sim forward
                interface.send_forces(net.u)

                # ----------------
                if net.reach_mode == "reach_target":
                    if error < net.reach["error_thresh"]:
                        model.geom_rgba[target_geom_id] = green
                    else:
                        model.geom_rgba[target_geom_id] = red
                else:
                    model.geom_rgba[target_geom_id] = red

                    # the reason we differentiate hold and n timesteps is that hold is how
                    # long we want to wait to allow for the action, mainly used for grasping,
                    # whereas n_timesteps determines the number of steps in the path planner.
                    # we check n_timesteps*2 to allow the arm to catch up to the path planner

                    if net.reach["hold_timesteps"] is not None:
                        if net.count >= net.reach["hold_timesteps"]:
                            net.next_reach = True
                    elif net.count > net.reach["n_timesteps"] * 2 and error < 0.07:
                        net.next_reach = True

                net.count += 1

                # toggle the path planner visualization -------------------------------
                if net.path_vis or net.path_vis != interface.viewer.path_vis:
                    if interface.viewer.path_vis:
                        interface.set_mocap_xyz("path_planner_orientation", target[:3])
                        interface.set_mocap_orientation(
                            "path_planner_orientation",
                            quaternion_from_euler(
                                orient[0], orient[1], orient[2], "rxyz"
                            ),
                        )
                    else:
                        interface.set_mocap_xyz(
                            "path_planner_orientation", np.array([0, 0, -100])
                        )
                    net.path_vis = interface.viewer.path_vis

                # print out information to mjviewer -----------------------------------
                interface.viewer.custom_print = (
                    "%s\n" % net.reach["label"]
                    + "Error: %.3fm\n" % error
                    + "Gripper toggle: %i\n" % interface.viewer.gripper
                    + "Dumbbell: %ikg\n"
                    % interface.model.body_mass[
                        interface.model.body_name2id("dumbbell")]
                    + "Gravity: %s" % (interface.viewer.planet)
                    )

                # check if the ADAPT sign should be on --------------------------------
                if not interface.viewer.adapt:
                    model.geom_rgba[adapt_geom_id] = adapt_off
                    interface.set_mocap_xyz("brain", [0, 0, -100])

                # display the planet
                if interface.viewer.planet != net.prev_planet:
                    interface.set_mocap_xyz(net.prev_planet, [0, 0, -100])
                    interface.set_mocap_xyz(interface.viewer.planet, [1, 1, 0.5])
                    interface.set_mocap_xyz('%s_floor' % net.prev_planet, [0, 0, -100])
                    interface.set_mocap_xyz('%s_floor' % interface.viewer.planet, [0, 0, 0.0])
                    net.prev_planet = interface.viewer.planet


            # we made it out of the loop, so the adapt sign should be on! -------------
            model.geom_rgba[adapt_geom_id] = adapt_on
            interface.set_mocap_xyz("brain", [0, 1, .2])

            # if adaptation is on, generate context signal for neural population ------
            feedback = interface.get_feedback()
            context = scale_inputs(
                spherical,
                means,
                variances,
                np.hstack([feedback["q"][:5], feedback["dq"][:5]]),
            )
            training_signal = -net.reach["ctrlr"].training_signal[:5]
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
            # TODO: account for scaling on the transform here
            nengo.Connection(arm[n_input:], conn_learn[ii].learning_rule, synapse=None)

    return net, interface, robot_config


if __name__ == '__main__':
    # if we're running outside of Nengo GUI
    # while 1:
        net, interface, robot_config = demo(backend)
        try:
            if backend == "loihi":
                with nengo_loihi.Simulator(
                    net, target="loihi", hardware_options=dict(snip_max_spikes_per_step=300)
                ) as sim:
                    while 1:
                        try:
                            sim.run(1e5)
                        except RestartMujoco:
                            interface = restart_mujoco(net, interface, robot_config)

            elif backend == "cpu":
                with nengo.Simulator(net) as sim:
                    while 1:
                        try:
                            sim.run(1e5)
                        except RestartMujoco:
                            interface = restart_mujoco(net, interface, robot_config)

        except ExitSim:
            pass

        finally:
            interface.disconnect()
else:
    # if we're running inside Nengo GUI
    try:
        model, interface = demo()
    finally:
        interface.disconnect()
