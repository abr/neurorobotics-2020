import glfw
import mujoco_py
import nengo
import nengo_loihi
import numpy as np
import sys
import time
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

backend = "loihi"
UI = 'keyboard'
demo_mode = False
if len(sys.argv) > 1:
    for arg in sys.argv:
        arg = str(arg)
        if arg == 'cpu':
            backend = 'cpu'
        elif arg == 'loihi':
            backend = 'loihi'
        elif arg == 'keyboard':
            UI = 'keyboard'
        elif arg == 'gamepad':
            UI = 'gamepad'
        elif arg == 'demo':
            demo_mode = True

print("Using %s as backend" % backend)


def initialize_mujoco(robot_config, UI='keyboard'):
    # create our Mujoco interface
    interface = Mujoco(robot_config, dt=0.001, visualize=True)
    interface.connect()
    if UI == 'gamepad':
        interface.viewer.setup_xbox_controller()
    interface.send_target_angles(robot_config.START_ANGLES)

    return interface


def restart_mujoco(net, robot_config, UI):
    net.interface.disconnect()
    glfw.destroy_window(net.interface.viewer.window)
    del net.interface
    time.sleep(.25)
    net.interface = initialize_mujoco(robot_config, UI)
    initialize_interface(net.interface)
    net.interface.set_mocap_xyz(name="target", xyz=net.interface.viewer.target)
    net.model = net.interface.sim.model
    net.data = net.interface.sim.data


def initialize_interface(interface):

    interface.set_mocap_xyz("moon", [0, 0, -100])
    interface.set_mocap_xyz("mars", [0, 0, -100])
    interface.set_mocap_xyz("jupiter", [0, 0, -100])
    interface.set_mocap_xyz("ISS", [0, 0, -100])
    # interface.set_mocap_xyz("moon_floor", [0, 0, -100])
    # interface.set_mocap_xyz("mars_floor", [0, 0, -100])
    # interface.set_mocap_xyz("jupiter_floor", [0, 0, -100])
    # interface.set_mocap_xyz("ISS_floor", [0, 0, -100])

    interface.set_mocap_xyz("earth", [1, 1, 0.5])
    interface.set_mocap_xyz("obstacle", [0, 0, -100])
    interface.set_mocap_xyz("path_planner", [0, 0, -100])
    interface.set_mocap_xyz("target_orientation", [0, 0, -100])
    interface.set_mocap_xyz("path_planner_orientation", [0, 0, -100])
    interface.set_mocap_xyz("elbow", [0, 0, -100])

    interface.viewer.target = np.array([-0.4, 0.5, 0.4])
    interface.target_moved = True

    hide_hotkeys(interface)


offset = 0.08
offset_x_right = np.array([offset, 0, 0])
offset_x_left = -1 * offset_x_right
offset_y_right = np.array([0, offset, 0])
offset_y_left = -1 * offset_y_right
offset_z_right = np.array([0, 0, offset])
offset_z_left = -1 * offset_z_right
def display_hotkeys(interface):
    target_xyz = interface.get_xyz('target')
    # move along x
    interface.set_mocap_xyz("r-arrow-double", target_xyz + offset_x_right)
    interface.set_mocap_xyz("l-arrow-double", target_xyz + offset_x_left)
    # move along y
    interface.set_mocap_xyz("u-arrow-double", target_xyz + offset_y_left)
    interface.set_mocap_xyz("d-arrow-double", target_xyz + offset_y_right)
    # move along z
    interface.set_mocap_xyz("alt1", target_xyz + offset_z_left + offset_x_left/3)
    interface.set_mocap_xyz("d-arrow-double2", target_xyz + offset_z_left + offset_x_right/3)
    interface.set_mocap_xyz("alt2", target_xyz + offset_z_right + offset_x_left/3)
    interface.set_mocap_xyz("u-arrow-double2", target_xyz + offset_z_right + offset_x_right/3)


def hide_hotkeys(interface):
    interface.set_mocap_xyz("alt1", [0, 0, -100])
    interface.set_mocap_xyz("alt2", [0, 0, -100])
    interface.set_mocap_xyz("l-arrow-double", [0, 0, -100])
    interface.set_mocap_xyz("r-arrow-double", [0, 0, -100])
    interface.set_mocap_xyz("u-arrow-double", [0, 0, -100])
    interface.set_mocap_xyz("d-arrow-double", [0, 0, -100])
    interface.set_mocap_xyz("u-arrow-double2", [0, 0, -100])
    interface.set_mocap_xyz("d-arrow-double2", [0, 0, -100])


def demo(backend, UI, demo_mode):
    rng = np.random.RandomState(9)

    n_input = 10
    n_output = 5

    n_neurons = 1000
    n_ensembles = 10
    pes_learning_rate = 1e-5 if backend == "cpu" else 1e-5
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

    net = nengo.Network(seed=seed)
    # Set the default neuron type for the network
    net.config[nengo.Ensemble].neuron_type = nengo.LIF()

    net.interface = initialize_mujoco(robot_config, UI)
    net.model = net.interface.sim.model
    net.data = net.interface.sim.data

    object_xyz = np.array([-0.5, 0.0, 0.02])

    reach_list = gen_reach_list(robot_config, object_xyz, net.interface.viewer.target)

    # max grip force
    max_grip = 8
    fkp = 144
    fkv = 15

    fingers = ["joint_thumb", "joint_index", "joint_pinky"]

    green = [0, 0.9, 0, 0.5]
    red = [0.9, 0, 0, 0.5]

    # yellow
    # adapt_on = [0.9, 0.75, 0.1, 1]
    # silver
    adapt_on = [0.77, 0.79, 0.81, 1]
    adapt_off = [0.5, 0.5, 0.5, 0.1]

    OUTPUT_ZEROS = np.zeros(n_input + n_output)
    adapt_geom_id = net.model.geom_name2id("adapt")
    target_geom_id = net.model.geom_name2id("target")
    net.weight_label_names = ["1lb", "2_5lb", "5lb"]
    dumbbell_body_id = net.model.body_name2id("dumbbell")

    net.path_vis = False  # start out not displaying path planner target
    net.gravities = {
        "earth": (np.array([0, 0, -9.81, 0, 0, 0]), "9_81N"),
        "moon": (np.array([0, 0, -1.62, 0, 0, 0]), "1_62N"),
        "mars": (np.array([0, 0, -3.71, 0, 0, 0]), "3_71N"),
        "jupiter": (np.array([0, 0, -24.92, 0, 0, 0]), "24_92N"),
        "ISS": (np.array([0, 0, 0, 0, 0, 0]), "0_00N"),
    }
    net.bodies = ["link1", "link2", "link3", "link4", "link5", "link6", "dumbbell"]
    net.base_gravity = np.hstack((net.interface.model.opt.gravity, np.zeros(3)))
    net.dumbbell_masses = [0.45, 1.13, 2.26]  # converted from 1, 5, 10 lbs

    initialize_interface(net.interface)

    def initialize_net(net):
        net.reach_index = -1
        net.next_reach = True
        net.reach = None
        net.u_gripper_prev = np.zeros(3)
        net.u = np.zeros(robot_config.N_JOINTS + 3)
        net.prev_planet = "earth"
        net.auto_reach_index = 0
        net.auto_target_index = 0
        net.count = 0
        net.at_target = 0
        net.dumbbell_mass_index = 0

    with net:

        net.auto_reach_modes = [
            "reach_target",
            "pick_up",
            "reach_target",
            "reach_target",
        ]

        net.auto_targets = np.array(
            [
                [-0.40, 0.50, 0.40],
                # [-0.10,  0.50,  0.50],
                [0.39, 0.50, 0.29],
                [ 0.39, -0.20,  0.39],
                # [ 0.09, -0.49,  0.69],
                # [ 0.10, -0.40,  0.50],
                [-0.30, -0.49,   0.70],
                # [-0.49, -0.30,  0.70 ],
                # [-0.39, -0.20,  0.60],
                [-0.40, 0.10, 0.30],
            ]
        )

        # make the target offset from that start position
        net.interface.set_mocap_xyz(name="target", xyz=net.interface.viewer.target)
        initialize_net(net)
        net.demo_mode = demo_mode
        net.path_vis = False  # start out not displaying path planner target
        net.pos = None

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
                if UI == 'gamepad':
                    viewer.xbox_callback()
                ran_at_least_once = True

                if viewer.exit:
                    glfw.destroy_window(viewer.window)
                    raise ExitSim()

                if net.demo_mode and viewer.key_pressed:
                    net.demo_mode = False
                    viewer.reach_mode_changed = True

                # if switching to demo script / auto mode, reset Mujoco
                if viewer.toggle_demo:
                    print("Toggle demo")
                    net.demo_mode = not net.demo_mode
                    if net.demo_mode:
                        print("Switching to demo mode")
                        viewer.restart_sim = True
                    viewer.toggle_demo = False

                if viewer.restart_sim:
                    initialize_net(net)
                    # raise RestartMujoco()
                    restart_mujoco(net, robot_config, UI)
                    interface = net.interface
                    viewer = interface.viewer
                    model = net.model
                    data = net.data

                if viewer.display_hotkeys:
                    display_hotkeys(interface)
                else:
                    hide_hotkeys(interface)

                if net.demo_mode:
                    net.reach_mode = net.auto_reach_modes[net.auto_reach_index]
                    viewer.target = net.auto_targets[net.auto_target_index]
                else:
                    net.reach_mode = viewer.reach_mode
                    net.auto_reach_index = 0
                    net.auto_target_index = 0

                if viewer.reach_mode_changed:
                    print("Reach mode changed")
                    net.next_reach = True
                    net.reach_index = -1
                    viewer.reach_mode_changed = False

                # if the reaching mode has changed, recalculate reaching parameters ---
                if net.next_reach:
                    print("Generating next reach")
                    net.reach_index += 1
                    if net.reach_index >= len(reach_list[net.reach_mode]):
                        if net.reach_mode != "reach_target" and net.demo_mode:
                            net.auto_reach_index += 1
                            if net.auto_reach_index == 3:
                                viewer.adapt = True
                            else:
                                viewer.adapt = False
                            net.auto_target_index = 0
                        viewer.reach_mode = "reach_target"
                        net.reach_mode = viewer.reach_mode
                        net.reach_index = -1

                    net.reach = reach_list[net.reach_mode][net.reach_index]
                    net.u_gripper_prev = np.zeros(3)

                    feedback = interface.get_feedback()

                    # if we're reaching to target, update with user changes
                    if net.reach_mode == "reach_target":
                        net.reach["target_pos"] = viewer.target

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

                # get arm feedback
                feedback = interface.get_feedback()
                hand_xyz = robot_config.Tx("EE")

                # update our path planner position and orientation --------------------
                if net.reach_mode == "reach_target":
                    error = np.linalg.norm(
                        hand_xyz - viewer.target + net.reach["offset"]
                    )
                    if viewer.target_moved:
                        net.n_timesteps = net.reach["n_timesteps"] - net.count
                        net.trajectory_planner.generate_path(
                            position=net.pos,
                            target_pos=viewer.target + net.reach["offset"],
                        )
                        interface.set_mocap_xyz("target", viewer.target)
                        viewer.target_moved = False
                    if error < 0.05:  # when close enough, don't use path planner
                        net.pos = viewer.target + net.reach["offset"]
                        net.vel = np.zeros(3)
                    else:
                        net.pos, net.vel = net.trajectory_planner.next()
                    orient = np.zeros(3)

                else:
                    error = np.linalg.norm((hand_xyz - net.target_data["approach_pos"]))
                    net.pos, net.vel = net.trajectory_planner.next()
                    orient = net.orientation_planner.next()

                # check if the user moved the target ----------------------------------
                if viewer.target_moved:
                    # update visualization of target
                    interface.set_mocap_xyz("target", viewer.target)
                    viewer.target_moved = False

                if net.pos is None:
                    # if net.pos hasn't been set somehow make sure it's set
                    net.pos = hand_xyz
                target = np.hstack([net.pos, orient])

                # apply force to elbow if one has been set! ---------------------------
                if viewer.move_elbow:
                    interface.set_mocap_xyz(
                        "elbow", robot_config.Tx("joint2", object_type="joint")
                    )
                else:
                    interface.set_mocap_xyz("elbow", [0, 0, -100])
                interface.set_external_force("ring2", viewer.elbow_force)

                # calculate our osc control signal ------------------------------------
                net.u[: robot_config.N_JOINTS] = net.reach["ctrlr"].generate(
                    q=feedback["q"], dq=feedback["dq"], target=target
                )

                if viewer.adapt:
                    # adaptive signal added (no signal for last joint)
                    net.u[: robot_config.N_JOINTS - 1] += u_adapt * adapt_scale

                # get our gripper command ---------------------------------------------
                finger_q = np.array(
                    [net.data.qpos[net.model.get_joint_qpos_addr(finger)] for finger in fingers]
                )
                finger_dq = np.array(
                    [net.data.qvel[net.model.get_joint_qpos_addr(finger)] for finger in fingers]
                )

                u_gripper = fkp * (net.reach["grasp_pos"] - finger_q) - fkv * finger_dq
                u_gripper = (
                    net.reach["f_alpha"] * u_gripper
                    + (1 - net.reach["f_alpha"]) * net.u_gripper_prev
                )
                u_gripper = np.clip(u_gripper, a_max=max_grip, a_min=-max_grip)
                net.u_gripper_prev[:] = np.copy(u_gripper)
                net.u[robot_config.N_JOINTS :] = u_gripper * viewer.gripper

                # set the world gravity
                gravity = net.gravities[viewer.planet][0]

                # incorporate dumbbell mass change
                if net.dumbbell_mass_index != viewer.dumbbell_mass_index:
                    net.dumbbell_mass_index = (viewer.dumbbell_mass_index
                                               % len(net.dumbbell_masses))
                    viewer.dumbbell_mass_index = net.dumbbell_mass_index
                    net.model.body_mass[net.model.body_name2id('dumbbell')] = net.dumbbell_masses[net.dumbbell_mass_index]

                for ii, name in enumerate(net.weight_label_names):
                    if ii == net.dumbbell_mass_index:
                        # model.geom_rgba[net.weight_label_ids[ii]] = adapt_on
                        position = (np.array([0.3, 1, 0.3])
                                    - net.model.body_ipos[net.model.body_name2id(name)])
                        interface.set_mocap_xyz(name, position)
                    else:
                        interface.set_mocap_xyz(name, [0, 0, -100])

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
                if net.reach_mode == "reach_target":
                    if error < net.reach["error_thresh"]:
                        net.model.geom_rgba[target_geom_id] = green
                    else:
                        net.model.geom_rgba[target_geom_id] = red

                    if net.demo_mode:
                        # TODO: if adapting add a hold timesteps
                        hold_timesteps = 0
                        if error < 0.02 or net.count >= hold_timesteps + 2000:
                            # add to our at target counter
                            if error < 0.02:
                                net.at_target += 1
                            # if we maxed out our hold + timesteps, or we've been at target
                            if net.count >= hold_timesteps + 2000 or net.at_target > 250:
                                net.next_reach = True
                                print("maxed out timesteps")

                                # if at our last target, go to the next part of the reach
                                if net.auto_target_index == len(net.auto_targets) - 1:
                                    # if at last part of reach, restart
                                    if (
                                        net.auto_reach_index
                                        == len(net.auto_reach_modes) - 1
                                    ):
                                        print(
                                            "last part and last target, restart auto mode"
                                        )
                                        initialize_net(net)
                                        # raise RestartMujoco()
                                        restart_mujoco(net, robot_config, UI)
                                        interface = net.interface
                                        viewer = interface.viewer
                                        model = net.model
                                        data = net.data
                                    else:
                                        print("going to next reach mode")
                                        net.auto_reach_index += 1
                                        if net.auto_reach_index == 3:
                                            viewer.adapt = True
                                        else:
                                            viewer.adapt = False
                                        net.auto_target_index = 0
                                # otherwise, go to next target
                                else:
                                    print("going to next target")
                                    net.auto_target_index += 1
                                viewer.target_moved = True
                        else:
                            net.at_target = 0

                else:
                    net.model.geom_rgba[target_geom_id] = red

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
                if net.path_vis or net.path_vis != viewer.path_vis:
                    if viewer.path_vis:
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
                    net.path_vis = viewer.path_vis

                # print out information to mjviewer -----------------------------------
                viewer.custom_print = (
                    "%s\n" % net.reach["label"]
                    + "Error: %.3fm\n" % error
                    # + "Gripper toggle: %i\n" % viewer.gripper
                    # + "Dumbbell: %i lbs\n"
                    # convert from kg to lbs for printout
                    # % np.round(interface.model.body_mass[dumbbell_body_id] / 2.2)
                    # + "Gravity: %s\n" % (viewer.planet)
                    + "Demonstration Mode: %s\n" % net.demo_mode
                    # + "Interface Mode: %s\n" % viewer.reach_type
                )

                # check if the ADAPT sign should be on --------------------------------
                if not viewer.adapt:
                    net.model.geom_rgba[adapt_geom_id] = adapt_off
                    interface.set_mocap_xyz("brain", [0, 0, -100])

                # display the planet
                if viewer.planet != net.prev_planet:
                    interface.set_mocap_xyz(net.prev_planet, [0, 0, -100])
                    interface.set_mocap_xyz(viewer.planet, [1, 1, 0.5])
                    # interface.set_mocap_xyz("%s_floor" % net.prev_planet, [0, 0, -100])
                    # interface.set_mocap_xyz(
                    #     "%s_floor" % viewer.planet, [0, 0, 0.0]
                    # )
                    net.prev_planet = viewer.planet

                for key in net.gravities.keys():
                    name = net.gravities[key][1]
                    if key == viewer.planet:
                        position = (np.array([0.3, 1, 0.45])
                                    - net.model.body_ipos[net.model.body_name2id(name)])
                        interface.set_mocap_xyz(name, position)
                    else:
                        interface.set_mocap_xyz(name, [0, 0, -100])

            # we made it out of the loop, so the adapt sign should be on! -------------
            net.model.geom_rgba[adapt_geom_id] = adapt_on
            interface.set_mocap_xyz("brain", [0, 0.75, 0.2])

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

    return net, robot_config


if __name__ == "__main__":
    # if we're running outside of Nengo GUI
    # while 1:
    net, robot_config = demo(backend, UI, demo_mode)
    net.UI = UI
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
else:
    # if we're running inside Nengo GUI
    try:
        model, robot_config = demo()
    finally:
        interface.disconnect()
