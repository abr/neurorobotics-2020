import nengo

# import nengo_loihi
import numpy as np
import timeit
import traceback

from nengo_loihi import decode_neurons

from abr_control.controllers import OSC, Damping
from abr_control.arms.mujoco_config import MujocoConfig as arm
from abr_control.interfaces.mujoco import Mujoco
from abr_control.utils import transformations

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
    calculate_reach_params,
)

from reach_list import gen_reach_list

rng = np.random.RandomState(9)

n_input = 10
n_output = 5

n_neurons = 1000
n_ensembles = 1
pes_learning_rate = 5e-5
seed = 0
spherical = True  # project the input onto the surface of a D+1 hypersphere
if spherical:
    n_input += 1

means = ([0.12, 2.14, 1.87, 4.32, 0.59, 0.12, -0.38, -0.42, -0.29, 0.36],)
variances = ([0.08, 0.6, 0.7, 0.3, 0.6, 0.08, 1.4, 1.6, 0.7, 1.2],)

# synapse time constants
tau_input = 0.012  # on input connection
tau_training = 0.012  # on the training signal
tau_output = 0.012  # on the output from the adaptive ensemble
# NOTE: the time constant on the neural activity used in the learning
# connection is the default 0.005, and can be set by specifying the
# pre_synapse parameter inside the PES rule instantiation

# set up neuron intercepts
intercepts_bounds = [-0.3, 0.1]
intercepts_mode = 0.1

intercepts_dist = AreaIntercepts(
    dimensions=n_input,
    base=Triangular(intercepts_bounds[0], intercepts_mode, intercepts_bounds[1]),
)
intercepts = intercepts_dist.sample(n=n_neurons * n_ensembles, rng=rng)
intercepts = intercepts.reshape(n_ensembles, n_neurons)

# TODO: add weight initialization ability to nengo_loihi
# weights = np.zeros((n_ensembles, n_output, n_neurons))
# print("Initializing connection weights to all zeros")

np.random.seed = seed
encoders_dist = ScatteredHypersphere(surface=True)
encoders = encoders_dist.sample(n_neurons * n_ensembles, n_input, rng=rng)
encoders = encoders.reshape(n_ensembles, n_neurons, n_input)

# initialize our robot config for the jaco2
robot_config = arm("jaco2_gripper")

# create our Mujoco interface
interface = Mujoco(robot_config, dt=0.001, visualize=True)
interface.connect()
interface.send_target_angles(robot_config.START_ANGLES)


net = nengo.Network(seed=seed)
# Set the default neuron type for the network
net.config[nengo.Ensemble].neuron_type = nengo.LIF()
# net.config[nengo.Connection].synapse = None

object_xyz = np.array([-0.5, 0.0, 0.02])
deposit_xyz = np.array([-0.4, 0.5, 0.4])
adapt_text = np.array([0, 1, 0])

scale = 0.05
xlim = [-0.5, 0.5]
ylim = [-0.5, 0.5]
zlim = [0.0, 0.7]


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

OUTPUT_ZEROS = np.zeros(n_input + n_output)
target_geom_id = interface.sim.model.geom_name2id("target")

with net:

    net.old_reach_mode = None
    net.reach_index = -1
    net.next_reach = False
    net.reach = None
    net.u_gripper_prev = np.zeros(3)
    net.adapt = False  # start out with adaptation off
    net.path_vis = False  # start out not displaying path planner target
    net.u = np.zeros(robot_config.N_JOINTS + 3)

    net.final_xyz = deposit_xyz
    # make the target offset from that start position
    interface.set_mocap_xyz(name="target", xyz=net.final_xyz)

    def arm_func(t, u_adapt):
        net.reach_mode = interface.viewer.reach_mode

        if net.reach_mode != net.old_reach_mode:
            print("Reach mode changed")
            net.next_reach = True
            net.old_reach_mode = net.reach_mode
            net.reach_index = -1

        # if the reaching mode has changed, recalculate reaching parameters -----------
        if net.next_reach:
            print("Generating next reach")
            net.reach_index += 1
            if net.reach_index >= len(reach_list[net.reach_mode]):
                interface.viewer.reach_mode = "reach_target"
                net.reach_mode = interface.viewer.reach_mode
                net.reach_index = 0

            net.reach = reach_list[net.reach_mode][net.reach_index]
            net.u_gripper_prev = np.zeros(3)

            (
                net.reach,
                net.trajectory_planner,
                net.orientation_planner,
                net.target_data,
            ) = calculate_reach_params(
                net.reach, net.reach_mode, net.final_xyz, interface, robot_config
            )

            net.next_reach = False
            net.count = 0

        # get the target location from the interface
        net.old_final_xyz = net.final_xyz

        # check if the user moved the target ------------------------------------------
        change = np.array(
            [
                interface.viewer.target_x,
                interface.viewer.target_y,
                interface.viewer.target_z,
            ]
        )
        if not np.allclose(change, 0):
            net.final_xyz = net.final_xyz + scale * change
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

        # update our path planner position and orientation ----------------------------
        if net.reach_mode == "reach_target":
            error = np.linalg.norm(hand_xyz - net.final_xyz + net.reach["offset"])
            if error < 0.05:  # when close enough, don't use path planner
                net.pos = net.final_xyz + net.reach["offset"]
                net.vel = np.zeros(3)
            else:
                if not np.allclose(net.final_xyz, net.old_final_xyz, atol=1e-5):
                    # if the target has moved, regenerate the path planner
                    net.trajectory_planner.reset(
                        position=net.pos,
                        target_pos=(net.final_xyz + net.reach["offset"]),
                    )
                net.pos, net.vel = net.trajectory_planner._step(error=error)
            orient = np.zeros(3)

        else:
            error = np.linalg.norm((hand_xyz - net.target_data["approach_pos"]))
            net.pos, net.vel = net.trajectory_planner.next()
            orient = net.orientation_planner.next()

        target = np.hstack([net.pos, orient])

        # calculate our osc control signal -------------------------------------------- 
        net.u[:robot_config.N_JOINTS] = net.reach["ctrlr"].generate(
            q=feedback["q"], dq=feedback["dq"], target=target
        )

        # if the adaptation state is toggled ------------------------------------------
        if net.adapt != interface.viewer.adapt:
            if interface.viewer.adapt:
                interface.set_mocap_xyz("adapt_on", adapt_text)
                interface.set_mocap_xyz("adapt_off", [0, 0, -1])
            else:
                interface.set_mocap_xyz("adapt_off", adapt_text)
                interface.set_mocap_xyz("adapt_on", [0, 0, -1])
            net.adapt = interface.viewer.adapt

        if net.adapt:
            # adaptive signal added (no signal for last joint)
            net.u[:robot_config.N_JOINTS - 1] += u_adapt

        # get our gripper command -----------------------------------------------------
        finger_q = np.array(
            [
                interface.sim.data.qpos[interface.sim.model.get_joint_qpos_addr(finger)]
                for finger in fingers
            ]
        )
        finger_dq = np.array(
            [
                interface.sim.data.qvel[interface.sim.model.get_joint_qpos_addr(finger)]
                for finger in fingers
            ]
        )

        # NOTE interface lets you toggle gripper status with the 'n' key
        # TODO remove the interface gripper control for actual demo
        u_gripper = fkp * (net.reach["grasp_pos"] - finger_q) - fkv * finger_dq
        u_gripper = (
            net.reach["f_alpha"] * u_gripper
            + (1 - net.reach["f_alpha"]) * net.u_gripper_prev
        )
        u_gripper = np.clip(u_gripper, a_max=max_grip, a_min=-max_grip)
        net.u_gripper_prev[:] = np.copy(u_gripper)
        net.u[robot_config.N_JOINTS:] = u_gripper * interface.viewer.gripper

        # send to mujoco, stepping the sim forward
        interface.send_forces(net.u)

        # ----------------
        if net.reach_mode == "reach_target":
            if error < net.reach["error_thresh"]:
                interface.sim.model.geom_rgba[target_geom_id] = green
            else:
                interface.sim.model.geom_rgba[target_geom_id] = red
        else:
            interface.sim.model.geom_rgba[target_geom_id] = red

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

        # toggle the path planner visualization ---------------------------------------
        if net.path_vis or net.path_vis != interface.viewer.path_vis:
            if interface.viewer.path_vis:
                interface.set_mocap_xyz("path_planner_orientation", target[:3])
                interface.set_mocap_orientation(
                    "path_planner_orientation",
                    transformations.quaternion_from_euler(
                        orient[0], orient[1], orient[2], "rxyz"
                    ),
                )
            else:
                interface.set_mocap_xyz(
                    "path_planner_orientation", np.array([0, 0, -1])
                )
            net.path_vis = interface.viewer.path_vis

        # print out information to mjviewer -------------------------------------------
        interface.viewer.custom_print = "%s\nerror: %.3fm\nGripper toggle: %i" % (
            net.reach["label"],
            error,
            interface.viewer.gripper,
        )

        # if adaptation is on, generate context signal for neural population ----------
        if net.adapt:
            feedback = interface.get_feedback()
            context = scale_inputs(
                spherical,
                means,
                variances,
                np.hstack([feedback["q"][:5], feedback["dq"][:5]]),
            )
            training_signal = -net.reach['ctrlr'].training_signal[:5]
            output_signal = np.hstack([context.flatten(), training_signal.flatten()])
        else:
            output_signal = OUTPUT_ZEROS

        # TODO: scale the training signal here
        return output_signal

    # ---------------------------------------------------------------------------------

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
            [np.eye(n_output), -(np.eye(n_output))] * output_decodeneurons.pairs_per_dim
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

try:
    # with nengo_loihi.Simulator(
    #         net,
    #         target='loihi',
    #         hardware_options=dict(snip_max_spikes_per_step=300)) as sim:
    with nengo.Simulator(net) as sim:
        sim.run(0.01)
        start = timeit.default_timer()
        sim.run(20)
        print("Run time: %0.5f" % (timeit.default_timer() - start))
        print("timers: ", sim.timers["snips"])
finally:
    interface.disconnect()
