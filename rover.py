"""
TODO
To run the demo with Nengo running on cpu:
    python nengo_rover.py cpu

To run the demo with Nengo on loihi
    NXSDKHOST=loihighrd python nengo_rover.py
"""
import glfw
import matplotlib.pyplot as plt
import numpy as np
import sys
import timeit
import tensorflow as tf

import mujoco_py
import nengo
import nengo_dl
import nengo_loihi

from abr_control.arms.mujoco_config import MujocoConfig
from abr_control.interfaces.mujoco import Mujoco
from abr_analyze import DataHandler

from rover_vision import RoverVision
from loihi_utils import LoihiRectifiedLinear, HetDecodeNeurons


class ExitSim(Exception):
    print("Ending simulation")
    pass


# set up parameters that depend on cpu or loihi as the backend
backend = "cpu"
if len(sys.argv) > 1:
    if sys.argv[1] == "loihi":
        backend = "loihi"
print("Using %s as backend" % backend)

generate_data = False  # should training / validation data be created
plot_mounted_camera_freq = None  # how often to plot image from cameras
collect_ground_truth = True  # track ideal vision and control network output
motor_neuron_type = nengo_loihi.neurons.LoihiSpikingRectifiedLinear()


def demo(backend="cpu", test_name="validation_0000", neural_vision=True):
    if backend == "loihi":
        nengo_loihi.set_defaults()

    seed = 9
    rng = np.random.RandomState(seed)
    # target generation limits
    dist_limit = [0.25, 3.5]
    angle_limit = [-np.pi, np.pi]

    # data collection parameters in steps (1ms per step)
    render_frequency = 1
    save_frequency = 10
    reaching_steps = 500
    total_images_saved = 100000
    max_time_to_target = 10000

    render_size = [32, 32]  # resolution of images from cameras
    res = [render_size[0], render_size[1] * 4]  # vstack feedback from 4 cameras
    subpixels = res[0] * res[1] * 3  # * 3 channels

    dat = DataHandler()
    dat.save(
        data={
            "render_frequency": render_frequency,
            "save_frequency": save_frequency,
            "reaching_steps": reaching_steps,
            "max_time_to_target": max_time_to_target,
            "render_size": render_size,
            "dist_limit": dist_limit,
            "angle_limit": angle_limit,
        },
        save_location="%s/params" % test_name,
        overwrite=True,
    )

    rover = MujocoConfig(folder="", xml_file="rover.xml", use_sim_state=True)

    net = nengo.Network()
    # create our Mujoco interface
    net.interface = Mujoco(rover, dt=0.001, create_offscreen_rendercontext=True)
    net.interface.connect()
    # shorthand
    interface = net.interface
    mujoco_model = interface.sim.model
    mujoco_data = interface.sim.data

    # get names of turning axel and back motors
    joint_names = ["steering_wheel"]
    interface.joint_pos_addrs = [
        mujoco_model.get_joint_qpos_addr(name) for name in joint_names
    ]
    interface.joint_vel_addrs = [
        mujoco_model.get_joint_qvel_addr(name) for name in joint_names
    ]

    # get target object id so we can change its colour
    target_geom_id = mujoco_model.geom_name2id("target")
    green = [0, 0.9, 0, 1]
    red = [0.9, 0, 0, 1]

    # set up the target position
    net.target = np.array([-0.0, 0.0, 0.2])

    vision = RoverVision(res=res, seed=0)

    with net:
        net.count = 0
        net.image_count = 0
        net.target_count = 0
        net.time_to_target = 0
        net.predicted_xy = [0, 0]

        # track position
        net.target_track = []
        net.rover_track = []
        net.turnerror_track = []
        net.q_track = []
        net.qdes_track = []
        net.input_track = []
        net.u_track = []
        net.localtarget_track = []

        image_input = np.zeros((res[0], res[1], 3))

        if neural_vision:
            converter_params = {
                "swap_activations": {
                    tf.nn.relu: nengo_loihi.neurons.LoihiSpikingRectifiedLinear()
                },
                "scale_firing_rates": 400,
            }
            visionsim, visionnet_1 = vision.convert(converter_params, add_probes=False,)
            visionsim.load_params("epoch_41")
            with visionsim:
                visionsim.freeze_params(visionnet_1)

                vision.nengo_innode.size_in = vision.subpixels
                vision.nengo_innode.output = None

        def simulate_world_func(t, x):
            # NOTE: radius of nengo_loihi decode_neurons is 1
            # so output from steering_wheel is in that range, scale back up here
            # turn = x[0] * 2 * np.pi * 5000  # also scale by kp = 5000
            turn = x[0] * 2 * 5000  # also scale by kp = 5000
            wheels = x[1] * 20 * 30
            u = np.array([turn, wheels])
            net.u_track.append(u)
            prediction = x[2:4]

            # get our image data
            if net.count % render_frequency == 0:
                for ii, jj in enumerate([4, 1, 3, 2]):
                    interface.offscreen.render(res[0], res[0], camera_id=jj)
                    image_input[
                        :, res[0] * ii : res[0] * (ii + 1)
                    ] = interface.offscreen.read_pixels(res[0], res[0])[0]
                    vision.image_input[:] = image_input.flatten()

            if (
                plot_mounted_camera_freq is not None
                and net.count % plot_mounted_camera_freq == 0
            ):
                # plot the mounted camera output every so often
                plt.figure()
                a = plt.subplot(1, 1, 1)
                a.imshow(vision.image_input.reshape((res[0], res[1], 3)) / 255)
                plt.show()

            error = net.target - rover.Tx("EE")
            mujoco_model.geom_rgba[target_geom_id] = (
                green if np.linalg.norm(error) < 0.02 else red
            )

            # error is in global coordinates, want it in local coordinates for rover
            # body_xmat will take from local coordinates to global
            R_raw = (
                np.copy(mujoco_data.body_xmat[mujoco_model.body_name2id("EE")])
                .reshape(3, 3)
                .T
            )  # R.T = R^-1
            # rotate it so the steering wheels point forward along y
            theta = 3 * np.pi / 2
            R90 = np.array(
                [
                    [np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1],
                ]
            )
            R = np.dot(R90, R_raw)

            # we also want the egocentric velocity of the rover
            body_com_vel_raw = mujoco_data.cvel[mujoco_model.body_name2id("base_link")][3:]
            body_com_vel = np.dot(R, body_com_vel_raw)

            local_target = np.dot(R, error)
            if net.count % save_frequency == 0:
                if generate_data:
                    print("Target count: %i " % (int(net.count / save_frequency)))
                    save_data = {
                        "rgb": image_input,
                        "target": local_target,
                    }

                    dat.save(
                        data=save_data,
                        save_location="%s/data/%04d" % (test_name, net.image_count),
                        overwrite=True,
                    )
                    print("%s/data/%04d" % (test_name, net.image_count))
                net.image_count += 1

            if interface.viewer.exit or net.image_count == total_images_saved:
                glfw.destroy_window(interface.viewer.window)
                raise ExitSim

            rover_xyz = rover.Tx("EE")
            dist = np.linalg.norm(rover_xyz - net.target)
            if dist < 0.2 or net.time_to_target > max_time_to_target or net.count == 0:
                # generate a new target at least .5m away from current position
                while dist < 0.5:
                    phi = np.random.uniform(low=angle_limit[0], high=angle_limit[1])
                    radius = 1
                    net.target = [np.cos(phi) * radius, np.sin(phi) * radius, 0.2]
                    dist = np.linalg.norm(rover_xyz - net.target)

                interface.set_mocap_xyz("target", net.target)
                net.target_count += 1
                net.time_to_target = 0

            # send to mujoco, stepping the sim forward --------------------------------
            interface.send_forces(u)

            feedback = interface.get_feedback()
            output_signal = np.array(
                [feedback["q"][0], local_target[0], local_target[1],]
            )

            # scale down for sending into motor_control ensemble
            output_signal[:3] = output_signal[:3] / 0.5

            net.count += 1
            net.time_to_target += 1

            # track data
            net.target_track.append(net.target)
            net.rover_track.append(np.copy(rover_xyz))
            net.localtarget_track.append(np.copy(local_target / np.pi))

            return np.hstack((output_signal, vision.image_input))

        # -----------------------------------------------------------------------------
        mujoco_node = nengo.Node(
            simulate_world_func, size_in=4, size_out=3 + vision.subpixels, label="mujoco"
        )

        # --- set up ensemble to calculate acceleration
        max_dist = 0.3

        def accel_function(x):
            error = x * np.pi / -max_dist  # scale back up error signal from vision
            return min(np.linalg.norm(error), 1) * np.sign(error[1]) * -1

        motor_control_accel = nengo.Ensemble(
            neuron_type=motor_neuron_type,
            n_neurons=4096,
            dimensions=2,
            max_rates=nengo.dists.Uniform(175, 220),
            radius=np.sqrt(2),
            label="motor_control_accel",
        )

        nengo.Connection(
            motor_control_accel, mujoco_node[1], function=lambda x: accel_function(x),
        )

        # --- set up ensemble to calculate torque to apply to steering wheel
        def steer_function(x, track=False):
            if net.count > 0 and track:
                net.input_track.append(x)

            error = x[1:] * np.pi  # take out the error signal from vision
            q = x[0] * 0.5  # scale normalized input back to original range

            # arctan2 input set this way to account for different alignment
            # of (x, y) axes of rover and the environment
            turn_des = np.arctan2(-error[0], abs(error[1]))
            u = (turn_des - q) / 2  # divide by 2 to get in -1 to 1 ish range

            # record input for finding mean and variance values
            if net.count > 0 and track:
                net.turnerror_track.append(turn_des - q)
                net.q_track.append(q)
                net.qdes_track.append(turn_des)

            return u

        motor_control_turn = nengo.Ensemble(
            neuron_type=motor_neuron_type,
            n_neurons=4096,
            dimensions=3,
            max_rates=nengo.dists.Uniform(175, 220),
            radius=np.sqrt(3),
            label="motor_control_turn",
        )

        nengo.Connection(
            motor_control_turn, mujoco_node[0], function=lambda x: steer_function(x),
        )

        relay_motor_input_turn = nengo.Node(size_in=3, label="relay_motor_input")
        nengo.Connection(relay_motor_input_turn, motor_control_turn, synapse=None)

        # -- send motor feedback to motor control
        nengo.Connection(mujoco_node[0], relay_motor_input_turn[0], synapse=None)

        # --- connect up vision to motor ensembles and mujoco node
        vision_synapse = 0.01
        if neural_vision:
            nengo.Connection(
                vision.nengo_output, motor_control_accel, synapse=vision_synapse
            )

            # send image input in to vision system
            nengo.Connection(mujoco_node[3:], vision.nengo_innode, synapse=None)

            nengo.Connection(
                vision.nengo_output, relay_motor_input_turn[1:], synapse=vision_synapse
            )
            nengo.Connection(vision.nengo_output, mujoco_node[2:4], synapse=vision_synapse)
        else:
            # connect up turning
            nengo.Connection(
                mujoco_node[1:3],
                motor_control_accel,
                synapse=vision_synapse,
                transform=1 / np.pi,
            )
            nengo.Connection(
                mujoco_node[1:3],
                relay_motor_input_turn[1:],
                synapse=vision_synapse,
                transform=1 / np.pi,
            )
            nengo.Connection(mujoco_node[1:3], mujoco_node[2:4], synapse=vision_synapse)

        if collect_ground_truth:
            # create a direct mode ensemble performing the same calculation for debugging
            motor_control_direct = nengo.Ensemble(
                neuron_type=nengo.Direct(),
                n_neurons=1,
                dimensions=3,
                label="motor_control_direct",
            )

            dead_end = nengo.Node(size_in=2, label="dead_end")
            # send motor signal to mujoco
            nengo.Connection(
                motor_control_direct,
                dead_end[1],
                function=lambda x: accel_function(x[1:]),
                transform=[[20 * 30]],
                synapse=0,
            )

            nengo.Connection(
                motor_control_direct,
                dead_end[0],
                function=lambda x: steer_function(x, track=True),
                transform=[[2 * 5000]],
                synapse=0,
            )

            # send vision prediction to motor control
            if neural_vision:
                nengo.Connection(
                    vision.nengo_output,
                    motor_control_direct[1:],
                    synapse=vision_synapse,
                )
            else:
                nengo.Connection(
                    mujoco_node[1:3],
                    motor_control_direct[1:],
                    synapse=vision_synapse,
                    transform=1 / np.pi * 0.5,
                )
            # send motor feedback to motor control
            nengo.Connection(mujoco_node[0], motor_control_direct[0], synapse=None)

            net.dead_end_probe = nengo.Probe(dead_end)

        if backend == "loihi":
            nengo_loihi.add_params(net)

            if neural_vision:
                net.config[vision.nengo_conv0.ensemble].on_chip = False

    return net


if __name__ == "__main__":
    ii = 0
    net = demo(backend, test_name="driving_%04i" % ii, neural_vision=True)
    try:
        if backend == "loihi":
            sim = nengo_loihi.Simulator(
                net,
                target="loihi",
                hardware_options=dict(snip_max_spikes_per_step=300),
            )
        elif backend == "cpu":
            sim = nengo.Simulator(net, progress_bar=False)

        with sim:
            start_time = timeit.default_timer()
            sim.run(10)
            print("\nRun time: %.5f\n" % (timeit.default_timer() - start_time))

    except ExitSim:
        pass

    finally:
        net.interface.disconnect()
        sim.close()

        fig0, axs0 = plt.subplots(2, 1)
        target = np.array(net.target_track)
        rover = np.array(net.rover_track)
        axs0[0].plot(target[:, 0], target[:, 1], "x", mew=3)
        axs0[0].plot(rover[:, 0], rover[:, 1], lw=2)

        error = np.array(net.turnerror_track)
        q = np.array(net.q_track)
        qdes = np.array(net.qdes_track)
        axs0[1].plot(q, label="q")
        axs0[1].plot(qdes, "--", label="qdes")
        axs0[1].plot(error, "r", label="error")
        axs0[1].legend()

        fig1, axs1 = plt.subplots(2, 1)
        u_track = np.array(net.u_track)
        plt.title("Control signal u")
        axs1[0].plot(u_track[:, 0], lw=2)
        axs1[1].plot(u_track[:, 1], lw=2)
        if collect_ground_truth:
            direct = sim.data[net.dead_end_probe]
            axs1[0].plot(direct[:, 0], "--", lw=3, alpha=0.5)
            axs1[1].plot(direct[:, 1], "--", lw=3, alpha=0.5)

            fig2, axs2 = plt.subplots(3, 1)
            inputs = np.array(net.input_track)
            for ii, ax in enumerate(axs2):
                ax.plot(inputs[:, ii])
            plt.suptitle("Turn motor inputs")
            local_target = np.array(net.localtarget_track)
            axs2[1].plot(local_target[:, 0], "r--", lw=2)
            axs2[2].plot(local_target[:, 1], "r--", lw=2)

            print(
                "Steering input means: ",
                [float("%.3f" % val) for val in np.mean(inputs, axis=0)],
            )
            print(
                "Steering input mins: ",
                [float("%.3f" % val) for val in np.min(inputs, axis=0)],
            )
            print(
                "Steering input maxs: ",
                [float("%.3f" % val) for val in np.max(inputs, axis=0)],
            )

        plt.show()
