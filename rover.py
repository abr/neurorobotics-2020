"""
To generate training data to train up the RoverVision system, set generate_data=True.


To run the demo with Nengo running on cpu:
    python nengo_rover.py cpu

To run the demo with Nengo on loihi
    NXSDKHOST=loihighrd python nengo_rover.py
"""
import matplotlib.pyplot as plt
import numpy as np
import timeit
import os
import tensorflow as tf

import nengo
import nengo_loihi
from nengo_loihi.neurons import LoihiSpikingRectifiedLinear
from nengo_interfaces.mujoco import Mujoco

from data_handler import DataHandler
from rover_vision import RoverVision


current_dir = os.path.abspath('.')
if not os.path.exists('%s/figures' % current_dir):
    os.makedirs('%s/figures' % current_dir)
if not os.path.exists('%s/data' % current_dir):
    save_folder = '%s/data' % current_dir

class ExitSim(Exception):
    pass


def demo(
    backend="cpu",
    collect_ground_truth=False,
    generate_data=False,
    motor_neuron_type=LoihiSpikingRectifiedLinear(),
    neural_vision=True,
    plot_mounted_camera_freq=None,
    weights_name='reference_weights'
):
    if backend == "loihi":
        nengo_loihi.set_defaults()

    seed = 0
    np.random.seed(seed)
    # target generation limits
    dist_limit = [0.25, 3.5]
    angle_limit = [-np.pi, np.pi]

    # data collection parameters in steps (1ms per step)
    save_frequency = 10
    # time steps
    render_frequency = 10
    total_images_saved = 50000
    if generate_data:
        max_time_to_target = 2000
    else:
        max_time_to_target = 10000

    net = nengo.Network(seed=seed)
    # create our Mujoco interface
    interface = Mujoco(
        xml_file="rover.xml",
        folder="",
        dt=0.001,
        render_params={
            "cameras": [4, 1, 3, 2],  # camera ids and order to render
            "resolution": [32, 32],
            "frequency": 1,  # render images from cameras every time step
            "plot_frequency": None,  # do not plot images from cameras
        },
        joint_names=["steering_wheel"],
        track_input=True,
    )

    # NOTE: why the slow rendering when defined before interface?
    vision = RoverVision(seed=0)

    # set up the target position
    net.target = np.array([-0.0, 0.0, 0.2])

    if generate_data:
        db_dir = None
        dat = DataHandler(db_dir=db_dir, db_name='rover')
        dat.save(
            data={
                "render_frequency": render_frequency,
                "save_frequency": save_frequency,
                "max_time_to_target": max_time_to_target,
                "render_size": vision.resolution,
                "dist_limit": dist_limit,
                "angle_limit": angle_limit,
            },
            save_location="params",
            overwrite=True,
        )

    net.config[nengo.Connection].synapse = None
    with net:
        net.image_count = 0
        net.val_img_count = -1
        net.train_img_count = -1

        # track position
        net.target_track = []
        net.rover_track = []
        net.vision_track = []
        net.localtarget_track = []
        net.ideal_motor_track = []

        if neural_vision:

            visionsim, visionnet = vision.convert(
                add_probes=False,
                swap_activations={tf.nn.relu: LoihiSpikingRectifiedLinear()},
                scale_firing_rates=400,
            )
            visionsim.load_params("%s/%s" % (save_folder, weights_name))
            with visionsim:
                visionsim.freeze_params(visionnet)
                vision.input.size_in = vision.subpixels
                vision.input.output = None

        if not neural_vision or collect_ground_truth:
            # rotation matrix for rotating error from world to rover frame
            theta = 3 * np.pi / 2
            R90 = np.array(
                [
                    [np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1],
                ]
            )

            def local_target(t):
                rover_xyz = interface.get_position("EE")
                error = net.target - rover_xyz
                # error in global coordinates, want it in local coordinates for rover
                R_raw = interface.get_orientation("EE").T  # R.T = R^-1
                # rotate it so y points forward toward the steering wheels
                R = np.dot(R90, R_raw)
                local_target = np.dot(R, error)
                net.localtarget_track.append(local_target / np.pi)

                if generate_data and int(t / interface.dt) % save_frequency == 0:
                    if np.linalg.norm(local_target) < 3.5:
                        # save 10% of our images for validation

                        if net.image_count % 10 == 0:
                            test_name = 'validation'
                            net.val_img_count += 1
                            img_num = net.val_img_count
                        else:
                            test_name = 'training'
                            net.train_img_count += 1
                            img_num = net.train_img_count

                        dat.save(
                            data={"rgb": interface.camera_feedback, "target": local_target},
                            save_location="%s/%04d" % (test_name, img_num),
                            overwrite=True,
                        )
                        net.image_count += 1
                        if net.image_count % 1000 == 0:
                            print('%i/%i images generated' % (net.image_count, total_images_saved))

                # scale by np.pi for consistency with neural vision output
                # used to roughly normalize the local target signal to -1:1 range
                output_signal = np.array([local_target[0], local_target[1]])
                return output_signal

            local_target = nengo.Node(output=local_target, size_out=2)

        accel_scale = 500
        steer_scale = 500

        def check_exit_and_track_data(t, x):
            if interface.exit or (
                generate_data and net.image_count == total_images_saved
            ):
                raise ExitSim

            rover_xyz = interface.get_position("EE")
            dist = np.linalg.norm(rover_xyz - net.target)
            if dist < 0.2 or int(t / interface.dt)%max_time_to_target == 0:
                # generate a new target 1-2.5m away from current position
                while dist < 1 or dist > 2.5:
                    phi = np.random.uniform(low=angle_limit[0], high=angle_limit[1])
                    radius = np.random.uniform(low=dist_limit[0], high=dist_limit[1])
                    net.target = [np.cos(phi) * radius, np.sin(phi) * radius, 0.2]
                    dist = np.linalg.norm(rover_xyz - net.target)
                interface.set_mocap_xyz("target", net.target)
                net.target_track.append(net.target)

            # track data
            net.rover_track.append(rover_xyz)
            net.vision_track.append(np.copy(x))

        check_exit_and_track_data = nengo.Node(
            check_exit_and_track_data, size_in=2, label="check_exit_and_track_data"
        )

        mujoco_node = interface.make_node()

        # -----------------------------------------------------------------------------

        # --- set up ensemble to calculate acceleration
        def accel_function(x):
            return min(np.linalg.norm(-x), 1)

        accel = nengo.Ensemble(
            neuron_type=motor_neuron_type,
            n_neurons=4096,
            dimensions=2,
            max_rates=nengo.dists.Uniform(50, 200),
            radius=1,
            label="accel",
        )

        nengo.Connection(
            accel,
            mujoco_node[1],
            function=accel_function,
            transform=accel_scale,
            synapse=0 if motor_neuron_type == nengo.Direct() else 0.1,
        )

        # --- set up ensemble to calculate torque to apply to steering wheel
        def steer_function(x):
            error = x[1:] * np.pi  # take out the error signal from vision
            q = x[0] * 0.7  # scale normalized input back to original range

            # arctan2 input set this way to account for different alignment
            # of (x, y) axes of rover and the environment
            # turn_des = np.arctan2(-error[0], abs(error[1]))
            turn_des = np.arctan2(-error[0], error[1])
            u = (turn_des - q) / 2  # divide by 2 to get in -1 to 1 ish range
            return u

        steer = nengo.Ensemble(
            neuron_type=motor_neuron_type,
            n_neurons=4096,
            dimensions=3,
            max_rates=nengo.dists.Uniform(50, 200),
            radius=np.sqrt(2),
            label="steer",
        )

        nengo.Connection(
            steer,
            mujoco_node[0],
            function=lambda x: steer_function(x),
            transform=steer_scale,
            # if using Direct mode, a synapse will cause oscillating control
            synapse=0 if motor_neuron_type == nengo.Direct() else 0.025,
        )

        # add a relay node to amalgamate input to steering ensemble
        steer_input = nengo.Node(size_in=3, label="relay_motor_input")
        nengo.Connection(steer_input, steer)

        # -- connect relevant motor feedback (joint angle of steering wheel)
        nengo.Connection(mujoco_node[0], steer_input[0])

        # --- connect vision up to motor ensembles and mujoco node
        vision_synapse = 0.05
        if neural_vision:
            # send image input in to vision system
            nengo.Connection(mujoco_node[2:], vision.input)
            # connect vision network prediction to steering ensemble
            nengo.Connection(vision.output, steer_input[1:], synapse=vision_synapse)
            # connect vision network prediction to accel ensemble
            nengo.Connection(vision.output, accel, synapse=vision_synapse)
            # connect vision network to data tracking node
            nengo.Connection(
                vision.output, check_exit_and_track_data, synapse=vision_synapse
            )

        else:
            # if we're not using neural vision, then hook up the local_target node
            # to our motor system to get the location of the target relative to rover
            nengo.Connection(
                local_target,
                steer_input[1:],
                synapse=vision_synapse,
                transform=1 / np.pi,
            )
            nengo.Connection(
                local_target, accel, synapse=vision_synapse, transform=1 / np.pi,
            )
            nengo.Connection(
                local_target, check_exit_and_track_data, transform=1 / np.pi
            )

        if collect_ground_truth:
            # create a direct mode ensemble performing the same calculation for debugging
            def calculate_ideal_motor_signals(t, x):
                net.ideal_motor_track.append(
                    [
                        steer_function(x) * steer_scale,
                        accel_function(x[1:]) * accel_scale,
                    ]
                )

            ground_truth = nengo.Node(
                output=calculate_ideal_motor_signals, size_in=3, size_out=0
            )

            # connect local target prediction or node
            if neural_vision:
                nengo.Connection(
                    vision.output, ground_truth[1:], synapse=vision_synapse,
                )
            else:
                nengo.Connection(
                    local_target,
                    ground_truth[1:],
                    synapse=vision_synapse,
                    transform=1 / np.pi,
                )
            # connect steering wheel angle feedback
            nengo.Connection(mujoco_node[0], ground_truth[0])

        if backend == "loihi":
            nengo_loihi.add_params(net)

            if neural_vision:
                net.config[vision.conv0.ensemble].on_chip = False

        net.control_track = interface.input_track

    return net


if __name__ == "__main__":

    # set up parameters that depend on cpu or loihi as the backend
    backend = "cpu"  # can be ["cpu"|"loihi"]

    collect_ground_truth = True  # track ideal network output, useful for plotting
    #TODO: change this to update weights_name to weights or reference_weights depending on user set flag for using self trained weights
    generate_data = False  # should training / validation data be created
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == 'generate_data':
            generate_data = True

    if generate_data:
        # when generating training data for vision network, run using ideal system
        print('\nGenerating Training Data\n')
        motor_neuron_type = nengo.Direct()
        neural_vision = False
        # set to a large value, once we collect all of our images it will stop the sim
        sim_runtime = 1e5
    else:
        # otherwise run use spiking neural networks
        motor_neuron_type = LoihiSpikingRectifiedLinear()
        neural_vision = True  # if True uses trained vision net, otherwise ground truth
        sim_runtime = 10

    net = demo(
        backend,
        collect_ground_truth=collect_ground_truth,
        generate_data=generate_data,
        motor_neuron_type=motor_neuron_type,
        neural_vision=neural_vision,
        plot_mounted_camera_freq=None,  # how often to plot image from cameras
        weights_name=weights_name
    )

    try:
        if backend == "loihi":
            sim = nengo_loihi.Simulator(
                net,
                target="sim",  # set equal to "loihi" to run on Loihi hardware
                hardware_options=dict(snip_max_spikes_per_step=300),
            )
        elif backend == "cpu":
            sim = nengo.Simulator(net, progress_bar=False)
        else:
            raise Exception("Invalid backend specified")

        with sim:
            start_time = timeit.default_timer()
            sim.run(sim_runtime)
            print("\nRun time: %.5f\n" % (timeit.default_timer() - start_time))

    except ExitSim:
        pass

    finally:
        sim.close()

        fig0 = plt.figure()
        target = np.array(net.target_track)
        rover = np.array(net.rover_track)
        plt.plot(target[:, 0], target[:, 1], "x", mew=3)
        plt.plot(rover[:, 0], rover[:, 1], lw=2)
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.gca().set_aspect("equal")
        plt.title("Rover trajectory in environment")
        fig0.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig("figures/rover_trajectory.pdf")

        fig1, axs1 = plt.subplots(2, 1, figsize=(8, 4))
        control_track = np.array(net.control_track)
        plt.suptitle("Control network output")
        axs1[0].plot(control_track[:, 0], lw=2, label="Steering torque")
        axs1[1].plot(control_track[:, 1], lw=2, label="Wheels torque")
        if collect_ground_truth:
            direct = np.array(net.ideal_motor_track)
            axs1[0].plot(
                direct[:, 0], "r--", alpha=0.5, lw=3, label="Ideal steering torque"
            )
            axs1[1].plot(
                direct[:, 1], "r--", alpha=0.5, lw=3, label="Ideal wheels torque"
            )
        axs1[0].legend()
        axs1[0].grid()
        axs1[1].legend()
        axs1[1].grid()
        axs1[0].set_ylabel("Steering torque")
        axs1[1].set_ylabel("Wheels torque")
        axs1[1].set_xlabel("Time (s)")
        fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig("figures/control_network_output.pdf")

        if collect_ground_truth:
            fig2, axs2 = plt.subplots(2, 1, figsize=(8, 4))
            local_target = np.array(net.localtarget_track)
            plt.suptitle("Vision network output")
            inputs = np.array(net.vision_track)
            axs2[0].plot(inputs[:, 0], label="Predicted x")
            axs2[0].plot(local_target[:, 0], "r--", lw=2, label="Ground truth x")
            axs2[0].legend()
            axs2[0].grid()
            axs2[1].plot(inputs[:, 1], label="Predicted y")
            axs2[1].plot(local_target[:, 1], "r--", lw=2, label="Ground truth y")
            axs2[1].legend()
            axs2[1].grid()
            axs2[0].set_ylabel("X (m)")
            axs2[1].set_ylabel("Y (m)")
            axs2[1].set_xlabel("Time (s)")
            fig2.tight_layout(rect=[0, 0.03, 1, 0.95])

            plt.savefig("figures/vision_network_output.pdf")
        plt.show()
