"""
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

import mujoco_py
import nengo
import nengo_dl
import nengo_loihi

from abr_control._vendor.nengolib.stats import ScatteredHypersphere, spherical_transform
from abr_control.arms.mujoco_config import MujocoConfig
from abr_control.interfaces.mujoco import Mujoco
from abr_analyze import DataHandler

from rover_vision import RoverVision
from loihi_utils import LoihiRectifiedLinear, HetDecodeNeurons


class ExitSim(Exception):
    print("Restarting simulation")
    pass

# set up parameters that depend on cpu or loihi as the backend
backend = "cpu"
if len(sys.argv) > 1:
    if sys.argv[1] == "loihi":
        backend = "loihi"
print("Using %s as backend" % backend)

generate_data = True
plot_mounted_camera_freq = None  # = None to not plot


def demo(backend="cpu", test_name="validation_0000", neural_vision=True):
    if backend == "loihi":
        nengo_loihi.set_defaults()

    seed = 9
    rng = np.random.RandomState(seed)
    # target generation limits
    dist_limit = [0.25, 3.5]
    angle_limit = [-np.pi, np.pi]

    # data collection parameters
    # in steps (1ms/step)
    render_frequency = 1
    save_frequency = 10
    reaching_steps = 500
    total_images_saved = 100000
    max_time_to_target = 10000

    # network parameters
    n_dof = 3  # 2 wheels to control
    n_input = 3  # input to neural net is body_com x, y velocity, error along (x, y) plane, and q feedback
    n_output = n_dof  # output from neural net is torque signals for the wheels
    # we stack feedback from 4 cameras to get a 2pi view
    render_size = [32, 32]
    res = [render_size[0], render_size[1] * 4]
    subpixels = res[0] * res[1] * 3

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

    # initialize our robot config for the jaco2
    rover_name = "rover.xml"
    # rover_name = 'rover4We-only.xml'
    robot_config = MujocoConfig(folder="", xml_file=rover_name, use_sim_state=True)

    net = nengo.Network()
    # create our Mujoco interface
    net.interface = Mujoco(
        robot_config, dt=0.001, visualize=True, create_offscreen_rendercontext=True
    )
    net.interface.connect()  # camera_id=0)
    # shorthand
    interface = net.interface
    viewer = interface.viewer
    model = net.interface.sim.model
    data = net.interface.sim.data
    EE_id = model.body_name2id("EE")

    # get names of turning axel and back motors
    if rover_name == "rover.xml":
        joint_names = ["steering_wheel"]
    else:
        joint_names = ["ghost-steer-hinge"]
    interface.joint_pos_addrs = [
        model.get_joint_qpos_addr(name) for name in joint_names
    ]
    interface.joint_vel_addrs = [
        model.get_joint_qvel_addr(name) for name in joint_names
    ]

    # get target object id so we can change its colour
    target_geom_id = model.geom_name2id("target")
    green = [0, 0.9, 0, 1]
    red = [0.9, 0, 0, 1]

    # set up the target position
    net.interface.viewer.target = np.array([-0.0, 0.0, 0.2])

    vision = RoverVision(res=res, seed=0)

    with net:
        net.count = 0
        net.image_count = 0
        net.target_count = 0
        net.time_to_target = 0
        net.predicted_xy = [0, 0]
        start = timeit.default_timer()

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
            visionsim, visionnet_1 = vision.convert(
                gain_scale=400,
                activation=nengo_loihi.neurons.LoihiSpikingRectifiedLinear(),
                # activation=LoihiRectifiedLinear(),
                synapse=None,  # 0.001,
                training=False,
            )
            visionsim.load_params(
                "/home/tdewolf/Downloads/data/abr_analyze/loihirelu/400/epoch_306"
                # "/home/tdewolf/Downloads/data/abr_analyze/loihirelu/400/epoch_218"
            )
            with visionsim:
                visionsim.freeze_params(visionnet_1)

                vision.nengo_innode.size_in = vision.subpixels
                vision.nengo_innode.output = None

        def sim_func(t, x):
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

            error = viewer.target - robot_config.Tx("EE")
            model.geom_rgba[target_geom_id] = (
                green if np.linalg.norm(error) < 0.02 else red
            )

            # error is in global coordinates, want it in local coordinates for rover --
            # body_xmat will take from local coordinates to global
            R_raw = np.copy(data.body_xmat[EE_id]).reshape(3, 3).T  # R.T = R^-1
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
            body_com_vel_raw = data.cvel[model.body_name2id("base_link")][3:]
            body_com_vel = np.dot(R, body_com_vel_raw)

            local_target = np.dot(R, error)
            if net.count % save_frequency == 0:
                # save relevant data
                if generate_data:
                    print("Target Count: %i " % (int(net.count / save_frequency)))
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

            if viewer.exit or net.image_count == total_images_saved:
                glfw.destroy_window(viewer.window)
                raise ExitSim

            rover_xyz = robot_config.Tx("EE")
            # update our target
            # if net.count % reaching_steps == 0:
            dist = np.linalg.norm(rover_xyz - viewer.target)
            if dist < 0.2 or net.time_to_target > max_time_to_target or net.count == 0:
                # generate test set
                # phis = np.linspace(-3.14, 3.14, 100)
                # phis = np.tile(phis, 10)
                # radii = np.linspace(0.5, 3.5, 1000)
                # phi = phis[net.target_count]
                # radius = radii[net.target_count]
                # phi = np.random.choice(phis)
                # radius = np.random.choice(radii)
                # generate training set

                # generate a new target at least .5m away from current position
                while dist < 0.5:
                    phi = np.random.uniform(low=angle_limit[0], high=angle_limit[1])
                    # radius = np.random.uniform(low=dist_limit[0], high=dist_limit[1])
                    radius = 1
                    viewer.target = [np.cos(phi) * radius, np.sin(phi) * radius, 0.2]
                    dist = np.linalg.norm(rover_xyz - viewer.target)

                interface.set_mocap_xyz("target", viewer.target)
                net.target_count += 1
                net.time_to_target = 0
            # theta = net.time_to_target * 0.001
            # r = 2
            # # theta = np.hstack((np.linspace(-.7, .7, 300), np.linspace(.7, -.7, 300)))[net.time_to_target % 600]
            # viewer.target = np.array([r * (np.cos(theta)), r * -np.sin(theta), .2])
            # interface.set_mocap_xyz("target", viewer.target)
            net.time_to_target += 1

            # track data
            net.target_track.append(viewer.target)
            net.rover_track.append(np.copy(rover_xyz))
            net.localtarget_track.append(np.copy(local_target / np.pi))

            # send to mujoco, stepping the sim forward --------------------------------
            interface.send_forces(u)

            feedback = interface.get_feedback()
            output_signal = np.array(
                [
                    # body_com_vel[0],
                    # body_com_vel[1],
                    feedback["q"][0] * (1 if rover_name == "rover.xml" else -1),
                    local_target[0],
                    local_target[1],
                ]
            )

            # scale down for sending into motor_control ensemble
            output_signal[:3] = output_signal[:3] / 0.5

            if net.count % 500 == 0:
                print("Time Since Start: ", timeit.default_timer() - start)

            net.count += 1
            return np.hstack((output_signal, vision.image_input))

        # -----------------------------------------------------------------------------
        sim = nengo.Node(
            sim_func, size_in=4, size_out=n_input + vision.subpixels, label="sim"
        )

        n_motor_neurons = 4096
        vision_synapse = 0.01

        # --- set up ensemble to calculate speed and direction

        def speed_function(x):
            error = x * np.pi  # scale back up error signal from vision
            dist = np.linalg.norm(error * 100)

            # set a max speed of the robot, but when we get close slow down
            # based on how far away the target is
            wheels = min(dist, 30)
            if error[1] > 0:  # if it's behind, go backwards
                wheels *= -1

            # normalize output for loihi decodeneurons by / max value possible
            wheels /= 30
            wheels *= -1 if rover_name == "rover.xml" else 1

            return wheels

        motor_control_speed = nengo.Ensemble(
            # neuron_type=nengo.Direct(),
            neuron_type=nengo_loihi.neurons.LoihiSpikingRectifiedLinear(),
            n_neurons=n_motor_neurons,
            dimensions=2,
            max_rates=nengo.dists.Uniform(175, 220),
            # the input is mean subtracted and scaled to be between -1 and 1,
            radius=np.sqrt(2),
            # seed=seed,
            label="motor_control_speed",
        )

        # output_decodeneurons1 = HetDecodeNeurons(pairs_per_dim=10)
        # onchip_output1 = output_decodeneurons1.get_ensemble(dim=1)
        # nengo.Connection(onchip_output1, sim[1], transform=1, synapse=0.001)

        nengo.Connection(
            motor_control_speed,
            # onchip_output1,
            sim[1],
            function=lambda x: speed_function(x),
            # synapse=0.005,
        )

        # connect up speed
        nengo.Connection(
            sim[1:3], motor_control_speed, synapse=vision_synapse, transform=1 / np.pi
        )

        # --- set up ensemble to calculate torque to apply to wheels

        def steer_function(x, track=False):
            if net.count > 0 and track:
                net.input_track.append(x)

            error = x[1:] * np.pi  # take out the error signal from vision
            # scale input from sim that has been normalized to -1 to 1 range
            q = x[0] * 0.5

            # input to arctan2 is modified to account for (x, y) axes of rover vs
            # the (x, y) axes of the environment
            turn_des = np.arctan2(-error[0], abs(error[1]))
            u = turn_des - q

            # normalize output for loihi decodeneurons by / max value possible
            u /= 2 * (1 if rover_name == "rover.xml" else -1)

            # record input for finding mean and variance values
            if net.count > 0 and track:
                net.turnerror_track.append(turn_des - q)
                net.q_track.append(q)
                net.qdes_track.append(turn_des)

            return u

        motor_control_turn = nengo.Ensemble(
            # neuron_type=nengo.Direct(),
            neuron_type=nengo_loihi.neurons.LoihiSpikingRectifiedLinear(),
            n_neurons=n_motor_neurons,
            dimensions=n_input,
            max_rates=nengo.dists.Uniform(175, 220),
            # the input is mean subtracted and scaled to be between -1 and 1,
            radius=np.sqrt(n_input),
            # seed=seed,
            label="motor_control_turn",
        )

        # output_decodeneurons0 = HetDecodeNeurons(pairs_per_dim=10)
        # onchip_output0 = output_decodeneurons0.get_ensemble(dim=1)
        # nengo.Connection(onchip_output0, sim[0], transform=1, synapse=0.001)

        nengo.Connection(
            motor_control_turn,
            # onchip_output0,
            sim[0],
            function=lambda x: steer_function(x),
            # synapse=0.005,
        )

        relay_motor_input_turn = nengo.Node(size_in=n_input, label="relay_motor_input")
        nengo.Connection(relay_motor_input_turn, motor_control_turn, synapse=None)

        # -- send motor feedback to motor control
        nengo.Connection(sim[0], relay_motor_input_turn[0], synapse=None)

        # --- connect up vision to motor ensembles and sim node
        if neural_vision:
            # create decode_neuron intermediate population to get output from conv1
            # without going off-chip
            # vision_decodeneurons = HetDecodeNeurons(pairs_per_dim=5)
            # vision_onchip = vision_decodeneurons.get_ensemble(dim=2)
            # vision_onchip.radius = np.sqrt(2)
            # the vision.nengo_output node will be optimized out, so the connection
            # is onchip -> onchip, not onchip -> offchip -> onchip
            # nengo.Connection(vision.nengo_output, vision_onchip, synapse=vision_synapse)

            nengo.Connection(vision.nengo_output, motor_control_speed, synapse=vision_synapse)

            # send image input in to vision system
            nengo.Connection(sim[3:], vision.nengo_innode, synapse=None)

            # TODO: connect directly from the last layer of vision net to the motor_control ensemble
            # so the system doesn't send the signal to the host and then back to the chip
            nengo.Connection(
                vision.nengo_output, relay_motor_input_turn[1:], synapse=vision_synapse
            )
            nengo.Connection(vision.nengo_output, sim[2:4], synapse=vision_synapse)
        else:
            # connect up turning
            nengo.Connection(
                sim[1:3],
                motor_control_speed,
                synapse=vision_synapse,
                transform=1 / np.pi,
            )
            nengo.Connection(
                sim[1:3],
                relay_motor_input_turn[1:],
                synapse=vision_synapse,
                transform=1 / np.pi,
            )
            nengo.Connection(sim[1:3], sim[2:4], synapse=vision_synapse)

        # create a direct mode ensemble performing the same calculation for debugging
        motor_control_direct = nengo.Ensemble(
            neuron_type=nengo.Direct(),
            n_neurons=n_motor_neurons,
            dimensions=n_input,
            label="motor_control_direct",
        )

        dead_end = nengo.Node(size_in=2, label="dead_end")
        # send motor signal to sim
        nengo.Connection(
            motor_control_direct,
            dead_end[1],
            function=lambda x: speed_function(x[1:]),
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
                vision.nengo_output, motor_control_direct[1:], synapse=vision_synapse,
            )
        else:
            nengo.Connection(
                sim[1:3],
                motor_control_direct[1:],
                synapse=vision_synapse,
                transform=1 / np.pi,
            )
        # send motor feedback to motor control
        nengo.Connection(sim[0], motor_control_direct[0], synapse=None)

        net.dead_end_probe = nengo.Probe(dead_end)

        if backend == "loihi":
            print("calling nengo_loihi.add_params(net)")
            nengo_loihi.add_params(net)

            if neural_vision:
                net.config[vision.nengo_conv0.ensemble].on_chip = False
                print(
                    "vision conv0 on chip: ",
                    net.config[vision.nengo_conv0.ensemble].on_chip,
                )

    return net


if __name__ == "__main__":
    for ii in range(47, 48):
        print("\n\nBeginning round ", ii)
        net = demo(backend, test_name="driving_%04i" % ii, neural_vision=True)
        for conn in net.all_connections:
            print(conn)
        for subnet in net.all_networks:
            print(subnet)
        # for ens in net.all_ensembles:
        #     print('%s.on_chip: %s' % (net.config[ens].on_chip, ens.on_chip))
        for node in net.all_nodes:
            print(node)
        try:
            if backend == "loihi":
                sim = nengo_loihi.Simulator(net, target="sim", remove_passthrough=False)
                #     , target="loihi", hardware_options=dict(snip_max_spikes_per_step=300)
                # )
            elif backend == "cpu":
                sim = nengo.Simulator(net, progress_bar=False)

            while 1:
                sim.run(0.1)

        except ExitSim:
            pass

        finally:
            net.interface.disconnect()
            direct = sim.data[net.dead_end_probe]
            sim.close()

            # plot data
            plt.subplot(2, 1, 1)
            target = np.array(net.target_track)
            rover = np.array(net.rover_track)
            plt.plot(target[:, 0], target[:, 1], "x", mew=3)
            plt.plot(rover[:, 0], rover[:, 1], lw=2)

            plt.subplot(2, 1, 2)
            error = np.array(net.turnerror_track)
            q = np.array(net.q_track)
            qdes = np.array(net.qdes_track)
            plt.plot(q, label="q")
            plt.plot(qdes, "--", label="qdes")
            plt.plot(error, "r", label="error")
            plt.legend()

            plt.figure()
            u_track = np.array(net.u_track)
            plt.title("Control signal u")
            plt.subplot(2, 1, 1)
            plt.plot(u_track[:, 0], alpha=0.5)
            plt.plot(direct[:, 0], "--", lw=5)
            plt.subplot(2, 1, 2)
            plt.plot(u_track[:, 1], alpha=0.5)
            plt.plot(direct[:, 1], "--", lw=3)

            # NOTE: ignore this plot if the motor_control ensemble is running with neurons
            # because it will just be the output from calculating the decoders
            plt.figure()
            inputs = np.array(net.input_track)
            for ii in range(inputs.shape[1]):
                plt.subplot(inputs.shape[1], 1, ii + 1)
                plt.plot(inputs[:, ii])
            plt.suptitle("Turn motor inputs")
            local_target = np.array(net.localtarget_track)
            plt.subplot(3, 1, 2)
            plt.plot(local_target[:, 0], 'r--', lw=2)
            plt.subplot(3, 1, 3)
            plt.plot(local_target[:, 1], 'r--', lw=2)

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

else:
    model, _ = demo()
