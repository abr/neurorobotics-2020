"""
To run the demo with Nengo running on cpu:
    python nengo_rover.py cpu

To run the demo with Nengo on loihi
    NXSDKHOST=loihighrd python nengo_rover.py
"""
import math
import tensorflow as tf
import keras
import nengo_dl
import glfw
import mujoco_py
import nengo
import nengo_loihi
import numpy as np
import os
import sys
import time
import timeit
import matplotlib.pyplot as plt
import cv2
import sys

import mujoco_py as mjp
from nengo_loihi import decode_neurons
import nengo_loihi
from abr_control.arms.mujoco_config import MujocoConfig
from abr_control.interfaces.mujoco import Mujoco
from abr_analyze import DataHandler

from rover_vision import RoverVision


class ExitSim(Exception):
    print("Restarting simulation")
    pass


# set up parameters that depend on cpu or loihi as the backend
backend = "cpu"
adapt_scale = 1
if len(sys.argv) > 1:
    if sys.argv[1] == "loihi":
        backend = "loihi"
        # TODO do we need to change adapt_scale for loihi

pes_learning_rate = 1e-5
if len(sys.argv) > 1:
    for arg in sys.argv:
        arg = str(arg)
        if arg == "loihi":
            backend = "loihi"
            adapt_scale = 10
            pes_learning_rate = 1e-5  # sometimes needs to be scaled
print("Using %s as backend" % backend)


def demo(backend="cpu"):
    if backend == "loihi":
        nengo_loihi.set_defaults()

    seed = 9
    rng = np.random.RandomState(seed)
    # target generation limits
    dist_limit = [0.5, 3.5]
    angle_limit = [-np.pi, np.pi]

    # data collection parameters
    # in steps (1ms/step)
    render_frequency = 1
    reaching_steps = 500
    imgs = []

    # network parameters
    n_dof = 3  # 2 wheels to control
    n_input = 5  # input to neural net is body_com y velocity, error along (x, y) plane, and q dq feedback
    n_output = n_dof  # output from neural net is torque signals for the wheels
    # we stack feedback from 4 cameras to get a 2pi view
    render_size = [32, 32]
    res = [render_size[0], render_size[1] * 4]
    subpixels = res[0] * res[1] * 3

    # initialize our robot config for the jaco2
    robot_config = MujocoConfig(folder="", xml_file="rover.xml", use_sim_state=True)

    net = nengo.Network()
    # create our Mujoco interface
    net.interface = Mujoco(robot_config, dt=0.001, visualize=True,
                           create_offscreen_rendercontext=True)
    net.interface.connect(camera_id=0)
    # shorthand
    interface = net.interface
    viewer = interface.viewer
    model = net.interface.sim.model
    data = net.interface.sim.data
    EE_id = model.body_name2id("EE")

    # get names of turning axel and back motors
    joint_names = ["joint%i" % ii for ii in [0, 1, 2]]
    interface.joint_pos_addrs = [
        model.get_joint_qpos_addr(name) for name in joint_names
    ]
    interface.joint_vel_addrs = [
        model.get_joint_qvel_addr(name) for name in joint_names
    ]

    # get target object id so we can change its colour
    target_geom_id = model.geom_name2id("target")
    green = [0, 0.9, 0, 0.5]
    red = [0.9, 0, 0, 0.5]

    # set up the target position
    net.interface.viewer.target = np.array([-0.4, 0.5, 0.4])

    with net:
        net.count = -1
        net.target_count = -1
        net.imgs = []
        net.predicted_xy = [0, 0]
        start = timeit.default_timer()

        vision = RoverVision(res=res, minibatch_size=1, dt=0.001, seed=0)
        _, visionnet = vision.convert(
            gain_scale=400,
            activation=nengo_loihi.neurons.LoihiSpikingRectifiedLinear(),
            synapse=0.001,
        )
        _, visionnet = vision.convert_nengodl_to_nengo(visionnet)
        visionnet.image_input = np.zeros((res[0], res[1], 3))

        def sim_func(t, x):
            u = x[:3]
            prediction = x[3:5]

            # get our image data
            if net.count % render_frequency == 0:
                # TODO rename the vision sensors so we can stack them sequentially by name
                for ii, jj in enumerate([4, 1, 3, 2]):
                    interface.offscreen.render(res[0], res[0], camera_id=jj)
                    visionnet.image_input[
                        :, res[0] * ii : res[0] * (ii + 1)
                    ] = interface.offscreen.read_pixels(res[0], res[0])[0] / 255

                if net.count % 1000 == 0:
                    plt.figure()
                    a = plt.subplot(1, 1, 1)
                    a.imshow(visionnet.image_input)
                    plt.show()

                # raw_img = np.hstack(
                #     [np.hstack((imgs[3], imgs[0])), np.hstack((imgs[2], imgs[1]))]
                # )

                # get predicted target from vision
                # vision.image_input = resize_images(
                #     raw_img, res=res, show_resized_image=False, flatten=True
                # ).squeeze()

                # scaled_image_data = []
                # for count, data in enumerate(raw_img):
                #     # normalize to 0-255
                #     # rgb = np.asarray(data) / 255
                #     # if np.any(rgb < 0):
                #     #     raise Exception("All data should be positive")
                #
                #     # resize image resolution
                #     if shape[1] != res[0] or shape[2] != res[1]:
                #         print("Resolution does not match desired value, resizing...")
                #         print("Desired Res: ", res)
                #         print("Input Res: ", [shape[1], shape[2]])
                #         rgb = cv2.resize(rgb, dsize=(res[1], res[0]), interpolation=cv2.INTER_CUBIC)
                #
                #     # # visualize scaling for debugging
                #     # if show_resized_image:
                #     #     plt.Figure()
                #     #     a = plt.subplot(121)
                #     #     a.set_title("Original")
                #     #     a.imshow(data, origin="lower")
                #     #     b = plt.subplot(122)
                #     #     b.set_title("Scaled")
                #     #     b.imshow(rgb, origin="lower")
                #     #     plt.show()
                #
                #     # flatten to 1D
                #     # NOTE should use np.ravel to maintain image order
                #     if flatten:
                #         rgb = rgb.flatten()
                #     # scale image from -1 to 1 and save to list
                #     # scaled_image_data.append(rgb*2 - 1)
                #     scaled_image_data.append(rgb)
                #
                # vision.image_input = np.asarray(scaled_image_data)

            error = viewer.target - robot_config.Tx("EE")
            model.geom_rgba[target_geom_id] = green if np.linalg.norm(error) < 0.02 else red

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
            local_target = np.dot(R, error)

            # we also want the egocentric velocity of the rover
            body_com_vel_raw = data.cvel[model.body_name2id("base_link")][3:]
            body_com_vel = np.dot(R, body_com_vel_raw)

            feedback = interface.get_feedback()
            q = feedback["q"]
            dq = feedback["dq"]

            local_target = local_target[:2]
            output_signal = np.array([body_com_vel[1], q[0], dq[0]])

            # TODO change this to index into an array
            # output_signal = np.hstack((output_signal, rendered_image))

            if viewer.exit:
                glfw.destroy_window(viewer.window)
                raise ExitSim

            # update our target
            if net.count % reaching_steps == 0:
                net.target_count += 1
                phis = np.linspace(-3.14, 3.14, 100)
                phis = np.tile(phis, 10)
                radii = np.linspace(0.5, 3.5, 1000)
                phi = phis[net.target_count]
                radius = radii[net.target_count]
                viewer.target = [np.cos(phi) * radius, np.sin(phi) * radius, 0.4]
                interface.set_mocap_xyz("target", viewer.target)

            # send to mujoco, stepping the sim forward --------------------------------
            interface.send_forces(np.asarray(u))

            if net.count % 500 == 0:
                print("Time Since Start: ", timeit.default_timer() - start)

            net.count += 1
            return output_signal

        def steering_function(x):
            body_com_vely = x[0]
            q = x[1]
            dq = x[2]
            error = x[3:]

            dist = np.linalg.norm(error)

            # input to arctan2 is modified to account for (x, y) axes of rover vs
            # the (x, y) axes of the environment
            turn_des = np.arctan2(-error[0], error[1])

            # set a max speed of the robot, but when we get close slow down
            # based on how far away the target is
            wheels = min(dist, 1)
            if error[1] < 0:
                wheels *= -1
            wheels -= body_com_vely

            kp = 1
            kv = 0.8
            u0 = kp * (turn_des - q) - kv * dq
            u = np.array([u0, wheels, wheels, error[0], error[1]])
            return u

        # -----------------------------------------------------------------------------
        sim = nengo.Node(sim_func, size_in=5, size_out=3)

        n_motor_neurons = 1000
        encoders = nengo.dists.UniformHypersphere(surface=True).sample(
            n_motor_neurons, d=n_input
        )
        motor_control = nengo.Ensemble(
            # neuron_type=nengo.Direct(),
            n_neurons=n_motor_neurons,
            dimensions=n_input,
            radius=np.sqrt(n_input),
            encoders=encoders,
            seed=seed,
        )

        # # send rendered image to vision network
        # nengo.Connection(
        #     sim[3:],
        #     vision.nengo_input,
        #     # synapse=None
        # )

        # send vision prediction to motor control
        nengo.Connection(vision.nengo_output, motor_control[3:])

        # send motor feedback to motor control
        nengo.Connection(sim[:3], motor_control[:3], synapse=None)

        # send motor signal to sim
        nengo.Connection(
            motor_control,
            # onchip_output,
            sim[:5],
            # function=steering_function
        )

    return net, robot_config


if __name__ == "__main__":
    net, robot_config = demo(backend)
    try:
        if backend == "loihi":
            sim = nengo_loihi.Simulator(net, target="sim")
            #     , target="loihi", hardware_options=dict(snip_max_spikes_per_step=300)
            # )
        elif backend == "cpu":
            sim = nengo.Simulator(net, progress_bar=False)

        while 1:
            sim.run(1e5)

    except ExitSim:
        pass

    finally:
        net.interface.disconnect()
        sim.close()

else:
    model, _ = demo()
