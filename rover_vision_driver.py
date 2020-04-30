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
from abr_control._vendor.nengolib.stats import ScatteredHypersphere, spherical_transform
from abr_analyze import DataHandler

from rover_vision import RoverVision
from loihi_rate_neuron import LoihiRectifiedLinear


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


generate_data = True
plot_mounted_camera_freq = None  # = None to not plot

def demo(backend="cpu", test_name='validation_0000', neural_vision=True):
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
    n_input = 6  # input to neural net is body_com x, y velocity, error along (x, y) plane, and q dq feedback
    n_output = n_dof  # output from neural net is torque signals for the wheels
    # we stack feedback from 4 cameras to get a 2pi view
    render_size = [32, 32]
    res = [render_size[0], render_size[1] * 4]
    subpixels = res[0] * res[1] * 3

    dat = DataHandler()
    dat.save(
        data={
            'render_frequency': render_frequency,
            'save_frequency': save_frequency,
            'reaching_steps': reaching_steps,
            'max_time_to_target': max_time_to_target,
            'render_size': render_size,
            'dist_limit': dist_limit,
            'angle_limit': angle_limit
        },
        save_location='%s/params' % test_name,
        overwrite=True)

    # initialize our robot config for the jaco2
    rover_name = 'rover.xml'
    # rover_name = 'rover4We-only.xml'
    robot_config = MujocoConfig(folder="", xml_file=rover_name, use_sim_state=True)

    net = nengo.Network()
    net.config[nengo.Connection].synapse = None
    # create our Mujoco interface
    net.interface = Mujoco(robot_config, dt=0.001, visualize=True,
                           create_offscreen_rendercontext=True)
    net.interface.connect()#camera_id=0)
    # shorthand
    interface = net.interface
    viewer = interface.viewer
    model = net.interface.sim.model
    data = net.interface.sim.data
    EE_id = model.body_name2id("EE")

    # get names of turning axel and back motors
    if rover_name == 'rover.xml':
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
    net.interface.viewer.target = np.array([-0.4, 0.5, 0.2])

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

        image_input = np.zeros((res[0], res[1], 3))
        vision = RoverVision(res=res, seed=0)
        if neural_vision:
            visionsim, visionnet = vision.convert(
                gain_scale=400,
                activation=nengo_loihi.neurons.LoihiSpikingRectifiedLinear(),
                # activation=LoihiRectifiedLinear(),
                synapse=None,#0.001,
            )
            visionsim.load_params(
                # "/home/tdewolf/Downloads/data/abr_analyze/loihirelu/400/epoch_484"
                # "/home/tdewolf/Downloads/data/abr_analyze/loihirelu/400/epoch_295"
                "/home/tdewolf/Downloads/data/abr_analyze/loihirelu/400/epoch_108"
            )
            with visionsim:
                visionsim.freeze_params(visionnet)
            _, visionnet = vision.convert_nengodl_to_nengo(visionnet)

        def sim_func(t, x):
            turn = x[0]
            wheels = x[1]
            u = np.array([turn, wheels])
            prediction = x[2:4]

            # get our image data
            if net.count % render_frequency == 0:
                for ii, jj in enumerate([4, 1, 3, 2]):
                    interface.offscreen.render(res[0], res[0], camera_id=jj)
                    image_input[
                        :, res[0] * ii : res[0] * (ii + 1)
                    ] = interface.offscreen.read_pixels(res[0], res[0])[0]# / 255
                    vision.image_input[:] = image_input.flatten()

            if plot_mounted_camera_freq is not None and net.count % plot_mounted_camera_freq == 0:
                # plot the mounted camera output every so often
                plt.figure()
                a = plt.subplot(1, 1, 1)
                a.imshow(vision.image_input.reshape((res[0], res[1], 3)) / 255)
                plt.show()

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

            # we also want the egocentric velocity of the rover
            body_com_vel_raw = data.cvel[model.body_name2id("base_link")][3:]
            body_com_vel = np.dot(R, body_com_vel_raw)

            local_target = np.dot(R, error)
            if net.count % save_frequency == 0:
                # save relevant data
                if generate_data:
                    print('Target Count: %i ' % (int(net.count/save_frequency)))
                    save_data={
                            'rgb': image_input,
                            'target': local_target,
                            }

                    dat.save(
                            data=save_data,
                            save_location='%s/data/%04d' % (test_name, net.image_count),
                        overwrite=True)
                    print("%s/data/%04d" % (test_name, net.image_count))
                net.image_count += 1

            if viewer.exit or net.image_count == total_images_saved:
                glfw.destroy_window(viewer.window)
                raise ExitSim

            rover_xyz = robot_config.Tx('EE')
            # update our target
            # if net.count % reaching_steps == 0:
            dist = np.linalg.norm(rover_xyz - viewer.target)
            if (dist < 0.2 or net.time_to_target > max_time_to_target or net.count == 0):
                # generate test set
                # phis = np.linspace(-3.14, 3.14, 100)
                # phis = np.tile(phis, 10)
                # radii = np.linspace(0.5, 3.5, 1000)
                # phi = phis[net.target_count]
                # radius = radii[net.target_count]
                # phi = np.random.choice(phis)
                # radius = np.random.choice(radii)
                # generate training set
                phi = np.random.uniform(low=angle_limit[0], high=angle_limit[1])
                # radius = np.random.uniform(low=dist_limit[0], high=dist_limit[1])
                radius = 1
                viewer.target = [
                    np.cos(phi) * radius,
                    np.sin(phi) * radius,
                    0.2
                ]
                interface.set_mocap_xyz("target", viewer.target)
                net.target_count += 1
                net.time_to_target = 0
            # theta = net.time_to_target * 0.001
            # r = 2
            # # theta = np.hstack((np.linspace(-.7, .7, 300), np.linspace(.7, -.7, 300)))[net.time_to_target % 600]
            # viewer.target = np.array([r * (np.cos(theta)), r * -np.sin(theta), .2])
            interface.set_mocap_xyz("target", viewer.target)
            net.time_to_target += 1

            # track data
            net.target_track.append(viewer.target)
            net.rover_track.append(np.copy(rover_xyz))

            # send to mujoco, stepping the sim forward --------------------------------
            interface.send_forces(u)

            feedback = interface.get_feedback()
            output_signal = np.array([
                body_com_vel[0],
                body_com_vel[1],
                feedback['q'][0] * (1 if rover_name == 'rover.xml' else -1),
                feedback['dq'][0] * (1 if rover_name == 'rover.xml' else -1),
                local_target[0],
                local_target[1]
            ])

            if net.count % 500 == 0:
                print("Time Since Start: ", timeit.default_timer() - start)

            net.count += 1
            return output_signal

        def steering_function(x):
            body_com_vel = np.linalg.norm(x[:2])
            q = x[2]
            dq = x[3]
            error = x[4:]
            # print('error: %2.3f, %2.3f' % (error[0], error[1]))

            dist = np.linalg.norm(error * 100)

            # input to arctan2 is modified to account for (x, y) axes of rover vs
            # the (x, y) axes of the environment
            # turn_des = np.arctan2(-error[0], abs(error[1]))
            turn_des = np.arctan2(-error[0], abs(error[1]))

            # set a max speed of the robot, but when we get close slow down
            # based on how far away the target is
            wheels = min(dist, 50)#(1 if rover_name == 'rover.xml' else 100))
            # if dist < 1:
            #     wheels = dist**2
            wheels -= body_com_vel
            wheels *= 20  # account for point mass
            if error[1] > 0:
                wheels *= -1

            print('turn des : %.3f' % turn_des)
            print('current angle: %.3f ' % q)
            print('error: %.3f' % (turn_des - q))

            net.turnerror_track.append(turn_des - q)
            net.q_track.append(q)
            net.qdes_track.append(turn_des)

            kp = 5000 #if rover_name == 'rover.xml' else 5000
            kv = 0#np.sqrt(kp)
            u0 = kp * (turn_des - q) - kv * dq
            u = np.array([
                u0 * (1 if rover_name == 'rover.xml' else -1),
                wheels * (-1 if rover_name == 'rover.xml' else 1)
            ])
            return u

        # -----------------------------------------------------------------------------
        sim = nengo.Node(sim_func, size_in=4, size_out=6)

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

        def print_func(t, x):
            # print(x)
            return x
        relay = nengo.Node(print_func, size_in=2)
        # send motor signal to sim
        nengo.Connection(
            motor_control,
            relay,
            function=steering_function,
            synapse=None,
        )
        nengo.Connection(
            relay,
            sim[:2],
            synapse=0,
        )

        # adapt = nengo.Ensemble(
        #     n_neurons=1000,
        #     # the context is the same input the motor_control ensemble gets
        #     dimensions=5,
        #     radius=np.sqrt(5),
        # )
        # set up learning connection
        # conn = nengo.Connection(
        #     adapt.neurons,
        #     sim[:1],
        #     transform=np.zeros((1, adapt.n_neurons)),
        #     learning_rule_type=nengo.PES(learning_rate=1e-3),
        # )
        # nengo.Connection(relay[:1], conn.learning_rule, transform=-1)


        # send vision prediction to motor control
        if neural_vision:
            nengo.Connection(vision.nengo_output, motor_control[4:], synapse=0.01)
            nengo.Connection(vision.nengo_output, sim[2:4], synapse=0.01)
            # nengo.Connection(vision.nengo_output, adapt[4:], synapse=0.01)
        else:
            nengo.Connection(sim[4:], motor_control[4:], synapse=0.01)
            nengo.Connection(sim[4:], sim[2:4], synapse=0.01)
            # nengo.Connection(sim[4:], adapt[4:], synapse=0.01)

        # send motor feedback to motor control
        nengo.Connection(sim[:4], motor_control[:4], synapse=None)
        # nengo.Connection(sim[:4], adapt[:4], synapse=None)

    return net, robot_config


if __name__ == "__main__":
    for ii in range(43, 44):
        print('\n\nBeginning round ', ii)
        net, robot_config = demo(backend, test_name='driving_%04i' % ii)
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

            # plot data
            plt.subplot(2, 1, 1)
            target = np.array(net.target_track)
            rover = np.array(net.rover_track)
            plt.plot(target[:, 0], target[:, 1], 'x', mew=3)
            plt.plot(rover[:, 0], rover[:, 1], lw=2)

            plt.subplot(2, 1, 2)
            error = np.array(net.turnerror_track)
            q = np.array(net.q_track)
            qdes = np.array(net.qdes_track)
            plt.plot(q, label='q')
            plt.plot(qdes, '--', label='qdes')
            plt.plot(error, 'r', label='error')
            plt.legend()
            plt.show()

else:
    model, _ = demo()
