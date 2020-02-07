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

from nengo_loihi import decode_neurons
from abr_control.arms.mujoco_config import MujocoConfig
from abr_control.interfaces.mujoco import Mujoco
from abr_analyze import DataHandler


class ExitSim(Exception):
    print("Restarting simulation")
    pass


# set up parameters that depend on cpu or loihi as the backend
backend = "cpu"
adapt_scale = 1
pes_learning_rate = 1e-5
if len(sys.argv) > 1:
    for arg in sys.argv:
        arg = str(arg)
        if arg == "loihi":
            backend = "loihi"
            adapt_scale = 10
            pes_learning_rate = 1e-5  # sometimes needs to be scaled
print("Using %s as backend" % backend)


def resize_images(
        image_data, res, rows=None, show_resized_image=False, flatten=True):
    # single image, append 1 dimension so we can loop through the same way
    image_data = np.asarray(image_data)
    if image_data.ndim == 3:
        shape = image_data.shape
        image_data = image_data.reshape((1, shape[0], shape[1], shape[2]))

    shape = image_data.shape

    # expect rgb image data
    assert image_data.shape[3] == 3

    scaled_image_data = []

    for count, data in enumerate(image_data):
        # normalize
        rgb = np.asarray(data)/255

        # select a subset of rows and update the vertical resolution
        if rows is not None:
            rgb = rgb[rows[0]:rows[1], :, :]
            res[0] = rows[1]-rows[0]

        # resize image resolution
        if shape[1] != res[0] or shape[2] != res[1]:
            print('Resolution does not match desired value, resizing...')
            print('Desired Res: ', res)
            print('Input Res: ', [shape[1], shape[2]])
            rgb = cv2.resize(
                rgb, dsize=(res[1], res[0]),
                interpolation=cv2.INTER_CUBIC)

        # visualize scaling for debugging
        if show_resized_image:
            plt.Figure()
            a = plt.subplot(121)
            a.set_title('Original')
            a.imshow(data, origin='lower')
            b = plt.subplot(122)
            b.set_title('Scaled')
            b.imshow(rgb, origin='lower')
            plt.show()

        # flatten to 1D
        #NOTE should use np.ravel to maintain image order
        if flatten:
            rgb = rgb.flatten()
        # scale image from -1 to 1 and save to list
        scaled_image_data.append(rgb*2 - 1)

    scaled_image_data = np.asarray(scaled_image_data)

    return scaled_image_data

def demo():
    seed = 9
    rng = np.random.RandomState(seed)
    # target generation limits
    dist_limit = [0.5, 3.5]
    angle_limit = [-np.pi, np.pi]

    # data collection parameters
    generate_training_data = False
    save_rendered_fig = False
    track_results = True
    target_track = []
    prediction_track = []
    motor_track = []
    n_targets = 100
    # in steps (1ms/step)
    render_frequency = 1
    reaching_steps = 1
    sim_length = reaching_steps * n_targets
    imgs = []


    # network parameters
    n_dof = 3  # 2 wheels to control
    n_input = 5  # input to neural net is body_com y velocity, error along (x, y) plane, and q dq feedback
    n_output = n_dof  # output from neural net is torque signals for the wheels
    weights='saved_net_32x128_learn_xy'
    # we stack feedback from 4 cameras to get a 2pi view
    render_size = [32, 32]
    res = [render_size[0], render_size[1] * 4]
    subpixels = res[0] * res[1] * 3
    # we are passing in one image at a time
    minibatch_size = 1
    filters = [32, 64, 128]
    kernel_size = [3, 3, 3]
    strides = [1, 1, 1]

    dat = DataHandler(db_name='rover_vis_comparison')
    test_name = 'combined_net_0004'
    # dat.save(
    #     data={
    #             'render_frequency': render_frequency,
    #             'reaching_steps': reaching_steps,
    #             'sim_length': sim_length,
    #             'render_size': render_size,
    #             'dist_limit': dist_limit,
    #             'angle_limit': angle_limit
    #            },
    #     save_location='%s/params' % test_name,
    #     overwrite=True)

    # initialize our robot config for the jaco2
    dir_path = os.path.dirname(os.path.realpath(__file__))
    robot_config = MujocoConfig(
        folder=dir_path,
        xml_file="rover.xml",
        use_sim_state=True
    )

    # set up our vision portion of the network in keras and convert to nengo_dl before adding other sections
    image_input = tf.keras.Input(shape=(res[0], res[1], 3), batch_size=minibatch_size)

    conv1 = tf.keras.layers.Conv2D(
        filters=filters[0],
        kernel_size=kernel_size[0],
        strides=strides[0],
        use_bias=True,
        activation=tf.nn.relu,
        data_format="channels_last",
        )

    conv1_out = conv1(image_input)

    conv2 = tf.keras.layers.Conv2D(
        filters=filters[1],
        kernel_size=kernel_size[1],
        strides=strides[1],
        use_bias=True,
        activation=tf.nn.relu,
        data_format="channels_last",
        )(conv1_out)

    conv3 = tf.keras.layers.Conv2D(
        filters=filters[2],
        kernel_size=kernel_size[2],
        strides=strides[2],
        use_bias=True,
        activation=tf.nn.relu,
        data_format="channels_last",
        )(conv2)

    flatten = tf.keras.layers.Flatten()(conv3)

    vis_output_probe = tf.keras.layers.Dense(
        units=2,
        )

    vis_output = vis_output_probe(flatten)

    model = tf.keras.Model(inputs=image_input, outputs=vis_output)

    converter = nengo_dl.Converter(model)
    net = converter.net
    # get our vision connections from the nengo_dl converter
    vision_input = converter.inputs[image_input]
    # vision_input = converter.layer_map[conv1][0][0]
    vision_output = converter.layer_map[vis_output_probe][0][0]
    # vision_output = converter.outputs[vis_output_probe]


    if weights is not None:
        with nengo_dl.Simulator(net, minibatch_size=minibatch_size, seed=seed) as sim:
            sim.load_params(weights)
            sim.freeze_params(net)

    # create our Mujoco interface
    net.interface = Mujoco(robot_config, dt=0.001, visualize=True)
    net.interface.connect(camera_id=0)
    # shorthand
    interface = net.interface
    viewer = interface.viewer
    model = net.interface.sim.model
    data = net.interface.sim.data
    EE_id = model.body_name2id('EE')

    # get names of turning axel and back motors
    joint_names = ['joint%i' % ii for ii in [0, 1, 2]]
    interface.joint_pos_addrs = [model.get_joint_qpos_addr(name)
                            for name in joint_names]
    interface.joint_vel_addrs = [model.get_joint_qvel_addr(name)
                            for name in joint_names]

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
        def sim_func(t, x):

            def render_vision_input():
                # get our image data
                if net.count % render_frequency == 0:
                    net.imgs = []
                    interface.sim.render(render_size[0], render_size[1], camera_name='vision1')
                    #TODO rename the vision sensors so we can stack them sequentially by name
                    net.imgs.append(interface.sim.render(render_size[0], render_size[1], camera_name='vision1', depth=False))
                    net.imgs.append(interface.sim.render(render_size[0], render_size[1], camera_name='vision2', depth=False))
                    net.imgs.append(interface.sim.render(render_size[0], render_size[1], camera_name='vision3', depth=False))
                    net.imgs.append(interface.sim.render(render_size[0], render_size[1], camera_name='vision4', depth=False))

                    net.imgs = (np.hstack(
                            (np.array(
                                np.hstack((net.imgs[3], net.imgs[0]))),
                                np.hstack((net.imgs[2], net.imgs[1])))
                        ))

                    # save relevant data
                    if generate_training_data:
                        #TODO update where we save data as this may be broken from moving the local error calc to node
                        print('Target Count: %i/%i ' % (int(net.count/render_frequency), n_targets))
                        raise NotImplementedError
                        save_data={
                                'rgb': net.imgs,
                                'target': viewer.target,
                                'EE': robot_config.Tx('EE'),
                                'EE_xmat': R_raw,
                                'target': target,
                            }

                        dat.save(
                            data=save_data,
                            save_location='%s/data/%04d' % (test_name, net.count),
                            overwrite=True)

                    # save figure
                    if save_rendered_fig:
                        plt.Figure()
                        plt.imshow(net.imgs, origin='lower')
                        plt.title('%i' % net.count)
                        plt.savefig('images/%04d.png'%net.count)
                        plt.show()


                    # get predicted target from vision
                    net.imgs = resize_images(net.imgs, res=res, rows=None, show_resized_image=False, flatten=True).squeeze()

                    # net.imgs = {
                    #     vision_input: net.imgs.reshape(
                    #         (1, 1, subpixels))
                    # }

                return net.imgs


            def get_feedback():

                error = viewer.target - robot_config.Tx('EE')
                # model.geom_rgba[target_geom_id] = green if np.linalg.norm(error) < 0.02 else red

                # error is in global coordinates, want it in local coordinates for rover --
                # body_xmat will take from local coordinates to global
                R_raw = np.copy(data.body_xmat[EE_id]).reshape(3, 3).T  # R.T = R^-1
                # rotate it so the steering wheels point forward along y
                theta = 3 * np.pi / 2
                R90 = np.array([
                    [np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]
                ])
                R = np.dot(R90, R_raw)
                local_target = np.dot(R, error)
                # print('-------------')
                # print('calc target')
                # print(net.count)
                # print(viewer.target)
                # print(robot_config.Tx)

                # we also want the egocentric velocity of the rover
                body_com_vel_raw = data.cvel[model.body_name2id('base_link')][3:]
                body_com_vel = np.dot(R, body_com_vel_raw)

                feedback = interface.get_feedback()
                q = feedback['q']
                dq = feedback['dq']

                return np.array([body_com_vel[1], q[0], dq[0], local_target[0], local_target[1]])


            net.count += 1
            u = x[:3]
            prediction = x[3:5]
            feedback = get_feedback()
            local_target = feedback[3:]
            output_signal = feedback[:3]
            rendered_image = render_vision_input()
            #TODO change this to index into an array
            output_signal = np.hstack((output_signal, rendered_image))

            # print("MAIN SIM COUNT: ", net.count)
            if viewer.exit or net.count == sim_length:
                if track_results:
                    dat.save(
                        data={'target': target_track,
                              'motor': motor_track,
                              'prediction': prediction_track
                        },
                        save_location=test_name,
                        overwrite=True
                    )
                glfw.destroy_window(viewer.window)
                raise ExitSim

            # update our target
            if net.count % reaching_steps == 0:
                net.target_count += 1
                # phis = [-1.45, 0.4, 1.1]
                # radii = [1.2, 0.8, 2.6]
                phis = np.linspace(-3.14, 3.14, 200)
                radii = np.linspace(0.8, 3.4, 200)
                # phi = np.random.uniform(low=angle_limit[0], high=angle_limit[1])
                # radius = np.random.uniform(low=dist_limit[0], high=dist_limit[1])
                phi = phis[net.target_count]
                radius = radii[net.target_count]
                viewer.target = [
                    np.cos(phi) * radius,
                    np.sin(phi) * radius,
                    0.4]
                interface.set_mocap_xyz("target", viewer.target)

            # send to mujoco, stepping the sim forward --------------------------------
            interface.send_forces(0*np.asarray(u))

            if net.count % 500 == 0:
                print('Time Since Start: ', timeit.default_timer() - start)

            if track_results and net.count % render_frequency == 0:
                print('NET COUNT: ', net.count)
                motor_track.append(u)
                prediction_track.append(prediction)
                target_track.append(local_target)

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
            u0 = kp * (turn_des- q) - kv * dq
            u = np.array([u0, wheels, wheels])
            return u


        # -----------------------------------------------------------------------------
        sim = nengo.Node(sim_func, size_in=5, size_out=subpixels+3)

        # feedback_node = nengo.Node(get_feedback, size_in=None, size_out=5)

        # render_node = nengo.Node(render_vision_input, size_in=None, size_out=subpixels)

        n_motor_neurons = 1000
        encoders = nengo.dists.UniformHypersphere(surface=True).sample(n_motor_neurons, d=n_input)
        motor_control = nengo.Ensemble(
            # neuron_type=nengo.Direct(),
            n_neurons=n_motor_neurons,
            dimensions=n_input,
            radius=np.sqrt(n_input),
            encoders=encoders,
            seed=seed
        )

        # send vision prediction to sim for data collection
        nengo.Connection(
            vision_output,
            sim[3:5],
            synapse=None
        )

        # send rendered image to vision network
        nengo.Connection(
            sim[3:],
            vision_input,
            synapse=None
        )

        # send vision prediction to motor control
        nengo.Connection(
            vision_output,
            motor_control[3:]
        )

        # send motor feedback to motor control
        nengo.Connection(
            sim[:3],
            motor_control[:3],
            synapse=None
        )

        # send motor signal to sim
        nengo.Connection(
            motor_control,
            # onchip_output,
            sim[:3],
            function=steering_function
        )

    return net, robot_config


if __name__ == "__main__":
    net, robot_config = demo()
    try:
        if backend == "loihi":
            sim = nengo_loihi.Simulator(
                net, target="loihi", hardware_options=dict(snip_max_spikes_per_step=300)
            )
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
