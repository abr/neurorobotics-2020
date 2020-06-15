"""
To generate training data to train up the RoverVision system, set generate_data=True.


To run the demo with Nengo running on cpu:
    python nengo_rover.py cpu

To run the demo with Nengo on loihi
    NXSDKHOST=loihighrd python nengo_rover.py
"""
import numpy as np
import timeit
import os

import nengo
from nengo_interfaces.mujoco import Mujoco

from data_handler import DataHandler


current_dir = os.path.abspath('.')
if not os.path.exists('%s/figures' % current_dir):
    os.makedirs('%s/figures' % current_dir)
if not os.path.exists('%s/data' % current_dir):
    save_folder = '%s/data' % current_dir

class ExitSim(Exception):
    pass


def generate_data(visualize):
    seed = 0
    np.random.seed(seed)
    # target generation limits
    dist_limit = [0.25, 3.5]
    angle_limit = [-np.pi, np.pi]

    # data collection parameters in steps (1ms per step)
    render_frequency = 100
    total_images_saved = 50000
    max_time_to_target = 2000

    resolution = [32, 32]
    cameras = [4, 1, 3, 2]
    dt = 0.001

    net = nengo.Network(seed=seed)
    # create our Mujoco interface
    interface = Mujoco(
        xml_file="rover.xml",
        folder="",
        dt=dt,
        update_display=100,
        render_params={
            "cameras": cameras,  # camera ids and order to render
            "resolution": resolution,
            "frequency": render_frequency,  # render images from cameras every time step
            "plot_frequency": None,  # do not plot images from cameras
        },
        joint_names=["steering_wheel"],
        track_input=True,
        visualize=visualize,
    )

    # set up the target position
    net.target = np.array([-0.0, 0.0, 0.2])

    if generate_data:
        db_dir = None
        dat = DataHandler(db_dir=db_dir, db_name='rover')
        dat.save(
            data={
                "render_frequency": render_frequency,
                "max_time_to_target": max_time_to_target,
                "cameras": cameras,
                "render_size": resolution,
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

            if np.linalg.norm(local_target) < 3.5:

                if int(t / dt) % render_frequency == 0:
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

            # used to roughly normalize the local target signal to -1:1 range
            output_signal = np.array([local_target[0], local_target[1]])
            return output_signal

        local_target = nengo.Node(output=local_target, size_out=2)

        accel_scale = 1
        steer_scale = 5

        def check_exit_and_gen_target(t, x):
            if net.image_count == total_images_saved:
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

        check_exit_and_gen_target = nengo.Node(
            check_exit_and_gen_target, size_in=2, label="check_exit_and_gen_target"
        )

        mujoco_node = interface.make_node()

        # -----------------------------------------------------------------------------

        # --- set up ensemble to calculate acceleration
        def accel_function(x):
            return min(np.linalg.norm(-x), 1)

        accel = nengo.Ensemble(n_neurons=1, dimensions=2, neuron_type=nengo.Direct())

        nengo.Connection(
            accel,
            mujoco_node[1],
            function=accel_function,
            transform=accel_scale,
            synapse=0,
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

        steer = nengo.Ensemble(n_neurons=1, dimensions=3, neuron_type=nengo.Direct())

        nengo.Connection(
            steer,
            mujoco_node[0],
            function=lambda x: steer_function(x),
            transform=steer_scale,
            synapse=0,
        )

        # add a relay node to amalgamate input to steering ensemble
        steer_input = nengo.Node(size_in=3, label="relay_motor_input")
        nengo.Connection(steer_input, steer)

        # -- connect relevant motor feedback (joint angle of steering wheel)
        nengo.Connection(mujoco_node[0], steer_input[0])

        # if we're not using neural vision, then hook up the local_target node
        # to our motor system to get the location of the target relative to rover
        nengo.Connection(
            local_target,
            steer_input[1:],
            transform=1 / np.pi,
        )
        nengo.Connection(
            local_target, accel, transform=1 / np.pi,
        )
        nengo.Connection(
            local_target, check_exit_and_gen_target, transform=1 / np.pi
        )


    return net


if __name__ == "__main__":
    print('\nGenerating Training Data\n')

    try:
        net = generate_data(visualize=False)
        sim = nengo.Simulator(net, progress_bar=False)
        with sim:
            start_time = timeit.default_timer()
            sim.run(1e5)  # i.e. run until all training images collected
            print("\nRun time: %.5f\n" % (timeit.default_timer() - start_time))
    except ExitSim:
        pass

