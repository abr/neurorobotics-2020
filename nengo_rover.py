"""
To run the demo with Nengo running on cpu:
    python nengo_rover.py cpu

To run the demo with Nengo on loihi
    NXSDKHOST=loihighrd python nengo_rover.py
"""
import glfw
import mujoco_py
import nengo
import nengo_loihi
import numpy as np
import os
import sys
import time

from nengo_loihi import decode_neurons
from abr_control.arms.mujoco_config import MujocoConfig
from abr_control.interfaces.mujoco import Mujoco


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


def demo():
    rng = np.random.RandomState(9)

    # initialize our robot config for the jaco2
    dir_path = os.path.dirname(os.path.realpath(__file__))
    robot_config = MujocoConfig(
        folder=dir_path,
        xml_file="rover.xml",
        use_sim_state=True
    )

    # create the Nengo network
    net = nengo.Network(seed=0)
    # create our Mujoco interface
    net.interface = Mujoco(robot_config, dt=0.001, visualize=True)
    #net.interface.connect(camera_id=0)
    net.interface.connect()
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

    n_dof = 3  # 2 wheels to control
    n_input = 3  # input to neural net is body_com y velocity and error along (x, y) plane
    n_output = n_dof  # output from neural net is torque signals for the wheels

    with net:
        net.count = 0
        def sim_func(t, u):
            if net.count%5 == 0:
                interface.sim.render(1000, 200, camera_name='vision1')
            if net.count%11 == 0:
                interface.sim.render(1000, 200, camera_name='vision2')
            # else:
            #     print((interface.sim.render(100, 100, camera_name='vision2')).shape)

            kp = 2
            feedback = interface.get_feedback()
            u0 = kp * (u[0]- feedback['q'][0]) - .8 * kp * feedback['dq'][0]
            u = [u0, u[1], u[2]]

            if viewer.exit:
                glfw.destroy_window(viewer.window)
                raise ExitSim()

            # send to mujoco, stepping the sim forward --------------------------------
            interface.send_forces(u)

            if net.count % 4000 == 0:
                viewer.target = np.random.rand(3) * np.array(
                    [4, 4, 0]
                ) - np.array([2, 2, 0])
                viewer.target[2] = 0.4
                interface.set_mocap_xyz("target", viewer.target)
                print('target location: ', viewer.target)

            # change target from red to green if within 0.02m -------------------------
            error = viewer.target - robot_config.Tx('EE')
            model.geom_rgba[target_geom_id] = green if np.linalg.norm(error) < 0.02 else red

            # error is in global coordinates, want it in local coordinates for rover --
            # body_xmat will take from local coordinates to global
            R = np.copy(data.body_xmat[EE_id]).reshape(3, 3).T  # R.T = R^-1
            # rotate it so the steering wheels point forward along y
            theta = 3 * np.pi / 2
            R90 = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])
            R = np.dot(R90, R)
            local_error = np.dot(R, error)

            # we also want the egocentric velocity of the rover
            body_com_vel = data.cvel[model.body_name2id('base_link')][3:]
            body_com_vel = np.dot(R, body_com_vel)

            net.count += 1
            output_signal = np.hstack([body_com_vel[1], local_error[:2]])
            return output_signal

        # -----------------------------------------------------------------------------

        sim = nengo.Node(sim_func, size_in=n_dof, size_out=n_input)

        # input_decodeneurons = decode_neurons.Preset5DecodeNeurons()
        # onchip_input = input_decodeneurons.get_ensemble(dim=n_input)
        # nengo.Connection(sim[:n_input], onchip_input, synapse=None)
        # inp2ens_transform = np.hstack(
        #     [np.eye(n_input), -(np.eye(n_input))] * input_decodeneurons.pairs_per_dim
        # )
        #
        # output_decodeneurons = decode_neurons.Preset5DecodeNeurons()
        # onchip_output = output_decodeneurons.get_ensemble(dim=n_output)
        # out2sim_transform = (
        #     np.hstack(
        #         [np.eye(n_output), -(np.eye(n_output))]
        #         * output_decodeneurons.pairs_per_dim
        #     ) / 2000.0
        # )  # divide by 100 (neuron firing rate) * 20 (on/off neurons per dim)
        # nengo.Connection(onchip_output.neurons, sim, transform=out2sim_transform)

        n_neurons = 1000
        encoders = nengo.dists.UniformHypersphere(surface=True).sample(n_neurons, d=n_input)
        brain = nengo.Ensemble(
            # neuron_type=nengo.Direct(),
            n_neurons=n_neurons,
            dimensions=n_input,
            radius=np.sqrt(n_input),
            encoders=encoders,
        )

        # # hook up Mujoco output signal to brain ensemble
        # inp2ens_transform = np.dot(encoders, inp2ens_transform)
        # nengo.Connection(
        #     onchip_input.neurons,
        #     brain.neurons,
        #     transform=inp2ens_transform,
        # )

        nengo.Connection(
            sim,
            brain,
        )

        # hook up the brain ensemble to the Mujoco input
        def steering_function(x):
            body_com_vely = x[0]
            error = x[1:]
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

            return np.array([turn_des, wheels, wheels])

        nengo.Connection(
            brain,
            # onchip_output,
            sim,
            function=steering_function,
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
            sim = nengo.Simulator(net)

        while 1:
            sim.run(1e5)

    except ExitSim:
        pass

    finally:
        net.interface.disconnect()
        sim.close()

else:
    model, _ = demo()
