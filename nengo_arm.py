"""
To run this demo you will need to download the required stl and texture files
Run the following two commands from the directory this file is in:

TO DOWNLOAD FILES
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1N4RSyJeHCMFKgtjdXOD_4A7izpFcFMKw' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1N4RSyJeHCMFKgtjdXOD_4A7izpFcFMKw" -O meshes && rm -rf /tmp/cookies.txt

TO EXTRACT FILES
tar -zxvf meshes

To run the demo with Nengo running on cpu:
    python nengo_arm.py cpu

To run the demo with Nengo on loihi
    NXSDKHOST=loihighrd python nengo_arm.py
"""
import glfw
import mujoco_py
import nengo
import nengo_loihi
import numpy as np
import sys
import time
import timeit

from nengo_loihi import decode_neurons

from abr_control.controllers import OSC
from abr_control.arms.mujoco_config import MujocoConfig
from abr_control.interfaces.mujoco import Mujoco
from abr_control.controllers import path_planners
from abr_control._vendor.nengolib.stats.ntmdists import (
    ScatteredHypersphere,
    spherical_transform,
)
from nengo_extras.dists import AreaIntercepts, Triangular


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
    # joint pos and vel as input, joint force as output
    n_dof = 5
    n_input = n_dof * 2 + 1  # input to adaptive ensemble
    n_output = n_dof  # adaptive signal sends torque to joints
    n_neurons = 1000  # total number of neurons is n_neurons * n_ensembles
    n_ensembles = 10  # 1 ensemble per Loihi core

    # variances are used to bring each dimension of the input signal to the
    # adaptive ensemble into the -1 to 1 range
    variances = np.array([[6.28,] * 5 + [1.25,] * 5]) / 2.0

    # initialize our robot config for the jaco2
    robot_config = MujocoConfig(xml_file="jaco2", use_sim_state=True)

    # create the Nengo network
    net = nengo.Network(seed=0)
    # create our Mujoco interface
    net.interface = Mujoco(robot_config, dt=0.001, visualize=True)
    net.interface.connect()
    net.interface.send_target_angles(robot_config.START_ANGLES)

    interface = net.interface
    viewer = interface.viewer
    model = net.interface.sim.model
    data = net.interface.sim.data

    # create operational space controller
    net.ctrlr = OSC(
        robot_config,
        kp=100,  # position gain
        kv=20,  # velocity gain
        # control all DOF [x, y, z, alpha, beta, gamma]
        ctrlr_dof=[True, True, True, False, False, False],
    )

    # set up neuron intercepts
    intercepts_bounds = [-0.4, -0.1]
    intercepts_mode = -0.3

    intercepts_dist = AreaIntercepts(
        dimensions=n_input,
        base=Triangular(intercepts_bounds[0], intercepts_mode, intercepts_bounds[1]),
    )
    intercepts = intercepts_dist.sample(n=n_neurons * n_ensembles, rng=rng)
    intercepts = intercepts.reshape(n_ensembles, n_neurons)

    # get target object id so we can change its colour
    target_geom_id = model.geom_name2id("target")
    green = [0, 0.9, 0, 0.5]
    red = [0.9, 0, 0, 0.5]

    bodies = ["link1", "link2", "link3", "link4", "link5", "link6"]
    extra_gravity_force = [
        np.array([0, 0, -9.81, 0, 0, 0]) * model.body_mass[model.body_name2id(body)]
        for body in bodies
    ]

    # set up the target position
    net.interface.viewer.target = np.array([-0.4, 0.5, 0.4])

    # make the default synapse time constant 0.012
    net.config[nengo.Connection].synapse = 0.012
    with net:
        net.u = np.zeros(robot_config.N_JOINTS + 3)
        net.u[robot_config.N_JOINTS:] = np.ones(3) * -0.2  # set gripper forces
        net.trajectory_planner = path_planners.Arc(n_timesteps=1000)
        net.count = 0
        # The simulation will stay inside this node while loop until adaptation is
        # turned on, at which point a signal will be sent out to the ensemble and the
        # neurons will be simulated.

        def arm_func(t, u_adapt):
            ran_at_least_once = False
            while not ran_at_least_once or not viewer.adapt:
                ran_at_least_once = True

                if viewer.exit:
                    glfw.destroy_window(viewer.window)
                    raise ExitSim()

                # change target every 2 simulated seconds
                if net.count % 2000 == 0:
                    # scale random target to +/- 0.6 in xy, and 0.2-0.6 in z
                    # limitting z above 0.2 so we don't have to have a complicated
                    # check to see if x and y are inside the base near the origin
                    viewer.target = np.random.rand(3) * np.array(
                        [1.2, 1.2, 0.4]
                    ) - np.array([0.6, 0.6, -0.2])

                    # generate our path to the new target
                    net.trajectory_planner.generate_path(
                        position=robot_config.Tx('EE'), target_position=viewer.target
                    )
                    interface.set_mocap_xyz("target", viewer.target)
                    net.next_reach = False

                # update our path planner position and orientation --------------------
                net.pos, net.vel = net.trajectory_planner.next()
                orient = np.zeros(3)
                target = np.hstack([net.pos, orient])

                # calculate our control signal ----------------------------------------
                feedback = interface.get_feedback()  # get arm feedback
                net.u[: robot_config.N_JOINTS] = net.ctrlr.generate(
                    q=feedback["q"], dq=feedback["dq"], target=target
                )
                if viewer.adapt:  # add adaptive signal if adaptation is on
                    net.u[:n_dof] += u_adapt * adapt_scale

                # apply our extra gravity term ----------------------------------------
                for ii, body in enumerate(bodies):
                    interface.set_external_force(body, extra_gravity_force[ii])

                # send to mujoco, stepping the sim forward ----------------------------
                interface.send_forces(net.u)

                # change target from red to green if within 0.02m ---------------------
                error = np.linalg.norm(robot_config.Tx('EE') - viewer.target)
                model.geom_rgba[target_geom_id] = green if error < 0.02 else red

                net.count += 1

            # if adaptation is on, generate context signal for neural population ------
            feedback = interface.get_feedback()
            context = np.hstack([feedback["q"][:n_dof], feedback["dq"][:n_dof]])
            # put into the 0-1 range for the spherical transformation
            context = context / variances + 0.5
            # project onto unit hypersphere in larger state space
            context = spherical_transform(context.reshape(1, 10))

            training_signal = -1 * net.ctrlr.training_signal[:n_output]
            output_signal = np.hstack([context.flatten(), training_signal.flatten()])
            return output_signal

        # -----------------------------------------------------------------------------

        arm = nengo.Node(arm_func, size_in=n_dof, size_out=n_input + n_output)

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
            ) / 2000.0
        )  # divide by 100 (neuron firing rate) * 20 (on/off neurons per dim)
        nengo.Connection(onchip_output.neurons, arm, transform=out2arm_transform)

        # set up encoders evently distributed around the unit hypersphere
        encoders_dist = ScatteredHypersphere(surface=True)
        encoders = encoders_dist.sample(n_neurons * n_ensembles, n_input, rng=rng)
        encoders = encoders.reshape(n_ensembles, n_neurons, n_input)

        adapt_ens = []
        conn_learn = []
        for ii in range(n_ensembles):
            adapt_ens.append(
                nengo.Ensemble(
                    n_neurons=n_neurons,
                    dimensions=n_input,
                    radius=np.sqrt(n_input),
                    encoders=encoders[ii],
                    intercepts=intercepts[ii],
                    label="ens%02d" % ii,
                )
            )

            # hook up input signal to adaptive population to provide context
            inp2ens_transform_ii = np.dot(encoders[ii], inp2ens_transform)
            nengo.Connection(
                onchip_input.neurons,
                adapt_ens[ii].neurons,
                transform=inp2ens_transform_ii,
            )

            conn_learn.append(
                nengo.Connection(
                    adapt_ens[ii],
                    onchip_output,
                    learning_rule_type=nengo.PES(pes_learning_rate),
                    transform=rng.uniform(-0.01, 0.01, size=(n_output, n_input)),
                )
            )

            # hook up the training signal to the learning rule
            nengo.Connection(arm[n_input:], conn_learn[ii].learning_rule, synapse=None)

    return net, robot_config


if __name__ == "__main__":
    print("---------------------------------------------")
    print("--------- Applying 2 x earth gravity --------")
    print("---- Use left shift to toggle adaptation ----")
    print("---------------------------------------------")
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
