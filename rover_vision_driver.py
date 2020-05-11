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
from nengo_loihi import decode_neurons

from abr_control.arms.mujoco_config import MujocoConfig
from abr_control.interfaces.mujoco import Mujoco
from abr_control._vendor.nengolib.stats import ScatteredHypersphere, spherical_transform
from abr_analyze import DataHandler

from rover_vision import RoverVision
from loihi_rate_neuron import LoihiRectifiedLinear


class ExitSim(Exception):
    print("Restarting simulation")
    pass

class HetDecodeNeurons(nengo_loihi.decode_neurons.OnOffDecodeNeurons):
    """Uses heterogeneous on/off pairs with pre-set values per dimension.

    The script for configuring these values can be found at:
        nengo-loihi-sandbox/utils/interneuron_unidecoder_design.py
    """

    def __init__(self, pairs_per_dim=500, dt=0.001, rate=None):
        super(HetDecodeNeurons, self).__init__(pairs_per_dim=pairs_per_dim, dt=dt, rate=rate)

        # Parameters determined by hyperopt
        intercepts = np.linspace(-1.053, 0.523, self.pairs_per_dim)
        max_rates = np.linspace(200, 250, self.pairs_per_dim)
        gain, bias = self.neuron_type.gain_bias(max_rates, intercepts)

        target_point = 0.947
        target_rate = np.sum(self.neuron_type.rates(target_point, gain, bias))
        self.scale = 1.05 * target_point / (self.dt * target_rate)

        self.gain = gain.repeat(2)
        self.bias = bias.repeat(2)

    def __str__(self):
        return "%s(dt=%0.3g, rate=%0.3g)" % (type(self).__name__, self.dt, self.rate)

# set up parameters that depend on cpu or loihi as the backend
backend = "cpu"
if len(sys.argv) > 1:
    if sys.argv[1] == "loihi":
        backend = "loihi"
print("Using %s as backend" % backend)

generate_data = False
plot_mounted_camera_freq = None  # = None to not plot

# scale down and bias the input from the sim node to be in -1 to 1 range
means = 0.1
scales = 0.75
# means = np.array([0.061, -0.203, 0.1])
# scales = np.array([1.0, 2.0, 0.75])
# Steering input mins:  [-0.92, -2.138, -0.694, -22.481, -2.849, -3.344]
# Steering input maxs:  [1.33, 2.224, 0.679, 18.685, 3.245, 2.112]


def get_weights(n_motor_neurons, n_input, function):
    """ Create decoders to use in the nengo_loihi simulation. Need to solve for them
    in nengo.Simulator and then break up the connection from neurons to 3 separete
    output in nengo_loihi.Simulator because of a 4096 connection limit.
    """
    nengo_loihi.set_defaults()

    seed = 9
    rng = np.random.RandomState(seed)

    net = nengo.Network()
    with net:
        nengo_loihi.add_params(net)

        motor_control = nengo.Ensemble(
            neuron_type=nengo_loihi.neurons.LoihiSpikingRectifiedLinear(),
            n_neurons=n_motor_neurons,
            dimensions=n_input,
            # the input is mean subtracted and scaled to be between -1 and 1,
            radius=np.sqrt(n_input),
            seed=seed,
            label='motor_control',
        )
        output = nengo.Node(size_in=2)

        conn = nengo.Connection(
            motor_control,
            output,
            function=function,
        )

    with nengo.Simulator(net) as sim:
        decoders = sim.signals[sim.model.sig[conn]["weights"]]

    return decoders



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
    n_input = 3  # input to neural net is body_com x, y velocity, error along (x, y) plane, and q feedback
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
        if backend == "loihi":
            nengo_loihi.add_params(net)

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
                "/home/tdewolf/Downloads/data/abr_analyze/loihirelu/400/epoch_306"
            )
            with visionsim:
                visionsim.freeze_params(visionnet)
            _, visionnet = vision.convert_nengodl_to_nengo(visionnet, loihi=True)

        # net.config[vision.nengo_conv0.ensemble].on_chip = False
        # print('vision conv0 on chip: ', net.config[vision.nengo_conv0.ensemble].on_chip)

        def sim_func(t, x):
            # NOTE: radius of nengo_loihi decoder_neurons is 1
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
                # body_com_vel[0],
                # body_com_vel[1],
                feedback['q'][0] * (1 if rover_name == 'rover.xml' else -1),
                local_target[0],
                local_target[1],
            ])

            # scale down for sending into motor_control ensemble
            output_signal[:3] = (output_signal[:3] - means) / scales

            if net.count % 500 == 0:
                print("Time Since Start: ", timeit.default_timer() - start)

            net.count += 1
            return output_signal

        # -----------------------------------------------------------------------------
        sim = nengo.Node(sim_func, size_in=4, size_out=n_input, label='sim')

        # def speed_function(x, track=False):
        #     if net.count > 0 and track:
        #         net.turn_input_track.append(x)
        #
        #     error = x[1:] * np.pi  # take out the error signal from vision
        #
        #     # set a max speed of the robot, but when we get close slow down
        #     # based on how far away the target is
        #     wheels = min(dist, 30)
        #     # wheels -= body_com_vel
        #     wheels *= 20  # account for point mass
        #     if error[1] > 0:
        #         wheels *= -1
        #
        #     dist = np.linalg.norm(error * 100)
        #     # normalize output for loihi decodeneurons by / max value possible
        #     wheels /= (30 * 20)
        #     wheels *= (-1 if rover_name == 'rover.xml' else 1)
        #
        #     return wheels


        def steering_function(x, dimension=None, track=False):
            if net.count > 0 and track:
                net.input_track.append(x)

            error = x[1:] * np.pi  # take out the error signal from vision
            # scale up input from sim that has been scaled down to -1 to 1 range
            q = x[0] * scales + means
            # body_com_vel = np.linalg.norm(x[:2])
            # q = x[2]

            dist = np.linalg.norm(error * 100)

            # input to arctan2 is modified to account for (x, y) axes of rover vs
            # the (x, y) axes of the environment
            turn_des = np.arctan2(-error[0], abs(error[1]))

            # set a max speed of the robot, but when we get close slow down
            # based on how far away the target is
            wheels = min(dist, 30)
            # wheels -= body_com_vel
            wheels *= 20  # account for point mass
            if error[1] > 0:
                wheels *= -1

            # print('turn des : %.3f' % turn_des)
            # print('current angle: %.3f ' % q)
            # print('error: %.3f' % (turn_des - q))

            u0 = (turn_des - q)

            # normalize output for loihi decodeneurons by / max value possible
            wheels /= (30 * 20)
            u0 /= 2

            u = np.array([
                u0 * (1 if rover_name == 'rover.xml' else -1),
                wheels * (-1 if rover_name == 'rover.xml' else 1)
            ])

            # record input for finding mean and variance values
            if net.count > 0 and track:
                net.turnerror_track.append(turn_des - q)
                net.q_track.append(q)
                net.qdes_track.append(turn_des)

            # if dimension == 0:
            #     return u[0]
            # elif dimension == 1:
            #     return u[1]
            return u

        n_motor_neurons = 4096
        from abr_control._vendor.nengolib.stats.ntmdists import ScatteredHypersphere
        # decoders = get_weights(n_motor_neurons, n_input, steering_function)
        motor_control = nengo.Ensemble(
            # neuron_type=nengo.Direct(),
            neuron_type=nengo_loihi.neurons.LoihiSpikingRectifiedLinear(),
            n_neurons=n_motor_neurons,
            dimensions=n_input,
            max_rates=nengo.dists.Uniform(175, 220),
            # the input is mean subtracted and scaled to be between -1 and 1,
            radius=np.sqrt(n_input),
            # eval_points=ScatteredHypersphere(surface=False).sample(n=n_motor_neurons, d=n_input) * np.sqrt(n_input),
            encoders=ScatteredHypersphere(surface=False).sample(n=n_motor_neurons, d=n_input),
            # seed=seed,
            label='motor_control',
        )

        # output_decodeneurons1 = HetDecodeNeurons(pairs_per_dim=1000)
        # onchip_output1 = output_decodeneurons1.get_ensemble(dim=2)
        # onchip_output1.radius = np.sqrt(2)

        # output_decodeneurons2 = HetDecodeNeurons()
        # onchip_output2 = output_decodeneurons2.get_ensemble(dim=1)
        # onchip_output2.radius = np.sqrt(2)

        # output_decodeneurons3 = HetDecodeNeurons(pairs_per_dim=1000)
        # onchip_output3 = output_decodeneurons3.get_ensemble(dim=1)
        # onchip_output3.radius = np.sqrt(2)

        # output_decodeneurons4 = decode_neurons.Preset10DecodeNeurons()
        # onchip_output4 = output_decodeneurons4.get_ensemble(dim=1)
        # onchip_output4.radius = np.sqrt(2)

        # n_conns = n_motor_neurons // 3
        nengo.Connection(
            # motor_control.neurons[:n_conns],
            motor_control,
            # onchip_output1,
            sim[:2],
            # transform=decoders[:, :n_conns],
            function=lambda x: steering_function(x),
            synapse=0.005,
        )
        # nengo.Connection(
        #     # motor_control.neurons[n_conns:2*n_conns],
        #     motor_control,
        #     onchip_output2,
        #     # transform=decoders[:, n_conns:2*n_conns],
        #     function=lambda x: steering_function(x, dimension=0),
        #     synapse=None,
        # )
        # nengo.Connection(
        #     # motor_control.neurons[2*n_conns:],
        #     motor_control,
        #     onchip_output3,
        #     # transform=decoders[:, 2*n_conns:],
        #     function=lambda x: steering_function(x, dimension=1),
        #     synapse=None,
        # )
        # nengo.Connection(
        #     # motor_control.neurons[2*n_conns:],
        #     motor_control,
        #     onchip_output3,
        #     # transform=decoders[:, 2*n_conns:],
        #     function=lambda x: steering_function(x, dimension=1),
        #     synapse=None,
        # )

        # nengo.Connection(onchip_output1, sim[:2], transform=1, synapse=0.001)
        # # nengo.Connection(onchip_output2, sim[0], transform=1/2, synapse=0.001)
        # nengo.Connection(onchip_output3, sim[1], transform=1, synapse=0.001)
        # # nengo.Connection(onchip_output4, sim[1], transform=1/2, synapse=0.001)

        relay_motor_input = nengo.Node(size_in=n_input)
        nengo.Connection(relay_motor_input, motor_control, synapse=None)

        # send vision prediction to motor control
        vision_synapse = 0#0.01
        # if neural_vision:
        # # TODO: connect directly from the last layer of vision net to the motor_control ensemble
        # # so the system doesn't send the signal to the host and then back to the chip
        #     relay = nengo.Node(size_in=2, label='relay')
        #     nengo.Connection(vision.nengo_output, relay, synapse=vision_synapse)
        #     nengo.Connection(relay, relay_motor_input[3:], synapse=None)
        #     nengo.Connection(relay, sim[2:4], synapse=None)
        # else:
        nengo.Connection(sim[1:], relay_motor_input[1:], synapse=vision_synapse, transform=1/np.pi)
        nengo.Connection(sim[1:], sim[2:4], synapse=vision_synapse)

        # send motor feedback to motor control
        nengo.Connection(sim[0], relay_motor_input[0], synapse=None)

        # create a direct mode ensemble performing the same calculation for debugging
        motor_control_direct = nengo.Ensemble(
            neuron_type=nengo.Direct(),
            n_neurons=n_motor_neurons,
            dimensions=n_input,
            label='motor_control_direct',
        )

        dead_end = nengo.Node(size_in=2, label='dead_end')
        # send motor signal to sim
        nengo.Connection(
            motor_control_direct,
            dead_end,
            function=lambda x: steering_function(x, track=True),
            transform=[[2 * 5000, 0], [0, 20 * 30]],
            synapse=0,
        )

        # send vision prediction to motor control
        # if neural_vision:
        #     nengo.Connection(relay, motor_control_direct[3:], synapse=None)
        # else:
        nengo.Connection(sim[1:], motor_control_direct[1:], synapse=vision_synapse, transform=1/np.pi)

        # send motor feedback to motor control
        nengo.Connection(sim[0], motor_control_direct[0], synapse=None)
        net.dead_end_probe = nengo.Probe(dead_end)

        # TODO: set up decodeneurons optimally tuned for 4d output, each d range -1:1

    return net


if __name__ == "__main__":
    for ii in range(45, 46):
        print('\n\nBeginning round ', ii)
        net = demo(backend, test_name='driving_%04i' % ii, neural_vision=False)
        for conn in net.all_connections:
            print(conn)
        try:
            if backend == "loihi":
                sim = nengo_loihi.Simulator(net, target="sim")
                #     , target="loihi", hardware_options=dict(snip_max_spikes_per_step=300)
                # )
            elif backend == "cpu":
                sim = nengo.Simulator(net, progress_bar=False)

            while 1:
                sim.run(.1)

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

            plt.figure()
            u_track = np.array(net.u_track)
            plt.title('Control signal u')
            plt.subplot(2, 1, 1)
            plt.plot(u_track[:, 0], alpha=.5)
            plt.plot(direct[:, 0], '--', lw=5)
            plt.subplot(2, 1, 2)
            plt.plot(u_track[:, 1], alpha=.5)
            plt.plot(direct[:, 1], '--', lw=3)

            # NOTE: ignore this plot if the motor_control ensemble is running with neurons
            # because it will just be the output from calculating the decoders
            plt.figure()
            inputs = np.array(net.input_track)
            for ii in range(inputs.shape[1]):
                plt.subplot(inputs.shape[1], 1, ii+1)
                plt.plot(inputs[:, ii])
            plt.suptitle('Motor inputs')

            print('Steering input means: ', [float('%.3f' % val) for val in np.mean(inputs, axis=0)])
            print('Steering input mins: ', [float('%.3f' % val) for val in np.min(inputs, axis=0)])
            print('Steering input maxs: ', [float('%.3f' % val) for val in np.max(inputs, axis=0)])

            plt.show()

else:
    model, _ = demo()
