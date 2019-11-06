import nengo
import nengo_loihi
import numpy as np
import timeit
import traceback

from nengo_loihi import decode_neurons

from abr_control.controllers import OSC, Damping
from abr_control.arms.mujoco_config import MujocoConfig as arm
from abr_control.interfaces.mujoco import Mujoco
from abr_control.utils import transformations

from utils import AreaIntercepts, Triangular, scale_inputs, ScatteredHypersphere


rng = np.random.RandomState(9)

n_input = 10
n_output = 5

n_neurons = 1000
n_ensembles = 5
pes_learning_rate = 5e-5
seed = 0
spherical = True  # project the input onto the surface of a D+1 hypersphere
if spherical:
    n_input += 1

means = ([0.12, 2.14, 1.87, 4.32, 0.59, 0.12, -0.38, -0.42, -0.29, 0.36],)
variances = ([0.08, 0.6, 0.7, 0.3, 0.6, 0.08, 1.4, 1.6, 0.7, 1.2],)

backend = nengo_loihi

# synapse time constants
tau_input = 0.012  # on input connection
tau_training = 0.012  # on the training signal
tau_output = 0.012  # on the output from the adaptive ensemble
# NOTE: the time constant on the neural activity used in the learning
# connection is the default 0.005, and can be set by specifying the
# pre_synapse parameter inside the PES rule instantiation

# set up neuron intercepts
intercepts_bounds = [-0.3, 0.1]
intercepts_mode = 0.1

intercepts_dist = AreaIntercepts(
    dimensions=n_input,
    base=Triangular(intercepts_bounds[0], intercepts_mode, intercepts_bounds[1]),
)
intercepts = intercepts_dist.sample(n=n_neurons * n_ensembles, rng=rng)
intercepts = intercepts.reshape(n_ensembles, n_neurons)

# TODO: add weight initialization ability to nengo_loihi
# weights = np.zeros((n_ensembles, n_output, n_neurons))
# print("Initializing connection weights to all zeros")

np.random.seed = seed
encoders_dist = ScatteredHypersphere(surface=True)
encoders = encoders_dist.sample(n_neurons * n_ensembles, n_input, rng=rng)
encoders = encoders.reshape(n_ensembles, n_neurons, n_input)

# initialize our robot config for the jaco2
robot_config = arm('jaco2')

# damp the movements of the arm
damping = Damping(robot_config, kv=10)
# instantiate controller
ctrlr = OSC(
    robot_config,
    kp=200,
    null_controllers=[damping],
    # vmax=[0.5, 0],  # [m/s, rad/s]
    # control (x, y, z) out of [x, y, z, alpha, beta, gamma]
    ctrlr_dof = [True, True, True, False, False, False])

# create our Mujoco interface
interface = Mujoco(robot_config, dt=.001, visualize=True)
interface.connect()
interface.send_target_angles(robot_config.START_ANGLES)

nengo_model = nengo.Network(seed=seed)
# Set the default neuron type for the network
nengo_model.config[nengo.Ensemble].neuron_type = nengo.LIF()
# nengo_model.config[nengo.Connection].synapse = None

feedback = interface.get_feedback()
start = robot_config.Tx('EE', feedback['q'])
# make the target offset from that start position
target_xyz = start + np.array([0.2, -0.2, -0.2])
interface.set_mocap_xyz(name='target', xyz=target_xyz)

with nengo_model:
    count = 0
    def arm_func(t, u_adapt):
        global count
        feedback = interface.get_feedback()

        target = np.hstack([
            interface.get_xyz('target', object_type='mocap'),
            transformations.euler_from_quaternion(
                interface.get_orientation('target', object_type='mocap'), 'rxyz')])

        u = ctrlr.generate(
            q=feedback['q'],
            dq=feedback['dq'],
            target=target,
        )

        # add an additional force for the controller to adapt to
        extra_gravity = robot_config.g(feedback['q']) * 2
        u_total = u + extra_gravity
        u_total[:5] += u_adapt  # add in adaptive term

        interface.send_forces(u_total)  #, update_display=True if count % 2 == 0 else False)
        feedback = interface.get_feedback()

        output = scale_inputs(
            spherical, means, variances,
            np.hstack([feedback['q'][:5], feedback['dq'][:5]]),
        )

        count += 1
        # TODO: scale the training signal here
        return np.hstack([output.flatten(), -ctrlr.training_signal[:5].flatten()])

    arm = nengo.Node(arm_func, size_in=n_output, size_out=n_input+n_output, label="arm")
    arm_probe = nengo.Probe(arm)

    input_decodeneurons = decode_neurons.Preset5DecodeNeurons()
    onchip_input = input_decodeneurons.get_ensemble(dim=n_input)
    nengo.Connection(arm[:n_input], onchip_input, synapse=None)
    inp2ens_transform = np.hstack(
        [np.eye(n_input), -np.eye(n_input)] * input_decodeneurons.pairs_per_dim
    )

    output_decodeneurons = decode_neurons.Preset5DecodeNeurons()
    onchip_output = output_decodeneurons.get_ensemble(dim=n_output)
    out2arm_transform = np.hstack(
        [np.eye(n_output), -np.eye(n_output)] * output_decodeneurons.pairs_per_dim
    ) / 2000.  # divide by 100 (neuron firing rate) * 20 (on/off neurons per dim)
    nengo.Connection(onchip_output.neurons, arm, transform=out2arm_transform, synapse=tau_output)

    adapt_ens = []
    conn_learn = []
    for ii in range(n_ensembles):
        adapt_ens.append(
            nengo.Ensemble(
                n_neurons=n_neurons,
                dimensions=n_input,
                intercepts=intercepts[ii],
                radius=np.sqrt(n_input),
                encoders=encoders[ii],
                label="ens%02d" % ii,
            )
        )

        # hook up input signal to adaptive population to provide context
        inp2ens_transform_ii = np.dot(encoders[ii], inp2ens_transform)
        nengo.Connection(onchip_input.neurons, adapt_ens[ii].neurons,
                         transform=inp2ens_transform_ii, synapse=tau_input)

        conn_learn.append(
            nengo.Connection(
                adapt_ens[ii],
                onchip_output,
                learning_rule_type=nengo.PES(
                    pes_learning_rate, pre_synapse=tau_training,
                ),
                transform=rng.uniform(-0.01, 0.01, size=(n_output, n_input)),
            )
        )

        # hook up the training signal to the learning rule
        # TODO: account for scaling on the transform here
        nengo.Connection(
            arm[n_input:], conn_learn[ii].learning_rule, synapse=None,
        )

try:
    # with nengo_loihi.Simulator(
    #         nengo_model,
    #         target='loihi',
    #         hardware_options=dict(snip_max_spikes_per_step=300)) as sim:
    with nengo.Simulator(nengo_model) as sim:
        sim.run(0.01)
        start = timeit.default_timer()
        sim.run(20)
        print('Run time: %0.5f' % (timeit.default_timer() - start))
        print('timers: ', sim.timers["snips"])
finally:
    interface.disconnect()



