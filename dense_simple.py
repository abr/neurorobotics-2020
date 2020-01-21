import nengo
import nengo_dl
import numpy as np
import cv2
import warnings
import tensorflow as tf
import matplotlib.pyplot as plt
import time

def get_angle(image):
    # only looking at R channel, so max last R channel the max index
    index = min((np.argmax(image), len(image)-3))
    # get normalized index
    norm_index1 = index/(len(image)-3)
    # scaled index from -1 to 1
    norm_index2 = norm_index1 * 2 -1
    # get angle
    angle = np.arccos(norm_index2)
    # scale from -pi to pi
    angle = angle*2 - np.pi
    return angle


warnings.simplefilter("ignore")
n_steps = 1
n_training = 100
minibatch_size = 100 #int(n_training/100)
n_validation = 1000
output_dims = 1
n_neurons = 1000
epochs = 10
res=[1, 10]
pixels = res[0] * res[1]
subpixels = pixels*3
seed = 0

# generate data
training_images = []
training_targets = []
training_angle_targets = []
zeros = np.zeros((res[0], res[1], 3))
targets = np.linspace(-3.14, 3.14, pixels)
assert output_dims == 1

for ii in range(0, n_training):
    index = np.random.randint(low=0, high=pixels)
    data = np.copy(zeros)
    data[0][index] = [1, 0, 0]
    # plt.figure()
    # plt.imshow(data)
    # plt.show()
    data = np.asarray(data).flatten()
    angle = get_angle(data)
    training_angle_targets.append(angle)
    training_images.append(data)
    training_targets.append(index)

training_images = np.asarray(training_images)
training_targets = np.array(training_targets)

# Define network
net = nengo.Network(seed=seed)

net.config[nengo.Ensemble].neuron_type = nengo.RectifiedLinear()
net.config[nengo.Ensemble].max_rates = nengo.dists.Choice([100])
net.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
net.config[nengo.Connection].synapse = None

with net:
    print('\n\nsubpixels: ', subpixels)
    print('n output dims: ', output_dims)
    image_input = nengo.Node(np.zeros(subpixels))
    dense_layer = nengo.Ensemble(
        n_neurons=n_neurons,
        dimensions=subpixels)

    image_output = nengo.Node(size_in=output_dims)

    nengo.Connection(
        image_input, dense_layer, label='input_conn')

    # weights = np.zeros((output_dims, dense_layer.n_neurons))
    #
    # nengo.Connection(
    #     dense_layer.neurons,
    #     image_output,
    #     label='output_conn',
    #     transform=weights)

    nengo.Connection(
        dense_layer,
        image_output,
        label='output_conn',
        function=get_angle)

    #input_probe = nengo.Probe(image_input)
    output_probe = nengo.Probe(image_output, synapse=None, label='output_filtered')
    #output_probe_no_filter = nengo.Probe(image_output, synapse=None, label='output_no_filter')

    # Training, turn off synapses for training to simplify
    for conn in net.all_connections:
        conn.synapse = None

# with nengo.Simulator(net) as sim:
#     sim.run(1)
#     np.set_printoptions(threshold=1e5)
#     print(sim.data[input_probe])
#     print(sim.data[output_probe])
#
#     import matplotlib.pyplot as plt
#     plt.plot(sim.data[input_probe])
#     plt.plot(sim.data[output_probe])
#     plt.savefig('learn_arccos')

# with nengo_dl.Simulator(net, minibatch_size=minibatch_size, seed=seed) as sim:
#     input_dict = {image_input: training_images.reshape(n_training, n_steps, subpixels)}
#     predictions = sim.predict(input_dict, n_steps=n_steps, stateful=False)[output_probe].squeeze()
#     print(predictions)
#     import matplotlib.pyplot as plt
#     plt.plot(predictions)
#     plt.plot(training_angle_targets, label='target', color='r')
#     plt.savefig('learn_arccos')

with nengo_dl.Simulator(net, minibatch_size=minibatch_size, seed=seed) as sim:

    training_images_dict = {
        image_input: training_images.reshape(
            (n_training, n_steps, subpixels))
    }

    # training_targets_dict = {
    #     #output_probe_no_filter: training_targets.reshape(
    #     output_probe: training_targets.reshape(
    #         (n_training, n_steps, output_dims))
    #     }

    # def _test_mse(y_true, y_pred):
    #     return tf.reduce_mean(tf.square(y_pred[:, -10:] - y_true[:, -10:]))
    #
    # print('Evaluation Before Training:')
    # sim.compile(loss={output_probe: _test_mse})
    # sim.evaluate(training_images_dict, training_targets_dict)
    #
    # print('Training')
    # sim.compile(optimizer=tf.optimizers.RMSprop(0.01),
    #             # loss={output_probe_no_filter: tf.losses.mse})
    #             loss={output_probe: tf.losses.mse})
    # sim.fit(training_images_dict, training_targets_dict, epochs=epochs)
    # # save parameters back into net
    # sim.freeze_params(net)
    #
    # print('Evaluation Before Training:')
    # sim.compile(loss={output_probe: _test_mse})
    # sim.evaluate(training_images_dict, training_targets_dict)
    # print('TRAINING NEURONS TYPE: ', net.config[nengo.Ensemble].neuron_type)


    # input is flattened, so total_pixels / pixels(per image)
    batch_size = int(training_images.shape[0]/pixels)

    predictions = sim.predict(training_images_dict, n_steps=n_steps, stateful=False)[output_probe]
    predictions = np.asarray(predictions).squeeze()

    fig = plt.Figure()
    #plt.plot(sorted(training_targets), label='target', color='r')
    plt.plot(training_angle_targets, label='target', color='r')
    plt.plot(predictions, label='predictions', color='k')
    plt.legend()
    plt.savefig('prediction_results.png')
    plt.show()
