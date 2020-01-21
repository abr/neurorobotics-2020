import nengo
import nengo_dl
import numpy as np
import cv2
import warnings
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from abr_analyze import DataHandler

def get_angle(image):
    # only looking at R channel, so max last R channel the max index
    image = image.reshape((int(len(image)/3), 3))
    #index = min((np.argmax(image), len(image)-3))
    index = np.argmax(image[:, 0])
    # get normalized index
    #norm_index1 = index/(len(image)-3)
    # scaled index from -1 to 1
    #norm_index2 = norm_index1 * 2 -1
    # subtract 1 to account for base 0
    norm_index = index/(image.shape[0]-1) *2 -1
    # # get angle
    #angle = np.arccos(norm_index2)
    angle = np.arcsin(norm_index)
    # scale from -pi to pi
    angle = angle*2
    #angle = angle*2 - np.pi
    return angle #, index

def preprocess_data(image_data, res, show_resized_image):
    # single image, append 1 dimension so we can loop through the same way
    image_data = np.asarray(image_data)
    if image_data.ndim == 3:
        shape = image_data.shape
        image_data = image_data.reshape((1, shape[0], shape[1], shape[2]))

    # expect rgb image data
    assert image_data.shape[3] == 3
    # expect square image
    assert image_data.shape[1] == image_data.shape[2]

    scaled_image_data = []
    scaled_target_data = []
    for data in image_data:
    #for ii in range(64):
        # zeros = np.zeros((64, 64, 3))
        # rgb = zeros
        # rgb[10][ii] = [1, 0, 0]
        # normalize
        rgb = np.asarray(data)/255
        # resize image resolution
        rgb = cv2.resize(
            rgb, dsize=(res[0], res[1]),
            interpolation=cv2.INTER_CUBIC)

        #TODO move this up so we only process one row
        # get a single row
        rgb = rgb[10, :, :].reshape((1, res[1], 3))

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
        rgb = rgb.flatten()
        # scale from -1 to 1 and save to list
        scaled_image_data.append(rgb*2 - 1)

        angle = get_angle(rgb)
        scaled_target_data.append(angle)

        # plt.figure()
        # plt.subplot(311)
        # plt.title('estimated: %i' % (index))
        # # scaled image
        # plt.imshow(rgb.reshape((1, 64, 3)), origin='lower')
        # plt.subplot(312)
        # # estimated location of target
        # a = np.zeros((1, 64, 3))
        # a[0][int(index)] = [1,1,1]
        # plt.imshow(a, origin='lower')
        # plt.subplot(313)
        # plt.xlim(-3.14, 3.14)
        # # estimated target angle
        # plt.scatter(angle, 1, c='r')
        # #plt.savefig('gif_cache/%04d'%ii)
        # plt.show()

    scaled_image_data = np.asarray(scaled_image_data).flatten()
    scaled_target_data = np.asarray(scaled_target_data)

    return scaled_image_data, scaled_target_data


def load_data(label='training_0000', n_imgs=None):
    dat = DataHandler('rover_training_0001')
    # training_images = dat.load(['row_image'], 'training_0000/data/single_row')['row_image']
    # # separate into 1000 original images, 64 pixels wide
    # training_images = training_images.reshape((1000, 192))
    # #validation_images = dat.load(['row_image'], 'validation_0000/data/single_row')['row_image']

    # load training images
    training_images = []
    training_targets = []
    for nn in range(1, n_imgs+1):
        data = dat.load(parameters=['rgb', 'local_error'], save_location='%s/data/%04d' % (label, nn))
        training_images.append(data['rgb'])
        training_targets.append(data['local_error'])

    # scale image resolution
    training_images, training_targets = preprocess_data(
        image_data=training_images,
        res=[64, 64], show_resized_image=False)

    return training_images, training_targets

def gen_data(n_images, res, pixels):
    # generate data
    zeros = np.zeros((res[0], res[1], 3))
    targets = np.linspace(-3.14, 3.14, pixels)
    images = []
    targets = []
    training_angle_targets = []
    for ii in range(0, n_images):
        index = np.random.randint(low=0, high=pixels)
        data = np.copy(zeros)
        data[0][index] = [1, 0, 0]
        # plt.figure()
        # plt.imshow(data)
        # plt.show()
        data = np.asarray(data).flatten()
        angle = get_angle(data)
        training_angle_targets.append(angle)
        images.append(data)
        #targets.append(index)

    images = np.asarray(images)
    #targets = np.array(targets)
    targets = np.array(training_angle_targets)
    return images, targets


warnings.simplefilter("ignore")
n_steps = 1
n_training = 10000
minibatch_size = 100 #int(n_training/100)
n_validation = 1000
output_dims = 1
n_neurons = 10000
epochs = 10
res=[1, 64]
pixels = res[0] * res[1]
subpixels = pixels*3
seed = 0
np.random.seed(seed)

assert output_dims == 1
# # generate data
# training_images, training_targets = gen_data(n_training, res=res, pixels=pixels)
# validation_images, validation_targets = gen_data(n_validation)

# load data
training_images, training_targets = load_data('training_0000', n_training)
validation_images, validation_targets = load_data('validation_0000', n_validation)

# ---------------------------------------Define network
net = nengo.Network(seed=seed)

net.config[nengo.Ensemble].neuron_type = nengo.RectifiedLinear()
net.config[nengo.Ensemble].max_rates = nengo.dists.Choice([100])
net.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
net.config[nengo.Connection].synapse = None

learn_func = True

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

    if learn_func:
        weights = np.zeros((output_dims, dense_layer.n_neurons))

        nengo.Connection(
            dense_layer.neurons,
            image_output,
            label='output_conn',
            transform=weights)
    else:
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

    validation_images_dict = {
        image_input: validation_images.reshape(
            (n_validation, n_steps, subpixels))
    }

    if learn_func:
        training_targets_dict = {
            #output_probe_no_filter: training_targets.reshape(
            output_probe: training_targets.reshape(
                (n_training, n_steps, output_dims))
            }

        print('Training')
        sim.compile(optimizer=tf.optimizers.RMSprop(0.01),
                    # loss={output_probe_no_filter: tf.losses.mse})
                    loss={output_probe: tf.losses.mse})
        sim.fit(training_images_dict, training_targets_dict, epochs=epochs)
        # save parameters back into net
        sim.freeze_params(net)

    # input is flattened, so total_pixels / pixels(per image)
    batch_size = int(training_images.shape[0]/pixels)

    #predictions = sim.predict(training_images_dict, n_steps=n_steps, stateful=False)[output_probe]
    predictions = sim.predict(validation_images_dict, n_steps=n_steps, stateful=False)[output_probe]
    predictions = np.asarray(predictions).squeeze()

    fig = plt.Figure()
    #plt.plot(training_targets[-100:], label='target', color='r')
    plt.plot(validation_targets[-100:], label='target', color='r')
    plt.plot(predictions[-100:], label='predictions', color='k', linestyle='--')
    plt.legend()
    plt.savefig('prediction_results.png')
    plt.show()
