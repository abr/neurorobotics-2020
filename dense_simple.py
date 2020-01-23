import nengo
import nengo_dl
import numpy as np
import cv2
import warnings
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import time
from abr_analyze import DataHandler

def get_angle(image):
    # print('----------')
    # print('raw img shape: ', image.shape)
    res = image.shape
    #print('res: ', res)
    # ignore green and blue channel
    image = image[:, :, 0]
    #print('new img shape: ', image.shape)
    # get max pixel
    index = np.argmax(image)
    #print('max index: ', index)
    # get remainder, which will be the position along the column
    index = index % res[1]
    #print('remainder: ', index)
    # normalize
    norm_index = index/res[1]
    #print('normalized: ', norm_index)
    # scale to -1 to 1
    index = norm_index * 2 -1
    # # get angle
    #angle = np.arccos(norm_index2)
    angle = np.arcsin(index)
    #print('angle: ', angle)
    # scale from -pi to pi
    angle = angle*2
    #print('scaled angle: ', angle)
    #angle = angle*2 - np.pi
    return angle, norm_index

def preprocess_data(image_data, res, show_resized_image, rows=None, debug=False):
    # single image, append 1 dimension so we can loop through the same way
    image_data = np.asarray(image_data)
    if image_data.ndim == 3:
        shape = image_data.shape
        image_data = image_data.reshape((1, shape[0], shape[1], shape[2]))

    # expect rgb image data
    assert image_data.shape[3] == 3
    # # expect square image
    # assert image_data.shape[1] == image_data.shape[2]

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
            rgb, dsize=(res[1], res[0]),
            interpolation=cv2.INTER_CUBIC)

        if rows is not None:
            # print(rgb.shape)
            # print(rgb[rows[0]:rows[1],:,:].shape)
            # print(rows)
            # print(rows[1]-rows[0])
            # print(res)
            rgb = rgb[rows[0]:rows[1], :, :] #.reshape((rows[1]-rows[0], res[1], 3))
        #TODO move this up so we only process one row
        # get a single row
        # rgb = rgb[10, :, :].reshape((1, res[1], 3))
        #
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

        # use unflattened image so we maintain the resolution information
        angle, index = get_angle(rgb)
        scaled_target_data.append(angle)

        # flatten to 1D
        #NOTE should use np.ravel to maintain image order
        rgb = rgb.flatten()
        # scale from -1 to 1 and save to list
        scaled_image_data.append(rgb*2 - 1)

        if debug:
            plt.figure()
            plt.subplot(311)
            plt.title('estimated: %i' % (index*res[1]))
            # scaled image
            if rows is None:
                plt.imshow(rgb.reshape((res[0], res[1], 3)), origin='lower')
            else:
                plt.imshow(rgb.reshape((rows[1]-rows[0], res[1], 3)), origin='lower')
            plt.subplot(312)
            # estimated location of target
            a = np.zeros((res[0], res[1], 3))
            a[:, int(index*res[1])] = [1,0,0]
            plt.imshow(a, origin='lower')
            plt.subplot(313)
            plt.xlim(-3.14, 3.14)
            # estimated target angle
            plt.scatter(angle, 1, c='r')
            #plt.savefig('gif_cache/%04d'%ii)
            plt.show()

    scaled_image_data = np.asarray(scaled_image_data).flatten()
    scaled_target_data = np.asarray(scaled_target_data)

    return scaled_image_data, scaled_target_data


def load_data(res, db_name, label='training_0000', n_imgs=None, rows=None, debug=False, show_resized_image=False):
    dat = DataHandler(db_name)
    # training_images = dat.load(['row_image'], 'training_0000/data/single_row')['row_image']
    # # separate into 1000 original images, 64 pixels wide
    # training_images = training_images.reshape((1000, 192))
    # #validation_images = dat.load(['row_image'], 'validation_0000/data/single_row')['row_image']

    # load training images
    training_images = []
    training_targets = []
    for nn in range(0, n_imgs):
        data = dat.load(parameters=['rgb', 'local_error'], save_location='%s/data/%04d' % (label, nn))
        training_images.append(data['rgb'])
        training_targets.append(data['local_error'])

    # scale image resolution
    training_images, training_targets = preprocess_data(
        image_data=training_images, debug=debug,
        res=res, show_resized_image=show_resized_image, rows=rows)

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
db_name = 'rover_training_0003'
load_net_params = True
save_net_params = True
debug = False
show_resized_image = debug
epochs = [16, 20]
n_steps = 1
n_training = 30000
minibatch_size = 100 #int(n_training/100)
n_validation = 1000
output_dims = 1
n_neurons = 20000
scale_res=[10, 64]
# select a subset of rows of the image to use, these will be used to index into the array
# rows = [0, 10]
rows = None
# if using a subset of the image, update the resolution used in the network
if rows is not None:
    assert rows[1] <= scale_res[0]
    res = [rows[1]-rows[0], scale_res[1]]
else:
    res = scale_res
pixels = res[0] * res[1]
subpixels = pixels*3
seed = 0
np.random.seed(seed)
# use training data in validation step
validate_with_training = False
params_file = 'saved_net_%ix%i' % (res[0], res[1])
save_folder = 'data/%s/%s' % (db_name, params_file)
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

assert output_dims == 1
# # generate data
# training_images, training_targets = gen_data(n_training, res=res, pixels=pixels)
# validation_images, validation_targets = gen_data(n_validation)

# load data
training_images, training_targets = load_data(
    res=scale_res, label='training_0000', n_imgs=n_training, rows=rows,
    debug=debug, show_resized_image=show_resized_image, db_name=db_name)

if validate_with_training:
    # images are flattened at this point, need to find how
    # many pixels per image
    n_val = int(n_validation * len(training_images) / n_training)
    validation_images=training_images[:n_val]
    n_val = int(n_validation * len(training_targets) / n_training)
    validation_targets=training_targets[:n_val]
else:
    validation_images, validation_targets = load_data(
        res=scale_res, label='validation_0000', n_imgs=n_validation, rows=rows,
        debug=debug, show_resized_image=show_resized_image, db_name=db_name)

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
        print('learning function, setting weights to zero')

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
    def predict(save_name='prediction_results.png'):
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
        plt.savefig(save_name)
        plt.show()

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

        if isinstance(epochs, int):
            epochs = [1, epochs]

        for epoch in range(epochs[0]-1, epochs[1]):
            print('\n\nEPOCH %i\n' % epoch)
            if load_net_params and epoch>0:
                sim.load_params('%s/%s' % (save_folder, params_file))
                print('loading pretrained network parameters')

            sim.fit(training_images_dict, training_targets_dict, epochs=1)

            # save parameters back into net
            sim.freeze_params(net)

            if save_net_params:
                print('saving network parameters to %s/%s' % (save_folder, params_file))
                sim.save_params('%s/%s' % (save_folder, params_file))

            predict(save_name='%s/prediction_epoch%i' % (save_folder, epoch))

    else:
        if load_net_params:
            sim.load_params('%s/%s' % (save_folder, params_file))
            print('loading pretrained network parameters')

        predict()

