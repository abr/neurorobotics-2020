import nengo
import keras
import nengo_dl
import numpy as np
import cv2
import warnings
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import time
from abr_analyze import DataHandler
import tensorflow as tf

def run_everything(single_ens):
    def get_angle(image):
        res = image.shape
        # ignore green and blue channel
        image = image[:, :, 0]
        # get max pixel
        index = np.argmax(image)
        # get remainder, which will be the position along the column
        index = index % res[1]
        # normalize
        norm_index = index/res[1]
        # scale to -1 to 1
        index = norm_index * 2 -1
        # # get angle
        angle = np.arcsin(index)
        # scale from -pi to pi
        angle = angle*2
        return angle, norm_index

    def preprocess_data(
            image_data, res, show_resized_image, rows=None, debug=False, flatten=True):
        # single image, append 1 dimension so we can loop through the same way
        image_data = np.asarray(image_data)
        if image_data.ndim == 3:
            shape = image_data.shape
            image_data = image_data.reshape((1, shape[0], shape[1], shape[2]))

        # expect rgb image data
        assert image_data.shape[3] == 3

        scaled_image_data = []
        scaled_target_data = []
        for data in image_data:
            # normalize
            rgb = np.asarray(data)/255

            if rows is not None:
                # select a subset of rows and update the vertical resolution
                rgb = rgb[rows[0]:rows[1], :, :]
                res[0] = rows[1]-rows[0]

            # resize image resolution
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

            # use unflattened image so we maintain the resolution information
            angle, index = get_angle(rgb)
            scaled_target_data.append(angle)

            # flatten to 1D
            #NOTE should use np.ravel to maintain image order
            if flatten:
                rgb = rgb.flatten()
            # scale image from -1 to 1 and save to list
            scaled_image_data.append(rgb*2 - 1)

            # if debug is on, show plots of the scaled image, predicted target
            # location, and target angle output
            if debug:
                plt.figure()
                plt.subplot(311)
                plt.title('estimated: %i' % (index*res[1]))
                # scaled image
                plt.imshow(rgb.reshape((res[0], res[1], 3)), origin='lower')
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

        scaled_image_data = np.asarray(scaled_image_data)
        scaled_target_data = np.asarray(scaled_target_data)

        return scaled_image_data, scaled_target_data


    def load_data(
            res, db_name, label='training_0000', n_imgs=None, rows=None, debug=False, show_resized_image=False,
            flatten=True):
        dat = DataHandler(db_name)

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
            res=res, show_resized_image=show_resized_image, rows=rows,
            flatten=flatten)

        return training_images, training_targets

    def gen_data(n_images, res, pixels):
        # generate array of white pixels with a randomly placed red pixel
        zeros = np.zeros((res[0], res[1], 3))
        targets = np.linspace(-3.14, 3.14, pixels)
        images = []
        targets = []
        training_angle_targets = []
        for ii in range(0, n_images):
            index = np.random.randint(low=0, high=pixels)
            data = np.copy(zeros)
            data[0][index] = [1, 0, 0]
            data = np.asarray(data).flatten()
            angle = get_angle(data)
            training_angle_targets.append(angle)
            images.append(data)

        images = np.asarray(images)
        targets = np.array(training_angle_targets)
        return images, targets


    warnings.simplefilter("ignore")
    db_name = 'rover_training_0003'
    save_db_name = '%s_results' % db_name
    # use keras model converted to nengo-dl
    use_keras = True
    # train or just validate
    # train_on_data = True
    # load params from previous epoch
    load_net_params = True
    # save net params
    save_net_params = True
    # show plots for processed data
    debug = False
    # show the images before and after scaling
    show_resized_image = debug
    # range of epochs
    epochs = [1, 5]
    n_steps = 1
    # training batch size
    n_training = 10000
    minibatch_size = 100
    # validation batch size
    n_validation = 1000
    output_dims = 1
    n_neurons = 20000
    gain = 100
    bias = 0
    # resolution to scale images to
    res=[64, 64]
    # select a subset of rows of the image to use, these will be used to index into the array
    # rows = [27, 28]
    rows = None
    pixels = res[0] * res[1]
    subpixels = pixels*3
    seed = 0
    np.random.seed(seed)
    flatten = True
    if use_keras:
        flatten = False

    # use training data in validation step
    # validate_with_training = False

    #save_name = 'image_input-to-dense_layer_neurons-norm_input_weights_10k-gain_bias'
    # save_name = 'version_that_works_specifying_enc_30k'
    save_name = '3conv_1dense'
    params_file = 'saved_net_%ix%i' % (res[0], res[1])
    if use_keras:
        save_folder = 'data/%s/keras/%s/%s' % (db_name, params_file, save_name)
    else:
        save_folder = 'data/%s/nengo_dl/%s/%s' % (db_name, params_file, save_name)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    notes = (
    '''
    \n- bias initializer on input to neurons set to zeros
    \n- 3 conv layers 1 dense
    '''
            )
    # '''
    # '''
    # )
    # '''
    # \n- having input node connect to dense_node seemed redundant, removed dense_nose
    # and now have connections from image_input>dense_layer.neurons, dense_layer.neurons>output_node
    # \n- was not normalizing input weights for input to dense_layer.neurons, as is default
    # in nengo
    # \n- set connection seeds
    # \n- setting gain to 100 and bias to 0
    # ''')

    test_params = {
        'use_keras': use_keras,
        'n_training': n_training,
        'minibatch_size': minibatch_size,
        'n_neurons': n_neurons,
        'res': res,
        'seed': seed,
        'notes': notes,
        'gain': gain,
        'bias': bias}

    dat_results = DataHandler(db_name=save_db_name)
    dat_results.save(data=test_params, save_location='%s/params'%save_folder, overwrite=True)

    # # generate data
    # training_images, training_targets = gen_data(n_training, res=res, pixels=pixels)
    # validation_images, validation_targets = gen_data(n_validation)

    # load data
    training_images, training_targets = load_data(
        res=res, label='training_0000', n_imgs=n_training, rows=rows,
        debug=debug, show_resized_image=show_resized_image, db_name=db_name,
        flatten=flatten)

    # if validate_with_training:
    #     # images are flattened at this point, need to find how
    #     # many pixels per image
    #     n_val = int(n_validation * len(training_images) / n_training)
    #     validation_images=training_images[:n_val]
    #     n_val = int(n_validation * len(training_targets) / n_training)
    #     validation_targets=training_targets[:n_val]
    # else:
    validation_images, validation_targets = load_data(
        res=res, label='validation_0000', n_imgs=n_validation, rows=rows,
        debug=debug, show_resized_image=show_resized_image, db_name=db_name,
        flatten=flatten)

    test_print = ('Running tests for %s' % save_folder)
    print('\n')
    print('Training Images: ', training_images.shape)
    print('Validation Images: ', validation_images.shape)
    print('-'*len(test_print))
    print(test_print)
    print('-'*len(test_print))
    print('\n')
    # ---------------------------------------Define network
    if not use_keras:
        # single_ens = True

        weights_in = nengo.dists.UniformHypersphere(
            surface=True).sample(n_neurons, subpixels, rng=np.random.RandomState(seed=0))

        if single_ens:
            #VERSION THAT WORKS
            net = nengo.Network(seed=seed)
            net.config[nengo.Ensemble].neuron_type = nengo.RectifiedLinear()
            net.config[nengo.Ensemble].gain = nengo.dists.Choice([gain])
            net.config[nengo.Ensemble].bias = nengo.dists.Choice([bias])
            net.config[nengo.Connection].synapse = None

            with net:
                image_input = nengo.Node(np.zeros(subpixels))
                dense_layer = nengo.Ensemble(
                    n_neurons=n_neurons,
                    dimensions=subpixels,
                    encoders=weights_in)
                image_output = nengo.Node(size_in=output_dims)

                input_conn = nengo.Connection(
                    image_input, dense_layer, label='input_to_ens', seed=0)
                # nengo_dl.configure_settings(trainable=None)
                # net.config[input_conn].trainable = False

                nengo.Connection(
                    dense_layer.neurons,
                    image_output,
                    label='output_conn',
                    transform=np.zeros(
                        (output_dims, dense_layer.n_neurons)),
                    seed=0
                    )

                output_probe = nengo.Probe(image_output, synapse=None, label='output_filtered')
                neuron_probe = nengo.Probe(dense_layer.neurons, synapse=None, label='single_layer_neuron_probe')

                # Training, turn off synapses for training to simplify
                for conn in net.all_connections:
                    conn.synapse = None

        else:
            # breaking up into dense_node and dense_layer
            net = nengo.Network(seed=seed)

            net.config[nengo.Ensemble].neuron_type = nengo.RectifiedLinear()
            net.config[nengo.Ensemble].gain = nengo.dists.Choice([1])
            net.config[nengo.Ensemble].bias = nengo.dists.Choice([bias])
            net.config[nengo.Connection].synapse = None

            with net:

                image_input = nengo.Node(np.zeros(subpixels))
                # dense_node = nengo.Node(size_in=subpixels)
                dense_layer = nengo.Ensemble(
                    n_neurons=n_neurons,
                    dimensions=subpixels,
                    encoders=weights_in)
                image_output = nengo.Node(size_in=output_dims)

                # nengo.Connection(
                #     image_input, dense_node, seed=0)

                # nengo.Connection(
                #     dense_node, dense_layer.neurons, transform=weights_in, label='input_to_neurons', seed=0)

                nengo.Connection(
                    image_input, dense_layer.neurons, transform=weights_in*gain, label='input_to_neurons', seed=0)

                nengo.Connection(
                    dense_layer.neurons,
                    image_output,
                    label='output_conn',
                    transform=np.zeros(
                        (output_dims, dense_layer.n_neurons)),
                    seed=0
                    )

                output_probe = nengo.Probe(image_output, synapse=None, label='output_filtered')
                neuron_probe = nengo.Probe(dense_layer.neurons, synapse=None, label='single_layer_neuron_probe')

                # Training, turn off synapses for training to simplify
                for conn in net.all_connections:
                    conn.synapse = None

    else:
        image_input = tf.keras.Input(shape=(res[0], res[1], 3), batch_size=minibatch_size)

        conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=3,
            strides=1,
            use_bias=True,
            activation=tf.nn.relu,
            # kernel_initializer=keras.initializers.Zeros(),
            data_format="channels_last",
            # padding='same'
            )(image_input)

        conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=3,
            strides=1,
            use_bias=True,
            activation=tf.nn.relu,
            # kernel_initializer=keras.initializers.Zeros(),
            data_format="channels_last",
            # padding='same'
            )(conv1)

        conv3 = tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=3,
            strides=1,
            use_bias=True,
            activation=tf.nn.relu,
            # kernel_initializer=keras.initializers.Zeros(),
            data_format="channels_last",
            # padding='same'
            )(conv2)

        flatten = tf.keras.layers.Flatten()(conv3)

        output_probe = tf.keras.layers.Dense(
            units=output_dims,
            use_bias=False,
            kernel_initializer=keras.initializers.Zeros()
            )(flatten)

        model = tf.keras.Model(inputs=image_input, outputs=output_probe)

        converter = nengo_dl.Converter(model)
        image_input = converter.inputs[image_input]
        output_probe = converter.outputs[output_probe]
        net = converter.net

    with nengo_dl.Simulator(net, minibatch_size=minibatch_size, seed=seed) as sim:

        training_images_dict = {
            image_input: training_images.reshape(
                (n_training, n_steps, subpixels))
        }

        validation_images_dict = {
            image_input: validation_images.reshape(
                (n_validation, n_steps, subpixels))
        }


        training_targets_dict = {
            #output_probe_no_filter: training_targets.reshape(
            output_probe: training_targets.reshape(
                (n_training, n_steps, output_dims))
        }

        def predict(save_folder='', save_name='prediction_results'):
            data = sim.predict(validation_images_dict, n_steps=n_steps, stateful=False)
            predictions = data[output_probe]
            predictions = np.asarray(predictions).squeeze()
            dat_results.save(
                data={save_name: predictions},
                save_location=save_folder,
                overwrite=True)

            fig = plt.Figure()
            plt.plot(validation_targets[-100:], label='target', color='r')
            plt.plot(predictions[-100:], label='predictions', color='k', linestyle='--')
            plt.legend()
            plt.savefig('%s/%s.png' % (save_folder, save_name))
            #plt.show()
            plt.close()
            fig = None
            return data

        # if train_on_data:

        print('Training')
        sim.compile(
            # optimizer=tf.optimizers.SGD(1e-5),
            optimizer=tf.optimizers.RMSprop(0.01),
            # # loss={output_probe_no_filter: tf.losses.mse})
            loss={output_probe: tf.losses.mse})

        if isinstance(epochs, int):
            epochs = [1, epochs]

        for epoch in range(epochs[0]-1, epochs[1]):
            print('\n\nEPOCH %i\n' % epoch)
            if load_net_params and epoch>0:
                sim.load_params('%s/%s' % (save_folder, params_file))
                print('loading pretrained network parameters')

            # if train_on_data:
            sim.fit(training_images_dict, training_targets_dict, epochs=1)

            # save parameters back into net
            sim.freeze_params(net)

            if save_net_params:
                print('saving network parameters to %s/%s' % (save_folder, params_file))
                sim.save_params('%s/%s' % (save_folder, params_file))

            predict_data = predict(save_folder=save_folder, save_name='prediction_epoch%i' % (epoch))

        dat_results.save(
            data={'targets': validation_targets[-100:]},
            save_location=save_folder,
            overwrite=True)
        # return predict_data[neuron_probe]

single_layer = run_everything(single_ens=True)
# split_layer = run_everything(single_ens=False)
# print('SINGLE LAYER')
# print(single_layer)
# print('\nSPLIT LAYER')
# print(split_layer)
# assert np.allclose(single_layer, split_layer)
#
