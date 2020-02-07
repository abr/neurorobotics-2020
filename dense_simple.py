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
import math

def run_everything(single_ens):
    def preprocess_data(
            image_data, res, show_resized_image, local_error, rows=None, debug=False, flatten=True):

        # single image, append 1 dimension so we can loop through the same way
        image_data = np.asarray(image_data)
        if image_data.ndim == 3:
            shape = image_data.shape
            image_data = image_data.reshape((1, shape[0], shape[1], shape[2]))

        # expect rgb image data
        assert image_data.shape[3] == 3

        scaled_image_data = []
        scaled_target_data = []
        for count, data in enumerate(image_data):
            # normalize
            rgb = np.asarray(data)/255

            if rows is not None:
                raise Exception
                # # select a subset of rows and update the vertical resolution
                # rgb = rgb[rows[0]:rows[1], :, :]
                # res[0] = rows[1]-rows[0]

            # resize image resolution
            rgb = cv2.resize(
                rgb, dsize=(res[1], res[0]),
                interpolation=cv2.INTER_CUBIC)

            # visualize scaling for debugging
            if show_resized_image and count%100 == 0:
                plt.Figure()
                a = plt.subplot(121)
                a.set_title('Original')
                a.imshow(data, origin='lower')
                b = plt.subplot(122)
                b.set_title('Scaled')
                b.imshow(rgb, origin='lower')
                plt.show()

            # scan the image for red pixels to determine target angle
            angle = math.atan2(local_error[count][1], local_error[count][0])

            scaled_target_data.append(angle)

            # if debug is on, show plots of the scaled image, predicted target
            # location, and target angle output
            if debug and count%100 == 0:
                plt.figure()
                plt.subplot(311)
                # scaled image
                plt.title('estimated: shifted to center at zero for viewing')
                shift_lim = 16
                plt.imshow(
                    np.hstack(
                        (rgb[:, -shift_lim:, :], rgb[:, :-shift_lim, :]))
                    , origin='lower')
                # if local_error is None:
                #     plt.title('estimated: %i' % (index*res[1]))
                #     plt.subplot(312)
                #     # estimated location of target
                #     a = np.zeros((res[0], res[1], 3))
                #     a[:, int(index*res[1])] = [1,0,0]
                #     plt.imshow(a, origin='lower')
                plt.subplot(313)
                plt.xlim(-3.14, 3.14)
                # estimated target angle
                plt.scatter(angle, 1, c='r')
                #plt.savefig('gif_cache/%04d'%ii)
                plt.show()

            # flatten to 1D
            #NOTE should use np.ravel to maintain image order
            if flatten:
                rgb = rgb.flatten()
            # scale image from -1 to 1 and save to list
            # scaled_image_data.append(rgb*2 - 1)
            scaled_image_data.append(rgb)


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
        # training_images, training_targets = preprocess_data(
        training_images, _ = preprocess_data(
            image_data=training_images, debug=debug,
            res=res, show_resized_image=show_resized_image, rows=rows,
            flatten=flatten, local_error=training_targets)
        training_targets = np.asarray(training_targets)[:, 0:2]

        return training_images, training_targets

    warnings.simplefilter("ignore")
    db_name = 'rover_training_0004'
    # db_name = 'rover_dist_range_0_marked'
    save_db_name = '%s_results' % db_name
    # use keras model converted to nengo-dl
    use_keras = True
    # train or just validate
    train_on_data = False
    # load params from previous epoch
    load_net_params = True
    # save net params
    save_net_params = False
    # show plots for processed data
    debug = False
    # show the images before and after scaling
    show_resized_image = debug
    # range of epochs
    epochs = [1, 100]
    n_steps = 1
    # training batch size
    n_training = 30000
    minibatch_size = 100
    # validation batch size
    n_validation = 1000
    output_dims = 2
    n_neurons = 20000
    gain = 100
    bias = 0
    # resolution to scale images to
    res=[32, 128]
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
    final_errors = []

    # use training data in validation step
    # validate_with_training = False

    #save_name = 'image_input-to-dense_layer_neurons-norm_input_weights_10k-gain_bias'
    # save_name = 'version_that_works_specifying_enc_30k'
    save_name = 'learning_xy_optimized_1_conv'
    params_file = 'saved_net_%ix%i' % (res[0], res[1])
    if use_keras:
        save_folder = 'data/%s/keras/%s/%s' % (db_name, params_file, save_name)
    else:
        save_folder = 'data/%s/nengo_dl/%s/%s' % (db_name, params_file, save_name)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    notes = (
    '''
    \n- bias off on all layer, loihi restriction
    \n- default kernel initializers
    \n- 1 conv layers 1 dense (removed conv 2 and 3)
    \n- adding relu layer between input and conv1 due to loihi comm restriction
    \n- changing image scaling from -1 to 1, to 0 to 1 due to loihi comm restriction
    '''
            )

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

    # for im in training_images:
    #     plt.figure()
    #     plt.imshow(im)
    #     plt.show()
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
        raise Exception
        # # single_ens = True
        #
        # weights_in = nengo.dists.UniformHypersphere(
        #     surface=True).sample(n_neurons, subpixels, rng=np.random.RandomState(seed=0))
        #
        # if single_ens:
        #     #VERSION THAT WORKS
        #     net = nengo.Network(seed=seed)
        #     net.config[nengo.Ensemble].neuron_type = nengo.RectifiedLinear()
        #     net.config[nengo.Ensemble].gain = nengo.dists.Choice([gain])
        #     net.config[nengo.Ensemble].bias = nengo.dists.Choice([bias])
        #     net.config[nengo.Connection].synapse = None
        #
        #     with net:
        #         image_input = nengo.Node(np.zeros(subpixels))
        #         dense_layer = nengo.Ensemble(
        #             n_neurons=n_neurons,
        #             dimensions=subpixels,
        #             encoders=weights_in)
        #         image_output = nengo.Node(size_in=output_dims)
        #
        #         input_conn = nengo.Connection(
        #             image_input, dense_layer, label='input_to_ens', seed=0)
        #         # nengo_dl.configure_settings(trainable=None)
        #         # net.config[input_conn].trainable = False
        #
        #         nengo.Connection(
        #             dense_layer.neurons,
        #             image_output,
        #             label='output_conn',
        #             transform=np.zeros(
        #                 (output_dims, dense_layer.n_neurons)),
        #             seed=0
        #             )
        #
        #         output_probe = nengo.Probe(image_output, synapse=None, label='output_filtered')
        #         neuron_probe = nengo.Probe(dense_layer.neurons, synapse=None, label='single_layer_neuron_probe')
        #
        #         # Training, turn off synapses for training to simplify
        #         for conn in net.all_connections:
        #             conn.synapse = None
        #
        # else:
        #     # breaking up into dense_node and dense_layer
        #     net = nengo.Network(seed=seed)
        #
        #     net.config[nengo.Ensemble].neuron_type = nengo.RectifiedLinear()
        #     net.config[nengo.Ensemble].gain = nengo.dists.Choice([1])
        #     net.config[nengo.Ensemble].bias = nengo.dists.Choice([bias])
        #     net.config[nengo.Connection].synapse = None
        #
        #     with net:
        #
        #         image_input = nengo.Node(np.zeros(subpixels))
        #         # dense_node = nengo.Node(size_in=subpixels)
        #         dense_layer = nengo.Ensemble(
        #             n_neurons=n_neurons,
        #             dimensions=subpixels,
        #             encoders=weights_in)
        #         image_output = nengo.Node(size_in=output_dims)
        #
        #         # nengo.Connection(
        #         #     image_input, dense_node, seed=0)
        #
        #         # nengo.Connection(
        #         #     dense_node, dense_layer.neurons, transform=weights_in, label='input_to_neurons', seed=0)
        #
        #         nengo.Connection(
        #             image_input, dense_layer.neurons, transform=weights_in*gain, label='input_to_neurons', seed=0)
        #
        #         nengo.Connection(
        #             dense_layer.neurons,
        #             image_output,
        #             label='output_conn',
        #             transform=np.zeros(
        #                 (output_dims, dense_layer.n_neurons)),
        #             seed=0
        #             )
        #
        #         output_probe = nengo.Probe(image_output, synapse=None, label='output_filtered')
        #         neuron_probe = nengo.Probe(dense_layer.neurons, synapse=None, label='single_layer_neuron_probe')
        #
        #         # Training, turn off synapses for training to simplify
        #         for conn in net.all_connections:
        #             conn.synapse = None
        #
    else:
        image_input = tf.keras.Input(shape=(res[0], res[1], 3), batch_size=minibatch_size)

        relu_layer = tf.keras.layers.Activation(tf.nn.relu)(image_input)

        conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=3,
            strides=1,
            # use_bias=True,
            use_bias=False,
            activation=tf.nn.relu,
            # kernel_initializer=keras.initializers.Zeros(),
            data_format="channels_last",
            # padding='same'
            # )(relu_layer)
            )(image_input)

        # conv2 = tf.keras.layers.Conv2D(
        #     filters=64,
        #     kernel_size=3,
        #     strides=1,
        #     # use_bias=True,
        #     use_bias=False,
        #     activation=tf.nn.relu,
        #     # kernel_initializer=keras.initializers.Zeros(),
        #     data_format="channels_last",
        #     # padding='same'
        #     )(conv1)

        # conv3 = tf.keras.layers.Conv2D(
        #     filters=128,
        #     kernel_size=3,
        #     strides=1,
        #     # use_bias=True,
        #     use_bias=False,
        #     activation=tf.nn.relu,
        #     # kernel_initializer=keras.initializers.Zeros(),
        #     data_format="channels_last",
        #     # padding='same'
        #     )(conv2)

        flatten = tf.keras.layers.Flatten()(conv1)

        output_probe = tf.keras.layers.Dense(
            units=output_dims,
            use_bias=False,
            #kernel_initializer=keras.initializers.Zeros()
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

        def predict(save_folder='', save_name='prediction_results', num_pts=100, final=False, final_errors=None):
            data = sim.predict(validation_images_dict, n_steps=n_steps, stateful=False)
            predictions = data[output_probe]
            predictions = np.asarray(predictions).squeeze()
            dat_results.save(
                data={save_name: predictions},
                save_location=save_folder,
                overwrite=True)
            # print(predictions.shape)

            x_err = np.linalg.norm(validation_targets[:, 0] - predictions[:, 0])
            y_err = np.linalg.norm(validation_targets[:, 1] - predictions[:, 1])
            final_errors.append([x_err, y_err])
            fig = plt.Figure()
            plt.subplot(211)
            plt.title('X: %.3f' % x_err)
            plt.plot(validation_targets[-num_pts:, 0], label='target', color='r')
            plt.plot(predictions[-num_pts:, 0], label='predictions', color='k', linestyle='--')
            plt.subplot(212)
            plt.title('Y: %.3f' % y_err)
            plt.plot(validation_targets[-num_pts:, 1], label='target', color='r')
            plt.plot(predictions[-num_pts:, 1], label='predictions', color='k', linestyle='--')
            plt.legend()
            plt.savefig('%s/%s.png' % (save_folder, save_name))
            #plt.show()
            plt.close()
            fig = None

            if final:
                final_errors = np.array(final_errors)
                plt.figure()
                plt.subplot(211)
                plt.title('X error over epochs')
                plt.plot(final_errors[:, 0])
                plt.subplot(212)
                plt.title('Y error over epochs')
                plt.plot(final_errors[:, 1])
                plt.savefig('%s/final_epoch_error.png' % (save_folder))
            return data

        print('Training')
        sim.compile(
            # optimizer=tf.optimizers.SGD(1e-5),
            optimizer=tf.optimizers.RMSprop(0.001),
            # # loss={output_probe_no_filter: tf.losses.mse})
            loss={output_probe: tf.losses.mse})

        if isinstance(epochs, int):
            epochs = [1, epochs]

        for epoch in range(epochs[0]-1, epochs[1]):
            num_pts = 100
            print('\n\nEPOCH %i\n' % epoch)
            if load_net_params and epoch>0:
                sim.load_params('%s/%s_%i' % (save_folder, params_file, epoch-1))
                print('loading pretrained network parameters')

            if train_on_data:
                sim.fit(training_images_dict, training_targets_dict, epochs=1)

            # save parameters back into net
            sim.freeze_params(net)

            if save_net_params:
                print('saving network parameters to %s/%s_%i' % (save_folder, params_file, epoch))
                sim.save_params('%s/%s_%i' % (save_folder, params_file, epoch))

            if epoch == epochs[1]-1:
                final = True
            else:
                final = False

            predict_data = predict(save_folder=save_folder, save_name='prediction_epoch%i' % (epoch),
                    num_pts=num_pts, final=final, final_errors=final_errors)

        dat_results.save(
            data={'targets': validation_targets},
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
