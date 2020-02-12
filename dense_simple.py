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
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

def run_everything():
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
        # scaled_target_data = []
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

            # scaled_target_data.append(angle)

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
                # rgb = rgb.flatten()
                rgb = np.ravel(rgb)
            # scale image from -1 to 1 and save to list
            # scaled_image_data.append(rgb*2 - 1)
            scaled_image_data.append(np.copy(rgb))


        scaled_image_data = np.asarray(scaled_image_data)
        # scaled_target_data = np.asarray(scaled_target_data)

        return scaled_image_data #, scaled_target_data


    def load_data(
            res, db_name, label='training_0000', n_imgs=None, rows=None, debug=False, show_resized_image=False,
            flatten=True, n_steps=1):
        dat = DataHandler(db_name)

        # load training images
        training_images = []
        training_targets = []
        for nn in range(0, n_imgs):
            data = dat.load(parameters=['rgb', 'local_error'], save_location='%s/data/%04d' % (label, nn))
            # for ii in range(n_steps):
            training_images.append(data['rgb'])
            training_targets.append(data['local_error'])

        # scale image resolution
        # training_images, training_targets = preprocess_data(
        training_images = preprocess_data(
            image_data=training_images, debug=debug,
            res=res, show_resized_image=show_resized_image, rows=rows,
            flatten=flatten, local_error=training_targets)
        training_targets = np.asarray(training_targets)[:, 0:2]

        print('images pre_tile: ', training_images.shape)
        print('targets pre_tile: ', training_targets.shape)
        training_images = np.tile(training_images[:, None, :], (1, n_steps, 1))
        training_targets = np.tile(training_targets[:, None, :], (1, n_steps, 1))
        print('images post_tile: ', training_images.shape)
        print('targets post_tile: ', training_targets.shape)

        return training_images, training_targets

    warnings.simplefilter("ignore")
    db_name = 'rover_training_0004'
    spiking = True
    # db_name = 'rover_dist_range_0_marked'
    save_db_name = '%s_results' % db_name
    # train or just validate
    train_on_data = True
    if spiking:
        train_on_data = False
    # load params from previous epoch
    load_net_params = True
    # save net params only if training
    save_net_params = train_on_data
    # show plots for processed data
    debug = False
    # show the images before and after scaling
    show_resized_image = debug
    # range of epochs
    epochs = [30, 30]
    if not spiking:
        n_steps = 1
        gain_scale = 1
        synapses = None
    else:
        n_steps = 500
        gain_scale = 200
        save_net_params = False
        synapses = 0.05
    # training batch size
    n_training = 30000
    minibatch_size = 100
    # validation batch size
    n_validation = 100
    minibatch_size = min(minibatch_size, n_validation)
    output_dims = 2
    gain = 1
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

    # use training data in validation step
    # validate_with_training = False

    # save_name = 'learning_xy_optimized_1_conv_with_relu_spiking'
    save_name = 'filters_64'
    params_file = 'saved_net_%ix%i' % (res[0], res[1])
    save_folder = 'data/%s/keras/%s/%s' % (db_name, params_file, save_name)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    notes = (
    '''
    \n- retraining with more filters so we get more neurons in our spiking net
    \n- bias off on all layer, loihi restriction
    \n- default kernel initializers
    \n- 1 conv layers 1 dense
    \n- adding relu layer between input and conv1 due to loihi comm restriction
    \n- changing image scaling from -1 to 1, to 0 to 1 due to loihi comm restriction
    '''
            )

    test_params = {
        'n_training': n_training,
        'minibatch_size': minibatch_size,
        'res': res,
        'seed': seed,
        'notes': notes,
        'gain': gain,
        'bias': bias}

    dat_results = DataHandler(db_name=save_db_name)
    dat_results.save(data=test_params, save_location='%s/params'%save_folder, overwrite=True)


    if train_on_data:
        # load data
        training_images, training_targets = load_data(
            res=res, label='training_0000', n_imgs=n_training, rows=rows,
            debug=debug, show_resized_image=show_resized_image, db_name=db_name,
            flatten=flatten, n_steps=n_steps)

    validation_images, validation_targets = load_data(
        res=res, label='validation_0000', n_imgs=n_validation, rows=rows,
        debug=debug, show_resized_image=show_resized_image, db_name=db_name,
        flatten=flatten, n_steps=n_steps)

    test_print = ('Running tests for %s' % save_folder)
    print('\n')
    if train_on_data:
        print('Training Images: ', training_images.shape)
    print('Validation Images: ', validation_images.shape)
    print('-'*len(test_print))
    print(test_print)
    print('-'*len(test_print))
    print('\n')

    # ---------------------------------------Define network
    image_input = tf.keras.Input(shape=(res[0], res[1], 3), batch_size=minibatch_size)
    # image_input = tf.keras.Input(shape=subpixels) #, batch_size=minibatch_size)

    relu_layer = tf.keras.layers.Activation(tf.nn.relu)
    relu_out = relu_layer(image_input)

    conv1 = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=3,
        strides=1,
        use_bias=False,
        activation=tf.nn.relu,
        data_format="channels_last",
        )

    conv1_out = conv1(relu_out)

    flatten = tf.keras.layers.Flatten()(conv1_out)

    keras_dense = tf.keras.layers.Dense(
        units=output_dims,
        use_bias=False,
        )
    keras_output_probe = keras_dense(flatten)

    model = tf.keras.Model(inputs=image_input, outputs=keras_output_probe)

    if not spiking:
        converter = nengo_dl.Converter(model)
    else:
        converter = nengo_dl.Converter(
            model,
            swap_activations={
                tf.nn.relu: nengo.SpikingRectifiedLinear(amplitude=1/gain_scale)
                }
        )

    # IO objects
    image_input = converter.inputs[image_input]
    output_probe = converter.outputs[keras_output_probe]

    # ensemble objects
    nengo_conv = converter.layer_map[conv1][0][0]
    nengo_relu = converter.layer_map[relu_layer][0][0]

    # adjust gains and synapses
    nengo_conv.ensemble.gain = nengo.dists.Choice([gain * gain_scale])
    nengo_relu.ensemble.gain = nengo.dists.Choice([gain * gain_scale])
    print('MAX RATES: ',nengo_conv.ensemble)

    net = converter.net
    for conn in net.all_connections:
        conn.synapse = synapses



    with nengo_dl.Simulator(net, minibatch_size=minibatch_size, seed=seed) as sim:
        scale = 1
        print('in sim gain: ', sim.data[nengo_conv.ensemble].gain)

        if train_on_data:
            # training_images *= scale
            training_images_dict = {
                image_input: training_images.reshape(
                    (n_training, n_steps, subpixels))
            }

        validation_images *= scale
        validation_images_dict = {
            image_input: validation_images.reshape(
                (n_validation, n_steps, subpixels))
        }


        if train_on_data:
            training_targets_dict = {
                #output_probe_no_filter: training_targets.reshape(
                output_probe: training_targets.reshape(
                    (n_training, n_steps, output_dims))
            }

        def predict(
                input_data_dict, target_vals,
                save_folder='', save_name='prediction_results', num_pts=100):

            print('input shape: ', input_data_dict[image_input].shape)
            data = sim.predict(input_data_dict, n_steps=n_steps, stateful=False)
            predictions = data[output_probe]
            predictions = np.asarray(predictions).squeeze()
            print('targets shape: ', target_vals.shape)
            print('prediction shape: ', predictions.shape)

            if predictions.ndim > 2:
                shape = np.asarray(predictions).shape
                predictions = np.asarray(predictions).reshape(shape[0]*shape[1], shape[2])
                print('pred reshape: ', predictions.shape)
            if target_vals.ndim > 2:
                shape = np.asarray(target_vals).shape
                target_vals = np.asarray(target_vals).reshape(shape[0]*shape[1], shape[2])
                print('targets reshape: ', target_vals.shape)

            x_err = np.linalg.norm(target_vals[:, 0] - predictions[:, 0])
            y_err = np.linalg.norm(target_vals[:, 1] - predictions[:, 1])


            fig = plt.Figure()
            plt.subplot(211)
            plt.title('X: %.3f' % x_err)
            plt.plot(target_vals[-num_pts:, 0], label='target', color='r')
            plt.plot(predictions[-num_pts:, 0], label='predictions', color='k', linestyle='--')
            plt.subplot(212)
            plt.title('Y: %.3f' % y_err)
            plt.plot(target_vals[-num_pts:, 1], label='target', color='r')
            plt.plot(predictions[-num_pts:, 1], label='predictions', color='k', linestyle='--')
            plt.legend()
            plt.savefig('%s/%s.png' % (save_folder, save_name))
            # plt.show()
            plt.close()
            fig = None

            if not spiking:
                #TODO: fix this because we keep appending, need to attach value to an epoch and overwrite
                final_errors = dat_results.load(
                    parameters=['final_errors'],
                    save_location=save_folder)['final_errors']
                if final_errors.ndim == 0:
                    final_errors = []

                final_errors = np.vstack((final_errors, [x_err, y_err]))

                final_errors = np.asarray(final_errors)
                plt.figure()
                plt.subplot(211)
                plt.title('X error over epochs')
                plt.plot(final_errors[:, 0])
                plt.subplot(212)
                plt.title('Y error over epochs')
                plt.plot(final_errors[:, 1])
                plt.savefig('%s/final_epoch_error.png' % (save_folder))

                save_data = {save_name: predictions, 'final_errors': final_errors}

            else:
                save_data = {save_name: predictions}

            dat_results.save(
                data=save_data,
                save_location=save_folder,
                overwrite=True)

        if train_on_data:
            print('Training')
            sim.compile(
                # optimizer=tf.optimizers.SGD(1e-5),
                optimizer=tf.optimizers.RMSprop(0.001),
                # # loss={output_probe_no_filter: tf.losses.mse})
                loss={output_probe: tf.losses.mse})

        if isinstance(epochs, int):
            epochs = [0, epochs]

        for epoch in range(epochs[0], epochs[1]):
            num_imgs_to_show = 10
            num_pts = num_imgs_to_show*n_steps #3*int(max(n_steps, min(n_validation, 100)))
            print('\n\nEPOCH %i\n' % epoch)
            if load_net_params and epoch>0:
                prev_params_loc = ('%s/%s_%i' % (save_folder, params_file, epoch-1))
                print('loading pretrained network parameters from \n%s' % prev_params_loc)
                sim.load_params(prev_params_loc)

            if train_on_data:
                print('Training...')
                sim.fit(training_images_dict, training_targets_dict, epochs=1)

            # save parameters back into net
            sim.freeze_params(net)

            if save_net_params:
                current_params_loc = '%s/%s_%i' % (save_folder, params_file, epoch)
                print('saving network parameters to %s' % current_params_loc)
                sim.save_params(current_params_loc)

            if train_on_data:
                # we're predicting using the weights from this epoch
                save_name='prediction_epoch%i' % (epoch)
            else:
                if spiking:
                    prefix = 'spiking_'
                else:
                    prefix = ''
                # we're running inference using the previous weights
                save_name='%sinference_epoch%i' % (prefix, epoch)

            predict(
                    input_data_dict=validation_images_dict, target_vals=validation_targets,
                    save_folder=save_folder, save_name=save_name,
                    num_pts=num_pts)

        dat_results.save(
            data={'targets': validation_targets},
            save_location=save_folder,
            overwrite=True)

single_layer = run_everything()
