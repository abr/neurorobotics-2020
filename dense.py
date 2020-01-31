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


warnings.simplefilter("ignore")

class rover_vision():
    """
    Parameters
    ----------
    n_training: int, Optional (Default: 10000)
        batch size for training data set
    n_validation: int, Optional (Default: 1000)
        batch size for validation data set
    seed: int, Optional (Default: 0)
        rng seed
    n_neurons: int, Optional (Default: 1000)
        number of neurons
    minibatch_size: int, Optional (Default: 100)
        size to break the batch up into
    n_steps: int, Optional (Default: 1)
        number of steps to run sim for, for training
        it is better to set it to 1 and change the
        batch size (n_training, n_validation) to be
        the number of images to train
    res: list of two ints
        resolution to scale the image to before flattening
        if no scaling then pass in the data image resolution
    """
    def __init__(
        self, res, minibatch_size=100, weights=None, filters=None, kernel_size=None, strides=None, seed=0):

        self.minibatch_size = minibatch_size
        self.seed = seed

        if filters is None:
            filters = [32, 64, 128]

        if kernel_size is None:
            kernel_size = [3, 3, 3]

        if strides is None:
            strides = [1, 1, 1]

        warnings.simplefilter("ignore")

        image_input = tf.keras.Input(shape=(res[0], res[1], 3), batch_size=self.minibatch_size)

        conv1 = tf.keras.layers.Conv2D(
            filters=filters[0],
            kernel_size=kernel_size[0],
            strides=strides[0],
            use_bias=True,
            activation=tf.nn.relu,
            data_format="channels_last",
            )(image_input)

        conv2 = tf.keras.layers.Conv2D(
            filters=filters[1],
            kernel_size=kernel_size[1],
            strides=strides[1],
            use_bias=True,
            activation=tf.nn.relu,
            data_format="channels_last",
            )(conv1)

        conv3 = tf.keras.layers.Conv2D(
            filters=filters[2],
            kernel_size=kernel_size[2],
            strides=strides[2],
            use_bias=True,
            activation=tf.nn.relu,
            data_format="channels_last",
            )(conv2)

        flatten = tf.keras.layers.Flatten()(conv3)

        output_probe = tf.keras.layers.Dense(
            units=1,
            )(flatten)

        model = tf.keras.Model(inputs=image_input, outputs=output_probe)

        converter = nengo_dl.Converter(model)
        self.image_input = converter.inputs[image_input]
        self.output_probe = converter.outputs[output_probe]
        self.net = converter.net

        if weights is not None:
            with nengo_dl.Simulator(self.net, minibatch_size=self.minibatch_size, seed=self.seed) as sim:
                sim.load_params(weights)
                sim.freeze_params(self.net)


    # def train(self, images, targets, epochs=25, preprocess_images=True):
    #     """
    #     Take in images in the shape of (batch_size, horizontal_pixels, vertical_pixels, 3)
    #     """
    #     with nengo_dl.Simulator(self.net, minibatch_size=self.minibatch_size, seed=self.seed) as sim:
    #
    #         training_images_dict = {
    #             image_input: training_images.reshape(
    #                 (n_training, n_steps, subpixels))
    #         }
    #
    #         validation_images_dict = {
    #             image_input: validation_images.reshape(
    #                 (n_validation, n_steps, subpixels))
    #         }
    #
    #
    #         training_targets_dict = {
    #             #output_probe_no_filter: training_targets.reshape(
    #             output_probe: training_targets.reshape(
    #                 (n_training, n_steps, output_dims))
    #         }
    #
    #         # if train_on_data:
    #
    #         print('Training')
    #         sim.compile(
    #             # optimizer=tf.optimizers.SGD(1e-5),
    #             optimizer=tf.optimizers.RMSprop(0.001),
    #             # # loss={output_probe_no_filter: tf.losses.mse})
    #             loss={output_probe: tf.losses.mse})
    #
    #         if isinstance(epochs, int):
    #             epochs = [1, epochs]
    #
    #         for epoch in range(epochs[0]-1, epochs[1]):
    #             num_pts = 100
    #             print('\n\nEPOCH %i\n' % epoch)
    #             if load_net_params and epoch>0:
    #                 sim.load_params('%s/%s' % (save_folder, params_file))
    #                 print('loading pretrained network parameters')
    #
    #             # if train_on_data:
    #             sim.fit(training_images_dict, training_targets_dict, epochs=1)
    #
    #             # save parameters back into net
    #             sim.freeze_params(net)
    #
    #             if save_net_params:
    #                 print('saving network parameters to %s/%s' % (save_folder, params_file))
    #                 sim.save_params('%s/%s' % (save_folder, params_file))
    #
    #             predict_data = predict(save_folder=save_folder, save_name='prediction_epoch%i' % (epoch), num_pts=num_pts)
    #
    #         dat_results.save(
    #             data={'targets': validation_targets},
    #             save_location=save_folder,
    #             overwrite=True)
    #         # return predict_data[neuron_probe]
    #
    #     db_name = 'rover_training_0004'
    #     # db_name = 'rover_dist_range_0_marked'
    #     save_db_name = '%s_results' % db_name
    #     # use keras model converted to nengo-dl
    #     use_keras = True
    #     # train or just validate
    #     # train_on_data = True
    #     # load params from previous epoch
    #     load_net_params = True
    #     # save net params
    #     save_net_params = True
    #     # show plots for processed data
    #     debug = False
    #     # show the images before and after scaling
    #     show_resized_image = debug
    #     # range of epochs
    #     epochs = [1, 5]
    #     n_steps = 1
    #     # training batch size
    #     n_training = 10000
    #     minibatch_size = 100
    #     # validation batch size
    #     n_validation = 1000
    #     output_dims = 1
    #     n_neurons = 20000
    #     gain = 100
    #     bias = 0
    #     # resolution to scale images to
    #     res=[32, 128]
    #     # select a subset of rows of the image to use, these will be used to index into the array
    #     # rows = [27, 28]
    #     rows = None
    #     pixels = res[0] * res[1]
    #     subpixels = pixels*3
    #     seed = 0
    #     np.random.seed(seed)
    #     flatten = True
    #     if use_keras:
    #         flatten = False
    #
    #     # use training data in validation step
    #     # validate_with_training = False
    #
    #     #save_name = 'image_input-to-dense_layer_neurons-norm_input_weights_10k-gain_bias'
    #     # save_name = 'version_that_works_specifying_enc_30k'
    #     save_name = 'target_from_local_error'
    #     params_file = 'saved_net_%ix%i' % (res[0], res[1])
    #     if use_keras:
    #         save_folder = 'data/%s/keras/%s/%s' % (db_name, params_file, save_name)
    #     else:
    #         save_folder = 'data/%s/nengo_dl/%s/%s' % (db_name, params_file, save_name)
    #
    #     if not os.path.exists(save_folder):
    #         os.makedirs(save_folder)
    #
    #     notes = (
    #     '''
    #     \n- bias on all layer
    #     \n- default kernel initializers
    #     \n- 3 conv layers 1 dense
    #     '''
    #             )
    #     # '''
    #     # '''
    #     # )
    #     # '''
    #     # \n- having input node connect to dense_node seemed redundant, removed dense_nose
    #     # and now have connections from image_input>dense_layer.neurons, dense_layer.neurons>output_node
    #     # \n- was not normalizing input weights for input to dense_layer.neurons, as is default
    #     # in nengo
    #     # \n- set connection seeds
    #     # \n- setting gain to 100 and bias to 0
    #     # ''')
    #
    #     test_params = {
    #         'use_keras': use_keras,
    #         'n_training': n_training,
    #         'minibatch_size': minibatch_size,
    #         'n_neurons': n_neurons,
    #         'res': res,
    #         'seed': seed,
    #         'notes': notes,
    #         'gain': gain,
    #         'bias': bias}
    #
    #     dat_results = DataHandler(db_name=save_db_name)
    #     dat_results.save(data=test_params, save_location='%s/params'%save_folder, overwrite=True)
    #
    #     # # generate data
    #     # training_images, training_targets = gen_data(n_training, res=res, pixels=pixels)
    #     # validation_images, validation_targets = gen_data(n_validation)
    #
    #     # load data
    #     training_images, training_targets = load_data(
    #         res=res, label='training_0000', n_imgs=n_training, rows=rows,
    #         debug=debug, show_resized_image=show_resized_image, db_name=db_name,
    #         flatten=flatten)
    #
    #     # if validate_with_training:
    #     #     # images are flattened at this point, need to find how
    #     #     # many pixels per image
    #     #     n_val = int(n_validation * len(training_images) / n_training)
    #     #     validation_images=training_images[:n_val]
    #     #     n_val = int(n_validation * len(training_targets) / n_training)
    #     #     validation_targets=training_targets[:n_val]
    #     # else:
    #     validation_images, validation_targets = load_data(
    #         res=res, label='validation_0000', n_imgs=n_validation, rows=rows,
    #         debug=debug, show_resized_image=show_resized_image, db_name=db_name,
    #         flatten=flatten)
    #
    #     test_print = ('Running tests for %s' % save_folder)
    #     print('\n')
    #     print('Training Images: ', training_images.shape)
    #     print('Validation Images: ', validation_images.shape)
    #     print('-'*len(test_print))
    #     print(test_print)
    #     print('-'*len(test_print))
    #     print('\n')


    def predict(self, images, targets=None, show_fig=False): #, save_folder='', save_name='prediction_results', num_pts=100):
        """
        Parameters
        ----------
        stateful: boolean, Optional (Default: False)
            True: learn during this pass, weights will be updated
            False: weights will revert to what they were at the start
            of the prediction
        """
        images = np.asarray(images)
        if images.ndim == 3:
            shape = images.shape
            images = images.reshape((1, shape[0], shape[1], shape[2]))

        shape = images.shape
        subpixels = shape[1]*shape[2]*shape[3]
        batch_size = shape[0]

        images_dict = {
            self.image_input: images.reshape(
                (batch_size, 1, subpixels))
        }

        with nengo_dl.Simulator(self.net, minibatch_size=self.minibatch_size, seed=self.seed) as sim:
            data = sim.predict(images_dict, n_steps=1, stateful=False)
            predictions = data[self.output_probe]
            predictions = np.asarray(predictions).squeeze()

            if show_fig:
                num_pts = 100
                fig = plt.Figure()
                plt.plot(targets[-num_pts:], label='target', color='r')
                plt.plot(predictions[-num_pts:], label='predictions', color='k', linestyle='--')
                plt.legend()
                # plt.savefig('%s/%s.png' % (save_folder, save_name))
                plt.show()
                plt.close()
                fig = None

        return predictions


    # def plot_results(self, predictions, targets, image_nums=None):
    #     predictions = np.asarray(predictions).squeeze()
    #     targets = np.asarray(targets).squeeze()
    #
    #
    #     # print('\nIN PLOT')
    #     # print(targets[-1, :])
    #
    #     fig = plt.Figure()
    #     if targets.ndim == 2:
    #         x = plt.subplot(311)
    #         x.set_title('X')
    #         x.plot(targets[:, 0], label='Target', color='r')
    #         x.plot(predictions[:, 0], label='Pred', color='k', linestyle='--')
    #         plt.legend()
    #
    #         y = plt.subplot(312)
    #         y.set_title('Y')
    #         y.plot(targets[:, 1], label='Target', color='g')
    #         y.plot(predictions[:, 1], label='Pred', color='k', linestyle='--')
    #         plt.legend()
    #
    #         z = plt.subplot(313)
    #         z.set_title('Z')
    #         z.plot(targets[:, 2], label='Target', color='b')
    #         z.plot(predictions[:, 2], label='Pred', color='k', linestyle='--')
    #
    #         plt.legend()
    #         plt.savefig('prediction_results.png')
    #         plt.show()
    #
    #     elif targets.ndim == 1:
    #         plt.plot(targets, label='target', color='r')
    #         plt.plot(predictions, label='predictions', color='k')
    #         plt.legend()
    #         plt.savefig('prediction_results.png')
    #         plt.show()
    #     else:
    #         raise Exception ('targets have unexpected shape ', targets.shape)
    #
    #     if image_nums is not None:
    #         raise NotImplementedError
    #         # rgb2 = cv2.resize(rgb, dsize=(res[0], res[1]), interpolation=cv2.INTER_CUBIC)
    #         # plt.Figure()
    #         # a = plt.subplot(121)
    #         # a.set_title('Original')
    #         # a.imshow(rgb, origin='lower')
    #         # b = plt.subplot(122)
    #         # b.set_title('Scaled')
    #         # b.imshow(rgb2, origin='lower')

    # def load_data(
    #         res, db_name, label='training_0000', n_imgs=None, rows=None, debug=False, show_resized_image=False,
    #         flatten=True):
    #     dat = DataHandler(db_name)
    #
    #     # load training images
    #     training_images = []
    #     training_targets = []
    #     for nn in range(0, n_imgs):
    #         data = dat.load(parameters=['rgb', 'local_error'], save_location='%s/data/%04d' % (label, nn))
    #         training_images.append(data['rgb'])
    #         training_targets.append(data['local_error'])
    #
    #     # scale image resolution
    #     training_images, training_targets = preprocess_data(
    #         image_data=training_images, debug=debug,
    #         res=res, show_resized_image=show_resized_image, rows=rows,
    #         flatten=flatten, local_error=training_targets)
    #
    #     return training_images, training_targets
    #

    def resize_images(
            self, image_data, res, rows=None, show_resized_image=False, flatten=True):
        # single image, append 1 dimension so we can loop through the same way
        image_data = np.asarray(image_data)
        if image_data.ndim == 3:
            shape = image_data.shape
            image_data = image_data.reshape((1, shape[0], shape[1], shape[2]))

        # expect rgb image data
        assert image_data.shape[3] == 3

        scaled_image_data = []

        for count, data in enumerate(image_data):
            # normalize
            rgb = np.asarray(data)/255

            # select a subset of rows and update the vertical resolution
            if rows is not None:
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

            # flatten to 1D
            #NOTE should use np.ravel to maintain image order
            if flatten:
                rgb = rgb.flatten()
            # scale image from -1 to 1 and save to list
            scaled_image_data.append(rgb*2 - 1)

        scaled_image_data = np.asarray(scaled_image_data)

        return scaled_image_data

    def target_angle_from_local_error(self, local_error):
        local_error = np.asarray(local_error)
        if local_error.ndim == 3:
            shape = local_error.shape
            local_error = local_error.reshape((1, shape[0], shape[1], shape[2]))

        angles = []
        for error in local_error:
            angles.append(math.atan2(error[1], error[0]))

        angles = np.array(angles)

        return angles



if __name__ == '__main__':
    from abr_analyze import DataHandler
    dat = DataHandler('rover_training_0004')
    weights = 'saved_net_32x128'
    res=[32, 128]

    # instantiate vision class
    rover = rover_vision(res=res, weights=weights)

    # load training images
    training_images = []
    training_targets = []
    for nn in range(0, 1000):
        data = dat.load(parameters=['rgb', 'local_error'], save_location='training_0000/data/%04d' % nn)
        training_images.append(data['rgb'])
        training_targets.append(data['local_error'])

    # scale image resolution
    training_images = rover.resize_images(
        image_data=training_images,
        res=res,
        show_resized_image=False,
        flatten=False)

    # get target angle
    training_targets = rover.target_angle_from_local_error(training_targets)

        #
        # validation_images = []
        # validation_targets = []
        # for nn in range(1, n_validation+1):
        #     data = dat.load(parameters=['rgb', 'local_error'], save_location='validation_0000/data/%04d' % nn)
        #     validation_images.append(data['rgb'])
        #     validation_targets.append(data['local_error'])
        # # scale validation images
        # validation_images, validation_targets = rover.preprocess_data(
        #     image_data=validation_images, target_data=validation_targets,
        #     res=res, show_resized_image=False)

    predictions = rover.predict(images=training_images, targets=training_targets, show_fig=True)
