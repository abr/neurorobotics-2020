import nengo
import keras
import nengo_dl
import nengo_loihi
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

class RoverVision():
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
            units=2,
            )(flatten)

        model = tf.keras.Model(inputs=image_input, outputs=output_probe)

        # converter = nengo_dl.Converter(model)
        converter = nengo_dl.Converter(model, swap_activations={tf.nn.relu: nengo.SpikingRectifiedLinear()})
        # self.image_input = converter.inputs[image_input]
        # self.output_probe = converter.outputs[output_probe]
        self.net = converter.net

        if weights is not None:
            # with nengo_dl.Simulator(self.net, minibatch_size=self.minibatch_size, seed=self.seed) as sim:
            with nengo_dl.Simulator(self.net) as sim:
                sim.load_params(weights)
                sim.freeze_params(self.net)


    # def predict(self, images, targets=None, show_fig=False): #, save_folder='', save_name='prediction_results', num_pts=100):
    #     """
    #     Parameters
    #     ----------
    #     stateful: boolean, Optional (Default: False)
    #         True: learn during this pass, weights will be updated
    #         False: weights will revert to what they were at the start
    #         of the prediction
    #     """
    #     images = np.asarray(images)
    #     shape = images.shape
    #     # print(images)
    #     if images.ndim == 3:
    #         batch_size = 1
    #         subpixels = shape[0]*shape[1]*shape[2]
    #         images = images.reshape(1, shape[0], shape[1], shape[2])
    #     elif images.ndim == 4:
    #         batch_size = shape[0]
    #         subpixels = shape[1]*shape[2]*shape[3]
    #
    #     images_dict = {
    #         self.image_input: images.reshape(
    #             (batch_size, 1, subpixels))
    #     }
    #
    #     with nengo_dl.Simulator(self.net, minibatch_size=self.minibatch_size, seed=self.seed) as sim:
    #         data = sim.predict(images_dict, n_steps=1, stateful=False)
    #         predictions = np.asarray(data[self.output_probe]).squeeze()
    #
    #         if show_fig:
    #             fig = plt.Figure()
    #             num_pts = 100
    #             plt.plot(targets[-num_pts:], label='target', color='r')
    #             plt.plot(predictions[-num_pts:], label='predictions', color='k', linestyle='--')
    #             plt.legend()
    #             plt.show()
    #             plt.close()
    #             fig = None
    #
    #     return predictions
    #
    #
    # def resize_images(
    #         self, image_data, res, rows=None, show_resized_image=False, flatten=True):
    #     # single image, append 1 dimension so we can loop through the same way
    #     image_data = np.asarray(image_data)
    #     if image_data.ndim == 3:
    #         shape = image_data.shape
    #         image_data = image_data.reshape((1, shape[0], shape[1], shape[2]))
    #
    #     # expect rgb image data
    #     assert image_data.shape[3] == 3
    #
    #     scaled_image_data = []
    #
    #     for count, data in enumerate(image_data):
    #         # normalize
    #         rgb = np.asarray(data)/255
    #
    #         # select a subset of rows and update the vertical resolution
    #         if rows is not None:
    #             rgb = rgb[rows[0]:rows[1], :, :]
    #             res[0] = rows[1]-rows[0]
    #
    #         # resize image resolution
    #         rgb = cv2.resize(
    #             rgb, dsize=(res[1], res[0]),
    #             interpolation=cv2.INTER_CUBIC)
    #
    #         # visualize scaling for debugging
    #         if show_resized_image:
    #             plt.Figure()
    #             a = plt.subplot(121)
    #             a.set_title('Original')
    #             a.imshow(data, origin='lower')
    #             b = plt.subplot(122)
    #             b.set_title('Scaled')
    #             b.imshow(rgb, origin='lower')
    #             plt.show()
    #
    #         # flatten to 1D
    #         #NOTE should use np.ravel to maintain image order
    #         if flatten:
    #             rgb = rgb.flatten()
    #         # scale image from -1 to 1 and save to list
    #         scaled_image_data.append(rgb*2 - 1)
    #
    #     scaled_image_data = np.asarray(scaled_image_data)
    #
    #     return scaled_image_data
    #
    # def target_angle_from_local_error(self, local_error):
    #     local_error = np.asarray(local_error)
    #     if local_error.ndim == 3:
    #         shape = local_error.shape
    #         local_error = local_error.reshape((1, shape[0], shape[1], shape[2]))
    #
    #     angles = []
    #     for error in local_error:
    #         angles.append(math.atan2(error[1], error[0]))
    #
    #     angles = np.array(angles)
    #
    #     # angles = math.atan2(local_error[1], local_error[0])
    #
    #     return angles


if __name__ == '__main__':
    from abr_analyze import DataHandler
    dat = DataHandler('rover_training_0004')
    weights = 'saved_net_32x128_learn_xy'
    res=[32, 128]
    minibatch_size = 1

    # instantiate vision class
    rover = RoverVision(res=res, weights=weights, minibatch_size=minibatch_size)

    # load training images
    # training_images = []
    # training_targets = []
    # for nn in range(0, 1000):
    #     data = dat.load(parameters=['rgb', 'local_error'], save_location='validation_0000/data/%04d' % nn)
    #     training_images.append(data['rgb'])
    #     training_targets.append(data['local_error'])

    # scale image resolution
    # training_images = rover.resize_images(
    #     image_data=training_images,
    #     res=res,
    #     show_resized_image=False,
    #     flatten=False)

    # get target angle
    # training_targets = rover.target_angle_from_local_error(training_targets)

    # predictions = rover.predict(images=training_images, targets=training_targets, show_fig=True)
    try:
        sim = nengo_loihi.Simulator(
            rover.net, target='sim')

        while 1:
            sim.run(1e5)

    except ExitSim:
        pass

    finally:
        sim.close()


