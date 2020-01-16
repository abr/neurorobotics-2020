import nengo
import nengo_dl
import numpy as np
import cv2
import warnings
import tensorflow as tf
import matplotlib.pyplot as plt

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
        self, res=None, n_training=10000, n_validation=1000,
        seed=0, n_neurons=1000, minibatch_size=100, n_steps=1,
        weights=None):

        self.n_training = n_training
        self.n_validation = n_validation
        self.seed = seed
        self.n_neurons = n_neurons
        self.minibatch_size = minibatch_size
        self.n_steps = n_steps

        # image parameters
        assert res is not None, (
            "Require resolution of images, or the desired scaled resolution")
        self.res = res
        self.img_size = res[0] * res[1] * 3

        self.net = nengo.Network(seed=self.seed)
        #self.net.config[nengo.Ensemble].neuron_type = nengo.LIF()

        with self.net:
            self.image_input = nengo.Node(size_out=self.img_size, output=np.zeros(self.img_size))
            self.dense_layer = nengo.Ensemble(
                n_neurons=self.n_neurons,
                dimensions=self.img_size)

            # def output_func(t, x):
            #     self.output = np.copy(x)

            self.image_output = nengo.Node(size_in=3)

            self.conn_learn = nengo.Connection(
                self.image_input, self.dense_layer, label='input_conn')

            if weights is None:
                weights = np.zeros((3, self.dense_layer.n_neurons))
                print('**No weights passed in, starting with zeros**')
            else:
                print('**Using pretrained weights**')

            nengo.Connection(
                self.dense_layer.neurons,
                self.image_output,
                label='output_conn',
                transform=weights)

            self.output_probe = nengo.Probe(self.image_output, synapse=0.01)

        self.sim = nengo_dl.Simulator(
                self.net, minibatch_size=self.minibatch_size, seed=self.seed)


    def preprocess_data(self, image_data, target_data=None, res=None, show_plot=False):
        """
        Resizes image to desired resolution and flattens it to input to node,
        target_data gets flattened

        Parameters
        ----------
        image_data: array of the shape (batch_size, res[0], res[1], 3)
            the rgb images for training / evaluation
        target_data: array of the shape (batch_size, 3)
            the cartesian target data
        """
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
        for data in image_data:
            # scale to -1 to 1
            rgb = np.asarray(data)/255 * 2 -1
            # resize image resolution
            rgb = cv2.resize(
                rgb, dsize=(self.res[0], self.res[1]),
                interpolation=cv2.INTER_CUBIC)

            # visualize scaling for debugging
            if show_plot:
                plt.Figure()
                a = plt.subplot(121)
                a.set_title('Original')
                a.imshow(data, origin='lower')
                b = plt.subplot(122)
                b.set_title('Scaled')
                b.imshow(rgb, origin='lower')
                plt.show()

            # flatten to 1D
            rgb = rgb.flatten()
            scaled_image_data.append(rgb)

        scaled_image_data = np.asarray(scaled_image_data)

        if target_data is not None:
            target_data = np.asarray(target_data)
            if target_data.ndim == 1:
                shape = target_data.shape
                target_data = target_data.reshape((1, shape[0]))

            # except cartesian locations as target data
            assert target_data.shape[1] == 3

            scaled_target_data = []
            for data in target_data:
                scaled_target_data.append(data.flatten())

            scaled_target_data = np.asarray(scaled_target_data)
        else:
            scaled_target_data = None

        return scaled_image_data, scaled_target_data


    def get_weights(self):
        return self.sim.data[self.conn_learn].weights


    def train(self, images, targets, epochs=25, get_weights=True):

        # resize images and flatten all data
        images, targets = self.preprocess_data(
            image_data=images, target_data=targets, res=self.res, show_plot=False)

        # Training
        # turn off synapses for training to simplify
        for conn in self.net.all_connections:
            conn.synapse = None

        with self.net:
            self.output_probe_no_filter = nengo.Probe(self.image_output)
        # increase probe filter to account for removing interal filters
        self.output_probe.synapse = 0.04

        # NOTE does not work with this for some reason, made a git issue
        # nengo_dl.configure_settings(trainable=False)
        # self.net.config[nengo.Ensemble].trainable = True

        training_images_dict = {
            self.image_input: images.reshape(
                (self.n_training, self.n_steps, self.img_size))
        }
        training_targets_dict = {
            self.output_probe_no_filter: targets.reshape(
                (self.n_training, self.n_steps, 3))
        }

        self.sim = nengo_dl.Simulator(
                self.net, minibatch_size=self.minibatch_size, seed=self.seed)

        with self.sim:
            # run the training
            self.sim.compile(optimizer=tf.optimizers.RMSprop(0.01),
                        loss={self.output_probe_no_filter: tf.losses.mse})
            self.sim.fit(training_images_dict, training_targets_dict, epochs=epochs)

        if get_weights:
            weights = self.get_weights()
            return weights

    def validate(self, images, targets):

        def _test_mse(y_true, y_pred):
            return tf.reduce_mean(tf.square(y_pred[:, -10:] - y_true[:, -10:]))

        # resize images and flatten all data
        images, targets = self.preprocess_data(
            image_data=images, target_data=targets, res=self.res, show_plot=False)

        validation_images_dict = {
            self.image_input: images.reshape(
                (self.n_validation, self.n_steps, self.img_size))
        }
        validation_targets_dict = {
            self.output_probe: targets.reshape(
                (self.n_validation, self.n_steps, 3))
        }


        self.sim = nengo_dl.Simulator(
                self.net, minibatch_size=self.minibatch_size, seed=self.seed)

        with self.sim:
            print("Validation Error:")
            self.sim.compile(
                loss={self.output_probe: _test_mse})
            self.sim.evaluate(validation_images_dict, validation_targets_dict)

    def predict(self, images, stateful=False):
        """
        Parameters
        ----------
        stateful: boolean, Optional (Default: False)
            True: learn during this pass, weights will be updated
            False: weights will revert to what they were at the start
            of the prediction
        """

        # this is repeated from the preprocessing function, but we need to check
        # this to get the number of images before preprocessing
        # we set this for training / validation on init to set network parameters
        # TODO think of a consistent way to get this value, maybe assert that
        # the batch size is always the first dim for data?
        images = np.asarray(images)
        if images.ndim == 3:
            n_images = 1
        elif images.ndim == 4:
            n_images = images.shape[0]

        # resize images and flatten all data
        images, _ = self.preprocess_data(
            image_data=images, res=self.res, show_plot=False)

        prediction_images_dict = {
            self.image_input: images.reshape(
                (n_images, self.n_steps, self.img_size))
        }
        self.sim = nengo_dl.Simulator(
                self.net, minibatch_size=self.minibatch_size, seed=self.seed)
        predictions = self.sim.predict(
            prediction_images_dict, n_steps=self.n_steps, stateful=stateful)

        return predictions

    def plot_results(self, predictions, targets, image_nums=None):
        predictions = np.asarray(predictions).squeeze()
        targets = np.asarray(targets).squeeze()

        fig = plt.Figure()

        x = plt.subplot(311)
        x.set_title('X')
        x.plot(targets[:, 0], label='Target', color='r')
        x.plot(predictions[:, 0], label='Pred', color='k', linestyle='--')
        plt.legend()

        y = plt.subplot(312)
        y.set_title('Y')
        y.plot(targets[:, 1], label='Target', color='g')
        y.plot(predictions[:, 1], label='Pred', color='k', linestyle='--')
        plt.legend()

        z = plt.subplot(313)
        z.set_title('Z')
        z.plot(targets[:, 2], label='Target', color='b')
        z.plot(predictions[:, 2], label='Pred', color='k', linestyle='--')

        plt.legend()
        plt.show()

        if image_nums is not None:
            raise NotImplementedError
            # rgb2 = cv2.resize(rgb, dsize=(res[0], res[1]), interpolation=cv2.INTER_CUBIC)
            # plt.Figure()
            # a = plt.subplot(121)
            # a.set_title('Original')
            # a.imshow(rgb, origin='lower')
            # b = plt.subplot(122)
            # b.set_title('Scaled')
            # b.imshow(rgb2, origin='lower')




if __name__ == '__main__':
    from abr_analyze import DataHandler
    dat = DataHandler('rover_training')

    n_training = 10000
    n_validation = 1000

    rover = rover_vision(res=[64, 64], n_training=n_training, n_validation=n_validation,
        seed=0, n_neurons=1000, minibatch_size=100, n_steps=1)

    print('\nTRAINING')
    training_images = []
    training_targets = []
    for nn in range(1, n_training+1):
        data = dat.load(parameters=['rgb', 'local_error'], save_location='training_0000/data/%04d' % nn)
        training_images.append(data['rgb'])
        training_targets.append(data['local_error'])

    weights = rover.train(
        images=training_images,
        targets=training_targets,
        get_weights=True,
        epochs=10)
    dat.save(data={'weights': weights}, save_location='prediction_0000/data/weights', overwrite=True)

    print('\nVALIDATION')
    validation_images = []
    validation_targets = []
    for nn in range(1, n_validation+1):
        data = dat.load(parameters=['rgb', 'local_error'], save_location='validation_0000/data/%04d' % nn)
        validation_images.append(data['rgb'])
        validation_targets.append(data['local_error'])

    rover.validate(images=validation_images, targets=validation_targets)

    print('\nPREDICTION')
    predictions = rover.predict(images=validation_images)[rover.output_probe]
    rover.plot_results(predictions=predictions, targets=validation_targets)
