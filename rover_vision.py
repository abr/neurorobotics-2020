import nengo
import nengo_dl
import nengo_loihi
import sys

from contextlib import nullcontext

import keras
import cv2
import warnings
import os
import dl_utils
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from abr_analyze import DataHandler
from nengo.utils.matplotlib import rasterplot

from loihi_utils import LoihiRectifiedLinear


# =================== Tensorflow settings to avoid OOM errors =================
warnings.simplefilter("ignore")
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            # tf.config.experimental.set_memory_growth(gpu, True)
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
# =============================================================================


class RoverVision:
    def __init__(self, res, minibatch_size=1, dt=0.001, seed=0):
        """
        A keras converted network for locating a red ball in a mujoco generated image.
        Returns the local error (x, y).

        To extend this network with other nengo objects you will need to run the
        extend_network() function. This creates an input node so you can manually
        inject images into the network. You can then use the sim or net objects from
        this class and extend them as desired.

        Pass input images to vision.net.validation_images - this updates the input node
        function
        Get outputs by connecting to the vision.output node

        vision = RoverVision(res=[32, 128], minibatch_size=10, dt=0.001, seed=0)

        with vision.net as net:
            def sim_fun(t, x):
                # x is the vision prediction
                # do something with the vision output
                # update our input image
                vision.net.validation_images = updated_image

            ens = nengo.Ensemble(...)
            nengo.Connection(vision.output, ens)

        # update our sim with our extended net
        sim = nengo.Simulator(vision.net)
        sim.run(1e5)


        Parameters
        ----------
        res: list of two ints
            the resolution, in vertical by horizontal pixels, of the input images
        minibatch_size: int
            specifies how to break up the training batches.
            NOTE: must be <= the batch_size
        dt: float
            simulation time step
        probe_neuron: boolean
            toggles the nengo probe on the convolutional layer's neurons
            NOTE: if you are running into OOM errors, try setting this to False
        """

        self.dt = dt
        self.seed = seed
        self.subpixels = res[0] * res[1] * 3
        self.minibatch_size = minibatch_size
        self.image_input = np.zeros(self.subpixels)

        # Define our keras network
        self.input = tf.keras.Input(shape=(res[0], res[1], 3))

        self.conv0 = tf.keras.layers.Conv2D(
            filters=3,
            kernel_size=1,
            strides=1,
            use_bias=False,
            activation=tf.nn.relu,
            data_format="channels_last",
        )(self.input)

        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=5,
            strides=1,
            use_bias=False,
            activation=tf.nn.relu,
            data_format="channels_last",
        )(self.conv0)

        flatten = tf.keras.layers.Flatten()(self.conv1)
        self.dense = tf.keras.layers.Dense(units=2, use_bias=False)(flatten)

        self.model = tf.keras.Model(inputs=self.input, outputs=self.dense)

    def convert(self, gain_scale, activation, synapse=None, training=True):
        """
        gain_scale: int
            scaling factor for spiking network to increase the amount of activity.
            gain gets scaled by gain_scale and neuron amplitudes get set to 1/gain_scale
        actiation: nengo.Neuron
            the neuron type to swap in for the tf.nn.relu in the Keras network
        synapse: float
            synapses set on nengo connections
        """

        converter = nengo_dl.Converter(
            self.model,
            swap_activations={tf.nn.relu: activation},
            scale_firing_rates=gain_scale,
            synapse=synapse,
        )

        # create references to some nengo objects in the network IO objects
        self.nengo_input = converter.inputs[self.input]
        self.nengo_dense = converter.outputs[self.dense]

        net = converter.net

        self.nengo_innode = converter.layers[self.input]
        self.nengo_conv0 = converter.layers[self.conv0]
        nengo_conv1 = converter.layers[self.conv1]
        self.nengo_output = converter.layers[self.dense]

        if training:
            with net:
                # set our biases to non-trainable to make sure they're always 0
                net.config[self.nengo_conv0].trainable = False
                net.config[nengo_conv1].trainable = False

                # set up probes so that we can add the firing rates to the cost function
                self.probe_conv0 = nengo.Probe(self.nengo_conv0, label="probe_conv0")
                self.probe_conv1 = nengo.Probe(nengo_conv1, label="probe_conv1")
                self.probe_dense = nengo.Probe(self.nengo_output, label="probe_dense", synapse=0.005)

        sim = nengo_dl.Simulator(
            net, minibatch_size=self.minibatch_size, seed=self.seed
        )
        return sim, net

    def convert_nengodl_to_nengo(self, net, image_array=False, loihi=False):
        # our input node function
        def send_image_in(t):
            # if updating an image over time, like rendering from a simulator
            # then we update this object each time
            if image_array:
                return self.image_input[int(t / 0.001) - 1]
            return self.image_input

        # if not using nengo dl we have to use a sim.run function
        # create a node so we can inject data into the network
        with net:
            if loihi:
                nengo_loihi.add_params(net)
                net.config[self.nengo_conv0.ensemble].on_chip = False
            # create our input node
            self.nengo_innode.size_out = self.subpixels
            self.nengo_innode.output = send_image_in

        # replace nengo_dl simulator with the nengo or nengo_loihi simulator
        if loihi:
            sim = nengo_loihi.Simulator(net, dt=self.dt)
        else:
            sim = nengo.Simulator(net, dt=self.dt)

        return sim, net
        # return None, net

    def train(
        self,
        sim,
        images,
        targets,
        epochs,
        save_folder=None,
        validation_images=None,
        validation_targets=None,
        num_pts=None,
        fr_loss_function=None,
    ):
        """
        We loop through from epochs[0] to epochs[1]. If epochs[0]
        is not 0 then this function assumes the weights to start
        with will be at '%s/epoch_%i' % (save_folder, epoch-1)

        Saves validation prediction plots if validation data is passed in
        (validation_images and targets)

        Parameters
        ----------
        images: array of floats (flattened images)
            shape (n_training_images, n_steps, subpixels)
        targets: array of floats (flattened targets)
            shape (n_training_targets, n_steps, target_dimensions)
        epochs: list of ints
            start and stop epochs
        validation_images: array of floats (flattened images)
            shape (n_validation_images, n_steps, subpixels)
        validation_targets: array of floats (flattened targets)
            shape (n_validation_targets, n_steps, target_dimensions)
        save_folder: string
            location where to save figures of results
        num_pts: int
            number of steps to plot from the predictions
        """
        if validation_images is not None:
            validation_images_dict = {self.nengo_input: validation_images}

        loss = {self.nengo_dense: tf.losses.mse}
        training_targets = {self.nengo_dense: targets}
        if fr_loss_function:
            loss[self.probe_conv0]: fr_loss_function
            loss[self.probe_conv1]: fr_loss_function
            train_data[self.probe_conv0]: np.zeros(targets.shape)
            train_data[self.probe_conv1]: np.zeros(targets.shape)

        with sim:
            sim.compile(optimizer=tf.optimizers.RMSprop(0.001), loss=loss)

            for epoch in range(epochs[0], epochs[1]):
                print("\nEPOCH %i" % epoch)
                if epoch > 0:
                    prev_params_loc = "%s/epoch_%i" % (save_folder, epoch - 1)
                    print(
                        "loading pretrained network parameters from \n%s"
                        % prev_params_loc
                    )
                    sim.load_params(prev_params_loc)

                print("using learning rate scheduler...")

                # if epoch < 2:
                #     def scheduler(x):
                #         return 0.001
                # else:
                def scheduler(x):
                    return 0.0001 * np.exp(0.1 * (0 - epoch//2))

                print("Learning rate: ", scheduler(None))

                lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

                print("fitting data...")
                sim.fit(
                    training_images,
                    training_targets,
                    epochs=1,
                    callbacks=[lr_scheduler],
                )

                current_params_loc = "%s/epoch_%i" % (save_folder, epoch)
                print("saving network parameters to %s" % current_params_loc)
                sim.save_params(current_params_loc)

                if validation_images is not None:
                    print("Running Prediction in nengo-dl")
                    data = sim.predict(
                        validation_images_dict,
                        n_steps=validation_images.shape[1],
                        stateful=False,
                    )
                    print(data)

                    dl_utils.plot_prediction_error(
                        predictions=np.asarray(data[vision.nengo_dense]),
                        target_vals=validation_targets,
                        save_folder=save_folder,
                        save_name="validation_error_epoch_%i" % epoch,
                        num_pts=num_pts,
                        show_plot=True,
                    )


if __name__ == "__main__":

    mode = sys.argv[1]
    assert mode in ["train", "predict", "run"]

    activations = {
        "relu": nengo.RectifiedLinear(),
        "srelu": nengo.SpikingRectifiedLinear(),
        "lif": nengo.LIF(),
        "loihirelu": nengo_loihi.neurons.LoihiSpikingRectifiedLinear(),
        "loihirelurate": LoihiRectifiedLinear(),
        "loihilif": nengo_loihi.neurons.LoihiLIF(),
    }

    # rate neurons are default
    if len(sys.argv) > 2:
        try:
            activation_name = sys.argv[2].lower()
            activation = activations[activation_name]
        except KeyError:
            print(activations.keys())
            raise KeyError

    # check if gain / amplitude scaling is set, default is 1
    gain_scale = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    use_dl_rate = False
    if len(sys.argv) > 4:
        use_dl_rate = sys.argv[4] == "use_dl_rate"

    use_new_driver_validation = True

    # ------------ set test parameters
    dt = 0.001
    # db_name = "rover_training_0004"
    db_name = "abr_analyze"
    res = [32, 128]
    n_training = 30000  # number of images to train on
    n_validation = 1200 # number of validation images
    n_validation_steps = 1
    seed = np.random.randint(1e5)
    data_dir = "/home/tdewolf/Downloads"
    driving_data = True  # use data collected while driving around
    normalize_targets = True  # change the target range to -1:1 instead of -pi:pi

    # using baseline of 1ms timesteps to define n_steps
    # adjust based on dt to keep sim time constant
    n_validation_steps = int(n_validation_steps * 0.001 / dt)

    minibatch_size = 10 if mode == "train" else 1
    # can't have a minibatch larger than our number of images
    minibatch_size = min(minibatch_size, n_validation)

    # plotting parameters
    num_pts = n_validation * n_validation_steps  # number of steps to plot

    # dl_utils.plot_data(db_name=db_name, label='validation_0000', n_imgs=100)

    # ------------ load and prepare our data
    # load our raw data
    validation_images, validation_targets = dl_utils.load_data(
        # db_name=db_name, label="validation_0000", n_imgs=n_validation
        db_name=db_name, label="driving_0047", #n_imgs=115,
    )

    # our saved targets are 3D but we only care about x and y
    validation_targets = validation_targets[:, 0:2]
    if normalize_targets:
        validation_targets /= np.pi

    # do our resizing, scaling, and flattening
    validation_images = dl_utils.preprocess_images(
        image_data=validation_images,
        show_resized_image=False,
        flatten=True,
        normalize=False,
        res=res,
    )

    # we have to set the minibatch_size when instantiating the network so if we want
    # to run sim.predict after each training session to track progress, we need to have
    # a consistent minibatch_size between training and predicting
    # this is also using rate neurons during training so it doesn't matter if the
    # prediction is batched (won't have the output zeroed between minibatches since
    # we aren't using spiking neurons)
    batch_data = True if mode == "train" else False

    # repeat and batch our data
    validation_images = dl_utils.repeat_data(
        data=validation_images, batch_data=batch_data, n_steps=n_validation_steps
    )
    validation_targets = dl_utils.repeat_data(
        data=validation_targets, batch_data=batch_data, n_steps=n_validation_steps
    )

    # ------------ set up data tracking
    save_db_name = "%s_results" % db_name  # for saving results to
    group_name = activation_name  # helps to define what this group of tests is doing
    test_name = str(gain_scale)
    # for saving figures and weights
    save_folder = "%s/data/%s/%s/%s" % (data_dir, db_name, group_name, test_name,)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # instantiate our keras converted network
    vision = RoverVision(res=res, minibatch_size=minibatch_size, dt=dt, seed=seed,)

    if mode == "train":

        n_training_steps = 1
        epochs = [284, 1000]

        # prepare training data ------------------------------------
        if driving_data:
            try:
                processed_data = np.load("%s/%s_driving_training_images_processed.npz" % (data_dir, db_name))
                training_images = processed_data["images"]
                training_targets = processed_data["targets"]
                if normalize_targets:
                    training_targets /= np.pi
                print("Processed training images loaded from file...")

            except FileNotFoundError:
                training_images, training_targets = dl_utils.consolidate_data(
                    db_name=db_name,
                    label_list=['driving_%04i' % ii for ii in range(45)],
                    step_size=5,
                )
                # our saved targets are 3D but we only care about x and y
                training_targets = training_targets[:, 0:2]

                # do our resizing, scaling, and flattening
                training_images = dl_utils.preprocess_images(
                    image_data=training_images,
                    show_resized_image=False,
                    flatten=True,
                    normalize=False,
                    res=res,
                )
                np.savez_compressed(
                    "%s/%s_driving_training_images_processed" % (data_dir, db_name),
                    images=training_images,
                    targets=training_targets,
                )
        else:
            try:
                processed_data = np.load("%s/%s_training_images_processed.npz" % (data_dir, db_name))
                training_images = processed_data["images"]
                training_targets = processed_data["targets"]
                if normalize_targets:
                    training_targets /= np.pi
                print("Processed training images loaded from file...")

            except FileNotFoundError:
                print("Processed training images not found, generating...")
                # load raw data
                training_images, training_targets = dl_utils.load_data(
                    db_name=db_name, label="training_0000", n_imgs=n_training
                )
                # our saved targets are 3D but we only care about x and y
                training_targets = training_targets[:, 0:2]

                # do our resizing, scaling, and flattening
                training_images = dl_utils.preprocess_images(
                    image_data=training_images,
                    show_resized_image=False,
                    flatten=True,
                    normalize=False,
                    res=res,
                )
                np.savez_compressed(
                    "%s/%s_training_images_processed" % (data_dir, db_name),
                    images=training_images,
                    targets=training_targets,
                )

        # repeat and batch our data
        training_images = dl_utils.repeat_data(
            data=training_images, batch_data=True, n_steps=n_training_steps
        )
        training_targets = dl_utils.repeat_data(
            data=training_targets, batch_data=True, n_steps=n_training_steps
        )

        sim, net = vision.convert(
            gain_scale=gain_scale, activation=activation, synapse=None
        )
        vision.train(
            sim=sim,
            images=training_images,
            targets=training_targets,
            epochs=epochs,
            # validation_images=validation_images,
            validation_images=None,
            save_folder=save_folder,
            num_pts=num_pts,
            validation_targets=validation_targets,
        )

        # Notes to save with this test
        notes = """
        \n- setting biases to non-trainable
        \n- 32 filters
        \n- bias off on all layer, loihi restriction
        \n- default kernel initializers
        \n- 2 conv layers 1 dense
        \n- first conv layer runs off chip
        \n- changing image scaling from -1 to 1, to 0 to 1 due to loihi comm restriction
        """

        test_params = {
            "dt": dt,
            "gain_scale": gain_scale,
            "n_training": n_training,
            "n_validation": n_validation,
            "res": res,
            "minibatch_size": minibatch_size,
            "seed": seed,
            "notes": notes,
            "epochs": epochs,
        }

        # Save our set up parameters
        dat_results = DataHandler(db_name=save_db_name)
        dat_results.save(
            data=test_params, save_location="%s/params" % save_folder, overwrite=True
        )

    else:

        synapse = None  # 0.001
        # weights = "/home/tdewolf/Downloads/data/abr_analyze/loihirelu/400/epoch_260"
        # weights = "/home/tdewolf/Downloads/data/abr_analyze/loihirelu/400/epoch_108"
        # weights = "/home/tdewolf/Downloads/data/abr_analyze/loihirelu/400/epoch_218"
        weights =  "/home/tdewolf/Downloads/data/abr_analyze/loihirelu/400/epoch_306"

        with tf.keras.backend.learning_phase_scope(1) if use_dl_rate else nullcontext():
            sim, net = vision.convert(
                gain_scale=gain_scale, activation=activation, synapse=synapse,
            )

            # load a specific set of weights
            if weights is not None:
                print("Received specific weights to use: ", weights)
                sim.load_params(weights)

            # if batched prediction
            if mode == "predict":
                data = sim.predict(
                    {vision.nengo_input: validation_images},
                    n_steps=validation_images.shape[1],
                    stateful=False,
                )

            elif mode == "run":
                with sim:
                    sim.freeze_params(net)
                # if we want to run this in Nengo and not Nengo DL
                # have to extend our network to manually inject input images
                sim, net = vision.convert_nengodl_to_nengo(net, image_array=True, loihi=True)
                print('simulator: ', sim)

                # the user needs to pass input image to rover_vision.image_input
                # should be [n_timesteps, image.flatten()]
                vision.image_input = validation_images[0]

                sim_steps = dt * n_validation_steps * n_validation
                sim.run(sim_steps)
                data = sim.data

            print("data shape: ", data[vision.nengo_dense].shape)
            print("validation targets size: ", validation_targets.shape)
            dl_utils.plot_prediction_error(
                # predictions=np.asarray(data[vision.nengo_dense]),
                predictions=np.asarray(data[vision.probe_dense]),
                target_vals=validation_targets,
                save_folder=save_folder,
                save_name="%s_prediction_error" % mode,
                num_pts=num_pts,
                show_plot=True,
            )
