import nengo
import nengo_dl
import nengo_loihi
import sys

from contextlib import nullcontext

# import nengo_loihi
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

# this line is crucial for linking the Loihi neuron implementations
# to their differentiable TensorFlow implementations!
nengo_loihi.builder.nengo_dl.install_dl_builders()


class RoverVision:
    def __init__(self, res, minibatch_size, dt, seed, probe_neurons=False):
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

        vision = RoverVision(res=[32, 128], minibatch_size=10, dt=0.001
            seed=0, probe_neurons=False)
        vision.extend_network()

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

        # Define our keras network
        self.input = tf.keras.Input(
            shape=(res[0], res[1], 3), batch_size=self.minibatch_size
        )

        self.conv1 = tf.keras.layers.Conv2D(
            filters=3,
            kernel_size=1,
            strides=1,
            use_bias=False,
            activation=tf.nn.relu,
            data_format="channels_last",
        )(self.input)

        self.conv2 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=3,
            strides=1,
            use_bias=False,
            activation=tf.nn.relu,
            data_format="channels_last",
        )(self.conv1)

        flatten1 = tf.keras.layers.Flatten()(self.conv2)

        self.dense1 = tf.keras.layers.Dense(units=2, use_bias=False)(flatten1)

        self.model = tf.keras.Model(inputs=self.input, outputs=self.dense1)

    def convert(self, gain_scale, activation, synapses=None, probe_neurons=False):
        """
        gain_scale: int
            scaling factor for spiking network to increase the amount of activity.
            gain gets scaled by gain_scale and neuron amplitudes get set to 1/gain_scale
        synapses: list of 4 floats or None's
            synapses set on nengo connections
        """

        converter = nengo_dl.Converter(
            self.model,
            swap_activations={tf.nn.relu: activation},
            scale_firing_rates=gain_scale,
        )

        # create references to some nengo objects in the network IO objects
        self.nengo_input = converter.inputs[self.input]
        self.nengo_dense1 = converter.outputs[self.dense1]

        net = converter.net

        nengo_conv1 = converter.layers[self.conv1]
        nengo_conv2 = converter.layers[self.conv2]

        with net:
            # set our biases to non-trainable to make sure they're always 0
            net.config[nengo_conv1].trainable = False
            net.config[nengo_conv2].trainable = False

            # set up probes so that we can add the firing rates to the cost function
            self.probe_conv1 = nengo.Probe(nengo_conv1, label="probe_conv1")
            self.probe_conv2 = nengo.Probe(nengo_conv2, label="probe_conv2")

        # set our synapses
        if synapses is not None:
            for cc, conn in enumerate(net.all_connections):
                conn.synapse = synapses[cc]
                print("Setting synapse to %s on " % str(synapses[cc]), conn)
            self.nengo_dense1.synapse = synapses[3]
            print("Setting synapse to %s on output probe" % str(synapses[3]))

        sim = nengo_dl.Simulator(
            net, minibatch_size=self.minibatch_size, seed=self.seed
        )
        return sim, net

    def convert_nengodl_to_nengo(self, net, loihi=False):
        # our input node function
        def send_image_in(t):
            # if updating an image over time, like rendering from a simulator
            # then we update this object each time
            return self.image_input[0, int(t / 0.001) - 1]

        # if not using nengo dl we have to use a sim.run function
        # create a node so we can inject data into the network
        with net:
            if loihi:
                nengo_loihi.add_params(net)
                net.config[self.conv1].on_chip = False
            # create out input node
            image_input_node = nengo.Node(send_image_in, size_out=self.subpixels)
            self.input_probe = nengo.Probe(image_input_node)
            nengo.Connection(image_input_node, self.nengo_input, synapse=None)

        # overwrite the nengo_dl simulator with the nengo simulator
        if loihi:
            sim = nengo_loihi.Simulator(net, dt=self.dt)
        else:
            sim = nengo.Simulator(net, dt=self.dt)

        return sim

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

        loss = {self.nengo_dense1: tf.losses.mse}
        training_targets = {self.nengo_dense1: targets}
        if fr_loss_function:
            loss[self.probe_conv1]: fr_loss_function
            loss[self.probe_conv2]: fr_loss_function
            train_data[self.probe_conv1]: np.zeros(targets.shape)
            train_data[self.probe_conv2]: np.zeros(targets.shape)

        with sim:
            sim.compile(optimizer=tf.optimizers.RMSprop(0.00001), loss=loss)

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

                def scheduler(epoch):
                    if epoch < 2:
                        return 0.001
                    else:
                        lr = 0.0001 * tf.math.exp(0.1 * (2 - epoch))
                        print("Learning rate: ", lr)
                        return lr

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
                        predictions=np.asarray(data[vision.nengo_dense1]),
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
        "loihirelu": nengo_loihi.neurons.LoihiSpikingRectifiedLinear(),  # amplitude),
        "loihirelunoise": nengo_loihi.neurons.LoihiSpikingRectifiedLinear(
            nengo_dl_noise=nengo_loihi.neurons.LowpassRCNoise(0.001)
        ),
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

    # ------------ set test parameters
    dt = 0.001
    db_name = "rover_training_0004"
    res = [32, 128]
    n_training = 30000  # number of images to train on
    n_validation = 50  # number of validation images
    n_validation_steps = 30
    seed = 0
    # probe_neurons = True

    # using baseline of 1ms timesteps to define n_steps
    # adjust based on dt to keep sim time constant
    n_validation_steps = int(n_validation_steps * 0.001 / dt)

    minibatch_size = 50 if mode == "train" else 1
    # can't have a minibatch larger than our number of images
    minibatch_size = min(minibatch_size, n_validation)

    # plotting parameters
    num_pts = n_validation * n_validation_steps  # number of steps to plot

    # ------------ load and prepare our data
    try:
        processed_data = np.load(
            "/home/tdewolf/Downloads/validation_images_processed.npz"
        )
        validation_images = processed_data["images"]
        validation_targets = processed_data["targets"]
    except:
        # load our raw data
        validation_images, validation_targets = dl_utils.load_data(
            db_name=db_name, label="validation_0000", n_imgs=n_validation
        )

        # our saved targets are 3D but we only care about x and y
        valiation_targets = validation_targets[:, 0:2]

        # do our resizing, scaling, and flattening
        validation_images = dl_utils.preprocess_images(
            image_data=validation_images,
            show_resized_image=False,
            flatten=True,
            normalize=False,
            res=res,
        )
        np.savez_compressed(
            "/home/tdewolf/Downloads/validation_images_processed",
            images=validation_images,
            targets=validation_targets,
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
    save_folder = "/home/tdewolf/Downloads/data/%s/%s/%s" % (
        db_name,
        group_name,
        test_name,
    )
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # instantiate our keras converted network
    vision = RoverVision(
        res=res,
        minibatch_size=minibatch_size,
        dt=dt,
        seed=seed,
        # probe_neurons=probe_neurons,
    )

    # if training
    if mode == "train":

        n_training_steps = 1
        epochs = [0, 100]

        # prepare training data ------------------------------------
        try:
            processed_data = np.load(
                "/home/tdewolf/Downloads/training_images_processed.npz"
            )
            training_images = processed_data["images"]
            training_targets = processed_data["targets"]
            print("Processed training images loaded from file...")
        except:
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
                "/home/tdewolf/Downloads/training_images_processed",
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
            gain_scale=gain_scale, activation=activation, synapses=None
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
            # "probe_neurons": probe_neurons,
        }

        # Save our set up parameters
        dat_results = DataHandler(db_name=save_db_name)
        dat_results.save(
            data=test_params, save_location="%s/params" % save_folder, overwrite=True
        )

    else:

        # synapses = [None, 0.005, 0.005, None]
        # synapses = [None,] * 4
        # synapses = [.003,] * 4
        # synapses = [None, None, None, 0.005]
        # synapses = [None, 0.001, 0.001, None]
        synapses = [0.001] * 4
        # weights = save_folder + "/epoch_99"
        # weights = "data/rover_training_0004/loihirelunoise_005/200/epoch_173"
        # weights = "data/rover_training_0004/loihirelu/200/epoch_99"
        weights = "/home/tdewolf/Downloads/data/rover_training_0004/loihirelunoise/400/epoch_14"

        with tf.keras.backend.learning_phase_scope(1) if use_dl_rate else nullcontext():
            sim, net = vision.convert(
                gain_scale=gain_scale, activation=activation, synapses=synapses,
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

            # if non-batched prediction using sim.run in Nengo (not NengoDL)
            # the extend function gives you access to the input keras layer so you can inject data
            elif mode == "run":
                # net = sim.model.toplevel
                print("Trying to freeze")
                with sim:
                    sim.freeze_params(net)
                # if we want to run this in Nengo and not Nengo DL
                # have to extend our network to manually inject input images
                sim = vision.convert_nengodl_to_nengo(net)
                # NOTE that the user will need to pass their input image to
                # rover_vision.image_input if using an external rover_vision.sim.run call
                print(n_validation_steps)
                print(n_validation)

                sim_steps = dt * n_validation_steps * n_validation
                vision.image_input = validation_images
                sim.run(sim_steps)
                data = sim.data

            print("data shape: ", data[vision.nengo_dense1].shape)
            print("vvalidation targets size: ", validation_targets.shape)
            dl_utils.plot_prediction_error(
                predictions=np.asarray(data[vision.nengo_dense1]),
                target_vals=validation_targets,
                save_folder=save_folder,
                save_name="%s_prediction_error" % mode,
                num_pts=num_pts,
                show_plot=True,
            )

            # if probe_neurons:
            #     # pass in the input images so we can see the neural activity next to the input image
            #     # since we repeat n_validation_steps times, just take the first column in the 2nd dim
            #     images = validation_images.reshape(
            #         (n_validation, n_validation_steps, res[0], res[1], 3)
            #     )[:, 0, :, :, :].squeeze()
            #     # skip showing the neurons next to images
            #     images = None
            #
            #     dl_utils.plot_neuron_activity(
            #         activity=data[vision.neuron_probe],
            #         num_pts=num_pts,
            #         save_folder=save_folder,
            #         save_name="%s_activity" % mode,
            #         num_neurons_to_plot=100,
            #         images=images,
            #     )
