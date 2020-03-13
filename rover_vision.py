import nengo
import nengo_dl
import sys

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

        self.relu1 = tf.keras.layers.Activation(tf.nn.relu)(self.input)

        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=3,
            strides=1,
            use_bias=False,
            activation=tf.nn.relu,
            data_format="channels_last",
        )(self.relu1)

        flatten1 = tf.keras.layers.Flatten()(self.conv1)

        self.dense1 = tf.keras.layers.Dense(units=2, use_bias=False)(flatten1)

        self.model = tf.keras.Model(inputs=self.input, outputs=self.dense1)

    def convert(self, gain_scale, spiking=False, synapses=None, probe_neurons=False):
        """
        gain_scale: int
            scaling factor for spiking network to increase the amount of activity.
            gain gets scaled by gain_scale and neuron amplitudes get set to 1/gain_scale
        synapses: list of 4 floats or None's
            synapses set on nengo connections
        spiking: boolean
            for inference only, switches between relu and spiking relu activations
        """

        # convert model to nengo and set neuron type
        activation = nengo.SpikingRectifiedLinear if spiking else nengo.RectifiedLinear

        converter = nengo_dl.Converter(
            self.model,
            swap_activations={tf.nn.relu: activation()},
            scale_firing_rates=gain_scale,
        )

        # create references to some nengo objects in the network IO objects
        self.nengo_input = converter.inputs[self.input]
        self.nengo_dense1 = converter.outputs[self.dense1]

        net = converter.net

        nengo_relu1 = converter.layers[self.relu1]
        nengo_conv1 = converter.layers[self.conv1]

        with net:
            # set our biases to non-trainable to make sure they're always 0
            net.config[nengo_relu1].trainable = False
            net.config[nengo_conv1].trainable = False

            # set up probes so that we can add the firing rates to the cost function
            self.probe_relu1 = nengo.Probe(nengo_relu1, label='probe_relu1')
            self.probe_conv1 = nengo.Probe(nengo_conv1, label='probe_conv1')

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

        return sim

    def extend_net(self, net, validation_images):
        # our input node function
        def send_image_in(t):
            # need to be set manually
            if net.count is not None:
                # if passing in a list of images to run through one step at a time
                img = validation_images[net.count]
                net.count += 1
            else:
                # if updating an image over time, like rendering from a simulator
                # then we update this object each time
                img = validation_images
            return img

        # if not using nengo dl we have to use a sim.run function
        # create a node so we can inject data into the network
        with net:
            # create out input node
            image_input_node = nengo.Node(send_image_in, size_out=self.subpixels)
            self.input_probe = nengo.Probe(image_input_node)
            nengo.Connection(image_input_node, self.nengo_input, synapse=None)

        # overwrite the nengo_dl simulator with the nengo simulator
        sim = nengo.Simulator(net, dt=self.dt)

    def train(
        self,
        sim,
        images,
        targets,
        epochs,
        validation_images=None,
        weights_loc=None,
        save_folder=None,
        validation_targets=None,
        num_pts=None,
    ):
        """
        We loop through from epochs[0] to epochs[1]. If epochs[0]
        is not 0 then this function assumes the weights to start
        with will be at '%s/epoch_%i' % (weights_loc, epoch-1)

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
        weights_loc: string
            location where to save and load epoch weights to / from
        save_folder: string
            location where to save figures of results
        num_pts: int
            number of steps to plot from the predictions
        """
        if validation_images is not None:
            validation_images_dict = {self.nengo_input: validation_images}

        def put_in_range(x, y, weight=0, min=200, max=300):
            assert len(y.shape) == 3
            index_greater = (y > max)
            index_lesser = (y < min)
            error = tf.reduce_sum(y[index_greater] - max) + tf.reduce_sum(min - y[index_lesser])
            return weight * error

        with sim:
            sim.compile(
                optimizer=tf.optimizers.RMSprop(0.00001),
                loss={
                    self.nengo_dense1: tf.losses.mse,
                    # self.probe_relu1: put_in_range,
                    # self.probe_conv1: put_in_range,
                },
            )

            for epoch in range(epochs[0], epochs[1]):
                print("\nEPOCH %i" % epoch)
                if epoch > 0:
                    prev_params_loc = "%s/epoch_%i" % (weights_loc, epoch - 1)
                    print(
                        "loading pretrained network parameters from \n%s"
                        % prev_params_loc
                    )
                    sim.load_params(prev_params_loc)

                print("fitting data...")
                sim.fit(
                    {self.nengo_input: training_images},
                    {
                        self.nengo_dense1: targets,
                        # self.probe_relu1: np.zeros(targets.shape),
                        # self.probe_conv1: np.zeros(targets.shape),
                    }, epochs=1)

                current_params_loc = "%s/epoch_%i" % (weights_loc, epoch)
                print("saving network parameters to %s" % current_params_loc)
                sim.save_params(current_params_loc)

                if validation_images is not None:
                    print("Running Prediction in nengo-dl")
                    data = sim.predict(
                        validation_images_dict,
                        n_steps=validation_images.shape[1],
                        stateful=False,
                    )

                # TODO fix plotting NEED access to num_pts

                dl_utils.plot_prediction_error(
                    predictions=np.asarray(data[vision.nengo_dense1]),
                    target_vals=validation_targets,
                    save_folder=save_folder,
                    save_name="validation_error_epoch_%i" % epoch,
                    num_pts=num_pts,
                )


if __name__ == "__main__":

    mode = sys.argv[1]  # valid options are 'train', 'predict', and 'run'
    # rate neurons are default
    spiking = (sys.argv[2] == 'spiking') if len(sys.argv) > 2 else False
    # check if gain / amplitude scaling is set, default is 1
    gain_scale = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    # ------------ set test parameters
    dt = 0.001
    db_name = "rover_training_0004"
    res = [32, 128]
    n_training = 30000  # number of images to train on
    n_validation = 10  # number of validation images
    n_validation_steps = 30
    seed = 0
    # probe_neurons = True

    # using baseline of 1ms timesteps to define n_steps
    # adjust based on dt to keep sim time constant
    n_validation_steps = int(n_validation_steps * 0.001 / dt)

    minibatch_size = 300 if mode == "train" else 1
    # can't have a minibatch larger than our number of images
    minibatch_size = min(minibatch_size, n_validation)

    # plotting parameters
    num_imgs_to_show = 10  # number of image predictions to show in the results plot
    num_imgs_to_show = min(num_imgs_to_show, n_validation)
    num_pts = num_imgs_to_show * n_validation_steps  # number of steps to plot

    # ------------ load and prepare our data
    try:
        processed_data = np.load('/home/tdewolf/Downloads/validation_images_processed.npz')
        validation_images = processed_data['images']
        validation_targets = processed_data['targets']
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
            '/home/tdewolf/Downloads/validation_images_processed',
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
    group_name = "test"  # helps to define what this group of tests is doing
    test_name = "test_1"
    # for saving figures and weights
    save_folder = "data/%s/%s/%s" % (db_name, group_name, test_name)
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
        epochs = [70, 80]

        # prepare training data ------------------------------------
        try:
            processed_data = np.load('/home/tdewolf/Downloads/training_images_processed.npz')
            training_images = processed_data['images']
            training_targets = processed_data['targets']
            print('Processed training images loaded from file...')
        except:
            print('Processed training images not found, generating...')
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
                '/home/tdewolf/Downloads/training_images_processed',
                images=training_images,
                targets=training_targets
            )

        # repeat and batch our data
        training_images = dl_utils.repeat_data(
            data=training_images, batch_data=True, n_steps=n_training_steps
        )
        training_targets = dl_utils.repeat_data(
            data=training_targets, batch_data=True, n_steps=n_training_steps
        )

        sim = vision.convert(gain_scale=1, spiking=False, synapses=None)
        vision.train(
            sim=sim,
            images=training_images,
            targets=training_targets,
            epochs=epochs,
            validation_images=validation_images,
            weights_loc=save_folder,
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
        \n- 1 conv layers 1 dense
        \n- adding relu layer between input and conv1 due to loihi comm restriction
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

        synapses = None  # [None, None, None, 0.05]
        weights = "data/rover_training_0004/test/test_1/epoch_73"

        sim = vision.convert(gain_scale=gain_scale, spiking=spiking, synapses=synapses)

        # load a specific set of weights
        if weights is not None:
            print("Received specific weights to use: ", weights)
            sim.load_params(weights)

        # if batched prediction
        if mode == "predict":
            data = sim.predict(
                {vision.nengo_input: validation_images},
                n_steps=validation_images.shape[1],
                # stateful=False
            )

        # if non-batched prediction using sim.run in Nengo (not NengoDL)
        # the extend function gives you access to the input keras layer so you can inject data
        elif mode == "run":
            raise Exception('Not properly implemented yet')
            # NOTE that the user will need to pass their input image to
            # rover_vision.net.validation_images if using an external rover_vision.sim.run call

            # we have to extend our network to be able to manually inject input images
            # and to switch from the nengo_dl to the nengo simulator
            vision.extend_net(net, validation_images)

            sim_steps = dt * n_validation_steps * n_validation
            sim.run(sim_steps)
            data = sim.data

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
