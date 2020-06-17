"""
The vision system for the rover. Takes in a 360 view of the Mujoco environment and
locates a red target. The network then outputs the target's (x, y) location relative
to the cameras.

To test the network on validation data, run

    python generate_training_data.py

to generate the validation test set, and then run

    python rover_vision.py

To run the network with your own trained weights, instead of the default reference
weights, after you generate the training data, run

    python train_rover_vision.py

and then

    python rover_vision.py data/weights
"""
import tensorflow as tf
import numpy as np
import os
import sys

import nengo
import nengo_dl
import nengo_loihi
from nengo_loihi.neurons import (
    LoihiSpikingRectifiedLinear,
    loihi_spikingrectifiedlinear_rates,
)

import os
import sys

sys.path.append("../")
import dl_utils


class LoihiRectifiedLinear(nengo.RectifiedLinear):
    """Non-spiking version of the LoihiSpikingRectifiedLinear neuron

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.probeable = ("rates",)

    def step_math(self, dt, J, output):
        """Implement the LoihiRectifiedLinear nonlinearity."""
        output[...] = loihi_spikingrectifiedlinear_rates(
            self, x=J, gain=1, bias=0, dt=dt
        ).squeeze()


class RoverVision:
    def __init__(self, minibatch_size=1, dt=0.001, seed=0):
        """
        Parameters
        ----------
        minibatch_size: int
            specifies how to break up the training batches.
            NOTE: must be <= the batch_size
        dt: float
            simulation time step
        seed: int
            random number generator seed
        """

        self.dt = dt
        self.seed = seed
        self.resolution = np.array([32, 32])
        # input size is larger than resolution because we receive input from 4 cameras
        self.input_size = np.array([self.resolution[0], self.resolution[1] * 4])
        self.subpixels = self.input_size[0] * self.input_size[1] * 3
        self.minibatch_size = minibatch_size
        self.image_input = np.zeros(self.subpixels)

        # Define our keras network
        self.input = tf.keras.Input(shape=(self.input_size[0], self.input_size[1], 3))

        self.conv0 = tf.keras.layers.Conv2D(
            filters=3,
            kernel_size=1,
            strides=1,
            use_bias=False,
            activation=tf.nn.relu,
            data_format="channels_last",
        )(self.input)

        self.conv1_layer = tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=10,
            strides=5,
            use_bias=False,
            activation=tf.nn.relu,
            data_format="channels_last",
        )
        self.conv1 = self.conv1_layer(self.conv0)

        flatten = tf.keras.layers.Flatten()(self.conv1)
        self.dense0_layer = tf.keras.layers.Dense(
            units=200, use_bias=False, activation=tf.nn.relu
        )
        self.dense0 = self.dense0_layer(flatten)
        self.dense1_layer = tf.keras.layers.Dense(
            units=100, use_bias=False, activation=tf.nn.relu
        )
        self.dense1 = self.dense1_layer(self.dense0)
        self.dense = tf.keras.layers.Dense(units=2, use_bias=False)(self.dense1)

        self.model = tf.keras.Model(inputs=self.input, outputs=self.dense)

    def convert(self, add_probes=True, synapse=None, **kwargs):
        """ Run the NengoDL Converter on the above Keras net

        add_probes : bool, optional (Default: True)
            if False, no probes are added to the model, reduces simulation overhead
        """
        converter = nengo_dl.Converter(self.model, **kwargs)

        # create references to some nengo objects in the network IO objects
        self.nengo_input = converter.inputs[self.input]
        self.nengo_dense = converter.outputs[self.dense]

        net = converter.net

        self.input = converter.layers[self.input]
        self.conv0 = converter.layers[self.conv0]
        self.conv1 = converter.layers[self.conv1]
        self.output = converter.layers[self.dense]

        with net:
            # set our biases to non-trainable to make sure they're always 0
            net.config[self.conv0].trainable = False
            net.config[self.conv1].trainable = False

            if add_probes:
                # set up probes so to add the firing rates to the cost function
                self.probe_conv0 = nengo.Probe(self.conv0, label="probe_conv0")
                self.probe_conv1 = nengo.Probe(self.conv1, label="probe_conv1")
                self.probe_dense = nengo.Probe(
                    self.output, label="probe_dense", synapse=synapse
                )

        sim = nengo_dl.Simulator(
            net, minibatch_size=self.minibatch_size, seed=self.seed
        )
        return sim, net


if __name__ == "__main__":
    current_dir = os.path.abspath(".")
    db_dir = "data"
    mode = "predict"  # should be ["predict"|"run"]
    if mode == "run":
        activation = LoihiSpikingRectifiedLinear()  # can be any Nengo neuron type
        n_steps = 100  # how many time steps to present the input for
        synapse = 0.005
    elif mode == "predict":
        activation = LoihiRectifiedLinear()
        n_steps = 2
        synapse = None

    scale_firing_rates = 400
    weights = sys.argv[1] if len(sys.argv) > 1 else "data/reference_weights"
    if weights[-4:] == ".npz":
        weights = weights[:-4]

    images, targets = dl_utils.load_data(
        db_dir=db_dir, db_name="rover", label="validation",
    )

    # saved targets are 3D but we only care about x and y
    # also want to normalize the targets to the -1:1 range so dividing by pi
    targets = targets[:, 0:2] / np.pi

    # do our resizing, scaling, and flattening
    images = dl_utils.preprocess_images(
        image_data=images,
        show_resized_image=False,
        flatten=True,
        normalize=False,
        res=[32, 128],
    )
    # choose random subset of 100 images for testing
    indices = np.random.permutation(np.arange(len(images)))[:100]
    images = images[indices]
    targets = targets[indices]

    # repeat and batch our data
    images = dl_utils.repeat_data(images, batch_data=False, n_steps=n_steps)
    targets = dl_utils.repeat_data(targets, batch_data=False, n_steps=n_steps)

    # instantiate our keras converted network
    dt = 0.001
    vision = RoverVision(minibatch_size=1, dt=dt, seed=np.random.randint(1e5))
    # convert from Keras to Nengo
    sim, net = vision.convert(
        synapse=synapse,
        swap_activations={tf.nn.relu: activation},
        scale_firing_rates=scale_firing_rates,
    )
    if weights is not None:
        sim.load_params(weights)
        print("Loaded in weights from: ", weights)

    if mode == "predict":
        # this mode uses the NengoDL simulator, and input images are presented
        # as batched input. for simulating spiking neurons, use 'run' mode
        data = sim.predict(
            {vision.input: images}, n_steps=images.shape[1], stateful=False,
        )

    elif mode == "run":
        # this mode uses the Nengo or NengoLoihi simulators, and input images are
        # presented sequentially to the network, appropriate for spiking neurons
        images = images.squeeze()

        def send_image_in(t):
            # dimensions should be (n_timesteps, image.flatten())
            return images[int(t / 0.001) - 1]

        with sim:
            sim.freeze_params(net)
            vision.input.output = send_image_in

        with net:
            nengo_loihi.add_params(net)
            net.config[vision.conv0.ensemble].on_chip = False

        sim = nengo_loihi.Simulator(net, dt=dt)
        with sim:
            sim.run(dt * images.shape[0])
        data = sim.data

    dl_utils.plot_prediction_error(
        predictions=np.asarray(data[vision.probe_dense]),
        target_vals=targets,
        save_name="%s_prediction_error" % mode,
        show_plot=True,
    )
