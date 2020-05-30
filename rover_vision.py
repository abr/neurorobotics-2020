"""
To run the demo with Nengo running on cpu:
    python nengo_rover.py cpu

To run the demo with Nengo on loihi
    NXSDKHOST=loihighrd python nengo_rover.py
"""
import tensorflow as tf
import numpy as np
import sys

import nengo
import nengo_dl
import nengo_loihi
from nengo_loihi.neurons import (
    LoihiSpikingRectifiedLinear,
    loihi_spikingrectifiedlinear_rates,
)

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
        self.res = [32, 128]  # size of input image in pixels
        self.subpixels = self.res[0] * self.res[1] * 3
        self.minibatch_size = minibatch_size
        self.image_input = np.zeros(self.subpixels)

        # Define our keras network
        self.input = tf.keras.Input(shape=(self.res[0], self.res[1], 3))

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

    def convert(self, converter_params, add_probes=True):
        """
        """
        converter = nengo_dl.Converter(self.model, **converter_params)

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
                    self.output, label="probe_dense", synapse=0.005
                )

        sim = nengo_dl.Simulator(
            net, minibatch_size=self.minibatch_size, seed=self.seed
        )
        return sim, net


if __name__ == "__main__":
    """
    """

    mode = "predict"  # should be ["predict"|"run"]
    if mode == "run":
        activation = LoihiSpikingRectifiedLinear()  # can be any Nengo neuron type
    elif mode == "predict":
        activation = LoihiRectifiedLinear()
    scale_firing_rates = 400
    weights = "epoch_41"

    images, targets = dl_utils.load_data(
        db_name="abr_analyze", label="driving_0047", n_imgs=10000, step_size=20,
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

    # repeat and batch our data
    images = dl_utils.repeat_data(images, batch_data=False, n_steps=1)
    targets = dl_utils.repeat_data(targets, batch_data=False, n_steps=1)

    # instantiate our keras converted network
    dt = 0.001
    vision = RoverVision(minibatch_size=1, dt=dt, seed=np.random.randint(1e5))
    # convert from Keras to Nengo
    sim, net = vision.convert(
        converter_params={
            "swap_activations": {tf.nn.relu: activation},
            "scale_firing_rates": scale_firing_rates,
        }
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
            return vision.image_input[int(t / 0.001) - 1]

        with sim:
            sim.freeze_params(net)
            vision.input.output = send_image_in

        with net:
            nengo_loihi.add_params(net)
            net.config[vision.conv0.ensemble].on_chip = False

        # dimensions should be (n_timesteps, image.flatten())
        vision.image_input = images[0]

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
