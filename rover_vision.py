import nengo
import nengo_dl
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


# =================== Tensorflow settings to avoid OOM errors =================
warnings.simplefilter("ignore")
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
# =============================================================================

class RoverVision():
    def __init__(
            self, res, minibatch_size, spiking, dt,
            gain, gain_scale, seed, synapses, probe_neurons=False):

        self.dt = dt
        self.seed = seed
        self.subpixels = res[0] * res[1] * 3
        self.spiking = spiking
        self.minibatch_size = minibatch_size

        # Define our keras network
        keras_image_input = tf.keras.Input(shape=(res[0], res[1], 3), batch_size=self.minibatch_size)

        relu_layer = tf.keras.layers.Activation(tf.nn.relu)
        relu_out = relu_layer(keras_image_input)

        conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=3,
            strides=1,
            use_bias=False,
            activation=tf.nn.relu,
            data_format="channels_last"
            )
        conv1_out = conv1(relu_out)

        flatten = tf.keras.layers.Flatten()(conv1_out)

        keras_dense = tf.keras.layers.Dense(
            units=2,
            use_bias=False
            )
        keras_dense_output = keras_dense(flatten)

        model = tf.keras.Model(inputs=keras_image_input, outputs=keras_dense_output)

        # convert model to nengo and set neuron type
        if not self.spiking:
            converter = nengo_dl.Converter(
                model,
                #TODO: make sure swapping the activation here doesn't affect results
                swap_activations={
                    tf.nn.relu: nengo.RectifiedLinear(amplitude=1/gain_scale)
                    }
                )
        else:
            converter = nengo_dl.Converter(
                model,
                swap_activations={
                    tf.nn.relu: nengo.SpikingRectifiedLinear(amplitude=1/gain_scale)
                    }
            )

        # create references to some nengo objects in the network
        # IO objects
        self.vision_input = converter.inputs[keras_image_input]
        self.dense_output = converter.outputs[keras_dense_output]

        # ensemble objects
        nengo_conv = converter.layer_map[conv1][0][0]
        nengo_relu = converter.layer_map[relu_layer][0][0]

        # adjust ensemble gains
        nengo_conv.ensemble.gain = nengo.dists.Choice([gain * gain_scale])
        nengo_relu.ensemble.gain = nengo.dists.Choice([gain * gain_scale])

        self.net = converter.net

        with self.net:
            #NOTE to avoid OOM errors, only have one neuron probe set at a time
            if probe_neurons:
                self.neuron_probe = nengo.Probe(nengo_conv.ensemble.neurons) #[:20000])
                # self.neuron_probe = nengo.Probe(nengo_relu.ensemble.neurons)

            # set our biases to non-trainable
            self.net.config[nengo_conv.ensemble.neurons].trainable = False
            self.net.config[nengo_relu.ensemble.neurons].trainable = False

        # set our synapses
        for cc, conn in enumerate(self.net.all_connections):
            conn.synapse = synapses[cc]
            print('Setting synapse to %s on ' % str(synapses[cc]), conn)
        self.dense_output.synapse = synapses[3]
        print('Setting synapse to %s on output probe' % str(synapses[3]))

        self.sim = nengo_dl.Simulator(self.net, minibatch_size=self.minibatch_size, seed=self.seed)


    def extend_net(self):
        # our input node function
        def send_image_in(t):
            #TODO catch exception if validation images don't exist and warn user they
            # need to be set manually
            if self.net.count is not None:
                # if passing in a list of images to run through one step at a time
                img = self.net.validation_images[self.net.count]
                self.net.count += 1
            else:
                # if updating an image over time, like rendering from a simulator
                # then we update this object each time
                img = self.net.validation_images
            return img

        # if not using nengo dl we have to use a sim.run function
        # create a node so we can inject data into the network
        with self.net:
            # create out input node
            image_input_node = nengo.Node(send_image_in, size_out=self.subpixels)
            self.input_probe = nengo.Probe(image_input_node)
            nengo.Connection(image_input_node, self.vision_input, synapse=None)

        # overwrite the nengo_dl simulator with the nengo simulator
        self.sim = nengo.Simulator(self.net, dt=self.dt)


    def train(
            self, images, targets, epochs, validation_images=None,
            n_validation_steps=10, weights_loc=None, save_folder=None,
            validation_targets=None, num_pts=None):
        """
        We loop through from epochs[0] to epochs[1], so if epochs[0]
        is not 0 then this function assumes the weights to start
        with will be at '%s/epoch_%i' % (weights_loc, epoch-1)

        In other words it is assumed that if our starting epoch is not
        0 then we should be loading weights from the previous epoch
        """
        if validation_images is not None:
            validation_images_dict = {
                self.vision_input: validation_images
            }
        training_images_dict = {
                self.vision_input: images
            }

        with self.sim:
            self.sim.compile(
                optimizer=tf.optimizers.RMSprop(0.001),
                loss={self.dense_output: tf.losses.mse})

            for epoch in range(epochs[0], epochs[1]):
                print('\nEPOCH %i' % epoch)
                if epoch>0:
                    prev_params_loc = '%s/epoch_%i' % (weights_loc, epoch-1)
                    print('loading pretrained network parameters from \n%s' % prev_params_loc)
                    self.sim.load_params(prev_params_loc)

                print('fitting data...')
                self.sim.fit(training_images_dict, targets, epochs=1)

                # save parameters back into net
                self.sim.freeze_params(self.net)

                current_params_loc = '%s/epoch_%i' % (weights_loc, epoch)
                print('saving network parameters to %s' % current_params_loc)
                self.sim.save_params(current_params_loc)

                if validation_images is not None:
                    print('Running Prediction in nengo-dl')
                    data = self.sim.predict(
                        validation_images_dict,
                        n_steps=validation_images.shape[1],
                        stateful=False)

                #TODO fix plotting NEED access to num_pts

                dl_utils.plot_prediction_error(
                        predictions=np.asarray(data[vision.dense_output]),
                        target_vals=validation_targets,
                        save_folder=save_folder,
                        save_name='validation_error_epoch_%i' % epoch,
                        num_pts=num_pts)


    def predict(self, images, n_validation_steps, weights=None):
        print('Running prediction for %i steps per image,  with images shape' % n_validation_steps, images.shape)
        # load a specific set of weights
        if weights is not None:
            with nengo_dl.Simulator(
                    self.net, minibatch_size=self.minibatch_size, seed=self.seed) as sim:
                print('Received specific weights to use: ', weights)
                sim.load_params(weights)
                sim.freeze_params(self.net)

            # update our class sim object with the new net with weights
            self.sim = nengo_dl.Simulator(self.net, minibatch_size=self.minibatch_size, seed=self.seed)

        validation_images_dict = {
            self.vision_input: images
        }
        with self.sim:
            data = self.sim.predict(validation_images_dict, n_steps=images.shape[1], stateful=False)

        return data


    def run(self, images, sim_steps, weights=None):
        # load a specific set of weights
        if weights is not None:
            with nengo_dl.Simulator(
                    self.net, minibatch_size=self.minibatch_size, seed=self.seed) as sim:
                print('Received specific weights to use: ', weights)
                sim.load_params(weights)
                sim.freeze_params(self.net)
                # since we've changed our net object we need to update our nengo sim
            self.sim = nengo.Simulator(self.net, dt=self.dt)

        # NOTE that the user will need to pass their input image to
        # rover_vision.net.validation_images if using an external rover_vision.sim.run call
        self.net.count = 0
        self.net.validation_images = images

        # we have to extend our network to be able to manually inject input images
        # and to switch from the nengo_dl to the nengo simulator
        self.extend_net()

        with self.sim:
            print('Starting Nengo sim...')
            # match our sim length for nengo-dl
            self.sim.run(sim_steps)

        return self.sim.data

if __name__ == '__main__':
    # ------------ set test parameters
    # mode = 'predict'
    # mode = 'train'
    mode = 'run'
    spiking = True
    gain = 1
    dt = 0.001
    db_name = 'rover_training_0004'
    res = [32, 128]
    n_training = 30000 # number of images to train on
    n_validation = 10 # number of validation images
    seed = 0
    probe_neurons = True

    if spiking:
        # typically for validation with spiking neurons
        n_validation_steps = 300
        n_training_steps = None # should not train with spikes, this will throw an error
        gain_scale = 1000
        synapses = [None, None, None, 0.05]
        weights = 'data/rover_training_0004/biases_non_trainable/filters_32/biases_non_trainable_23'
        epochs = None
    else:
        # typically for training with rate neurons
        n_validation_steps = 1
        n_training_steps = 1
        gain_scale = 1
        synapses = [None, None, None, None]
        weights = None
        epochs = [0, 10]

    # using baseline of 1ms timesteps to define n_steps
    n_validation_steps = int(n_validation_steps * 0.001/dt) # adjust based on dt to keep sim time constant

    if mode == 'train':
        minibatch_size = 100
        # we can't run our inference after training with fewer images than our minibatch_size
        n_validation = minibatch_size
    else:
        minibatch_size = 1
    minibatch_size = min(minibatch_size, n_validation) # can't have a minibatch larger than our number of images

    # plotting parameters
    custom_save_tag = '' # if you want to prepend some tag to saved images
    num_imgs_to_show = 10 # number of image predictions to show in the results plot
    num_imgs_to_show = min(num_imgs_to_show, n_validation)
    num_pts = num_imgs_to_show*n_validation_steps # number of steps to plot

    # ------------ load and prepare our data
    if mode == 'train':
        # load our raw data
        training_images, training_targets = dl_utils.load_data(
            db_name=db_name, label='training_0000', n_imgs=n_training)
        # our saved targets are 3D but we only care about x and y
        training_targets = training_targets[:, 0:2]

        # do our resizing, scaling, and flattening
        training_images = dl_utils.preprocess_images(
            image_data=training_images,
            show_resized_image=False,
            flatten=True,
            normalize=True,
            res=res)

        # repeat and batch our data
        training_images = dl_utils.repeat_data(
            data=training_images,
            batch_data=True,
            n_steps=n_training_steps)
        training_targets = dl_utils.repeat_data(
            data=training_targets,
            batch_data=True,
            n_steps=n_training_steps)

    # load our raw data
    validation_images, validation_targets = dl_utils.load_data(
        db_name=db_name, label='validation_0000', n_imgs=n_validation)

    # our saved targets are 3D but we only care about x and y
    validation_targets = validation_targets[:, 0:2]

    # do our resizing, scaling, and flattening
    validation_images = dl_utils.preprocess_images(
            image_data=validation_images,
            show_resized_image=False,
            flatten=True,
            normalize=True,
            res=res)

    # we have to set the minibatch_size when instantiating the network so if we want
    # to run sim.predict after each training session to track progress, we need to have
    # a consistent minibatch_size between training and predicting
    # this is also using rate neurons during training so it doesn't matter if the
    # prediction is batched (won't have the output zeroed between minibatches since
    # we aren't using spiking neurons)
    if mode == 'train':
        batch_data = True
    else:
        batch_data = False

    # repeat and batch our data
    validation_images = dl_utils.repeat_data(
        data=validation_images,
        batch_data=batch_data,
        n_steps=n_validation_steps)
    validation_targets = dl_utils.repeat_data(
        data=validation_targets,
        batch_data=batch_data,
        n_steps=n_validation_steps)


    # ------------ set up data tracking
    save_db_name = '%s_results' % db_name # for saving results to
    group_name = 'test' # helps to define what this group of tests is doing
    test_name = 'test_1'
    # for saving figures and weights
    save_folder = 'data/%s/%s/%s' % (db_name, group_name, test_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Notes to save with this test
    notes = (
    '''
    \n- setting biases to non-trainable
    \n- 32 filters
    \n- bias off on all layer, loihi restriction
    \n- default kernel initializers
    \n- 1 conv layers 1 dense
    \n- adding relu layer between input and conv1 due to loihi comm restriction
    \n- changing image scaling from -1 to 1, to 0 to 1 due to loihi comm restriction
    '''
    )

    test_params = {
        'spiking': spiking,
        'gain': gain,
        'dt': dt,
        'n_validation_steps': n_validation_steps,
        'n_training_steps': n_training_steps,
        'gain_scale': gain_scale,
        'synapses': synapses,
        'starting_weights_loc': weights,
        'epochs': epochs,
        'n_training': n_training,
        'n_validation': n_validation,
        'res': res,
        'minibatch_size': minibatch_size,
        'seed': seed,
        'notes': notes,
        'probe_neurons': probe_neurons,
        }

    # Save our set up parameters
    dat_results = DataHandler(db_name=save_db_name)
    dat_results.save(data=test_params, save_location='%s/params'%save_folder, overwrite=True)

    # instantiate our keras converted network
    vision = RoverVision(
        res=res, minibatch_size=minibatch_size, spiking=spiking, dt=dt,
        gain=gain, gain_scale=gain_scale, seed=seed, synapses=synapses,
        probe_neurons=probe_neurons)

    # if training
    if mode == 'train':
        vision.train(
            images=training_images,
            targets=training_targets,
            epochs=epochs,
            validation_images=validation_images,
            n_validation_steps=n_validation_steps,
            weights_loc=save_folder,
            save_folder=save_folder,
            num_pts=num_pts,
            validation_targets=validation_targets)

    # if batched prediction
    if mode == 'predict':
        data = vision.predict(images=validation_images, n_validation_steps=n_validation_steps, weights=weights)

        dl_utils.plot_prediction_error(
                predictions=np.asarray(data[vision.dense_output]), target_vals=validation_targets,
                save_folder=save_folder, save_name='%s_prediction_error'%mode, num_pts=num_pts)

    # if non-batched prediction using sim.run
    # the extend function gives you access to the input keras layer so you can inject data
    if mode == 'run':
        print('VAL SHAPE: ', validation_images.shape)
        sim_steps = dt*n_validation_steps*n_validation
        data = vision.run(
            images=np.asarray(validation_images).squeeze(), sim_steps=sim_steps, weights=weights)

        dl_utils.plot_prediction_error(
                predictions=np.asarray(data[vision.dense_output]), target_vals=validation_targets,
                save_folder=save_folder, save_name='%s_prediction_error'%mode, num_pts=num_pts)

    if probe_neurons:
        # pass in the input images so we can see the neural activity next to the input image
        # since we repeat n_validation_steps times, just take the first column in the 2nd dim
        images = validation_images.reshape(
                (n_validation, n_validation_steps, res[0], res[1], 3))[:, 0, :, :, :].squeeze()
        # skip showing the neurons next to images
        images = None

        dl_utils.plot_neuron_activity(
                activity=data[vision.neuron_probe],
                num_pts=num_pts,
                save_folder=save_folder,
                save_name='%s_activity'%mode,
                num_neurons_to_plot=100,
                images=images)
