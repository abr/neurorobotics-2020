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
from nengo.utils.matplotlib import rasterplot

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
        # print(np.array(local_error).shape)
        angle = math.atan2(local_error[count][1], local_error[count][0])

        # scaled_target_data.append(angle)

        # if debug is on, show plots of the scaled image, predicted target
        # location, and target angle output
        if debug and count%10 == 0:
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
        data = dat.load(parameters=['rgb', 'target'], save_location='%s/data/%04d' % (label, nn))
        # for ii in range(n_steps):
        training_images.append(data['rgb'])
        training_targets.append(data['target'])

    # scale image resolution
    # training_images, training_targets = preprocess_data(
    training_images = preprocess_data(
        image_data=training_images, debug=debug,
        res=res, show_resized_image=show_resized_image,
        flatten=flatten, local_error=training_targets)
    training_targets = np.asarray(training_targets)[:, 0:2]

    print('images pre_tile: ', training_images.shape)
    print('targets pre_tile: ', training_targets.shape)
    training_images = np.tile(training_images[:, None, :], (1, n_steps, 1))
    # training_images = np.vstack(training_images)[None, :, :]
    training_targets = np.tile(training_targets[:, None, :], (1, n_steps, 1))
    # training_targets = np.vstack(training_targets)[None, :, :]
    print('images post_tile: ', training_images.shape)
    print('targets post_tile: ', training_targets.shape)

    return training_images, training_targets

# ------------------------------ DEFINE PARAMETERS
db_name = 'rover_training_0004'
# db_name = 'circular_targets'
save_db_name = '%s_results' % db_name

# params_file = 'minimizing_filters'
params_file = 'biases_non_trainable'
save_name = 'filters_32'
save_folder = 'data/%s/%s/%s' % (db_name, params_file, save_name)

# show plots for processed data
debug = False
# show the images before and after scaling
show_resized_image = debug

#==================
spiking = True
load_net_params = True # false if pretrained weights not None
train_on_data = False # false if spiking
# with biases trainable
# pretrained_weights = 'data/rover_training_0004/spiking_conversion/filters_32/spiking_conversion_26'
pretrained_weights = 'data/rover_training_0004/biases_non_trainable/filters_32/biases_non_trainable_23'
# pretrained_weights = None
#==================


if pretrained_weights is not None:
    # don't use auto weight loading, use the specified file
    print('Custom weights passed in, turning off auto weight loading...')
    load_net_params = False

if spiking:
    # backend = 'nengo_loihi'
    backend = 'nengo_dl'
    # backend = 'nengo'
    train_on_data = False
    # n_steps = 500
    n_steps = 300
    gain_scale = 100
    synapses = [None, None, None, 0.05]
else:
    # backend = 'nengo_dl'
    backend = 'nengo'
    n_steps = 1
    gain_scale = 1
    synapses = [None, None, None, None]

print('using %s backend' % backend)
# save net params only if training
save_net_params = train_on_data

dt = 0.001
# using baseline of 1ms timesteps to define n_steps
n_steps = int(n_steps * 0.001/dt)
custom_save_tag = 'gain_scale_%i_' % gain_scale
# training batch size
epochs = [23, 24]
n_training = 30000
minibatch_size = 100
# validation batch size
n_validation = 10
num_imgs_to_show = 10
minibatch_size = min(minibatch_size, n_validation)
output_dims = 2
gain = 1
bias = 0
# resolution to scale images to
res=[32, 128]
pixels = res[0] * res[1]
subpixels = pixels*3
seed = 0
np.random.seed(seed)
flatten = True

# if backend is not 'nengo_loihi':
#     pretrained_weights = None

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

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
        res=res, label='training_0000', n_imgs=n_training,
        debug=debug, show_resized_image=show_resized_image, db_name=db_name,
        flatten=flatten, n_steps=n_steps)

valid_label = 'validation_0000'
# valid_label = 'training_0000'

validation_images, validation_targets = load_data(
    res=res, label=valid_label, n_imgs=n_validation,
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
    filters=32,
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
    # activation=tf.nn.relu
    )
keras_output_probe = keras_dense(flatten)

model = tf.keras.Model(inputs=image_input, outputs=keras_output_probe)

if not spiking:
    converter = nengo_dl.Converter(
        model,
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

# IO objects
vision_input = converter.inputs[image_input]
output_probe = converter.outputs[keras_output_probe]
output_layer = converter.layer_map[keras_dense][0][0]

# ensemble objects
nengo_conv = converter.layer_map[conv1][0][0]
nengo_relu = converter.layer_map[relu_layer][0][0]

# adjust gains, synapses, and bias trainable parameters
nengo_conv.ensemble.gain = nengo.dists.Choice([gain * gain_scale])
nengo_relu.ensemble.gain = nengo.dists.Choice([gain * gain_scale])

net = converter.net
with net:
    conv_neuron_probe = nengo.Probe(nengo_conv.ensemble.neurons) #[:20000])
    # relu_neuron_probe = nengo.Probe(nengo_relu.ensemble.neurons)
    net.config[nengo_conv.ensemble.neurons].trainable = False
    net.config[nengo_relu.ensemble.neurons].trainable = False

output_probe.synapse = synapses[3]
print('Setting synapse to %s on output probe' % str(synapses[3]))
for cc, conn in enumerate(net.all_connections):
    conn.synapse = synapses[cc]
    print('Setting synapse to %s on ' % str(synapses[cc]), conn)

if backend is not 'nengo_dl':
    net.count = 0
    shape = validation_images.shape
    net.validation_images = validation_images.reshape(shape[0]*shape[1], shape[2])
    print('Image input node has image array of shape ', net.validation_images.shape)
    print('Running loihi sim with images shape: ', net.validation_images.shape)
    def send_image_in(t):
        img = net.validation_images[net.count]
        # print('count: ', net.count)
        # print('img shape: ', img.shape)
        net.count += 1
        return img

    with net:
        image_input_node = nengo.Node(send_image_in, size_out=subpixels)
        nengo.Connection(image_input_node, vision_input, synapse=None)
        # overwrite the output probe for non batched runs
        # prediction_probe = nengo.Probe(output_layer, synapse=None)
        output_probe = nengo.Probe(output_layer, synapse=synapses[3])
        n_conv_neurons =  nengo_conv.ensemble.n_neurons
        print('Convolutional neurons: ', n_conv_neurons)
        if backend == 'nengo_loihi':
            nengo_loihi.add_params(net)
            for cc, conn in enumerate(net.all_connections):
                if cc == 1:
                    print('setting pop_type 16 on conn: ', conn)
                    net.config[conn].pop_type = 16

            net.config[nengo_conv.ensemble].block_shape = nengo_loihi.BlockShape(
                (800,), (n_conv_neurons,))
            # net.config[image_input_node].on_chip = False


if pretrained_weights is not None:
    with nengo_dl.Simulator(net, minibatch_size=minibatch_size, seed=seed) as sim:
        print('Received specific weights to use: ', pretrained_weights)
        sim.load_params(pretrained_weights)
        sim.freeze_params(net)

if backend == 'nengo_loihi':
    sim = nengo_loihi.Simulator(net, target='sim', dt=dt)
elif backend == 'nengo_dl':
    sim = nengo_dl.Simulator(net, minibatch_size=minibatch_size, seed=seed)
elif backend == 'nengo':
    sim = nengo.Simulator(net, dt=dt)


with sim:
    def plot_predict(
            prediction_data, target_vals,
            save_folder='', save_name='prediction_results', num_pts=100):

        predictions = prediction_data[output_probe]
        predictions = np.asarray(predictions).squeeze()
        print('plotting %i timesteps' % num_pts)
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
        plt.subplot(221)
        plt.title('X: %.3f' % x_err)
        # x = np.linspace(0.5, 3.5, len(predictions[-num_pts:]))
        x = np.linspace(0, len(predictions[-num_pts:]), len(predictions[-num_pts:]))
        print('x: ', x)
        print('y: ', predictions[-num_pts:, 0])
        plt.plot(x, predictions[-num_pts:, 0], label='predictions', color='k', linestyle='--')
        plt.plot(x, target_vals[-num_pts:, 0], label='target', color='r')
        plt.legend()
        plt.subplot(222)
        plt.title('Y: %.3f' % y_err)
        plt.plot(x, predictions[-num_pts:, 1], label='predictions', color='k', linestyle='--')
        plt.plot(x, target_vals[-num_pts:, 1], label='target', color='r')
        plt.legend()
        plt.subplot(223)
        plt.title('Target XY')
        plt.xlim([-3, 3])
        plt.ylim([-3, 3])
        plt.scatter(0, 0, color='r', label='rover', s=2)
        plt.scatter(
            target_vals[-num_pts:, 0],
            target_vals[-num_pts:, 1], label='target', s=1)
        plt.legend()
        a4 = plt.subplot(224)
        activity = np.array(prediction_data[conv_neuron_probe])
        # activity = np.array(prediction_data[relu_neuron_probe])
        shape = activity.shape
        print('activity shape: ', shape)
        if activity.ndim == 3:
            activity = activity.reshape(shape[0]*shape[1], shape[2])
        print('reshaped activity: ', activity.shape)
        activity_over_time = np.sum(activity, axis=0)
        print('if I did this right %i should match' % (shape[-1]), activity_over_time.shape)
        zero_count = 0
        non_zero_neurons = []
        for index, neuron in enumerate(activity_over_time):
            if neuron == 0:
                zero_count += 1
            else:
                non_zero_neurons.append(index)
        non_zero_activity = [activity[:, i] for i in non_zero_neurons]
        # keep time/batch_size as our first dimension
        non_zero_activity = np.asarray(non_zero_activity).T
        print('%i neurons never fire' % zero_count)
        print('%i neurons fire' % np.array(non_zero_activity).shape[1])
        print('non zero activity shape: ', non_zero_activity.shape)
        neurons_to_plot = 100
        neuron_step = int(non_zero_activity.shape[1]/neurons_to_plot)
        print('choosing every %i neuron' % neuron_step)
        # plt.plot(activity[-num_pts:, :neurons_to_plot])
        # plt.plot(non_zero_activity[-num_pts:, ::neuron_step])
        activity_to_plot =  non_zero_activity[-num_pts:, ::neuron_step]
        print(activity_to_plot)
        t = np.arange(0, num_pts, 1)
        rasterplot(t, activity_to_plot)
        plt.tight_layout()
        plt.savefig('%s/%s.png' % (save_folder, save_name))
        print('saving figure to %s/%s' % (save_folder, save_name))
        # plt.show()
        plt.close()
        fig = None

        show_hist = True
        if show_hist:
            plt.figure()
            plt.title('ACTIVE: %i | INACTIVE: %i' % (np.array(non_zero_activity).shape[1], zero_count))
            shape = activity_to_plot.shape
            x = activity_to_plot.reshape(shape[0]*shape[1])
            plt.hist(x, bins=40)
            plt.savefig('%s/activity_%s.png' % (save_folder, save_name))
            # plt.show()

        if not spiking:
            #TODO: fix this because we keep appending, need to attach value to an epoch and overwrite
            final_errors = dat_results.load(
                parameters=['final_errors'],
                save_location=save_folder)['final_errors']
            if final_errors.ndim == 0:
                final_errors = np.array([[x_err, y_err]])
            else:
                final_errors = np.vstack((final_errors, [x_err, y_err]))

            final_errors = np.asarray(final_errors)
            plt.figure()
            plt.subplot(311)
            plt.title('X error over epochs')
            plt.plot(final_errors[:, 0])
            plt.subplot(312)
            plt.title('Y error over epochs')
            plt.plot(final_errors[:, 1])
            plt.savefig('%s/final_epoch_error.png' % (save_folder))
            plt.close()

            save_data = {save_name: predictions, 'final_errors': final_errors}

        else:
            save_data = {save_name: predictions}

        dat_results.save(
            data=save_data,
            save_location=save_folder,
            overwrite=True)

    if backend == 'nengo_loihi' or backend == 'nengo':
        print('Starting %s sim...' % backend)
        sim.run(dt*n_steps*n_validation)
        prediction_data = sim.data #[prediction_probe]
        # np.savez_compressed('predictions2', predictions)
        #TODO should this be steps*n_val?
        target_vals = validation_targets[:n_steps*n_validation]
        num_pts = num_imgs_to_show*n_steps
        if spiking:
            prefix = 'spiking_'
        else:
            prefix = ''
        save_name = '%s%s_%sinference_epoch%i' % (prefix, backend, custom_save_tag, epochs[0])
        plot_predict(
                prediction_data=prediction_data, target_vals=target_vals,
                save_folder=save_folder, save_name=save_name,
                num_pts=num_pts)

    elif backend == 'nengo_dl':
        validation_images_dict = {
            vision_input: validation_images.reshape(
                (n_validation, n_steps, subpixels))
        }


        if train_on_data:
            print('Training')
            training_images_dict = {
                vision_input: training_images.reshape(
                    (n_training, n_steps, subpixels))
            }

            training_targets_dict = {
                output_probe: training_targets.reshape(
                    (n_training, n_steps, output_dims))
            }

            sim.compile(
                optimizer=tf.optimizers.RMSprop(0.001),
                loss={output_probe: tf.losses.mse})

        if isinstance(epochs, int):
            epochs = [0, epochs]

        for epoch in range(epochs[0], epochs[1]):
            num_pts = num_imgs_to_show*n_steps #3*int(max(n_steps, min(n_validation, 100)))
            # num_pts=100
            print('\n\nEPOCH %i\n' % epoch)
            if load_net_params and epoch>0:
                prev_params_loc = ('%s/%s_%i' % (save_folder, params_file, epoch-1))
                print('loading pretrained network parameters from \n%s' % prev_params_loc)
                sim.load_params(prev_params_loc)

            if train_on_data:
                print('Training in nengo-dl...')
                sim.fit(training_images_dict, training_targets_dict, epochs=1)

            # save parameters back into net
            sim.freeze_params(net)

            if save_net_params:
                current_params_loc = '%s/%s_%i' % (save_folder, params_file, epoch)
                print('saving network parameters to %s' % current_params_loc)
                sim.save_params(current_params_loc)

            if train_on_data:
                # we're predicting using the weights from this epoch
                save_name='%sprediction_epoch%i' % (custom_save_tag, epoch)
            else:
                if spiking:
                    prefix = 'spiking_'
                else:
                    prefix = ''
                # we're running inference using the previous weights
                save_name='%s%sinference_epoch%i' % (prefix, custom_save_tag, epoch-1)


            print('input shape: ', validation_images_dict[vision_input].shape)
            print('Running Prediction in nengo-dl')
            data = sim.predict(validation_images_dict, n_steps=n_steps, stateful=False)
            predictions = data[output_probe]
            predictions = np.asarray(predictions).squeeze()
            np.savez_compressed('dl_predictions', predictions, validation_images_dict)

            plot_predict(
                    prediction_data=data, target_vals=validation_targets,
                    save_folder=save_folder, save_name=save_name,
                    num_pts=num_pts)

        # NOTE this seems unnecessary to save
        # dat_results.save(
        #     data={'targets': validation_targets},
        #     save_location=save_folder,
        #     overwrite=True)
