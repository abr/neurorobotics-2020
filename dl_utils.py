import cv2
import numpy as np
from abr_analyze import DataHandler
import matplotlib.pyplot as plt
from nengo.utils.matplotlib import rasterplot

def preprocess_images(
        image_data, res, show_resized_image=False, flatten=True, normalize=True):
    """
    Accepts a 3D array (single image) or 4D array (multiple images) of shape
    (n_imgs, vertical_pixels, horizontal_pixels, rgb_data).

    Returns the image array with the following Optional processing
        - reshaped to the specified res if not matching
        - normalizing image to be from 0-1 instead of 0-255
        - flattening of images

    If flattening: returns image array of shape (n_images, n_subpixels)
    else: returns image array of shape (n_imgs, horizontal_pixels, vertical_pixels, rgb_data).

    Note that the printouts will only be output for the first image

    Parameters
    ----------
    image_data: 3D or 4D array of floats
        a single, or array of rgb image(s)
        shape (n_imgs, vertical pixels, horizontal pixels, 3)
    res: list of 2 floats
        the desired resolution of the output images
    show_resized_image: boolean, Optional (Default: False)
        plots the image before and after rescaling
    flatten: boolean, Optional (Default: True)
        flattens the image to a vector
        if array of images passed in, will output (n_images, subpixels) shape
    normalize: boolean, Optional (Default: True)
        normalize data from 0-255 to 0-1
    """

    scaled_image_data = []

    # single image, append 1 dimension so we can loop through the same way
    image_data = np.asarray(image_data)
    shape = image_data.shape
    if image_data.ndim == 3:
        image_data = image_data.reshape((1, shape[0], shape[1], shape[2]))

    # expect rgb image data
    assert image_data.shape[3] == 3

    for count, data in enumerate(image_data):
        rgb = np.asarray(data)
        # normalize
        if normalize:
            if np.mean(data) > 1:
                if count == 0: # only print for the first image
                    print('Image passed in 0-255, normalizing to 0-1')
                rgb = rgb/255
            else:
                if count == 0: # only print for the first image
                    print('Image passed in 0-1, skipping normalizing')


        # resize image resolution
        if shape[1] != res[0] or shape[2] != res[1]:
            if count == 0: # only print for the first image
                print('Resolution does not match desired value, resizing...')
                print('Desired Res: ', res)
                print('Input Res: ', [shape[1], shape[2]])
            rgb = cv2.resize(
                rgb, dsize=(res[1], res[0]),
                interpolation=cv2.INTER_CUBIC)
        else:
            if count == 0: # only print for the first image
                print('Resolution already at desired value, skipping resizing...')

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
        if flatten:
            rgb = np.ravel(rgb)

        scaled_image_data.append(np.copy(rgb))

    scaled_image_data = np.asarray(scaled_image_data)

    return scaled_image_data


def repeat_data(data, batch_data=False, n_steps=1):
    """
    Accepts flattened data of shape (number images / targets, flattened data length)
    Repeats the data n_steps times and batches the images based on batch_size

    Parameters
    ----------
    data: array of floats
        inputs data of shape (number of imgs / targets, flattened data dimensionality)
    batch_data: boolean, Optional (Default: False)
        True: output shape (number imgs / targets, n_steps, flattened dimensionality)
        False: output shape (1, number imgs / targets * n_steps, flattened dimensionality)
    n_steps: int, Optional (Default: 1)
        number of times to repeat each input target / image
    """
    print('Data pre_tile: ', data.shape)
    if batch_data:
        # batch our images for training
        data = np.tile(data[:, None, :], (1, n_steps, 1))

    else:
        # run like nengo sim.run without batching
        data = np.repeat(data, n_steps, 0)
        data = data[np.newaxis, :]

    print('Data post_tile: ', data.shape)

    return data


def load_data(db_name, label='training_0000', n_imgs=None):
    """
    loads rgb images and targets from an hdf5 database and returns them as a np array

    Expects data to be saved in the following group stucture:
        Training Data
        training_0000/data/0000     using %04d to increment data name

        Validation Data
        validation_0000/data/0000       using %04d to increment data name

        Both return an array with the rgb image saved under the 'rgb' key
        and the target saved under the 'target' key

    Parameters
    ----------
    db_name: string
        name of database to load from
    label: string, Optional (Default: 'training_0000')
        location in database to load from
    n_imgs: int
        how many images to load
    """
    #TODO: specify the data format expected in the comment above
    dat = DataHandler(db_name)

    # load training images
    training_images = []
    training_targets = []
    for nn in range(0, n_imgs):
        data = dat.load(parameters=['rgb', 'target'], save_location='%s/data/%04d' % (label, nn))
        training_images.append(data['rgb'])
        training_targets.append(data['target'])

    training_images = np.asarray(training_images)
    training_targets = np.asarray(training_targets)

    return training_images, training_targets

# very long function for plotting results and debugging fun
def plot_prediction_error(
        predictions, target_vals, num_pts,
        save_folder='', save_name='prediction_results', show_plot=False):
    """
    Accepts predictions and targets, plots the x and y error, along with the target location

    Parameters
    ----------
    predictions: array of floats
        nengo sim data[output] array
    target_vals: array of float
        flattened target data that was passed in during inferece
    num_pts: int
        how many steps to plot from the end of the sim
    save_folder: string
        location to save figures
    save_name: string, Optional (Default: 'prediction_results')
        name to save plot under
    """

    print('plotting %i timesteps' % num_pts)
    print('targets shape: ', np.asarray(target_vals).shape)
    print('prediction shape: ', np.asarray(predictions).shape)

    if predictions.ndim > 2:
        shape = np.asarray(predictions).shape
        predictions = np.asarray(predictions).reshape(shape[0]*shape[1], shape[2])
        print('pred reshape: ', predictions.shape)

    if target_vals.ndim > 2:
        shape = np.asarray(target_vals).shape
        target_vals = np.asarray(target_vals).reshape(shape[0]*shape[1], shape[2])
        print('targets reshape: ', target_vals.shape)

    # calculate our error to target val
    x_err = np.linalg.norm(target_vals[:, 0] - predictions[:, 0])
    y_err = np.linalg.norm(target_vals[:, 1] - predictions[:, 1])

    fig = plt.Figure()
    x = np.arange(predictions.shape[0])

    # plot our X predictions
    plt.subplot(311)
    plt.title('X: %.3f' % x_err)
    plt.plot(x, predictions[:, 0], label='predictions', color='k', linestyle='--')
    plt.plot(x, target_vals[:, 0], label='target', color='r')
    # plt.plot(x, predictions[-num_pts:, 0], label='predictions', color='k', linestyle='--')
    # plt.plot(x, target_vals[-num_pts:, 0], label='target', color='r')
    plt.legend()

    # plot our Y predictions
    plt.subplot(312)
    plt.title('Y: %.3f' % y_err)
    plt.plot(x, predictions[:, 1], label='predictions', color='k', linestyle='--')
    plt.plot(x, target_vals[:, 1], label='target', color='r')
    # plt.plot(x, predictions[-num_pts:, 1], label='predictions', color='k', linestyle='--')
    # plt.plot(x, target_vals[-num_pts:, 1], label='target', color='r')
    plt.legend()

    # plot our targets in the xy plane to get an idea of their coverage range
    plt.subplot(313)
    plt.title('Target XY')
    plt.xlim([-3, 3])
    plt.ylim([-3, 3])
    plt.scatter(0, 0, color='r', label='rover', s=2)
    plt.scatter(
        target_vals,
        target_vals, label='target', s=1)
        # target_vals[-num_pts:, 0],
        # target_vals[-num_pts:, 1], label='target', s=1)
    plt.legend()
    plt.tight_layout()
    plt.savefig('%s/%s.png' % (save_folder, save_name))
    print('Saving prediction results to %s/%s.png' % (save_folder, save_name))

    if show_plot:
        plt.show()

    plt.close()


def plot_neuron_activity(
        activity, num_pts, save_folder='', save_name='activity', num_neurons_to_plot=100,
        images=None):
    """
    Accepts the nengo sim data[neuron_probe] array and saves the raster plot
    If the input data is passed in then the individual neuron activity will also be
    plotted next to the input image. This second figure is currently set to plot
    num_neurons_to_plot figures, with each figure showing that neurons activity for each
    input image passed in

    Parameters
    ----------
    activity: array of floats
        nengo sim.data[neuron_probe] array
    num_pts: int
        how many steps to plot from the end of the sim
    save_folder: string
        location to save figures
    save_name: string, Optional (Default: 'activity')
        name to save plot under
    num_neurons_to_lot: int, Optional(Default: 100)
        This function will check each neurons activity over time and only
        plot neurons that were active. The nerons are then evenly selected
        based on this int
        ex: if 1000 active neurons and num_neurons_to_plot=10, then each 100th neuron
        will be plotted
    """

    print('\nNEURON ACTIVITY')
    # flatten our activity over time so we can extract only neurons that have some activity
    shape = activity.shape
    print('activity shape: ', shape)
    if activity.ndim == 3:
        activity = activity.reshape(shape[0]*shape[1], shape[2], order='C')
    print('reshaped activity: ', activity.shape)

    # sum our activity over time, should have an array of length n_neurons
    activity_over_time = np.sum(activity, axis=0)
    assert shape[-1] == activity_over_time.shape[0]
    # print('if I did this right %i should match' % (shape[-1]), activity_over_time.shape)

    # track how many neurons never activate
    zero_count = 0
    non_zero_neurons = []

    # loop through our activities and find the index of active neurons
    for index, neuron in enumerate(activity_over_time):
        if neuron == 0:
            zero_count += 1
        else:
            non_zero_neurons.append(index)

    # extract non zero activities
    non_zero_activity = [activity[:, i] for i in non_zero_neurons]

    # keep time/batch_size as our first dimension
    non_zero_activity = np.asarray(non_zero_activity).T
    assert non_zero_activity.shape[1] == activity.shape[1] - zero_count

    print('%i neurons never fire' % zero_count)
    print('%i neurons fire' % len(non_zero_neurons))

    print('non zero activity shape: ', non_zero_activity.shape)
    print('only plotting %i neurons out of %i active neurons' %(num_neurons_to_plot, non_zero_activity.shape[1]))
    print('only showing the last %i timesteps out of %i' %(num_pts, non_zero_activity.shape[0]))

    # evenly select our neurons instead of grabbing them sequentially
    neuron_step = int(non_zero_activity.shape[1]/num_neurons_to_plot)
    print('choosing every %i neuron' % neuron_step)
    activity_to_plot =  non_zero_activity[-num_pts:, ::neuron_step]
    t = np.arange(0, num_pts, 1)

    # rasterplot of our neural activity for a subset of neurons
    plt.figure(figsize=(10, 30))
    plt.subplot(111)
    rasterplot(t, activity_to_plot)
    plt.tight_layout()
    plt.savefig('%s/%s_rasterplot.png' % (save_folder, save_name))
    print('saving figure to %s/%s_rasterplot.png' % (save_folder, save_name))
    plt.close()

    # plots neural activity of individual neurons over n_steps for each image, next to the input image
    #TODO may want to reorder this to have all neurons plotted together instead of in separate figures
    if images is not None:
        for neuron in range(0, num_neurons_to_plot):
            # x is activity for a single neuron over num_pts steps
            # use the same neurons selected in the rasterplot
            single_activity = non_zero_activity[:, neuron_step*neuron]
            print('single neuron activity shape: ', single_activity.shape)
            # reshape so we can separate by image
            num_images = images.shape[0]
            # we show the images for some number of timesteps, divide the total steps
            # by the number of images so we can split the activity by image change
            n_val_steps = int(single_activity.shape[0] / num_images)
            single_activity = single_activity.reshape(num_images, n_val_steps)
            print('reshaped by image: ', single_activity.shape)

            plt.figure(figsize=(10, 10))
            plt.title('Neuron %i Activity over %i steps' % (neuron, n_val_steps))
            for ii, image_activity in enumerate(single_activity):
                plt.subplot2grid((len(single_activity), 4), (ii, 0), colspan=2, rowspan=1)
                plt.imshow(images[ii], origin='lower')
                plt.title('Image %i' % ii)
                # plt.subplot(len(x), 1, ii+1)
                plt.subplot2grid((len(single_activity), 4), (ii, 2), colspan=2, rowspan=1)
                # plt.plot(non_zero_activity[:, 0])
                plt.plot(image_activity, label=ii)
            plt.tight_layout()
            plt.savefig('%s/%s_neuron%i_spikes.png' % (save_folder, save_name, neuron))
            # plt.show()

def plot_training_errors():
    raise NotImplementedError
    #TODO this is for tracking the final error after each epoch
    # tracks our final error as we train so we can see our progression
    #TODO need to rewrite this as we don't have access to train_on_data
    if not train_on_data:
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
