import cv2
import numpy as np
from abr_analyze import DataHandler
import matplotlib.pyplot as plt

def preprocess_images(
        image_data, res, show_resized_image=False, flatten=True, normalize=True):
    """
    Accepts a 3D array (single image) or 4D array (multiple images) of shape
    (n_imgs, horizontal_pixels, vertical_pixels, rgb_data).

    Returns the image array with the following optional processing
        - reshaped to the specified res if not matching
        - normalizing image to be from 0-1 instead of 0-255
        - flattening of images

    If flattening: returns image array of shape (n_images, n_subpixels)
    else: returns image array of shape (n_imgs, horizontal_pixels, vertical_pixels, rgb_data).

    Note that the printouts will only be output for the first image
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
        # normalize
        if normalize:
            if np.mean(data) > 1:
                if count == 0: # only print for the first image
                    print('Image passed in 0-255, normalizing to 0-1')
                rgb = np.asarray(data)/255
            else:
                if count == 0: # only print for the first image
                    print('Image passed in 0-1, skipping normalizing')
                rgb = np.asarray(data)


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
    """
    #TODO generalize this so it work with the parameters being passed in
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


def load_data(res, db_name, label='training_0000', n_imgs=None):
    """
    loads rgb images and targets from an hdf5 database and returns them as a np array

    Expects data to be saved in the following group stucture:
        Training Data
        training_0000/data/0000     using %04d to increment data name

        Validation Data
        validation_0000/data/0000       using %04d to increment data name

        Both return an array with the rgb image saved under the 'rgb' key
        and the target saved under the 'target' key
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
    #NOTE: kind of hacky taking the desired xy dim here, data should be saved accordingly
    training_targets = np.asarray(training_targets)[:, 0:2]

    return training_images, training_targets

# very long function for plotting results and debugging fun
def plot_prediction_error(
        predictions, target_vals, num_pts,
        save_folder='', save_name='prediction_results'):

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
    x = np.linspace(0, len(predictions[-num_pts:]), len(predictions[-num_pts:]))

    # plot our X predictions
    plt.subplot(311)
    plt.title('X: %.3f' % x_err)
    plt.plot(x, predictions[-num_pts:, 0], label='predictions', color='k', linestyle='--')
    plt.plot(x, target_vals[-num_pts:, 0], label='target', color='r')
    plt.legend()

    # plot our Y predictions
    plt.subplot(312)
    plt.title('Y: %.3f' % y_err)
    plt.plot(x, predictions[-num_pts:, 1], label='predictions', color='k', linestyle='--')
    plt.plot(x, target_vals[-num_pts:, 1], label='target', color='r')
    plt.legend()

    # plot our targets in the xy plane to get an idea of their coverage range
    plt.subplot(313)
    plt.title('Target XY')
    plt.xlim([-3, 3])
    plt.ylim([-3, 3])
    plt.scatter(0, 0, color='r', label='rover', s=2)
    plt.scatter(
        target_vals[-num_pts:, 0],
        target_vals[-num_pts:, 1], label='target', s=1)
    plt.legend()
    plt.tight_layout()
    plt.savefig('%s/%s.png' % (save_folder, save_name))
    print('Saving prediction results to %s/%s.png' % (save_folder, save_name))

def plot_neuron_activity(
        activity, num_pts, save_folder='', save_name='activity'):

    raise NotImplementedError
    # rasterplot of our neural activity for a subset of neurons
    neurons_to_plot = 100

    a4 = plt.subplot(224)
    shape = activity.shape
    print('activity shape: ', shape)
    # flatten our activity over time so we can extract only neurons that have some activity
    if activity.ndim == 3:
        activity = activity.reshape(shape[0]*shape[1], shape[2], order='C')
    print('reshaped activity: ', activity.shape)
    # sum our activity over time, should have an array of length n_neurons
    activity_over_time = np.sum(activity, axis=0)
    print('if I did this right %i should match' % (shape[-1]), activity_over_time.shape)
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
    print('%i neurons never fire' % zero_count)
    print('%i neurons fire' % len(non_zero_neurons))
    print('non zero activity shape: ', non_zero_activity.shape)
    print('only plotting %i neurons out of %i' %(neurons_to_plot, non_zero_activity.shape[1]))
    print('only showing the last %i timesteps out of %i' %(num_pts, non_zero_activity.shape[0]))
    # evenly select our neurons instead of grabbing them sequentially
    neuron_step = int(non_zero_activity.shape[1]/neurons_to_plot)
    print('choosing every %i neuron' % neuron_step)
    activity_to_plot =  non_zero_activity[-num_pts:, ::neuron_step]
    t = np.arange(0, num_pts, 1)
    print('t: ', t.shape)
    print('act: ', activity_to_plot.shape)
    rasterplot(t, activity_to_plot)
    plt.tight_layout()
    plt.savefig('%s/%s.png' % (save_folder, group_name))
    print('saving figure to %s/%s' % (save_folder, group_name))
    plt.close()

    # plots neural activity of individual neurons over n_steps for each image, next to the input image
    show_hist = False
    if show_hist:
        if not spiking:
            plt.figure()
            plt.subplot(111)
            plt.title('ACTIVE: %i | INACTIVE: %i' % (np.array(non_zero_activity).shape[1], zero_count))
            shape = non_zero_activity.shape
            x = non_zero_activity.reshape(shape[0]*shape[1])
            plt.ylabel('Rate Neuron Outputs')
            plt.hist(x, bins=40)
            plt.legend()
            plt.savefig('%s/activity_%s.png' % (save_folder, group_name))
        else:
            n_neuron_activities_to_plot = 10
            for neuron in range(0, n_neuron_activities_to_plot):
                # neuron = np.random.randint(0, non_zero_activity.shape[1])
                # neuron = [87, 88, 89 , 90, 91, 92, 93, 94, 95, 96, 97, 98][neurons]
                # x is activity for a single neuron over n steps for all images
                x = non_zero_activity[:, neuron]
                print('single neuron activity shape: ', x.shape)
                # reshape so we can separate by image
                x = x.reshape(num_imgs_to_show, n_val_steps)
                print('reshaped: ', x.shape)

                plt.figure(figsize=(10, 10))
                plt.title('Neuron %i Activity over %i steps' % (neuron, n_val_steps))
                for ii, x_img in enumerate(x):
                    # plt.subplot2grid((len(x), 6), (ii, 0), colspan=2, rowspan=1)
                    # plt.imshow(prediction_data[input_probe][int(ii), :].reshape((res[0], res[1], 3)), origin='lower')
                    # plt.title('Probed Input %i' % int(ii))
                    plt.subplot2grid((len(x), 4), (ii, 0), colspan=2, rowspan=1)
                    # image ii, timestep 0 of n_val_steps, all subpixels
                    # plt.imshow(validation_images[ii, 0, :].reshape((res[0], res[1], 3)), origin='lower')
                    plt.imshow(validation_images[:, ii*n_val_steps, :].reshape((res[0], res[1], 3)), origin='lower')
                    plt.title('Image %i' % ii)
                    # plt.subplot(len(x), 1, ii+1)
                    plt.subplot2grid((len(x), 4), (ii, 2), colspan=2, rowspan=1)
                    # plt.plot(non_zero_activity[:, 0])
                    plt.plot(x_img, label=ii)
                plt.tight_layout()
                plt.savefig('%s/%s_img_spikes_neuron%i_%s.png' % (save_folder, backend, neuron, group_name))
                # plt.show()

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

        save_data = {group_name: predictions, 'final_errors': final_errors}

    else:
        save_data = {group_name: predictions}

    dat_results.save(
        data=save_data,
        save_location=save_folder,
        overwrite=True)
