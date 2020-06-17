import cv2
import numpy as np
from data_handler import DataHandler
import matplotlib.pyplot as plt
from nengo.utils.matplotlib import rasterplot


def preprocess_images(
    image_data, res, show_resized_image=False, flatten=True, normalize=True
):
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
                if count == 0:  # only print for the first image
                    print("Image passed in 0-255, normalizing to 0-1")
                rgb = rgb / 255
            else:
                if count == 0:  # only print for the first image
                    print("Image passed in 0-1, skipping normalizing")

        # resize image resolution
        if shape[1] != res[0] or shape[2] != res[1]:
            if count == 0:  # only print for the first image
                print("Resolution does not match desired value, resizing...")
                print("Desired Res: ", res)
                print("Input Res: ", [shape[1], shape[2]])
            rgb = cv2.resize(rgb, dsize=(res[1], res[0]), interpolation=cv2.INTER_CUBIC)
        else:
            if count == 0:  # only print for the first image
                print("Resolution already at desired value, skipping resizing...")

        # visualize scaling for debugging
        if show_resized_image:
            plt.Figure()
            a = plt.subplot(121)
            a.set_title("Original")
            a.imshow(data, origin="lower")
            b = plt.subplot(122)
            b.set_title("Scaled")
            b.imshow(rgb, origin="lower")
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
    print("Data pre_tile: ", data.shape)
    if batch_data:
        # batch our images for training
        data = np.tile(data[:, None, :], (1, n_steps, 1))

    else:
        # run like nengo sim.run without batching
        data = np.repeat(data, n_steps, 0)
        data = data[np.newaxis, :]

    print("Data post_tile: ", data.shape)

    return data


def load_data(
    db_name, label="training_0000", n_imgs=None, thresh=1e5, step_size=1, db_dir=None
):
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
    # TODO: specify the data format expected in the comment above
    dat = DataHandler(db_dir=db_dir, db_name=db_name)

    # load training images
    images = []
    targets = []

    skip_list = ["datestamp", "timestamp"]
    keys = np.array(
        [int(val) for val in dat.get_keys("%s" % label) if val not in skip_list]
    )
    n_imgs = max(keys) if n_imgs is None else n_imgs
    print("Total number of images in dataset: ", max(keys))

    for nn in range(0, n_imgs, step_size):
        data = dat.load(
            parameters=["rgb", "target"], save_location="%s/%04d" % (label, nn)
        )
        if np.linalg.norm(data["target"]) < thresh:
            images.append(data["rgb"])
            targets.append(data["target"])

    images = np.asarray(images)
    targets = np.asarray(targets)

    print("Total number of images within threshold: ", images.shape[0])

    return images, targets


def plot_data(db_name, label="training_0000", n_imgs=None, db_dir=None):
    """
    loads rgb images and targets from an hdf5 database and plots the images, prints
    the targets

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
    # TODO: specify the data format expected in the comment above
    dat = DataHandler(db_name=db_name, db_dir=db_dir)

    keys = np.array([int(val) for val in dat.get_keys("%s/data" % label)])
    print("Total number of images in dataset: ", max(keys))

    for nn in range(n_imgs):
        data = dat.load(
            parameters=["rgb", "target"], save_location="%s/data/%04d" % (label, nn)
        )

        print("Target: ", data["target"])

        plt.figure()
        a = plt.subplot(1, 1, 1)
        a.imshow(data["rgb"] / 255)
        plt.show()


def plot_prediction_error(
    predictions,
    target_vals,
    save_folder=".",
    save_name="prediction_results",
    show_plot=False,
):
    """
    Accepts predictions and targets, plots the x and y error, along with the target location

    Parameters
    ----------
    predictions: array of floats
        nengo sim data[output] array
    target_vals: array of float
        flattened target data that was passed in during inferece
    save_folder: string
        location to save figures
    save_name: string, Optional (Default: 'prediction_results')
        name to save plot under
    """

    print("targets shape: ", np.asarray(target_vals).shape)
    print("prediction shape: ", np.asarray(predictions).shape)

    if predictions.ndim > 2:
        shape = np.asarray(predictions).shape
        predictions = np.asarray(predictions).reshape(shape[0] * shape[1], shape[2])
        print("pred reshape: ", predictions.shape)

    if target_vals.ndim > 2:
        shape = np.asarray(target_vals).shape
        target_vals = np.asarray(target_vals).reshape(shape[0] * shape[1], shape[2])
        print("targets reshape: ", target_vals.shape)

    # calculate our error to target val
    x_err = np.linalg.norm(target_vals[:, 0] - predictions[:, 0])
    y_err = np.linalg.norm(target_vals[:, 1] - predictions[:, 1])

    fig = plt.Figure()
    x = np.arange(predictions.shape[0])

    # plot our X predictions
    plt.subplot(311)
    plt.title("X: %.3f" % x_err)
    plt.plot(x, predictions[:, 0], label="predictions", color="k", linestyle="--")
    plt.plot(x, target_vals[:, 0], label="target", color="r")
    plt.legend()

    # plot our Y predictions
    plt.subplot(312)
    plt.title("Y: %.3f" % y_err)
    plt.plot(x, predictions[:, 1], label="predictions", color="k", linestyle="--")
    plt.plot(x, target_vals[:, 1], label="target", color="r")
    plt.legend()

    # plot our targets in the xy plane to get an idea of their coverage range
    plt.subplot(313)
    plt.title("Target XY")
    plt.xlim([-3, 3])
    plt.ylim([-3, 3])
    plt.scatter(0, 0, color="r", label="rover", s=2)
    plt.scatter(target_vals[:, 0], target_vals[:, 1], label="target", s=1)
    plt.gca().set_aspect("equal")
    plt.tight_layout()
    plt.savefig("%s/%s.png" % (save_folder, save_name))
    print("Saving prediction results to %s/%s.png" % (save_folder, save_name))

    if show_plot:
        plt.show()

    plt.close()


def consolidate_data(db_name, label_list, thresh=3.5, step_size=1, db_dir=None):
    """
    loads rgb images and targets from multiple hdf5 database and consolidates them
    into a single np array, saves back to the database under the specified label

    Parameters
    ----------
    db_name: string
        name of database to load from
    label_list: list
        list of locations in database to load from
    """
    dat = DataHandler(db_dir=db_dir, db_name=db_name)

    all_images = []
    all_targets = []

    for ii, label in enumerate(label_list):
        print("db_name: ", db_name)
        print("label: ", label)
        images, targets = load_data(
            db_name=db_name,
            db_dir=db_dir,
            label=label,
            thresh=thresh,
            step_size=step_size,
        )
        all_images.append(images)
        all_targets.append(targets)

    all_images = np.vstack(all_images)
    all_targets = np.vstack(all_targets)

    print("Total images shape: ", all_images.shape)
    print("Total targets shape: ", all_targets.shape)

    return all_images, all_targets
