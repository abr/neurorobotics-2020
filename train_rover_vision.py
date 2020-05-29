"""
Train up the RoverVision network. Applies the scale_firing_rates = 400 heuristic to
initialize the network with higher firing rates.

Expecting (images, targets) to be stored in the abr_analyze database in the
abr_analyze.paths.database_dir folder. To generate training data, run rover.py with the
generate_data variable set to True.
"""
import nengo_loihi

import warnings
import os
import tensorflow as tf
import numpy as np
from abr_analyze import DataHandler, paths

import dl_utils
from rover_vision import RoverVision, LoihiRectifiedLinear


# =================== Tensorflow settings to avoid OOM errors =================
warnings.simplefilter("ignore")
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
# =============================================================================


activation = nengo_loihi.neurons.LoihiSpikingRectifiedLinear()
dt = 0.001
db_name = "abr_analyze"  # database we're reading images and targets from
epochs = [0, 1000]  # how many training iterations to run
minibatch_size = 10
scale_firing_rates = 400  # gain on first layer connection weights
seed = np.random.randint(1e5)

# for saving figures and weights
save_folder = "%s/data/%s/%s/%s" % (
    paths.database_dir,
    db_name,
    activation,
    str(scale_firing_rates),
)
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# instantiate our keras converted network
vision = RoverVision(minibatch_size=minibatch_size, dt=dt, seed=seed)

try:
    # try to load in saved processed data
    processed_data = np.load(
        "%s/%s_training_images_processed.npz" % (paths.database_dir, db_name)
    )
    images = processed_data["images"]
    targets = processed_data["targets"]
    print("Processed training images loaded from file...")

except FileNotFoundError:
    print("Processed training images not found, processing now...")
    images, targets = dl_utils.consolidate_data(
        db_name=db_name,
        # assuming data collected is all saved as training_0000. if you run multiple
        # sessions and save under more test names, e.g. training_0001, training_0002,
        # then update this parameter to include those test names.
        label_list=["training_%04i" % ii for ii in range(1)],
        # how often to sample the images saved, works in conjunction with the
        # save_frequency parameter in rover.py used when saving data
        step_size=5,
    )
    targets = targets[:, 0:2]  # saved targets are 3D but only care about x and y

    # do our resizing, scaling, and flattening
    images = dl_utils.preprocess_images(
        image_data=images,
        show_resized_image=False,
        flatten=True,
        normalize=False,
        res=[32, 128],
    )
    # save processed training data to speed up future runs
    np.savez_compressed(
        "%s/%s_training_images_processed" % (paths.database_dir, db_name),
        images=images,
        targets=targets,
    )

# repeat and batch training data
images = dl_utils.repeat_data(images, batch_data=True, n_steps=1)

targets /= np.pi  # change target range to -1:1 instead of -pi:pi
targets = dl_utils.repeat_data(targets, batch_data=True, n_steps=1)

# convert from Keras to Nengo
sim, net = vision.convert(
    converter_params={
        "swap_activations": {tf.nn.relu: activation},
        "scale_firing_rates": scale_firing_rates,
    }
)

loss = {vision.nengo_dense: tf.losses.mse}  # set up cost function
targets = {vision.nengo_dense: targets}  # set up target values for training

with sim:
    sim.compile(optimizer=tf.optimizers.RMSprop(0.0001), loss=loss)

    for epoch in range(epochs[0], epochs[1]):
        print("\nEPOCH %i" % epoch)
        if epoch > 0:
            prev_params_loc = "%s/epoch_%i" % (save_folder, epoch - 1)
            print("Loading pretrained network parameters from \n%s" % prev_params_loc)
            sim.load_params(prev_params_loc)

        print("Fitting data...")
        sim.fit(images, targets, epochs=1)

        current_params_loc = "%s/epoch_%i" % (save_folder, epoch)
        print("Saving network parameters to %s" % current_params_loc)
        sim.save_params(current_params_loc)

test_params = {
    "dt": dt,
    "scale_firing_rates": scale_firing_rates,
    "n_training": images.shape[0],
    "minibatch_size": minibatch_size,
    "seed": seed,
    "epochs": epochs,
}

# Save our set up parameters
dat_results = DataHandler("%s_results" % db_name)
dat_results.save(
    data=test_params, save_location="%s/params" % save_folder, overwrite=True
)
