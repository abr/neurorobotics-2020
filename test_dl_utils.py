import pytest
import os
import numpy as np
from abr_analyze import DataHandler
from abr_analyze.paths import database_dir

def gen_images(res, img_scale, n_imgs):
    img = []
    for _ in range(n_imgs):
        img.append(np.random.rand((res[0], res[1], 3)) * img_scale)
    img = np.array(img)

    return img


@pytest.mark.parametrize(
    ("res"), (
        ([10, 20]),
        ([20, 20]),
    )
)

@pytest.mark.parametrize(
    ("flatten"), (
        (True),
        (False),
    )
)

@pytest.mark.parametrize(
    ("normalize"), (
        (True),
        (False),
    )
)

@pytest.mark.parametrize(
    ("n_imgs"), (
        (1),
        (2),
    )
)

@pytest.mark.parametrize(
    ("img_scale"), (
        (1),
        (255),
    )
)

def test_preprocess_image(res, flatten, normalize, n_imgs, img_scale):
    img = gen_images(res=[30, 30]), img_scale=img_scale, n_imgs=n_imgs)

    proc_img = dl_utils.preprocess_images(
        image_data=img, res=res, flatten=flatten,
        normalize=normalize)

    for img in proc_img:
        # check that we've resized to the expected resolution
        if flatten:
            assert img.shape == (res[0]*res[1]*3)
        else:
            assert img.shape == (res[0], res[1], 3)

        if normalize:
            assert np.mean(img) > 0 and np.mean(img) < 1

            # if our images are already from 0-1 make sure they don't get divided by 255 again
            if img_scale == 1:
                assert np.mean(img) > 1/255

    # check or entire output shape
    if flatten:
        assert proc_img.shape == (n_imgs, res[0]*res[1]*3)
    else:
        assert proc_img.shape == (n_imgs, res[0], res[1], 3)

@pytest.mark.parametrize(
    ("batch_data"), (
        (True),
        (False),
    )
)

@pytest.mark.parametrize(
    ("n_steps"), (
        (1),
        (10),
    )
)

@pytest.mark.parametrize(
    ("n_imgs"), (
        (1),
        (2),
    )
)

def test_repeat_data(batch_data, n_steps, n_imgs):
    # since we flatten here we lose these 1 dims and will have the rgb dimension
    # should be (n_imgs, 3)
    imgs = gen_imgs(res=[1, 1], img_scale=1, n_imgs=n_imgs).flatten()

    data = dl_utils.repeat_data(data=imgs, batch_data=batch_data, n_steps=n_steps)

    if batch_data:
        # our images should be along the first dimension
        assert data.shape == (n_imgs, n_steps, 3)
    else:
        # our images should be stacked with the step dimension
        assert data.shape == (1, n_imgs*n_steps, 3)

def test_load_data():
    # create a random db name and make sure it doesn't exist already since we'll be deleting it
    db_exists = True
    while db_exists:
        db_name = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(8))
        db_loc = os.path.abspath(os.path.join(database_dir, db_name))
        if os.path.isfile(db_loc):
            db_exists = True
        else:
            db_exists = False

    # generate our data
    dat = DataHandler(db_name)
    label = 'test_data'
    n_data_points = 4
    rgb_track = []
    target_track = []
    for ii n range(n_data_points):
        rgb = gen_imgs(res=[10, 10], img_scale=1, n_imgs=1)
        rgb_track.append(rgb)
        target = np.random.rand(3)
        target_track.append(target)
        dat.save(
            data={'rgb': rgb, 'target': target},
            save_location='%s/data/%04d' % (label, ii))

    imgs, targets = dl_utils.load_data(db_name=db_name, label=label, n_imgs=n_data_points)

    # check that our data matches what was saved
    assert imgs == np.array(rgb_track)
    assert targets == np.array(target_track)

    # remove our generated database
    os.remove(db_loc)

def test_plot_prediction_errors():
