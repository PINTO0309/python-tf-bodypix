from pathlib import Path
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tf_bodypix.api import download_model, load_model, BodyPixModelPaths
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

# setup input and output paths
output_path = Path('./data/example-output')
output_path.mkdir(parents=True, exist_ok=True)
input_url = (
    'https://www.dropbox.com/s/7tsaqgdp149d8aj/serious-black-businesswoman-sitting-at-desk-in-office-5669603.jpg?dl=1'
)
local_input_path = tf.keras.utils.get_file(origin=input_url)

# load model (once)
bodypix_model = load_model(download_model(
    BodyPixModelPaths.MOBILENET_FLOAT_50_STRIDE_16
))

# get prediction result
image = tf.keras.preprocessing.image.load_img(local_input_path)
image_array = tf.keras.preprocessing.image.img_to_array(image)
result = bodypix_model.predict_single(image_array)

# simple mask
mask = result.get_mask(threshold=0.75)
tf.keras.preprocessing.image.save_img(
    f'{output_path}/output-mask.jpg',
    mask
)

# colored mask (separate colour for each body part)
colored_mask = result.get_colored_part_mask(mask)
tf.keras.preprocessing.image.save_img(
    f'{output_path}/output-colored-mask.jpg',
    colored_mask
)

# poses
from tf_bodypix.draw import draw_poses  # utility function using OpenCV

poses = result.get_poses()
image_with_poses = draw_poses(
    image_array.copy(),  # create a copy to ensure we are not modifing the source image
    poses,
    keypoints_color=(255, 100, 100),
    skeleton_color=(100, 100, 255)
)
tf.keras.preprocessing.image.save_img(
    f'{output_path}/output-poses.jpg',
    image_with_poses
)