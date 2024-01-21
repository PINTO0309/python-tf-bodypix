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
"""
segments = logits = image = [1, 14, 21, 1]
image = resized_and_padded.shape = [1, 209, 321, 1]
sigmoid(resized_and_padded) = [1, 209, 321, 1]

remove_padding_and_resize_backed = [426, 640, 1]
mask = (remove_padding_and_resize_backed > 0.75).astype(np.int32), 0 or 1
"""
mask = result.get_mask(threshold=0.75)
tf.keras.preprocessing.image.save_img(
    f'{output_path}/output-mask.jpg',
    mask
)

# colored mask (separate colour for each body part)
"""
part_heatmaps = logits = image = [1, 14, 21, 24]
image = resized_and_padded.shape = [1, 209, 321, 24]
sigmoid(resized_and_padded) = [1, 209, 321, 24]

remove_padding_and_resize_backed = [426, 640, 24]
argmaxed = argmax(remove_padding_and_resize_backed, axis=3)

# mask のうち argmax に該当するインデックス以外を -1 に置き換える
argmaxed_replaced = \
    np.where(
        np.squeeze(mask, axis=-1),
        argmaxed,
        np.asarray([-1])
    )
colored_mask = argmaxed_replaced # 本家はカラーマップを充てたあとのRGB画像を返しているが必要なし
"""
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