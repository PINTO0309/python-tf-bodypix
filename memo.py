"""
- segments
    segments = logits = image = [1, 14, 21, 1]
    image = resized_and_padded.shape = [1, 209, 321, 1]
    sigmoid(resized_and_padded) = [1, 209, 321, 1]

    remove_padding_and_resize_backed = [426, 640, 1]
    mask = (remove_padding_and_resize_backed > 0.75).astype(np.int32), 0 or 1
"""