import torch


def load_and_preprocess(image_path):
    # 1. PIL: Initial image loading
    from PIL import Image
    image = Image.open(image_path).convert('RGB')

    # 2. NumPy: Convert to array
    import numpy as np
    image_array = np.array(image)

    # 3. OpenCV: Color space conversions
    import cv2
    bgr_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    hsv_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)

    return {
        'pil_image': image,
        'numpy_array': image_array,
        'bgr_image': bgr_image,
        'hsv_image': hsv_image
    }