from PIL import Image
import numpy as np
import cv2


class ImageLoader:
    @staticmethod
    def load_image(image_path):
        """Load an image and return multiple formats"""
        # Load with PIL
        pil_image = Image.open(image_path).convert('RGB')

        # Convert to numpy array
        np_image = np.array(pil_image)

        # Convert to BGR (OpenCV format)
        bgr_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)

        # Convert to HSV
        hsv_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2HSV)

        return {
            'pil': pil_image,
            'numpy': np_image,
            'bgr': bgr_image,
            'hsv': hsv_image
        }