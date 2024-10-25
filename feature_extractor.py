import torch
import torchvision.models as models
import torchvision.transforms as transforms
import imagehash
from PIL import Image
import cv2
import numpy as np
from image_loader import ImageLoader


class FeatureExtractor:
    def __init__(self):
        # Initialize model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet50(pretrained=True)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model = self.model.to(self.device)
        self.model.eval()

        # Setup image transformation
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def extract_features(self, image_path):
        """Extract all features from an image"""
        # Load image
        image_data = ImageLoader.load_image(image_path)

        # Extract different types of features
        features = {
            'visual': self._extract_visual_features(image_data['pil']),
            'shape': self._extract_shape_features(image_data['bgr']),
            'color': self._extract_color_features(image_data['hsv']),
            'hash': str(imagehash.average_hash(image_data['pil']))
        }

        return features

    def _extract_visual_features(self, pil_image):
        """Extract CNN features"""
        image_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(image_tensor)
        return features.cpu().numpy().squeeze()

    def _extract_shape_features(self, bgr_image):
        """Extract shape-based features"""
        # Convert to grayscale
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

        # Get edges
        edges = cv2.Canny(gray, 100, 200)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        return {
            'num_contours': len(contours),
            'contour_areas': [cv2.contourArea(c) for c in contours],
            'aspect_ratio': bgr_image.shape[1] / bgr_image.shape[0]
        }

    def _extract_color_features(self, hsv_image):
        """Extract color-based features"""
        # Calculate color histogram
        hist = cv2.calcHist([hsv_image], [0, 1], None, [8, 8], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        return {
            'histogram': hist.tolist(),
            'avg_saturation': float(np.mean(hsv_image[:, :, 1]))
        }