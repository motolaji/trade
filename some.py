class FeatureExtractor:
    def __init__(self):
        # 1. PyTorch/TorchVision: Setup model and transforms
        import torch
        import torchvision.models as models
        import torchvision.transforms as transforms

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet50(pretrained=True).to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def extract_features(self, image_data):
        # 2. Visual Features (PyTorch)
        with torch.no_grad():
            image_tensor = self.transform(image_data['pil_image']).unsqueeze(0)
            visual_features = self.model(image_tensor.to(self.device))
            visual_features = visual_features.cpu().numpy().squeeze()

        # 3. Shape Features (OpenCV)
        edges = cv2.Canny(image_data['bgr_image'], 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # 4. Color Features (OpenCV + NumPy)
        color_hist = cv2.calcHist([image_data['hsv_image']], [0, 1], None,
                                  [8, 8], [0, 180, 0, 256])
        color_hist = cv2.normalize(color_hist, color_hist).flatten()

        # 5. Image Hash (Imagehash)
        import imagehash
        phash = str(imagehash.average_hash(image_data['pil_image']))

        return {
            'visual': visual_features,
            'shape': {
                'num_contours': len(contours),
                'contour_areas': [cv2.contourArea(c) for c in contours]
            },
            'color': {
                'histogram': color_hist.tolist()
            },
            'hash': phash
        }