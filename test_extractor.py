from feature_extractor import FeatureExtractor
from PIL import Image



def test_extractor():
    extractor = FeatureExtractor()
    image_path = './L3D-dataset/dataset/images/c4b31b5e-5e91-41ff-926e-81406b65835a.JPG' 
    image = Image.open(image_path)
    image.show() # Add your test image

    try:
        features = extractor.extract_features(image_path)
        print("Features extracted successfully!")
        print("\nFeature dimensions:")
        print(f"Visual features: {features['visual'].shape}")
        print(f"Number of contours: {features['shape']['num_contours']}")
        print(f"Color histogram bins: {len(features['color']['histogram'])}")
        print(f"Image hash: {features['hash']}")
    except Exception as e:
        print(f"Error extracting features: {str(e)}")


if __name__ == "__main__":
    test_extractor()

