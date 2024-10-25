from feature_extractor import FeatureExtractor


def test_extractor():
    extractor = FeatureExtractor()
    image_path = '../trademark/L3D_dataset/images/fffff6ec-af49-4d8e-98df-b7eb0f53067a.jpg'  # Add your test image

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