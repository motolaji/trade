from feature_extractor import FeatureExtractor
from visualizer import ResultVisualizer


def test_visualizer():
    # Initialize components
    extractor = FeatureExtractor()
    visualizer = ResultVisualizer()

    try:
        # Test feature visualization
        image_path = '../trademark/L3D_dataset/Bold/3.jpg'  # Add your test image
        features = extractor.extract_features(image_path)
        feature_viz_path = visualizer.visualize_features(features)
        print(f"Feature visualization saved to: {feature_viz_path}")

        # Test results visualization
        # Simulate search results
        similar_results = [
            {'path': '../trademark/L3D_dataset/Bold/3.jpg', 'similarity': 1},
            {'path': '../trademark/L3D_dataset/Bold/20.jpg', 'similarity': 0.093},
            {'path': '../trademark/L3D_dataset/Bold/12.jpg', 'similarity': 0.082}
        ]
        results_viz_path = visualizer.visualize_results(
            image_path,
            similar_results
        )
        print(f"Results visualization saved to: {results_viz_path}")

    except Exception as e:
        print(f"Error during visualization: {str(e)}")


if __name__ == "__main__":
    test_visualizer()