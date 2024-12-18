from feature_extractor import FeatureExtractor
# from visualizer import ResultVisualizer
from pdf_visualiser import ResultVisualizer


def test_visualizer(image_path, similar_results):
    # Initialize components
    extractor = FeatureExtractor()
    visualizer = ResultVisualizer()

    try:
        # Test feature visualization
        # image_path = 'C:/Users/jajim/Desktop/dissertation/trade/L3D-dataset/dataset/images/00000e23-4448-46d3-944c-799316bf0e7c.jpg'  # Add your test image
        features = extractor.extract_features(image_path)
        feature_viz_path = visualizer.visualize_features(features)
        print(f"Feature visualization saved to: {feature_viz_path}")

        # Test results visualization
        # Simulate search results
        # similar_results = [
        #     {'path': 'C:/Users/jajim/Desktop/dissertation/trade/L3D-dataset/dataset/images/00000e23-4448-46d3-944c-799316bf0e7c.jpg', 'similarity': 1},
        #     {'path': 'C:/Users/jajim/Desktop/dissertation/trade/L3D-dataset/dataset/images/02136da8-3606-45d0-b5be-6651abb1a74e.jpg', 'similarity': 0.352},
        #     {'path': 'C:/Users/jajim/Desktop/dissertation/trade/L3D-dataset/dataset/images/013b9cdf-0c40-4b02-a85e-f71750f849fb.jpg', 'similarity': 0.318},
        #     {'path': 'C:/Users/jajim/Desktop/dissertation/trade/L3D-dataset/dataset/images/0350adbe-abc0-4344-8320-9a507febbc5f.jpg', 'similarity': 0.317},
        #     {'path': 'C:/Users/jajim/Desktop/dissertation/trade/L3D-dataset/dataset/images/0171597c-50c5-42aa-82a4-7fee7a2423d8.jpg', 'similarity': 0.313}
        # ]
    # moving to test_main.py
        results_viz_path = visualizer.visualize_results(
            image_path,
            similar_results,
            output_name="test_results"
        )
        print(f"Results visualization saved to: {results_viz_path}")

    except Exception as e:
        print(f"Error during visualization: {str(e)}")


if __name__ == "__main__":
    test_visualizer()