from feature_extractor import FeatureExtractor
from index_builder import IndexBuilder
from similarity_searcher import SimilaritySearcher
from PIL import Image
from io import BytesIO
import requests
import tempfile
from pdf_visualiser import ResultVisualizer





def visual(image, similar_results):
    # Initialize components
    extractor = FeatureExtractor()
    visualizer = ResultVisualizer()

    try:
        # Test feature visualization
        # image_path = 'C:/Users/jajim/Desktop/dissertation/trade/L3D-dataset/dataset/images/00000e23-4448-46d3-944c-799316bf0e7c.jpg'  # Add your test image
        features = extractor.extract_features(image)
        feature_path = visualizer.visualize_features(features)
        print(f"Feature visualization saved to: {feature_path}")

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
        results_path = visualizer.visualize_results(
            image,
            similar_results,
            output_name="similar_results"
        )
        print(f"Results visualization saved to: {results_path}")

    except Exception as e:
        print(f"Error during visualization: {str(e)}")


def searcher(query_path):
    # Initialize components
    extractor = FeatureExtractor()
    indexer = IndexBuilder()
    visualizer = ResultVisualizer()
    

    try:
        # Load index
        index, features_file = indexer.load_index()
        searcher = SimilaritySearcher(index, features_file)

        # Test search with a query image
        # query_path = 'C:/Users/jajim/Desktop/dissertation/trade/L3D-dataset/dataset/images/00000e23-4448-46d3-944c-799316bf0e7c.jpg'  # Add your test image
        query_features = extractor.extract_features(query_path)

        # Search
        results = searcher.search(query_features, k=5)
        new_results = []

        print("Search Results:")
        for i, result in enumerate(results, 1):
            new_results.append({'path': result['path'], 'similarity': result['similarity'], 'distance': result['distance']}) 

        print(new_results)
        # visual = visual(query_path, new_results)

        features = extractor.extract_features(query_path)
        feature_path = visualizer.visualize_features(features)
        print(f"Feature visualization saved to: {feature_path}")

        results_path = visualizer.visualize_results(
        query_path,
        new_results,
        output_name="similar_results"
    )
        print(f"Results visualization saved to: {results_path}")

        return new_results, results_path
    
    

    


# taking this out placing in test_main.py
    except Exception as e:
        print(f"Error during search: {str(e)}")


# if __name__ == "__main__":
#     test_searcher()
