from feature_extractor import FeatureExtractor
from index_builder import IndexBuilder
from similarity_searcher import SimilaritySearcher


def test_searcher():
    # Initialize components
    extractor = FeatureExtractor()
    indexer = IndexBuilder()

    try:
        # Load index
        index, features_file = indexer.load_index()
        searcher = SimilaritySearcher(index, features_file)

        # Test search with a query image
        query_path = '../trademark/L3D_dataset/Bold/3.jpg'  # Add your test image
        query_features = extractor.extract_features(query_path)

        # Search
        results = searcher.search(query_features, k=5)

        print("Search Results:")
        for i, result in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"Path: {result['path']}")
            print(f"Similarity: {result['similarity']:.3f}")
            print(f"Distance: {result['distance']:.3f}")

    except Exception as e:
        print(f"Error during search: {str(e)}")


if __name__ == "__main__":
    test_searcher()