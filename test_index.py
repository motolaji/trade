from index_builder import IndexBuilder
from feature_extractor import FeatureExtractor
from pathlib import Path


def test_indexer():
    # Initialize components
    extractor = FeatureExtractor()
    indexer = IndexBuilder() 

    # Get test images
    dataset_path = Path('C:/Users/jajim/Desktop/dissertation/trade/new_images_dataset')
    image_paths = list(dataset_path.glob('*.[jp][pn][g]'))

    try:
        # Build index
        num_indexed = indexer.build_index(image_paths, extractor)
        print(f"Successfully indexed {num_indexed} images")

        # Test loading index
        index, features_file = indexer.load_index()
        print("Successfully loaded index")
        print(f"Index contains {index.ntotal} vectors")

    except Exception as e:
        print(f"Error during indexing: {str(e)}")


if __name__ == "__main__":
    test_indexer()