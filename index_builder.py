import faiss
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm


class IndexBuilder:
    def __init__(self, dimension=2048):  # ResNet50 feature dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.feature_dimension = dimension

    def build_index(self, image_paths, extractor, save_dir='index'):
        """Build search index from a list of images"""
        # Create save directory
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)

        # Process each image
        features_list = []
        valid_paths = []

        print("Building index...")
        for path in tqdm(image_paths):
            try:
                # Extract features
                features = extractor.extract_features(str(path))

                features_list.append(features['visual'])
                valid_paths.append(str(path))

            except Exception as e:
                print(f"Error processing {path}: {str(e)}")

        # Convert to numpy array
        features_array = np.array(features_list).astype('float32')

        # Add to index
        self.index.add(features_array)

        # Save index and features
        self._save_index(save_dir, features_array, valid_paths)

        return len(valid_paths)

    def _save_index(self, save_dir, features_array, paths):
        """Save index and features to disk"""
        # Save FAISS index
        faiss.write_index(self.index, str(save_dir / 'trademark.index'))

        # Save features and paths
        with h5py.File(save_dir / 'features.h5', 'w') as f:
            f.create_dataset('features', data=features_array)
            f.create_dataset('paths', data=np.array(paths, dtype='S'))

    def load_index(self, index_dir='index'):
        """Load index from disk"""
        index_dir = Path(index_dir)

        # Load FAISS index
        self.index = faiss.read_index(str(index_dir / 'trademark.index'))

        # Load features file
        features_file = h5py.File(index_dir / 'features.h5', 'r')

        return self.index, features_file