import numpy as np
from scipy.spatial.distance import cosine
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class SimilaritySearcher:
    def __init__(self, index, features_file):
        self.index = index
        self.features_file = features_file

    def search(self, query_features, k=5):
        """Search for similar logos"""
        # Prepare query features
        query_vector = query_features['visual'].reshape(1, -1).astype('float32')

        # Search the index
        distances, indices = self.index.search(query_vector, k)

        # Prepare results
        results = []
        for i, idx in enumerate(indices[0]):
            # Get path and features
            path = self.features_file['paths'][idx].decode('utf-8')
            ref_features = self.features_file['features'][idx]

            # Calculate comprehensive similarity
            similarity = self._calculate_similarity(query_features, {
                'visual': ref_features,
                'path': path
            })

            results.append({
                'path': path,
                'similarity': similarity,
                'distance': float(distances[0][i])
            })

        return results

    def _calculate_similarity(self, query_features, ref_features):
        """Calculate comprehensive similarity score"""
        # Visual similarity (using L2 distance)
        visual_sim = 1 / (1 + np.linalg.norm(
            query_features['visual'] - ref_features['visual']
        ))

        return float(visual_sim)