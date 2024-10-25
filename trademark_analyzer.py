from pathlib import Path
from feature_extractor import FeatureExtractor
from index_builder import IndexBuilder
from similarity_searcher import SimilaritySearcher
from visualizer import ResultVisualizer
from tqdm import tqdm


class TrademarkAnalyzer:
    def __init__(self, index_dir='index', results_dir='results'):
        # Initialize components
        self.extractor = FeatureExtractor()
        self.indexer = IndexBuilder()
        self.searcher = None
        self.visualizer = ResultVisualizer(results_dir)

        # Setup directories
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(exist_ok=True)

    def build_dataset_index(self, dataset_path):
        """Build search index from dataset"""
        dataset_path = Path(dataset_path)
        image_paths = []

        # Collect image paths
        print("Collecting images...")
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            image_paths.extend(dataset_path.glob(ext))

        if not image_paths:
            raise ValueError(f"No images found in {dataset_path}")

        # Build index
        n_indexed = self.indexer.build_index(
            image_paths,
            self.extractor,
            self.index_dir
        )

        print(f"Successfully indexed {n_indexed} images")

    def load_index(self):
        """Load existing index"""
        index, features_file = self.indexer.load_index(self.index_dir)
        self.searcher = SimilaritySearcher(index, features_file)
        print("Index loaded successfully")

    def find_similar(self, query_path, k=5):
        """Find similar logos to query image"""
        if self.searcher is None:
            raise ValueError("Index not loaded. Call load_index() first")

        # Extract features from query
        print("Extracting features from query image...")
        query_features = self.extractor.extract_features(query_path)

        # Search for similar images
        print("Searching for similar logos...")
        results = self.searcher.search(query_features, k)

        # Visualize results
        print("Generating visualizations...")
        viz_path = self.visualizer.visualize_results(query_path, results)

        return results, viz_path

    def analyze_logo(self, image_path):
        """Analyze a single logo"""
        # Extract features
        features = self.extractor.extract_features(image_path)

        # Visualize features
        viz_path = self.visualizer.visualize_features(features)

        return features, viz_path