import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import seaborn as sns
from pathlib import Path


class ResultVisualizer:
    def __init__(self, output_dir='results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def visualize_results(self, query_path, similar_results, output_name='results.png'):
        """Visualize search results"""
        # Set up the plot
        n_results = len(similar_results)
        fig = plt.figure(figsize=(15, 5))

        # Plot query image
        plt.subplot(1, n_results + 1, 1)
        self._plot_image(query_path, "Query Image")

        # Plot similar images
        for i, result in enumerate(similar_results, 1):
            plt.subplot(1, n_results + 1, i + 1)
            self._plot_image(
                result['path'],
                f"Sim: {result['similarity']:.2f}"
            )

        # Save result
        output_path = self.output_dir / output_name
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

        return output_path

    def visualize_features(self, features, output_name='features.png'):
        """Visualize extracted features"""
        fig = plt.figure(figsize=(15, 10))

        # Plot visual features distribution
        plt.subplot(2, 2, 1)
        sns.histplot(features['visual'])
        plt.title('Visual Features Distribution')

        # Plot shape features
        plt.subplot(2, 2, 2)
        if 'contour_areas' in features['shape']:
            sns.boxplot(y=features['shape']['contour_areas'])
            plt.title('Contour Areas Distribution')

        # Plot color histogram
        plt.subplot(2, 2, 3)
        plt.plot(features['color']['histogram'])
        plt.title('Color Histogram')

        # Save visualization
        output_path = self.output_dir / output_name
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

        return output_path

    def _plot_image(self, image_path, title):
        """Helper function to plot a single image"""
        img = Image.open(image_path)
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')