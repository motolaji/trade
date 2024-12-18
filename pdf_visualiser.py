# src/visualizer.py
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import seaborn as sns
from pathlib import Path
import logging
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages

class ResultVisualizer:
    def __init__(self, output_dir='results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _generate_unique_filename(self, base_name, extension):
        """Generate unique filename using timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        counter = 0
        
        while True:
            if counter == 0:
                filename = f"{base_name}_{timestamp}.{extension}"
            else:
                filename = f"{base_name}_{timestamp}_{counter}.{extension}"
            
            if not (self.output_dir / filename).exists():
                return filename
            counter += 1

    def visualize_results(self, query_path, similar_results, output_name=None):
        """Visualize search results and save as PDF"""
        # Generate unique filename for PDF
        if output_name is None:
            filename = self._generate_unique_filename("results", "pdf")
        else:
            base_name = Path(output_name).stem
            filename = self._generate_unique_filename(base_name, "pdf")
        
        output_path = self.output_dir / filename
        
        # Create PDF with multiple pages
        with PdfPages(output_path) as pdf:
            # First page: Overview with all images
            self._create_overview_page(query_path, similar_results, pdf)
            
            # Second page: Detailed comparison
            self._create_detail_page(query_path, similar_results, pdf)
            
            # Third page: Feature comparison
            self._create_feature_comparison_page(similar_results, pdf)
            
        self.logger.info(f"Results PDF saved to: {output_path}")
        return output_path

    def _create_overview_page(self, query_path, similar_results, pdf):
        """Create overview page with all images"""
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
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

    def _create_detail_page(self, query_path, similar_results, pdf):
        """Create detail page with pairwise comparisons"""
        for i, result in enumerate(similar_results):
            fig = plt.figure(figsize=(12, 6))
            
            # Plot query image
            plt.subplot(1, 2, 1)
            self._plot_image(query_path, "Query Image")
            
            # Plot similar image with details
            plt.subplot(1, 2, 2)
            self._plot_image(
                result['path'],
                f"Match {i+1}\nSimilarity: {result['similarity']:.3f}"
            )
            
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()

    def _create_feature_comparison_page(self, similar_results, pdf):
        """Create feature comparison page"""
        fig = plt.figure(figsize=(12, 8))
        
        # Plot similarity scores
        similarities = [result['similarity'] for result in similar_results]
        plt.bar(range(len(similarities)), similarities)
        plt.title('Similarity Scores Comparison')
        plt.xlabel('Match Number')
        plt.ylabel('Similarity Score')
        plt.ylim(0, 1)
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

    def _plot_image(self, image_path, title):
        """Helper function to plot a single image"""
        try:
            img = Image.open(image_path)
            plt.imshow(img)
            plt.title(title)
            plt.axis('off')
        except Exception as e:
            self.logger.error(f"Error plotting image {image_path}: {str(e)}")
            plt.text(0.5, 0.5, 'Image Load Error',
                    horizontalalignment='center',
                    verticalalignment='center')
            plt.axis('off')

    def visualize_features(self, features, output_name=None):
        """Visualize extracted features"""
        # Generate unique filename
        if output_name is None:
            filename = self._generate_unique_filename("features", "png")
        else:
            base_name = Path(output_name).stem
            filename = self._generate_unique_filename(base_name, "png")
        
        output_path = self.output_dir / filename
        
        # Create visualization
        fig = plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        sns.histplot(features['visual'])
        plt.title('Visual Features Distribution')
        
        plt.subplot(2, 2, 2)
        if 'contour_areas' in features['shape']:
            sns.boxplot(y=features['shape']['contour_areas'])
            plt.title('Contour Areas Distribution')
        
        plt.subplot(2, 2, 3)
        plt.plot(features['color']['histogram'])
        plt.title('Color Histogram')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        self.logger.info(f"Feature visualization saved to: {output_path}")
        return output_path

# Test function
def test_visualizer():
    visualizer = ResultVisualizer()
    
    # Create dummy data
    query_path = "path/to/query.jpg"
    similar_results = [
        {"path": "path/to/result1.jpg", "similarity": 0.95},
        {"path": "path/to/result2.jpg", "similarity": 0.85},
        {"path": "path/to/result3.jpg", "similarity": 0.75}
    ]
    
    # Test PDF generation
    try:
        pdf_path = visualizer.visualize_results(
            query_path,
            similar_results,
            output_name="test_results"
        )
        print(f"PDF saved to: {pdf_path}")
    except Exception as e:
        print(f"Error generating PDF: {str(e)}")

if __name__ == "__main__":
    test_visualizer()