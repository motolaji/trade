from feature_extractor import FeatureExtractor
from index_builder import IndexBuilder
from similarity_searcher import SimilaritySearcher
from PIL import Image
from io import BytesIO
import requests
import tempfile







# response = requests.get("https://www.freeiconspng.com/uploads/twitter-icon--simple-rounded-social-iconset--graphicsvibe-4.png")
# image_bytes = BytesIO(response.content)
# image_data = Image.open(image_bytes)
# image_convert = image_data.convert('RGB')
# image_convert.show()


# def temp_file(image):
#     try:
#         temp = tempfile.NamedTemporaryFile(delete=False)
#         image.save(temp.name, format="JPEG")
#         return temp.name
#     except Exception as e:
#         print(f"Error saving image to temp file: {str(e)}")
#         raise ValueError("Error saving image to temp file")

# image_convert.save('C:/Users/jajim/Desktop/dissertation/trade/L3D-dataset/dataset/images/new.jpg')


def test_searcher(query_path):
    # Initialize components
    extractor = FeatureExtractor()
    indexer = IndexBuilder()

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
            # print(f"\nResult {i}:")
            # print(f"Path: {result['path']}")
            # print(f"Similarity: {result['similarity']:.3f}")
            # print(f"Distance: {result['distance']:.3f}")
            new_results.append({'path': result['path'], 'similarity': result['similarity'], 'distance': result['distance']}) 

        print(new_results)    
        return new_results
# taking this out placing in test_main.py
    except Exception as e:
        print(f"Error during search: {str(e)}")


# if __name__ == "__main__":
#     test_searcher()
