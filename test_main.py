from feature_extractor import FeatureExtractor
from index_builder import IndexBuilder
from similarity_searcher import SimilaritySearcher
import json
from test_similarity import test_searcher
from test_visualizer import test_visualizer
from json_search import JsonSearch
from fpdf import FPDF
import io
import boto3
import matplotlib.image as mpimg
import requests
from PIL import Image
from io import BytesIO
import tempfile
import os
from image_parse import bucketImageParse


pdf = FPDF()

pdf.add_page()

pdf.set_font("Arial", size = 14, style = 'B')
pdf.cell(200, 10, txt = "Trademark Information", ln = True, align = 'C')
pdf.cell(200, 10, txt = "The Trademark design is similar to: ", ln = True, align = 'C')




# input_path = str(input("Enter the path to the input image: "))
input_path = bucketImageParse('trademarkdesign', 'logo2.png')
# array = []
json_path = './L3D-dataset/dataset/results.json'
# def test_searcher():
#     # Initialize components
#     extractor = FeatureExtractor()
#     indexer = IndexBuilder()

#     try:
#         # Load index
#         index, features_file = indexer.load_index()
#         searcher = SimilaritySearcher(index, features_file)

#         # Test search with a query image
#         query_path = str(input_path)  # Add your test image
#         query_features = extractor.extract_features(query_path)

#         # Search
#         results = searcher.search(query_features, k=5)

#         print("Search Results:")
#         for i, result in enumerate(results, 1):
#             # print(f"\nResult {i}:")
#             array.append({"Result": i})
#             # array.append([result['path'], result['similarity'], result['distance']])
#             # print(f"Path: {result['path']}")
#             array.append({"Path": result['path']})
#             # print(f"Similarity: {result['similarity']:.3f}")
#             array.append({"Similarity": result['similarity']})
#             # print(f"Distance: {result['distance']:.3f}")
#             array.append({"Distance": result['distance']})

#     except Exception as e:
#         print(f"Error during search: {str(e)}")

#     print (array)
#     json.dumps(array)
# if __name__ == "__main__":
#     test_searcher(input_path)
count = 0
results = test_searcher(input_path)
new_results = []

print("Search Results:")
for i, result in enumerate(results, 1):
    count += 1
    print(f"\nResult {i}:")
    print(f"Path: {result['path']}")
    print(f"Similarity: {result['similarity']:.3f}")
    print(f"Distance: {result['distance']:.3f}")
    new_results.append({"path": result['path'], "similarity": result['similarity']})
# print(new_results.get("Path"))

#for exporting image
test_visualizer(input_path, new_results)

search = JsonSearch(json_path)
# result = search.find_obj("file","3d454bc0-eeb7-4dc5-bb82-22d36f361e95.JPG")
print (result)

# with open(new_results, 'r') as file:
#     data = json.load(file)



#trim path to return just image name
    
def trim_path(path):
       return path[61:-4] if len(path) > 61 else ""  
  
empty_result = []

print(f'trimming test{trim_path(result["path"])}')

for i in new_results:
    result = search.find_obj("file", trim_path(i["path"])+".JPG")
    # print (result)
    # if result is not None:
    #     empty_result.append(result)
    # else:
    #     print("No result found")
    # print(empty_result)
    print(f"trademark name: {result['text']} \n registration year: {result['year']} \n vienna codes: {result['vienna_codes']} \n image path: {result['file']} \n similarity score: {i['similarity']}")
    empty_result.append({"Trademark Name": result['text'], "Registration Year": result['year'], "Vienna Codes": result['vienna_codes'], "Image Path": result['file'], "Similarity Score": i['similarity']}) 

print(empty_result)
    # vienna_codes
    # year
    # file   

image_path ='C:/Users/jajim/Desktop/dissertation/trade/L3D-dataset/dataset/images/'    

for i in empty_result:
    # pdf.set_font("Arial", size = 14, style = 'B')
    #pdf.cell(200, 10, txt = "Trademark Information", ln = True, align = 'C')
    #pdf.cell(200, 10, txt = "The Trademark design is similar to: ", ln = True, align = 'C')
    pdf.cell(200, 10, txt = f"Trademark Name: {i['Trademark Name']}", ln = True, align = 'l')
    pdf.cell(200, 10, txt = f"Registration Year: {i['Registration Year']}", ln = True, align = 'l')
    pdf.cell(200, 10, txt = f"Vienna Codes: {i['Vienna Codes']}", ln = True, align = 'l')
    # pdf.cell(200, 10, txt = f"Image Path: {i['Image Path']}", ln = True, align = 'l')
    pdf.image(image_path+i['Image Path'], x = 10, y = None, w = 100, h = 100, type = '', link = '')
    pdf.cell(200, 10, txt = f"Similarity Score: {i['Similarity Score']}", ln = True, align = 'l')
    pdf.cell(200, 10, txt = " ", ln = True, align = 'C')

pdf.output("Result.pdf")
# os.remove(temp)


