import json

json_path = './L3D-dataset/dataset/results.json'


class JsonSearch:
    def __init__(self, file_path):
        self.file_path = file_path
        self.json_data = self.load_json()

    def load_json(self):
        # in an error handler

        try:
            with open(self.file_path, 'r') as file:
                print("Loading json file")
                return json.load(file)
        except Exception as e:
            print(f"Error loading json: {str(e)}")
            return None

    def find_obj(self, key, value):

        try:
            if not self.json_data:
                print("No json data")
                return None
            return next((obj for obj in self.json_data if obj.get(key) == value), None)
        except Exception as e:
            print(f"Error finding object: {str(e)}")
            return None        
        

# search = JsonSearch(json_path)
# result = search.find_obj("file","01b22df4-e107-4ab9-b97a-6d5246210c74.JPG")
# print (result)

