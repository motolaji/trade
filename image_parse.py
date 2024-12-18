import os
import boto3
from PIL import Image
import io
from io import BytesIO
import requests
from bucket import bucketImageLoader
import tempfile


# image = load_image('trademarkdesign', 'logo1.png').show()

# image_convert = image.convert('RGB')

def bucketImageParse(bucket, key):
    try:
        image = bucketImageLoader(bucket, key)
        converted_image = image.convert('RGB')
        temp = tempfile.NamedTemporaryFile(delete=False)
        converted_image.save(temp.name, format="JPEG")
        return temp.name
    except Exception as e:
        print(f"Error saving image to temp file: {str(e)}")
        raise ValueError("Error saving image to temp file")






# image = ('trademarkdesign', 'logo1.png').show()
