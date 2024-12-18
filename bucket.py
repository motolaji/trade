import boto3
from PIL import Image
import io

def bucketImageLoader(bucket, key):
    try:
        s3 = boto3.resource('s3',
           aws_access_key_id='',
           aws_secret_access_key=''       
                  )


        bucket = s3.Bucket(bucket)
        image = bucket.Object(key)
        img_data = image.get().get('Body').read()
    

        return Image.open(io.BytesIO(img_data))
    except Exception as e:
        print(f"Error loading image from aws s3: {str(e)}")
        raise

        

       



