import boto3
import os
from botocore.exceptions import ClientError
from PIL import Image



def s3_upload(image, filename=None):
    # object_name = os.path.basename(image)
    bucket = 'trademarkdesign'
    try:
        s3_client = boto3.client('s3',
           aws_access_key_id='',
           aws_secret_access_key='')
            
        
        # s3_client.upload_file(image, bucket, filename)
        
        s3_client.upload_fileobj(image, bucket, filename)

        return "Success"
    except Exception as e:
        print(f"Error uploading image to aws s3: {str(e)} \n {filename}")
        raise
    

# image = Image.open('../logo1.png', 'rb') 

# s3_upload(f.read, filename='testy')
    










# def bucketImageLoader(bucket, key):
#     try:
#         s3 = boto3.resource('s3',
#            aws_access_key_id='',
#            aws_secret_access_key=''       
#                   )


#         bucket = s3.Bucket(bucket)
#         image = bucket.Object(key)
#         img_data = image.get().get('Body').read()
    

#         return Image.open(io.BytesIO(img_data))
#     except Exception as e:
#         print(f"Error loading image from aws s3: {str(e)}")
#         raise
