from fastapi import FastAPI, HTTPException, Depends, status, File, UploadFile, Form
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Optional, List, Any
import jwt
from pydantic import BaseModel, EmailStr
from bson import ObjectId
import uuid
from pathlib import Path
import shutil
import io
from PIL import Image
import tempfile
from uploadBucket import s3_upload
from checker import searcher
from test_similarity import test_searcher

app = FastAPI()

# MongoDB Connection
MONGODB_DB_URL = 'mongodb://localhost:27017'
client = AsyncIOMotorClient(MONGODB_DB_URL)
db = client.user_db
collection = db.users
image_collection = db.images
similarity_collection = db.similarity

# JWT Configuration

SECRET_KEY = "7/BNmfw5EARHgTjF"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 300

# Password Hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="signin")

#Pydantic Models

class UserCreate(BaseModel):
    username: str
    firstname: str
    lastname: str
    email: EmailStr
    password: str

class UserResponse(BaseModel):
   username: str
   firstname: str
   lastname: str
   email: EmailStr

class Token(BaseModel):
    access_token: str
    token_type: str


class ImageUpload(BaseModel):
    title: str
    description: str    

class ImageResponse(BaseModel):
    id:str
    title: str
    description: str
    image_url: str
    processed_results:Optional[dict]
    created_at: datetime   
class Test(BaseModel):
    result:dict   

class ProfileUpdate(BaseModel):
    username: Optional[str] = None
    firstname: Optional[str] = None
    lastname: Optional[str] = None
    email: Optional[EmailStr] = None  


class UploadData(BaseModel):
    title: str
    description: str
    processed_results: List[Any]
    novelty: List[Any]
    created_at: datetime  

class UserUploadsResponse(BaseModel):
    status: str
    total_uploads: int
    data: List[Any]         

#helper functions

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    ) 
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception 
    except jwt.PyJWTError:
        raise credentials_exception
    user = await collection.find_one({"email": email})
    if user is None:
        raise credentials_exception
    return user    


def generate_unique_filename(original_filename:str) -> str:
    ext = original_filename.lower().split(".")[-1]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"{timestamp}-{unique_id}.{ext}"


async def process_image(image_path:str) -> dict:
    """
    place holder for image processing logic
    replace with your own image processing logic
    """
    search = searcher(image_path)

    return search

    # return {"processed": True, 
    #         "results": "Sample processing results",
    #         "timestamp": datetime.utcnow().isoformat()}
            

@app.post("/signup", response_model=UserResponse)
async def signup(user: UserCreate):
    if await collection.find_one({"email": user.email}):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
         detail="Email already registered")
    
    # Hashing the password creating user document
    hashed_password = pwd_context.hash(user.password)
    user_dict = user.dict()
    user_dict["password"] = hashed_password
    user_dict["created_at"] = datetime.utcnow()

    # Inserting user document into the database

    await collection.insert_one(user_dict)

    return UserResponse(email=user.email,
                         username=user.username, 
                         firstname=user.firstname, 
                         lastname=user.lastname)

@app.post("/signin", response_model=Token)
async def signin(form_data: OAuth2PasswordRequestForm = Depends()):

    # find user in db
    user = await collection.find_one({"email": form_data.username})
    if not user or not pwd_context.verify(form_data.password, user["password"]):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
         detail="Incorrect email or password",
         headers={"WWW-Authenticate": "Bearer"})
    
    # create access token

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user["email"]},
                                       expires_delta=access_token_expires)
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/profile", response_model=UserResponse)
async def read_users_me(current_user: UserResponse = Depends(get_current_user)):
    return UserResponse(email=current_user["email"], username=current_user["username"],
                        firstname=current_user["firstname"], lastname=current_user["lastname"]
                        )

@app.post("/uplaod-image", response_model=ImageResponse)
async def upload_image(
    title:str = Form(...),
    description:str = Form(...),
    image: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
    ):
    try:
        if not image.content_type.startswith("image/"):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
             detail="Only images allowed") 

        # generate unique filename
        contents = image.file.read()
        image.file.seek(0)
        image_data = Image.open(io.BytesIO(contents))
        unique_filename = generate_unique_filename(image.filename)

        # read and upload file to s3
        #  contents = await image.read()   
        # 
        bucket_name = 'trademarkdesign'
        # upload to s3
        s3_upload(image.file, bucket_name, unique_filename)  

        # s3 url
        image_url = f"https://trademarkdesign.s3.eu-north-1.amazonaws.com/{unique_filename}"

        # create image document  

        processing_results = await process_image(image.file)

        image_doc = {
            "user_id": str(current_user["_id"]),
            "title": title,
            "description": description,
            "image_url": image_url,
            "filename": unique_filename,
            "processed_results": processing_results,
            "created_at": datetime.utcnow()
        }

        result = await image_collection.insert_one(image_doc)

        return {
            "id": str(result.inserted_id),
            "title": title,
            "description": description,
            "image_url": image_url,
            "processed_results": processing_results,
            "created_at": image_doc["created_at"]
        }

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=str(e))
    finally:
        image.file.close()

# @app.post("/searcher")
# async def upload_image(
#     title:str = Form(...),
#     description:str = Form(...),
#     current_user: dict = Depends(get_current_user),
#     image: UploadFile = File(...)
#     ):
#     try:
#          if not image.content_type.startswith("image/"):
#             raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
#              detail="Only images allowed")
         
#          contents = await image.read()
#          new_image = image.file
#         #  image.file.seek(0)
#          image_data = Image.open(io.BytesIO(contents))
#          unique_filename = generate_unique_filename(image.filename)

#         # read and upload file to s3
#         #  contents = await image.read()   
#         # 
#          bucket_name = 'trademarkdesign'
#         # upload to s3
         
#          processing_results =  searcher(new_image)
#          formatted_results = []
#          is_novel = []

#          if any(result['novel'] == False for result in processing_results):
#              is_novel.append({'is_novel': False})
#          else:
#              is_novel.append({'is_novel': True})

#          for result in processing_results:
#              formatted_result = {
#                     'path': result['path'],
#                     'metadata': {
#                         'original_filename': result['metadata']['original_filename'],
#                         's3_filename': result['metadata']['s3_filename'],
#                         'vienna_codes': result['metadata']['vienna_codes'],
#                         'year': result['metadata']['year'],
#                         'text': result['metadata']['text']
#                     },
#                     'similarity': result['similarity'],
#              }
#              formatted_results.append(formatted_result)
#          s3_upload(new_image, bucket_name, unique_filename)  

#         # s3 url
#          image_url = f"https://trademarkdesign.s3.eu-north-1.amazonaws.com/{unique_filename}"
#             # add to DB
#          similarity_result = {
#             "user_id": str(current_user["_id"]),
#             "title": title,
#             "description": description,
#             "queried_image":image_url,
#             "processed_results": formatted_results,
#             "novelty": is_novel,
#             "created_at": datetime.utcnow()
#         } 
         
#          result = await similarity_collection.insert_one(similarity_result)


#          # Return statement moved outside the loop
#          return {
#             "status": "success",
#             "queried_image":image_url,
#             "results": formatted_results,
#             "novelty": is_novel,
#             "message": f"Found {len(formatted_results)} similar images"
#         }
    
    

#     except Exception as e:
#         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#                             detail=str(e))
#     finally:
#         image.file.close()


@app.post("/searcher")
async def upload_image(
    title: str = Form(...),
    description: str = Form(...),
    current_user: dict = Depends(get_current_user),
    image: UploadFile = File(...)
):
    # Initialize file_object to None outside try block
    file_object = None
    
    try:
        if not image.content_type.startswith("image/"):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
             detail="Only images allowed")
         
        # Read contents and create a new file-like object
        contents = image.file.read()
        file_object = io.BytesIO(contents)
        
        # Get unique filename
        unique_filename = generate_unique_filename(image.filename)
        bucket_name = 'trademarkdesign'

        # Process with searcher
        processing_results = searcher(file_object)
        
        # Reset file position for S3 upload
        file_object.seek(0)
        
        # Upload to S3
        s3_upload(file_object, bucket_name, unique_filename)

        formatted_results = []
        is_novel = []

        if any(result['novel'] == False for result in processing_results):
            is_novel.append({'is_novel': False})
        else:
            is_novel.append({'is_novel': True})

        for result in processing_results:
            formatted_result = {
                'path': result['path'],
                'metadata': {
                    'original_filename': result['metadata']['original_filename'],
                    's3_filename': result['metadata']['s3_filename'],
                    'vienna_codes': result['metadata']['vienna_codes'],
                    'year': result['metadata']['year'],
                    'text': result['metadata']['text']
                },
                'similarity': result['similarity'],
            }
            formatted_results.append(formatted_result)

        # s3 url
        image_url = f"https://trademarkdesign.s3.eu-north-1.amazonaws.com/{unique_filename}"
        
        # add to DB
        similarity_result = {
            "user_id": str(current_user["_id"]),
            "title": title,
            "description": description,
            "queried_image": image_url,
            "processed_results": formatted_results,
            "novelty": is_novel,
            "created_at": datetime.utcnow()
        }
        
        result = await similarity_collection.insert_one(similarity_result)

        return {
            "status": "success",
            "queried_image": image_url,
            "results": formatted_results,
            "novelty": is_novel,
            "message": f"Found {len(formatted_results)} similar images"
        }

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=str(e))
    finally:
        # Only close if file_object was created
        if file_object:
            file_object.close()
        image.file.close()


@app.put("/update_profile")
async def update_profile(
    update_data: ProfileUpdate,
    current_user: dict = Depends(get_current_user)
):
    try:
        # Prepare update data
        update_fields = {}

        # Handle firstname update
        if update_data.firstname is not None:
            update_fields["firstname"] = update_data.firstname

        # Handle lastname update
        if update_data.lastname is not None:
            update_fields["lastname"] = update_data.lastname

        # Handle username update
        if update_data.username is not None:
            # Check if username already exists
            if update_data.username != current_user.get('username'):
                existing_user = await db.users.find_one({"username": update_data.username})
                if existing_user:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Username already taken"
                    )
            update_fields["username"] = update_data.username

        # Handle email update
        if update_data.email is not None:
            # Check if email already exists
            if update_data.email != current_user.get('email'):
                existing_user = await db.users.find_one({"email": update_data.email})
                if existing_user:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Email already registered"
                    )
            update_fields["email"] = update_data.email

        # If no fields to update, return error
        if not update_fields:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No fields to update provided"
            )

        # Update user profile
        result = await db.users.update_one(
            {"_id": current_user.get('_id')},  # Changed to dictionary access
            {"$set": update_fields}
        )

        if result.modified_count == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Update failed"
            )

        # Get updated user data
        updated_user = await db.users.find_one({"_id": current_user.get('_id')})
        
        return {
            "status": "success",
            "message": "Profile updated successfully",
            "data": {
                "firstname": updated_user.get("firstname"),
                "lastname": updated_user.get("lastname"),
                "username": updated_user.get("username"),
                "email": updated_user.get("email")
            }
        }

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating profile: {str(e)}"
        )
    

@app.get("/past-search", response_model=UserUploadsResponse)
async def get_user_uploads(
    current_user: dict = Depends(get_current_user)
):
    try:
        # Get all user's uploads

        user_id = str(current_user.get('_id'))

        cursor = db.similarity.find(
            {"user_id": user_id}  # Use string user_id
        ).sort("created_at", -1)

        # cursor = db.similarity.find(
        #     {"user_id": current_user.get('_id')},
        #     {
        #         "_id": 1,
        #         "title": 1,
        #         "description": 1,
        #         "processed_results": 1,
        #         "novelty": 1,
        #         "created_at": 1
        #     }
        # ).sort("created_at", -1)
        
        # Convert cursor to list
        user_uploads = await cursor.to_list(None)  # None means no limit
        
        # Get total count
        total_uploads = len(user_uploads)

        if not user_uploads:
            return {
                "status": "success",
                "total_uploads": 0,
                "data": []
            }

        # Format the response
        formatted_uploads = []
        for upload in user_uploads:
            formatted_upload = {
                "id": str(upload.get("_id")),
                "title": upload.get("title"),
                "description": upload.get("description"),
                "processed_results": upload.get("processed_results", []),
                "novelty": upload.get("novelty", []),
                "created_at": upload.get("created_at")
            }
            formatted_uploads.append(formatted_upload)

        return {
            "status": "success",
            "total_uploads": total_uploads,
            "data": formatted_uploads
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving uploads: {str(e)}"
        )