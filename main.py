from fastapi import FastAPI, HTTPException, Depends, status, File, UploadFile, Form
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Optional
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

@app.get("/me", response_model=UserResponse)
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

        # upload to s3
        s3_upload(image.file, unique_filename)  

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

@app.post("/searcher")
async def upload_image(
    image: UploadFile = File(...)
    ):
    try:
         if not image.content_type.startswith("image/"):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
             detail="Only images allowed")
         
         contents = image.file.read()
         image.file.seek(0)
         image_data = Image.open(io.BytesIO(contents))
        
         
         processing_results = searcher(image.file)
         new_results = processing_results

        #  for i, result in enumerate(processing_results, 1):
        #      new_results.append({'path': result['path'], 'similarity': result['similarity']}) 

         return {
             
            #  "path":str (processing_results['path']),
            #  "similarity":str (processing_results['similarity']),
            #  "distance":str( processing_results['distance']),
            
            # "result":dict(new_results),
            "result": [
                {
                    'path': str(result['path']),
                    'similarity': str(result['similarity']),
                    'distance': str(result['distance'])
                }
                for result in new_results
            ]

         }
             


    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e))      

