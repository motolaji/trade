from fastapi import FastAPI, HTTPException, Depends, status, File, UploadFile, Form, Body
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
import asyncio
import os
from revised_index import IndexBuilder
from feature_extractor import FeatureExtractor


app = FastAPI()

# MongoDB Connection
MONGODB_DB_URL = 'mongodb://localhost:27017'
client = AsyncIOMotorClient(MONGODB_DB_URL)
db = client.user_db
collection = db.users
image_collection = db.images
similarity_collection = db.similarity
index_history_collection = db.index_history

# JWT Configuration

SECRET_KEY = "7/BNmfw5EARHgTjF"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 300

# Password Hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="signin")


 #Global lock and status tracking
indexing_lock = asyncio.Lock()
indexing_status = {
    "is_indexing": False,
    "current_indexer": None,
    "start_time": None
}

#Pydantic Models

class UserCreate(BaseModel):
    username: str
    firstname: str
    lastname: str
    email: EmailStr
    password: str

class UserSignin(BaseModel):
    username: str
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



# Admin models
class AdminCreate(BaseModel):
    email: EmailStr
    password: str
    firstname: str
    lastname: str

class AdminLogin(BaseModel):
    email: EmailStr
    password: str

class AdminResponse(BaseModel):
    id: str
    email: str
    firstname: str
    lastname: str
    created_at: datetime          

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


async def check_admin(current_user: dict = Depends(get_current_user)):
    if not current_user.get("is_admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized. Admin access required."
        )
    return current_user

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
async def signin(form_data: UserSignin ):

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

@app.post("/searcher")
async def upload_image(
    title: str = Form(...),
    description: str = Form(...),
    image: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
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
            "message": f"Found {len(formatted_results)} similar trademark designs"
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
    
@app.post("/admin/index")
async def index_images(
    dataset: UploadFile = File(...),
    metadata_json: UploadFile = File(...),
    current_user: dict = Depends(check_admin)
):
    # Create temporary directories to store uploads
    temp_dataset_dir = None
    temp_json_path = None
    
    try:
        if indexing_status["is_indexing"]:
            return {
                "status": "error",
                "message": f"Indexing already in progress by {indexing_status['current_indexer']} since {indexing_status['start_time']}"
            }

        if not await indexing_lock.acquire():
            return {
                "status": "error",
                "message": "Could not acquire indexing lock"
            }

        start_time = datetime.utcnow()
        indexing_details = {
            "start_time": start_time,
            "indexed_by": str(current_user["_id"]),
            "indexer_email": current_user.get("email", "Unknown"),
            "status": "in_progress",
            "dataset_name": dataset.filename,
            "metadata_file": metadata_json.filename
        }

        # Insert initial record
        history_id = await index_history_collection.insert_one(indexing_details)

        # Create temporary directory for dataset
        temp_dataset_dir = tempfile.mkdtemp()
        dataset_path = Path(temp_dataset_dir)

        # Create temporary file for JSON
        temp_json_fd, temp_json_path = tempfile.mkstemp(suffix='.json')
        os.close(temp_json_fd)

        try:
            # Update indexing status
            indexing_status.update({
                "is_indexing": True,
                "current_indexer": current_user.get("email", "Unknown"),
                "start_time": start_time
            })

            # Save JSON file
            with open(temp_json_path, 'wb') as json_file:
                shutil.copyfileobj(metadata_json.file, json_file)

            # Extract dataset if it's a zip file
            if dataset.filename.endswith('.zip'):
                import zipfile
                with zipfile.ZipFile(dataset.file) as zip_ref:
                    zip_ref.extractall(temp_dataset_dir)
            else:
                # Save individual file
                dataset_file_path = os.path.join(temp_dataset_dir, dataset.filename)
                with open(dataset_file_path, 'wb') as dataset_file:
                    shutil.copyfileobj(dataset.file, dataset_file)

            # Initialize components
            extractor = FeatureExtractor()
            indexer = IndexBuilder()

            # Get image paths
            image_paths = []
            for ext in ('*.jpg', '*.jpeg', '*.png'):
                image_paths.extend(dataset_path.glob(ext))

            if not image_paths:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No images found in uploaded dataset"
                )

            # Build index
            num_indexed = indexer.build_index(image_paths, temp_json_path, extractor)

            # Load and verify index
            index, features_file = indexer.load_index()
            total_vectors = index.ntotal
            features_file.close()

            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()

            # Update index history with results
            await index_history_collection.update_one(
                {"_id": history_id.inserted_id},
                {
                    "$set": {
                        "end_time": end_time,
                        "duration_seconds": duration,
                        "images_indexed": num_indexed,
                        "total_vectors": total_vectors,
                        "status": "completed",
                        "success": True,
                        "image_count": len(image_paths)
                    }
                }
            )

            return {
                "status": "success",
                "message": f"Successfully indexed {num_indexed} images",
                "total_vectors": total_vectors,
                "duration_seconds": duration,
                "index_id": str(history_id.inserted_id)
            }

        finally:
            # Cleanup temporary files
            if temp_dataset_dir and os.path.exists(temp_dataset_dir):
                shutil.rmtree(temp_dataset_dir)
            if temp_json_path and os.path.exists(temp_json_path):
                os.remove(temp_json_path)

            # Reset indexing status and release lock
            indexing_status.update({
                "is_indexing": False,
                "current_indexer": None,
                "start_time": None
            })
            indexing_lock.release()

    except Exception as e:
        # Update index history with error
        if 'history_id' in locals():
            await index_history_collection.update_one(
                {"_id": history_id.inserted_id},
                {
                    "$set": {
                        "end_time": datetime.utcnow(),
                        "status": "failed",
                        "error": str(e),
                        "success": False
                    }
                }
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during indexing: {str(e)}"
        )    
    

@app.get("/admin/index-history")
async def get_index_history(
    current_admin: dict = Depends(check_admin),
    limit: int = 10
):
    try:
        cursor = index_history_collection.find({}).sort("start_time", -1).limit(limit)
        history = await cursor.to_list(length=limit)
        
        formatted_history = []
        for record in history:
            formatted_record = {
                "id": str(record["_id"]),
                "start_time": record["start_time"],
                "end_time": record.get("end_time"),
                "duration_seconds": record.get("duration_seconds"),
                "indexed_by": record["indexer_email"],
                "status": record["status"],
                "images_indexed": record.get("images_indexed"),
                "total_vectors": record.get("total_vectors"),
                "success": record.get("success", False),
                "error": record.get("error")
            }
            formatted_history.append(formatted_record)

        return {
            "status": "success",
            "data": formatted_history
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving index history: {str(e)}"
        )

@app.get("/admin/index-history/{index_id}")
async def get_index_details(
    index_id: str,
    current_admin: dict = Depends(check_admin)
):
    try:
        record = await index_history_collection.find_one({"_id": ObjectId(index_id)})
        
        if not record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Index record not found"
            )

        return {
            "status": "success",
            "data": {
                "id": str(record["_id"]),
                "start_time": record["start_time"],
                "end_time": record.get("end_time"),
                "duration_seconds": record.get("duration_seconds"),
                "indexed_by": record["indexer_email"],
                "status": record["status"],
                "images_indexed": record.get("images_indexed"),
                "total_vectors": record.get("total_vectors"),
                "success": record.get("success", False),
                "error": record.get("error")
            }
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving index details: {str(e)}"
        )
    
# Admin Auth Routes
@app.post("/admin/signup", response_model=AdminResponse)
async def admin_signup(admin: AdminCreate):
    try:
        existing_admin = await collection.find_one({"email": admin.email})
        if existing_admin:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )


        hashed_password = pwd_context.hash(admin.password)
        # hashed_password = bcrypt.hashpw(admin.password.encode('utf-8'), bcrypt.gensalt())
        
        admin_doc = {
            "email": admin.email,
            "password": hashed_password,
            "firstname": admin.firstname,
            "lastname": admin.lastname,
            "is_admin": True,
            "created_at": datetime.utcnow()
        }

        result = await collection.insert_one(admin_doc)
        created_admin = await collection.find_one({"_id": result.inserted_id})

        return {
            "id": str(created_admin["_id"]),
            "email": created_admin["email"],
            "firstname": created_admin["firstname"],
            "lastname": created_admin["lastname"],
            "created_at": created_admin["created_at"]
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/admin/login")
async def admin_login(form_data: OAuth2PasswordRequestForm = Depends()):
    try:
        user = await collection.find_one({"email": form_data.username})
        
        if not user or not user.get("is_admin", False):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid admin credentials"
            )
        if not user or not pwd_context.verify(form_data.password, user["password"]):
        # if not bcrypt.checkpw(form_data.password.encode('utf-8'), user["password"]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid admin credentials"
            )

        token_data = {
            "sub": str(user["_id"]),
            "email": user["email"],
            "is_admin": True
        }
        
        access_token = create_access_token(token_data)

        return {
            "access_token": access_token,
            "token_type": "bearer"
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )    