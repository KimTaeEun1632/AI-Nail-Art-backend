# backend/app.py
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from database import SessionLocal, get_db, engine, Base
from models import User, Image, RefreshToken
from pydantic import BaseModel, EmailStr, Field
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from fastapi.middleware.cors import CORSMiddleware
from diffusers import StableDiffusionPipeline
import torch
import io
import base64
from fastapi.responses import JSONResponse
import os
import shutil
from dotenv import load_dotenv
import secrets
from pathlib import Path
from fastapi.staticfiles import StaticFiles

# .env 파일 로드
load_dotenv()

# 테이블 생성
Base.metadata.create_all(bind=engine)

app = FastAPI()

# 정적 파일 제공 설정
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Stable Diffusion 설정
model_id = "runwayml/stable-diffusion-v1-5"
try:
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    pipe.to("cpu")
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise e

# JWT 설정
SECRET_KEY = os.getenv("SECRET_KEY")
print(f"SECRET_KEY: {SECRET_KEY}")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Pydantic 모델
class UserCreate(BaseModel):
    email: EmailStr
    nickname: str  # nickname 필드 추가
    password: str = Field(..., min_length=8)

class UserOut(BaseModel):
    id: int
    email: str
    nickname: str  # nickname 필드 추가

    class Config:
        from_attributes = True

class UserInToken(BaseModel):
    id: int
    email: str
    nickname: str

    class Config:
        from_attributes = True

class ImageOut(BaseModel):
    id: int
    file_path: str
    prompt: str
    created_at: datetime
    is_bookmarked: bool

    class Config:
        from_attributes = True

class Token(BaseModel):
    user: UserInToken
    refreshToken: str
    accessToken: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class RefreshRequest(BaseModel):
    refresh_token: str

# 유틸리티 함수
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def create_refresh_token(user_id: int, db: Session) -> str:
    print(f"Creating refresh token for user_id: {user_id}")
    token = secrets.token_urlsafe(32)
    expires_at = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    db_token = RefreshToken(
        user_id=user_id,
        token=token,
        expires_at=expires_at
    )
    db.add(db_token)
    db.commit()
    db.refresh(db_token)
    return token

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
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
    except JWTError:
        raise credentials_exception
    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise credentials_exception
    return user

# 기본 엔드포인트
@app.get("/")
def home():
    return {"message": "AI Nail Art Generator is running!"}

# 이미지 생성
@app.get("/generate")
def generate_image(
    prompt: str,
    num_images: int = 4,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not prompt or not prompt.strip():
        return JSONResponse(status_code=400, content={"error": "Prompt is required"})
    try:
        print(f"Generating images for prompt: {prompt}, num_images: {num_images}")
        generated_images = pipe(prompt, num_images_per_prompt=num_images).images
        images_response = []

        # Set up storage directory
        base_dir = Path(__file__).resolve().parent
        upload_dir = base_dir / "uploads"
        os.makedirs(upload_dir, exist_ok=True)

        for idx, image in enumerate(generated_images):
            # Create unique filename with timestamp
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            file_name = f"{current_user.id}_{timestamp}_{idx}.png"
            file_path = os.path.join(upload_dir, file_name)

            # Save image to server
            image.save(file_path, format="PNG")

            # Record in database
            relative_file_path = f"uploads/{file_name}"
            new_image = Image(
                user_id=current_user.id,
                file_path=relative_file_path,
                prompt=prompt,
                created_at=datetime.utcnow()
            )
            db.add(new_image)
            db.commit()
            db.refresh(new_image)

            # Convert to base64 for response
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format="PNG")
            img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")
            images_response.append({
                "id": new_image.id,
                "base64": img_base64,
                "file_path": relative_file_path,
                "prompt": prompt,
                "created_at": new_image.created_at.isoformat()
            })

        return JSONResponse(content={"images": images_response})
    except Exception as e:
        print(f"Error generating images: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})
    
    # 이미지 저장    
@app.post("/images/save")
async def save_image(prompt: str = Form(...),file: UploadFile = File(...),current_user: User = Depends(get_current_user),db: Session = Depends(get_db)):
    try:
        # backend 디렉토리를 기준으로 uploads 디렉토리 설정
        base_dir = Path(__file__).resolve().parent
        upload_dir = base_dir / "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, f"{current_user.id}_{file.filename}")
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 데이터베이스에는 상대 경로 저장 (uploads/1_image.png)
        relative_file_path = f"uploads/{current_user.id}_{file.filename}"
        new_image = Image(user_id=current_user.id, file_path=relative_file_path, prompt=prompt)
        db.add(new_image)
        db.commit()
        db.refresh(new_image)
        return {"message": "Image saved successfully", "file_path": relative_file_path}
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error in save_image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save image: {str(e)}")    

# 회원가입
@app.post("/signup", response_model=UserOut)
def signup(user: UserCreate, db: Session = Depends(get_db)):
    try:
        print(f"Received signup request: email={user.email}, nickname={user.nickname}, password={user.password}")
        db_user = db.query(User).filter(User.email == user.email).first()
        if db_user:
            raise HTTPException(status_code=400, detail="Email already registered")
        hashed_password = get_password_hash(user.password)
        print(f"Hashed password: {hashed_password}")
        new_user = User(email=user.email, nickname=user.nickname, hashed_password=hashed_password)
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        return new_user
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error in signup: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to sign up: {str(e)}")

# 로그인
@app.post("/login", response_model=Token)
def login(request: LoginRequest, db: Session = Depends(get_db)):
    try:
        print(f"Received login request: email={request.email}, password={request.password}")
        user = db.query(User).filter(User.email == request.email).first()
        print(f"User found: {user}")
        if not user:
            print("User not found")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        print(f"Verifying password for user: {user.email}")
        if not verify_password(request.password, user.hashed_password):
            print("Password verification failed")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        print("Password verified successfully")
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        print(f"Creating access token with SECRET_KEY: {SECRET_KEY}")
        access_token = create_access_token(
            data={"sub": user.email}, expires_delta=access_token_expires
        )
        print(f"Access token created: {access_token}")
        refresh_token = create_refresh_token(user.id, db)
        print(f"Refresh token created: {refresh_token}")
        return {
            "user": user,
            "refreshToken": refresh_token,
            "accessToken": access_token,
        }
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error in login: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to login: {str(e)}")

# 리프레시 토큰으로 새 액세스 토큰 발급
@app.post("/refresh", response_model=Token)
def refresh_token(request: RefreshRequest, db: Session = Depends(get_db)):
    try:
        refresh_token = request.refresh_token
        print(f"Received refresh token: {refresh_token}")
        db_token = db.query(RefreshToken).filter(RefreshToken.token == refresh_token).first()
        if not db_token:
            raise HTTPException(status_code=401, detail="Invalid refresh token")
        
        print(f"db_token.expires_at: {db_token.expires_at}, type: {type(db_token.expires_at)}")
        print(f"datetime.utcnow(): {datetime.utcnow()}, type: {type(datetime.utcnow())}")
        if db_token.expires_at < datetime.utcnow():
            print("Token is expired, deleting...")
            db.delete(db_token)
            db.commit()
            raise HTTPException(status_code=401, detail="Refresh token expired")
        
        user = db.query(User).filter(User.id == db_token.user_id).first()
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.email}, expires_delta=access_token_expires
        )
        db.delete(db_token)
        db.commit()
        new_refresh_token = create_refresh_token(user.id, db)
        
        return {
            "user": user,
            "refreshToken": new_refresh_token,
            "accessToken": access_token,
        }
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error in refresh_token: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to refresh token: {str(e)}")

# 로그아웃
@app.post("/logout")
def logout(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        db.query(RefreshToken).filter(RefreshToken.user_id == current_user.id).delete()
        db.commit()
        return {"message": "Logged out successfully"}
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error in logout: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to logout: {str(e)}")

# 내 라이브러리 조회
@app.get("/images/my-library", response_model=list[ImageOut])
def get_my_library(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        images = db.query(Image).filter(Image.user_id == current_user.id).order_by(Image.created_at.desc()).all()
        return images
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch library: {str(e)}")

# 북마크 
@app.post("/images/bookmark/{image_id}")
def toggle_bookmark(image_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        image = db.query(Image).filter(Image.id == image_id, Image.user_id == current_user.id).first()
        if not image:
            raise HTTPException(status_code=404, detail="이미지를 찾을 수 없습니다.")
        
        image.is_bookmarked = not image.is_bookmarked  # 북마크 상태 토글
        db.commit()
        db.refresh(image)
        return {"message": "북마크 상태가 업데이트되었습니다.", "is_bookmarked": image.is_bookmarked}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"북마크 업데이트 실패: {str(e)}")
    
# 북마크된 이미지 조회
@app.get("/images/bookmarked", response_model=list[ImageOut])
def get_bookmarked_images(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        images = db.query(Image).filter(Image.user_id == current_user.id, Image.is_bookmarked == True).all()
        return images
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"북마크된 이미지 조회 실패: {str(e)}")