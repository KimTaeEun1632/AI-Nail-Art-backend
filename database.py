# backend/database.py
import os
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from fastapi import HTTPException

# 프로젝트 루트 디렉토리 기준으로 절대 경로 설정
BASE_DIR = Path(__file__).resolve().parent
SQLALCHEMY_DATABASE_URL = f"sqlite:///{BASE_DIR}/nail_art.db"

# 데이터베이스 엔진 생성
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

# 세션 팩토리 생성
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base 클래스 정의
Base = declarative_base()

# 의존성 주입 함수
def get_db():
    db = SessionLocal()
    try:
        yield db
    except HTTPException as http_exc:
        # HTTPException은 변환하지 않고 그대로 전달
        raise http_exc
    except Exception as e:
        db.rollback()
        raise Exception(f"Database error: {str(e)}")
    finally:
        db.close()