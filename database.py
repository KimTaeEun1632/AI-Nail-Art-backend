import os
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from fastapi import HTTPException

# 프로젝트 루트 디렉토리 기준으로 절대 경로 설정
base_dir = Path(__file__).resolve().parent
SQLALCHEMY_DATABASE_URL = f"sqlite:///{base_dir / 'nail_art.db'}"
print(f"Database URL: {SQLALCHEMY_DATABASE_URL}")  # 디버깅용 출력

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        db.rollback()
        raise Exception(f"Database error: {str(e)}")
    finally:
        db.close()