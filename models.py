# backend/models.py
from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Boolean
from sqlalchemy.orm import relationship
from backend.database import Base  # database.py에서 Base 임포트
from datetime import datetime

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    nickname = Column(String, nullable=False)
    hashed_password = Column(String)
    images = relationship("Image", back_populates="user")
    refresh_tokens = relationship("RefreshToken", back_populates="user")

class Image(Base):
    __tablename__ = "images"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    file_path = Column(String)
    prompt = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="images")
    is_bookmarked = Column(Boolean, default=False) #북마크 필드 추가, 기본 false

class RefreshToken(Base):
    __tablename__ = "refresh_tokens"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    token = Column(String, index=True)
    expires_at = Column(DateTime)
    user = relationship("User", back_populates="refresh_tokens")