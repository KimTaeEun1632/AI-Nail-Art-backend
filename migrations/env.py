import sys
import os
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context

# 프로젝트 루트 디렉토리를 sys.path에 추가
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_dir)
print(f"Added to sys.path: {base_dir}")

# 로깅 설정
import logging
logging.basicConfig()
logging.getLogger('alembic').setLevel(logging.INFO)
logger = logging.getLogger('alembic.runtime')

# 모델과 데이터베이스 임포트
from backend.database import Base, engine
from backend.models import User, Image, RefreshToken

# 모델 임포트 확인
logger.info("Imported models: %s", [cls.__name__ for cls in [User, Image, RefreshToken]])

# Alembic 설정 로드
config = context.config
fileConfig(config.config_file_name)

# 타겟 메타데이터 설정
target_metadata = Base.metadata

# 디버깅 출력: 로드된 테이블 확인
logger.info("Tables in Base.metadata: %s", [table.name for table in Base.metadata.tables.values()])

def run_migrations_offline():
    """오프라인 모드에서 마이그레이션 실행"""
    url = config.get_main_option("sqlalchemy.url")
    logger.info("Running offline migrations with URL: %s", url)
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
    )
    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    """온라인 모드에서 마이그레이션 실행"""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    logger.info("Running online migrations")
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )
        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()