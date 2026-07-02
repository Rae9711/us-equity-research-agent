from __future__ import annotations

import os
from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

DEFAULT_DB_PATH = os.environ.get(
    "DATABASE_URL",
    "sqlite:////data/trading_os.db",
)


@lru_cache
def get_engine():
    url = DEFAULT_DB_PATH
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, connect_args=connect_args)


def get_session() -> Session:
    factory = sessionmaker(bind=get_engine(), autoflush=False, autocommit=False)
    return factory()
