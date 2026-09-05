from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
import threading

from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from reconpilot.config import DATABASE_URL, DB_PATH
from reconpilot.models import Base


REQUIRED_TABLES = (
    "runs",
    "cases",
    "ground_truth",
    "case_results",
    "exceptions",
    "audit_log",
)


_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None

# Prevent concurrent SQLite initialization from racing on CREATE TABLE.
_db_init_lock = threading.Lock()


def get_engine(url: str | None = None) -> Engine:
    """
    Return the SQLAlchemy engine.

    When a URL is explicitly supplied, create a fresh engine for that URL.
    Otherwise, reuse the application's shared engine.
    """
    global _engine, _SessionLocal

    if url is not None:
        return create_engine(
            url,
            echo=False,
            future=True,
        )

    if _engine is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)

        _engine = create_engine(
            DATABASE_URL,
            echo=False,
            future=True,
        )

        _SessionLocal = sessionmaker(
            bind=_engine,
            expire_on_commit=False,
            future=True,
        )

    return _engine


def init_db(url: str | None = None) -> Engine:
    """
    Initialize all database tables and return the engine.

    The initialization is protected by a process-level lock so that
    concurrent Streamlit/test initialization cannot race on SQLite
    CREATE TABLE statements.

    Explicit URLs are also initialized. This is important for tests
    using temporary SQLite databases.
    """
    global _SessionLocal

    engine = get_engine(url)

    with _db_init_lock:
        if url is None:
            DB_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Create all required tables if they do not already exist.
        Base.metadata.create_all(engine)

        # The application's default database uses the shared sessionmaker.
        if url is None and _SessionLocal is None:
            _SessionLocal = sessionmaker(
                bind=engine,
                expire_on_commit=False,
                future=True,
            )

    return engine


def list_tables(engine: Engine | None = None) -> set[str]:
    """
    Return the set of tables currently present in the database.
    """
    eng = engine or get_engine()

    return set(
        inspect(eng).get_table_names()
    )


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """
    Provide a transactional database session.

    Commits on success, rolls back on failure, and always closes
    the session.
    """
    if _SessionLocal is None:
        init_db()

    assert _SessionLocal is not None

    session = _SessionLocal()

    try:
        yield session
        session.commit()

    except Exception:
        session.rollback()
        raise

    finally:
        session.close()
