"""
Database session management and connection configuration.

Provides thread-safe session management using SQLAlchemy's modern patterns.
Designed for local-first, single-user operation with SQLite.
"""

from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from db.models import Base
from config import config


# Enable foreign keys for SQLite (disabled by default)
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Enable foreign key constraints for SQLite connections."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


class DatabaseManager:
    """
    Manages database connections and sessions.

    Usage:
        db = DatabaseManager()
        with db.session() as session:
            assets = session.query(Asset).all()
    """

    def __init__(self, db_url: Path | str | None = None):
        """
        Initialize database manager.

        Args:
            db_url: Optional custom database URL.
        """
        self.db_url = db_url if db_url else config.database.url
        self.engine = create_engine(
            self.db_url,
            echo=False,  # Set True for SQL debugging
            future=True,
        )
        self._session_factory = sessionmaker(
            bind=self.engine,
            expire_on_commit=False,
        )

    def create_tables(self) -> None:
        """Create all tables defined in models."""
        Base.metadata.create_all(self.engine)

    def drop_tables(self) -> None:
        """Drop all tables. Use with caution!"""
        Base.metadata.drop_all(self.engine)

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """
        Provide a transactional scope for database operations.

        Automatically commits on success, rolls back on exception.

        Yields:
            SQLAlchemy Session object.

        Example:
            with db.session() as session:
                session.add(new_asset)
        """
        session: Session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_session(self) -> Session:
        """
        Get a new session for manual management.

        Caller is responsible for commit/rollback/close.
        Prefer using session() context manager instead.

        Returns:
            New SQLAlchemy Session object.
        """
        return self._session_factory()


# Global database manager instance (singleton pattern)
_db_manager: DatabaseManager | None = None


def get_db(db_url: Path | str | None = None) -> DatabaseManager:
    """
    Get or create the global database manager.

    Args:
        db_url: Optional custom URL (only used on first call).

    Returns:
        Global DatabaseManager instance.
    """
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager(db_url)
    return _db_manager


def init_db(db_url: Path | str | None = None, if_drop: bool = False) -> DatabaseManager:
    """
    Initialize the database with all tables.

    Args:
        db_url: Optional custom database URL.
        if_drop: If True, drop existing tables before creating.

    Returns:
        Initialized DatabaseManager instance.
    """
    db = get_db(db_url)
    if if_drop:
        db.drop_tables()
    db.create_tables()
    return db
