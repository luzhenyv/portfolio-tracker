"""
Note Service - Orchestrates note creation and management.

Provides a service layer for creating, editing, and querying notes.
Notes can be attached to assets, trades, market symbols, or be general journal entries.
"""

import logging
from dataclasses import dataclass, field
from typing import Sequence

from db import get_db
from db.models import Note, NoteTarget, NoteTargetKind, NoteType
from db.repositories import AssetRepository, NoteRepository, NoteTargetRepository


logger = logging.getLogger(__name__)


@dataclass
class NoteResult:
    """Result object for note operations."""
    note: Note | None
    created: bool
    errors: list[str] = field(default_factory=list)
    status_message: str = ""
    
    @property
    def success(self) -> bool:
        """Whether the operation was successful."""
        return self.note is not None and len(self.errors) == 0


def create_note_for_asset(
    ticker: str,
    body_md: str,
    note_type: NoteType = NoteType.JOURNAL,
    title: str | None = None,
    summary: str | None = None,
    key_points: str | None = None,
    tags: str | None = None,
) -> NoteResult:
    """
    Create a note attached to an asset.
    
    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")
        body_md: Note content in Markdown
        note_type: Type of note (THESIS, RISK, TRADE_PLAN, etc.)
        title: Optional title
        summary: Brief summary for table display
        key_points: Key takeaways
        tags: Comma-separated tags
        
    Returns:
        NoteResult with the created note
    """
    ticker = ticker.upper()
    
    db = get_db()
    with db.session() as session:
        asset_repo = AssetRepository(session)
        target_repo = NoteTargetRepository(session)
        note_repo = NoteRepository(session)
        
        # Find the asset
        asset = asset_repo.get_by_ticker(ticker)
        if not asset:
            return NoteResult(
                note=None,
                created=False,
                errors=[f"Asset {ticker} not found"],
                status_message=f"❌ Asset {ticker} not found",
            )
        
        # Get or create target
        target = target_repo.get_or_create_asset_target(asset.id)
        
        # Create note
        note = note_repo.create(
            target_id=target.id,
            body_md=body_md,
            note_type=note_type,
            title=title,
            summary=summary,
            key_points=key_points,
            tags=tags,
        )
        
        logger.info(f"Created note {note.id} for asset {ticker}")
        
        return NoteResult(
            note=note,
            created=True,
            status_message=f"✅ Note created for {ticker}",
        )


def create_note_for_trade(
    trade_id: int,
    body_md: str,
    note_type: NoteType = NoteType.TRADE_REVIEW,
    title: str | None = None,
    summary: str | None = None,
    key_points: str | None = None,
    tags: str | None = None,
) -> NoteResult:
    """
    Create a note attached to a trade.
    
    Args:
        trade_id: Trade ID
        body_md: Note content in Markdown
        note_type: Type of note (usually TRADE_REVIEW or TRADE_PLAN)
        title: Optional title
        summary: Brief summary for table display
        key_points: Key takeaways
        tags: Comma-separated tags
        
    Returns:
        NoteResult with the created note
    """
    db = get_db()
    with db.session() as session:
        target_repo = NoteTargetRepository(session)
        note_repo = NoteRepository(session)
        
        # Get or create target
        target = target_repo.get_or_create_trade_target(trade_id)
        
        # Create note
        note = note_repo.create(
            target_id=target.id,
            body_md=body_md,
            note_type=note_type,
            title=title,
            summary=summary,
            key_points=key_points,
            tags=tags,
        )
        
        logger.info(f"Created note {note.id} for trade {trade_id}")
        
        return NoteResult(
            note=note,
            created=True,
            status_message=f"✅ Note created for trade #{trade_id}",
        )


def create_market_note(
    symbol: str,
    body_md: str,
    name: str | None = None,
    note_type: NoteType = NoteType.MARKET_VIEW,
    title: str | None = None,
    summary: str | None = None,
    key_points: str | None = None,
    tags: str | None = None,
) -> NoteResult:
    """
    Create a note about a market/index symbol.
    
    Args:
        symbol: Market symbol (e.g., "^GSPC" for S&P 500, "^VIX" for VIX)
        body_md: Note content in Markdown
        name: Display name for the symbol (e.g., "S&P 500")
        note_type: Type of note (usually MARKET_VIEW)
        title: Optional title
        summary: Brief summary for table display
        key_points: Key takeaways
        tags: Comma-separated tags
        
    Returns:
        NoteResult with the created note
    """
    symbol = symbol.upper()
    
    db = get_db()
    with db.session() as session:
        target_repo = NoteTargetRepository(session)
        note_repo = NoteRepository(session)
        
        # Get or create target
        target = target_repo.get_or_create_market_target(symbol, name)
        
        # Create note
        note = note_repo.create(
            target_id=target.id,
            body_md=body_md,
            note_type=note_type,
            title=title,
            summary=summary,
            key_points=key_points,
            tags=tags,
        )
        
        logger.info(f"Created market note {note.id} for {symbol}")
        
        return NoteResult(
            note=note,
            created=True,
            status_message=f"✅ Market note created for {symbol}",
        )


def create_journal_entry(
    body_md: str,
    note_type: NoteType = NoteType.JOURNAL,
    title: str | None = None,
    summary: str | None = None,
    key_points: str | None = None,
    tags: str | None = None,
) -> NoteResult:
    """
    Create a general journal entry (not attached to any specific entity).
    
    Args:
        body_md: Note content in Markdown
        note_type: Type of note (default JOURNAL)
        title: Optional title
        summary: Brief summary for table display
        key_points: Key takeaways
        tags: Comma-separated tags
        
    Returns:
        NoteResult with the created note
    """
    db = get_db()
    with db.session() as session:
        target_repo = NoteTargetRepository(session)
        note_repo = NoteRepository(session)
        
        # Get or create journal target
        target = target_repo.get_or_create_journal_target()
        
        # Create note
        note = note_repo.create(
            target_id=target.id,
            body_md=body_md,
            note_type=note_type,
            title=title,
            summary=summary,
            key_points=key_points,
            tags=tags,
        )
        
        logger.info(f"Created journal entry {note.id}")
        
        return NoteResult(
            note=note,
            created=True,
            status_message="✅ Journal entry created",
        )


def update_note(
    note_id: int,
    body_md: str | None = None,
    title: str | None = None,
    summary: str | None = None,
    key_points: str | None = None,
    tags: str | None = None,
    note_type: NoteType | None = None,
) -> NoteResult:
    """
    Update an existing note.
    
    Args:
        note_id: ID of the note to update
        body_md: New content (None to keep existing)
        title: New title (None to keep existing)
        summary: New summary (None to keep existing)
        key_points: New key points (None to keep existing)
        tags: New tags (None to keep existing)
        note_type: New type (None to keep existing)
        
    Returns:
        NoteResult with the updated note
    """
    db = get_db()
    with db.session() as session:
        note_repo = NoteRepository(session)
        
        note = note_repo.update(
            note_id=note_id,
            body_md=body_md,
            title=title,
            summary=summary,
            key_points=key_points,
            tags=tags,
            note_type=note_type,
        )
        
        if not note:
            return NoteResult(
                note=None,
                created=False,
                errors=[f"Note {note_id} not found"],
                status_message=f"❌ Note {note_id} not found",
            )
        
        logger.info(f"Updated note {note.id}")
        
        return NoteResult(
            note=note,
            created=False,
            status_message="✅ Note updated",
        )


def get_note(note_id: int) -> Note | None:
    """Get a note by ID."""
    db = get_db()
    with db.session() as session:
        note_repo = NoteRepository(session)
        return note_repo.get_by_id(note_id)


def get_notes_for_asset(
    ticker: str,
    include_archived: bool = False,
    limit: int | None = None,
) -> Sequence[Note]:
    """Get all notes for an asset."""
    ticker = ticker.upper()
    
    db = get_db()
    with db.session() as session:
        asset_repo = AssetRepository(session)
        note_repo = NoteRepository(session)
        
        asset = asset_repo.get_by_ticker(ticker)
        if not asset:
            return []
        
        return note_repo.list_by_asset(
            asset_id=asset.id,
            include_archived=include_archived,
            limit=limit,
        )


def get_recent_notes(
    limit: int = 20,
    include_archived: bool = False,
) -> Sequence[Note]:
    """Get recent notes across all targets."""
    db = get_db()
    with db.session() as session:
        note_repo = NoteRepository(session)
        return note_repo.list_recent(limit=limit, include_archived=include_archived)


def get_notes_by_type(
    note_type: NoteType,
    include_archived: bool = False,
    limit: int | None = None,
) -> Sequence[Note]:
    """Get notes by type."""
    db = get_db()
    with db.session() as session:
        note_repo = NoteRepository(session)
        return note_repo.list_by_type(
            note_type=note_type,
            include_archived=include_archived,
            limit=limit,
        )


def search_notes_by_tag(
    tag: str,
    include_archived: bool = False,
    limit: int | None = None,
) -> Sequence[Note]:
    """Search notes by tag."""
    db = get_db()
    with db.session() as session:
        note_repo = NoteRepository(session)
        return note_repo.search_by_tag(
            tag=tag,
            include_archived=include_archived,
            limit=limit,
        )


def archive_note(note_id: int) -> NoteResult:
    """Archive a note."""
    db = get_db()
    with db.session() as session:
        note_repo = NoteRepository(session)
        note = note_repo.set_status(note_id, "ARCHIVED")
        
        if not note:
            return NoteResult(
                note=None,
                created=False,
                errors=[f"Note {note_id} not found"],
                status_message=f"❌ Note {note_id} not found",
            )
        
        return NoteResult(
            note=note,
            created=False,
            status_message="✅ Note archived",
        )


def delete_note(note_id: int) -> NoteResult:
    """Soft delete a note."""
    db = get_db()
    with db.session() as session:
        note_repo = NoteRepository(session)
        success = note_repo.delete(note_id)
        
        if not success:
            return NoteResult(
                note=None,
                created=False,
                errors=[f"Note {note_id} not found"],
                status_message=f"❌ Note {note_id} not found",
            )
        
        return NoteResult(
            note=None,
            created=False,
            status_message="✅ Note deleted",
        )


def pin_note(note_id: int, pinned: bool = True) -> NoteResult:
    """Pin or unpin a note."""
    db = get_db()
    with db.session() as session:
        note_repo = NoteRepository(session)
        note = note_repo.set_pinned(note_id, pinned)
        
        if not note:
            return NoteResult(
                note=None,
                created=False,
                errors=[f"Note {note_id} not found"],
                status_message=f"❌ Note {note_id} not found",
            )
        
        action = "pinned" if pinned else "unpinned"
        return NoteResult(
            note=note,
            created=False,
            status_message=f"✅ Note {action}",
        )


def get_market_symbols() -> list[dict]:
    """
    Get all market symbols that have notes.
    
    Returns:
        List of dicts with 'symbol' and 'name' keys
    """
    db = get_db()
    with db.session() as session:
        target_repo = NoteTargetRepository(session)
        targets = target_repo.list_market_targets()
        
        return [
            {"symbol": t.symbol, "name": t.symbol_name or t.symbol}
            for t in targets
        ]
