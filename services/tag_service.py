"""
Tag Service - Manages asset tags with case-insensitive uniqueness.

This service layer provides the business logic for:
- Tag CRUD with case-insensitive uniqueness enforcement
- Attach/detach tags to/from assets
- Filter assets by tags (OR semantics)
- Query untagged assets

Designed to be consumed by CLI, Streamlit, FastAPI, etc.
"""

import logging
from dataclasses import dataclass, field
from typing import Sequence

from db import get_db, Asset, AssetStatus, Tag
from db.repositories import AssetRepository, TagRepository


logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Result DTOs
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class TagResult:
    """Result object for tag operations."""
    tag: Tag | None
    success: bool
    message: str
    errors: list[str] = field(default_factory=list)


@dataclass
class TagWithCount:
    """Tag with associated asset count."""
    tag: Tag
    asset_count: int


@dataclass
class TagListResult:
    """Result for tag list queries."""
    tags: list[TagWithCount]
    total: int


@dataclass
class AssetTagsResult:
    """Result for asset tag operations."""
    asset: Asset | None
    tags: list[Tag]
    success: bool
    message: str


# ──────────────────────────────────────────────────────────────────────────────
# Tag CRUD Operations
# ──────────────────────────────────────────────────────────────────────────────


def create_tag(name: str, description: str | None = None) -> TagResult:
    """
    Create a new tag with case-insensitive uniqueness.
    
    Args:
        name: Tag name (will be stripped)
        description: Optional description
        
    Returns:
        TagResult with created tag or error if name exists.
        
    Example:
        >>> result = create_tag("AI", "Artificial Intelligence companies")
        >>> print(result.message)
        "✅ Tag 'AI' created"
    """
    name = name.strip()
    
    if not name:
        return TagResult(
            tag=None,
            success=False,
            message="❌ Tag name cannot be empty",
            errors=["Tag name is required"],
        )
    
    if len(name) > 100:
        return TagResult(
            tag=None,
            success=False,
            message="❌ Tag name too long (max 100 chars)",
            errors=["Tag name exceeds maximum length"],
        )
    
    db = get_db()
    with db.session() as session:
        repo = TagRepository(session)
        
        # Check for case-insensitive duplicate
        existing = repo.get_by_name(name)
        if existing:
            return TagResult(
                tag=existing,
                success=False,
                message=f"❌ Tag '{existing.name}' already exists",
                errors=[f"A tag with name '{name}' already exists (case-insensitive)"],
            )
        
        tag = repo.create(name, description)
        session.commit()
        
        logger.info(f"Created tag: {tag.name}")
        return TagResult(
            tag=tag,
            success=True,
            message=f"✅ Tag '{tag.name}' created",
        )


def get_tag_by_id(tag_id: int) -> Tag | None:
    """Get a tag by its ID."""
    db = get_db()
    with db.session() as session:
        repo = TagRepository(session)
        return repo.get_by_id(tag_id)


def get_tag_by_name(name: str) -> Tag | None:
    """Get a tag by name (case-insensitive)."""
    db = get_db()
    with db.session() as session:
        repo = TagRepository(session)
        return repo.get_by_name(name)


def get_all_tags() -> list[Tag]:
    """Get all tags ordered by name."""
    db = get_db()
    with db.session() as session:
        repo = TagRepository(session)
        return list(repo.get_all())


def get_all_tags_with_counts() -> TagListResult:
    """
    Get all tags with their asset counts.
    
    Returns:
        TagListResult with list of TagWithCount and total count.
    """
    db = get_db()
    with db.session() as session:
        repo = TagRepository(session)
        results = repo.get_all_with_asset_counts()
        
        tags_with_counts = [
            TagWithCount(tag=tag, asset_count=count)
            for tag, count in results
        ]
        
        return TagListResult(
            tags=tags_with_counts,
            total=len(tags_with_counts),
        )


def rename_tag(tag_id: int, new_name: str) -> TagResult:
    """
    Rename a tag with case-insensitive uniqueness check.
    
    Args:
        tag_id: ID of tag to rename
        new_name: New name for the tag
        
    Returns:
        TagResult with updated tag or error.
    """
    new_name = new_name.strip()
    
    if not new_name:
        return TagResult(
            tag=None,
            success=False,
            message="❌ Tag name cannot be empty",
            errors=["Tag name is required"],
        )
    
    if len(new_name) > 100:
        return TagResult(
            tag=None,
            success=False,
            message="❌ Tag name too long (max 100 chars)",
            errors=["Tag name exceeds maximum length"],
        )
    
    db = get_db()
    with db.session() as session:
        repo = TagRepository(session)
        
        tag = repo.get_by_id(tag_id)
        if not tag:
            return TagResult(
                tag=None,
                success=False,
                message=f"❌ Tag with ID {tag_id} not found",
                errors=["Tag not found"],
            )
        
        old_name = tag.name
        
        # Check for case-insensitive duplicate (excluding self)
        existing = repo.get_by_name(new_name)
        if existing and existing.id != tag_id:
            return TagResult(
                tag=None,
                success=False,
                message=f"❌ Tag '{existing.name}' already exists",
                errors=[f"A tag with name '{new_name}' already exists"],
            )
        
        repo.rename(tag_id, new_name)
        session.commit()
        
        logger.info(f"Renamed tag '{old_name}' to '{new_name}'")
        return TagResult(
            tag=tag,
            success=True,
            message=f"✅ Tag renamed from '{old_name}' to '{new_name}'",
        )


def delete_tag(tag_id: int) -> TagResult:
    """
    Delete a tag (hard delete, removes all associations).
    
    Args:
        tag_id: ID of tag to delete
        
    Returns:
        TagResult indicating success or failure.
    """
    db = get_db()
    with db.session() as session:
        repo = TagRepository(session)
        
        tag = repo.get_by_id(tag_id)
        if not tag:
            return TagResult(
                tag=None,
                success=False,
                message=f"❌ Tag with ID {tag_id} not found",
                errors=["Tag not found"],
            )
        
        tag_name = tag.name
        
        if repo.delete(tag_id):
            session.commit()
            logger.info(f"Deleted tag: {tag_name}")
            return TagResult(
                tag=None,
                success=True,
                message=f"✅ Tag '{tag_name}' deleted",
            )
        
        return TagResult(
            tag=None,
            success=False,
            message=f"❌ Failed to delete tag '{tag_name}'",
            errors=["Delete operation failed"],
        )


# ──────────────────────────────────────────────────────────────────────────────
# Asset-Tag Association Operations
# ──────────────────────────────────────────────────────────────────────────────


def get_tags_for_asset(asset_id: int) -> list[Tag]:
    """Get all tags attached to an asset."""
    db = get_db()
    with db.session() as session:
        repo = TagRepository(session)
        return list(repo.get_tags_for_asset(asset_id))


def get_tags_for_ticker(ticker: str) -> list[Tag]:
    """Get all tags for an asset by ticker."""
    db = get_db()
    with db.session() as session:
        asset_repo = AssetRepository(session)
        tag_repo = TagRepository(session)
        
        asset = asset_repo.get_by_ticker(ticker)
        if not asset:
            return []
        
        return list(tag_repo.get_tags_for_asset(asset.id))


def attach_tag_to_asset(asset_id: int, tag_id: int) -> AssetTagsResult:
    """
    Attach a tag to an asset.
    
    Args:
        asset_id: The asset to tag
        tag_id: The tag to attach
        
    Returns:
        AssetTagsResult with updated tags list.
    """
    db = get_db()
    with db.session() as session:
        asset_repo = AssetRepository(session)
        tag_repo = TagRepository(session)
        
        asset = asset_repo.get_by_id(asset_id)
        if not asset:
            return AssetTagsResult(
                asset=None,
                tags=[],
                success=False,
                message=f"❌ Asset with ID {asset_id} not found",
            )
        
        tag = tag_repo.get_by_id(tag_id)
        if not tag:
            return AssetTagsResult(
                asset=asset,
                tags=list(tag_repo.get_tags_for_asset(asset_id)),
                success=False,
                message=f"❌ Tag with ID {tag_id} not found",
            )
        
        attached = tag_repo.attach_tag(asset_id, tag_id)
        session.commit()
        
        tags = list(tag_repo.get_tags_for_asset(asset_id))
        
        if attached:
            logger.info(f"Attached tag '{tag.name}' to {asset.ticker}")
            return AssetTagsResult(
                asset=asset,
                tags=tags,
                success=True,
                message=f"✅ Tag '{tag.name}' attached to {asset.ticker}",
            )
        else:
            return AssetTagsResult(
                asset=asset,
                tags=tags,
                success=True,
                message=f"ℹ️ Tag '{tag.name}' was already on {asset.ticker}",
            )


def detach_tag_from_asset(asset_id: int, tag_id: int) -> AssetTagsResult:
    """
    Detach a tag from an asset.
    
    Args:
        asset_id: The asset to untag
        tag_id: The tag to detach
        
    Returns:
        AssetTagsResult with updated tags list.
    """
    db = get_db()
    with db.session() as session:
        asset_repo = AssetRepository(session)
        tag_repo = TagRepository(session)
        
        asset = asset_repo.get_by_id(asset_id)
        if not asset:
            return AssetTagsResult(
                asset=None,
                tags=[],
                success=False,
                message=f"❌ Asset with ID {asset_id} not found",
            )
        
        tag = tag_repo.get_by_id(tag_id)
        tag_name = tag.name if tag else f"ID {tag_id}"
        
        detached = tag_repo.detach_tag(asset_id, tag_id)
        session.commit()
        
        tags = list(tag_repo.get_tags_for_asset(asset_id))
        
        if detached:
            logger.info(f"Detached tag '{tag_name}' from {asset.ticker}")
            return AssetTagsResult(
                asset=asset,
                tags=tags,
                success=True,
                message=f"✅ Tag '{tag_name}' detached from {asset.ticker}",
            )
        else:
            return AssetTagsResult(
                asset=asset,
                tags=tags,
                success=True,
                message=f"ℹ️ Tag '{tag_name}' was not on {asset.ticker}",
            )


def set_asset_tags(asset_id: int, tag_ids: list[int]) -> AssetTagsResult:
    """
    Set the tags for an asset (replaces all existing tags).
    
    Args:
        asset_id: The asset to update
        tag_ids: List of tag IDs to set (empty list removes all tags)
        
    Returns:
        AssetTagsResult with updated tags list.
    """
    db = get_db()
    with db.session() as session:
        asset_repo = AssetRepository(session)
        tag_repo = TagRepository(session)
        
        asset = asset_repo.get_by_id(asset_id)
        if not asset:
            return AssetTagsResult(
                asset=None,
                tags=[],
                success=False,
                message=f"❌ Asset with ID {asset_id} not found",
            )
        
        tag_repo.set_asset_tags(asset_id, tag_ids)
        session.commit()
        
        tags = list(tag_repo.get_tags_for_asset(asset_id))
        
        logger.info(f"Set {len(tags)} tags for {asset.ticker}")
        return AssetTagsResult(
            asset=asset,
            tags=tags,
            success=True,
            message=f"✅ Set {len(tags)} tags for {asset.ticker}",
        )


def set_asset_tags_by_names(asset_id: int, tag_names: list[str]) -> AssetTagsResult:
    """
    Set the tags for an asset by tag names (creates missing tags).
    
    Args:
        asset_id: The asset to update
        tag_names: List of tag names to set (will create if not exists)
        
    Returns:
        AssetTagsResult with updated tags list.
    """
    db = get_db()
    with db.session() as session:
        asset_repo = AssetRepository(session)
        tag_repo = TagRepository(session)
        
        asset = asset_repo.get_by_id(asset_id)
        if not asset:
            return AssetTagsResult(
                asset=None,
                tags=[],
                success=False,
                message=f"❌ Asset with ID {asset_id} not found",
            )
        
        # Get or create each tag
        tag_ids = []
        for name in tag_names:
            name = name.strip()
            if name:
                tag, _ = tag_repo.get_or_create(name)
                tag_ids.append(tag.id)
        
        tag_repo.set_asset_tags(asset_id, tag_ids)
        session.commit()
        
        tags = list(tag_repo.get_tags_for_asset(asset_id))
        
        logger.info(f"Set {len(tags)} tags for {asset.ticker}")
        return AssetTagsResult(
            asset=asset,
            tags=tags,
            success=True,
            message=f"✅ Set {len(tags)} tags for {asset.ticker}",
        )


# ──────────────────────────────────────────────────────────────────────────────
# Asset Filtering by Tags
# ──────────────────────────────────────────────────────────────────────────────


def get_assets_by_tags(
    tag_ids: list[int],
    status: AssetStatus | None = None,
) -> list[Asset]:
    """
    Get assets that have ANY of the specified tags (OR semantics).
    
    Args:
        tag_ids: List of tag IDs to filter by
        status: Optional status filter (OWNED, WATCHLIST)
        
    Returns:
        List of matching assets ordered by ticker.
    """
    if not tag_ids:
        return []
    
    db = get_db()
    with db.session() as session:
        repo = TagRepository(session)
        return list(repo.get_assets_by_tags(tag_ids, status))


def get_assets_by_tag_names(
    tag_names: list[str],
    status: AssetStatus | None = None,
) -> list[Asset]:
    """
    Get assets that have ANY of the specified tags by name (OR semantics).
    
    Args:
        tag_names: List of tag names to filter by (case-insensitive)
        status: Optional status filter
        
    Returns:
        List of matching assets ordered by ticker.
    """
    if not tag_names:
        return []
    
    db = get_db()
    with db.session() as session:
        repo = TagRepository(session)
        return list(repo.get_assets_by_tag_names(tag_names, status))


def get_untagged_assets(status: AssetStatus | None = None) -> list[Asset]:
    """
    Get assets that have no tags attached.
    
    Args:
        status: Optional status filter
        
    Returns:
        List of untagged assets ordered by ticker.
    """
    db = get_db()
    with db.session() as session:
        repo = TagRepository(session)
        return list(repo.get_untagged_assets(status))


def get_assets_for_tag(tag_id: int) -> list[Asset]:
    """Get all assets with a specific tag."""
    db = get_db()
    with db.session() as session:
        repo = TagRepository(session)
        return list(repo.get_assets_for_tag(tag_id))
