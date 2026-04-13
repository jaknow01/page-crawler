"""
Pydantic return types for MCP server tools.
All tools return instances of these models instead of plain dicts.
"""

from pydantic import BaseModel


class AddPageResult(BaseModel):
    success: bool
    id: str
    metadata_mode: str


class DomainExistsResult(BaseModel):
    exists: bool
    page_count: int


class PageItem(BaseModel):
    url: str
    title: str


class OwnPagesResult(BaseModel):
    pages: list[PageItem]


class DbStatsResult(BaseModel):
    total: int
    own_pages: int
    competitor_pages: int
    unique_domains: list[str]
