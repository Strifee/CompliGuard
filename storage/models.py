from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime

class Document(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    source_url: Optional[str] = None
    origin: str
    title: str
    type: str
    version_label: Optional[str] = None
    published_at: Optional[str] = None
    ingested_at: datetime
    language: str
    content_hash: str
    raw_path: str

class Section(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    document_id: int
    section_type: str
    section_ref: Optional[str] = None
    text: str
    order_index: int
    page_span: Optional[str] = None
