from pydantic import BaseModel
from typing import Optional


class Chunk(BaseModel):

    chunk_index: int                    
    source: str                         
    page: int                           

    livre:       Optional[str] = None   
    titre:       Optional[str] = None   
    chapitre:    Optional[str] = None   
    section:     Optional[str] = None   
    article_ref: Optional[str] = None   

    text: str                           
