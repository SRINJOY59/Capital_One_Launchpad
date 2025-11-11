from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class WebScrappingRequest(BaseModel):
    query: str = Field(..., description="Query for web scraping agent")

class WebScrappingResponse(BaseModel):
    success: bool = Field(..., description="Whether scraping was successful")
    data: Optional[Any] = Field(None, description="Scraped data or summary")
    sources: Optional[List[str]] = Field(None, description="List of web sources/links")
    error: Optional[str] = Field(None, description="Error message if scraping failed")
