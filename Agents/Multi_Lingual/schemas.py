from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from enum import Enum

class SupportedLanguage(str, Enum):
    AUTO = "auto"
    ENGLISH = "en"
    BENGALI = "bn"
    TELUGU = "te"
    HINDI = "hi"
    TAMIL = "ta"

class TranslationRequest(BaseModel):
    text: str = Field(..., description="Text to translate", min_length=1, max_length=5000)
    source_lang: SupportedLanguage = Field(default=SupportedLanguage.AUTO, description="Source language code")
    target_lang: SupportedLanguage = Field(default=SupportedLanguage.ENGLISH, description="Target language code")

class TranslationResponse(BaseModel):
    success: bool = Field(..., description="Whether translation was successful")
    translated_text: Optional[str] = Field(None, description="Translated text")
    source_language: Optional[str] = Field(None, description="Detected or specified source language")
    target_language: str = Field(..., description="Target language")
    error: Optional[str] = Field(None, description="Error message if translation failed")

class AgricultureQueryRequest(BaseModel):
    query: str = Field(..., description="Agricultural question in any supported language", min_length=1, max_length=2000)
    detect_language: bool = Field(default=True, description="Whether to auto-detect query language")
    preferred_response_lang: Optional[SupportedLanguage] = Field(None, description="Preferred response language (defaults to query language)")

class AgricultureQueryResponse(BaseModel):
    success: bool = Field(..., description="Whether the query was processed successfully")
    response: Optional[str] = Field(None, description="Agricultural advice response")
    detected_language: Optional[str] = Field(None, description="Detected language of the query")
    response_language: str = Field(..., description="Language of the response")
    processing_steps: Optional[Dict[str, Any]] = Field(None, description="Steps taken during processing")
    error: Optional[str] = Field(None, description="Error message if processing failed")

class HealthCheckResponse(BaseModel):
    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    supported_languages: list = Field(..., description="List of supported languages")

class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    details: Optional[str] = Field(None, description="Additional error details")
    error_code: Optional[str] = Field(None, description="Error code for categorization")
