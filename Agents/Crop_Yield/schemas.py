from pydantic import BaseModel

class CropYieldRequest(BaseModel):
    query: str

class CropYieldResponse(BaseModel):
    result: str
