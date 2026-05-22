"""
Shared FastAPI dependencies — database engine, API key verification.
"""
import os
from fastapi import Header, HTTPException
from data.db import get_engine

def verify_api_key(x_api_key: str = Header(...)):
    expected = os.getenv("ADMIN_API_KEY")
    if not expected or x_api_key != expected:
        raise HTTPException(status_code=403, detail="Invalid API key")
