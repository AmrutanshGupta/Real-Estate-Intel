import urllib.request
import json
import asyncio
from datetime import datetime, timedelta, timezone
import jwt
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from app.config import Config
from app.logger import logger

security = HTTPBearer()
auth_router = APIRouter()

class LoginRequest(BaseModel):
    org_id: str
    password: str

class OAuthLoginRequest(BaseModel):
    provider: str      
    access_token: str  
    org_id: str        

def create_access_token(data: dict, expires_delta: timedelta = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=Config.JWT_EXPIRE_MINS))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, Config.JWT_SECRET, algorithm=Config.JWT_ALGORITHM)

def get_current_tenant(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    token = credentials.credentials
    try:
        payload = jwt.decode(token, Config.JWT_SECRET, algorithms=[Config.JWT_ALGORITHM])
        org_id: str = payload.get("sub")
        if org_id is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
        return org_id
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Could not validate credentials")

@auth_router.post("/login")
async def login(req: LoginRequest):
    logger.info(f"Tenant authenticated via standard login: {req.org_id}")
    access_token = create_access_token(data={"sub": req.org_id})
    return {"access_token": access_token, "token_type": "bearer"}

def _fetch_oauth_profile(req: urllib.request.Request) -> dict:
    """Synchronous network call isolated for threading."""
    with urllib.request.urlopen(req, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))

@auth_router.post("/oauth/callback")
async def oauth_callback(req: OAuthLoginRequest):
    user_email = None
    
    try:
        if req.provider == "google":
            url = f"{Config.GOOGLE_USERINFO_URL}?access_token={req.access_token}"
            request = urllib.request.Request(url)
            profile = await asyncio.to_thread(_fetch_oauth_profile, request)
            user_email = profile.get("email")
                
        elif req.provider == "github":
            request = urllib.request.Request(
                Config.GITHUB_USERINFO_URL,
                headers={
                    "Authorization": f"token {req.access_token}",
                    "User-Agent": Config.PROJECT_NAME
                }
            )
            profile = await asyncio.to_thread(_fetch_oauth_profile, request)
            user_email = profile.get("email") or f"{profile.get('login')}@github.sys"
        else:
            raise HTTPException(status_code=400, detail="Unsupported OAuth provider")
            
    except Exception as e:
        logger.error(f"OAuth verification handshake failed: {str(e)}")
        raise HTTPException(status_code=401, detail="OAuth verification handshake failed")

    if not user_email:
        raise HTTPException(status_code=400, detail="Could not resolve unique identity from provider")

    logger.info(f"OAuth identity verified: {user_email} scoped to org: {req.org_id}")
    
    internal_jwt = create_access_token(data={"sub": req.org_id, "user": user_email})
    
    return {
        "access_token": internal_jwt,
        "token_type": "bearer",
        "org_id": req.org_id
    }