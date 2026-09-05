from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session

from app.auth import create_access_token, get_current_user, hash_password, verify_password
from app.database import User, get_db

router = APIRouter(prefix="/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    name: str = Field(min_length=2, max_length=160)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    role: str
    facility: str = Field(min_length=1, max_length=200)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class ProfileUpdate(BaseModel):
    name: str = Field(min_length=2, max_length=160)
    role: str
    facility: str = Field(min_length=1, max_length=200)


def profile(user: User) -> dict:
    return {
        "id": str(user.id), "name": user.name, "email": user.email,
        "role": user.role, "facility": user.facility, "is_admin": user.is_admin,
        "created_at": user.created_at.isoformat(), "last_active": user.last_active.isoformat(),
    }


def auth_response(user: User) -> dict:
    return {"access_token": create_access_token(user), "token_type": "bearer", "user": profile(user)}


@router.post("/register")
def register(request: RegisterRequest, db: Session = Depends(get_db)):
    email = request.email.lower()
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(status_code=409, detail="An account with this email already exists")
    user = User(name=request.name.strip(), email=email, password_hash=hash_password(request.password),
                role=request.role, facility=request.facility.strip(), is_admin=False)
    db.add(user)
    db.commit()
    db.refresh(user)
    return auth_response(user)


@router.post("/login")
def login(request: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == request.email.lower()).first()
    if user is None or not verify_password(request.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    user.last_active = datetime.utcnow()
    db.commit()
    return auth_response(user)


@router.get("/me")
def me(user: User = Depends(get_current_user)):
    return profile(user)


@router.put("/profile")
def update_profile(request: ProfileUpdate, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    user.name, user.role, user.facility = request.name.strip(), request.role, request.facility.strip()
    db.commit()
    db.refresh(user)
    return profile(user)