from __future__ import annotations

import uuid
from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, Optional

import bcrypt
import jwt

from . import config


@dataclass(frozen=True)
class UserRecord:
    user_id: str
    username: str
    email: str
    password_hash: bytes


_lock = Lock()
_users_by_id: Dict[str, UserRecord] = {}
_user_id_by_username: Dict[str, str] = {}
_user_id_by_email: Dict[str, str] = {}


def _normalize_username(s: str) -> str:
    return (s or "").strip().lower()


def _normalize_email(s: str) -> str:
    return (s or "").strip().lower()


def register_user(*, username: str, email: str, password: str) -> UserRecord:
    username_n = _normalize_username(username)
    email_n = _normalize_email(email)

    if not username_n:
        raise ValueError("username required")
    if not email_n or "@" not in email_n:
        raise ValueError("valid email required")
    if not password or len(password) < 6:
        raise ValueError("password must be at least 6 chars")

    with _lock:
        if username_n in _user_id_by_username:
            raise ValueError("username already exists")
        if email_n in _user_id_by_email:
            raise ValueError("email already exists")

        password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
        user_id = uuid.uuid4().hex
        rec = UserRecord(user_id=user_id, username=username_n, email=email_n, password_hash=password_hash)

        _users_by_id[user_id] = rec
        _user_id_by_username[username_n] = user_id
        _user_id_by_email[email_n] = user_id

        # Persist users so bandit state can be recovered across restarts.
        try:
            from . import db as persistence

            persistence.persist_user(rec)
        except Exception:
            # For local dev, allow auth to work even if DB isn't ready yet.
            pass

        return rec


def authenticate_login(*, username_or_email: str, password: str) -> Optional[UserRecord]:
    ident = (username_or_email or "").strip()
    if not ident or not password:
        return None

    username_n = _normalize_username(ident)
    email_n = _normalize_email(ident)

    with _lock:
        user_id = _user_id_by_email.get(email_n) or _user_id_by_username.get(username_n)
        if not user_id:
            return None

        rec = _users_by_id.get(user_id)

    if not rec:
        return None

    ok = bcrypt.checkpw(password.encode("utf-8"), rec.password_hash)
    return rec if ok else None


def get_user_by_id(user_id: str) -> Optional[UserRecord]:
    if not user_id:
        return None
    with _lock:
        return _users_by_id.get(user_id)


def get_user_by_email(email: str) -> Optional[UserRecord]:
    email_n = _normalize_email(email)
    if not email_n:
        return None
    with _lock:
        user_id = _user_id_by_email.get(email_n)
        return _users_by_id.get(user_id) if user_id else None


def update_password(*, user_id: str, new_password: str) -> bool:
    """
    Update password hash in-memory and persist to SQLite.
    Returns True if updated, False if user not found.
    """
    if not user_id or not new_password or len(new_password) < 6:
        raise ValueError("password must be at least 6 chars")

    new_hash = bcrypt.hashpw(new_password.encode("utf-8"), bcrypt.gensalt())
    with _lock:
        rec = _users_by_id.get(user_id)
        if not rec:
            return False

        updated = UserRecord(user_id=rec.user_id, username=rec.username, email=rec.email, password_hash=new_hash)
        _users_by_id[user_id] = updated

    # Persist outside the lock.
    try:
        from . import db as persistence

        persistence.persist_user(updated)
    except Exception:
        # Allow dev/test if DB isn't ready.
        pass

    return True


def seed_users_from_db(rows: list[dict[str, Any]]) -> None:
    """
    Replace the in-memory user store with persisted users from SQLite.
    Intended to be called once during FastAPI startup.
    """
    with _lock:
        _users_by_id.clear()
        _user_id_by_username.clear()
        _user_id_by_email.clear()

        for r in rows:
            rec = UserRecord(
                user_id=str(r["user_id"]),
                username=str(r["username"]),
                email=str(r["email"]),
                password_hash=r["password_hash"],
            )
            _users_by_id[rec.user_id] = rec
            _user_id_by_username[_normalize_username(rec.username)] = rec.user_id
            _user_id_by_email[_normalize_email(rec.email)] = rec.user_id


def create_access_token(*, user_id: str) -> str:
    import time as _time
    now = int(_time.time())
    exp = now + int(getattr(config, "JWT_EXPIRE_SECONDS", 60 * 60 * 24 * 7))

    payload = {"sub": user_id, "iat": now, "exp": exp}
    return jwt.encode(payload, config.JWT_SECRET, algorithm=getattr(config, "JWT_ALGORITHM", "HS256"))


def decode_access_token(token: str) -> str:
    if not token:
        raise ValueError("missing token")
    payload = jwt.decode(token, config.JWT_SECRET, algorithms=[getattr(config, "JWT_ALGORITHM", "HS256")])
    sub = payload.get("sub")
    if not sub:
        raise ValueError("invalid token payload")
    return str(sub)

