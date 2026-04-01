from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np
from sqlalchemy import Column, Integer, LargeBinary, String, Text, create_engine, func, select
from sqlalchemy.orm import declarative_base, sessionmaker

from . import config

Base = declarative_base()


class UserRow(Base):
    __tablename__ = "users"

    user_id = Column(String, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    email = Column(String, unique=True, nullable=False)
    password_hash = Column(LargeBinary, nullable=False)


class BanditGlobalRow(Base):
    __tablename__ = "bandit_global"

    # single row
    id = Column(Integer, primary_key=True, default=1)
    global_n = Column(Integer, nullable=False, default=0)
    global_mu_json = Column(Text, nullable=False)
    global_sinv_json = Column(Text, nullable=False)


class BanditUserRow(Base):
    __tablename__ = "bandit_users"

    user_id = Column(String, primary_key=True)
    mu_json = Column(Text, nullable=False)
    sigma_inv_json = Column(Text, nullable=False)
    history_json = Column(Text, nullable=False)
    reward_log_json = Column(Text, nullable=False)

    last_message = Column(Text, default="", nullable=False)
    last_response = Column(Text, default="", nullable=False)
    last_strategy = Column(String, nullable=True)
    last_x_json = Column(Text, nullable=True)

    msg_count = Column(Integer, nullable=False, default=0)
    prefs_json = Column(Text, nullable=False, default="[]")
    locked_strategy = Column(String, nullable=True)
    pending_strategy = Column(String, nullable=True)


class ConversationLogRow(Base):
    __tablename__ = "conversation_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, nullable=False, index=True)
    pane = Column(String, nullable=False)  # "adaptive" | "baseline"
    role = Column(String, nullable=False)  # "user" | "assistant"
    content = Column(Text, nullable=False)

    strategy = Column(String, nullable=True)
    elapsed = Column(String, nullable=True)
    widget = Column(Integer, nullable=False, default=0)
    ts = Column(Integer, nullable=False, default=0)


class UserPrimitiveRow(Base):
    __tablename__ = "user_primitives"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, nullable=False, index=True)
    name = Column(String, nullable=False)
    instruction = Column(Text, nullable=False)
    created_ts = Column(Integer, nullable=False, default=0)
    updated_ts = Column(Integer, nullable=False, default=0)


_PRIMITIVES_LOCK = threading.Lock()


def _read_primitives_json() -> dict:
    """
    primitives.json format:
      {
        "<user_id>": {
          "next_id": 1,
          "items": [{ id, name, instruction, created_ts, updated_ts }, ...]
        }
      }
    """
    path = getattr(config, "PRIMITIVES_JSON_PATH", str(config.HERE.parent / "user_primitives.json"))
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def _atomic_write_primitives_json(data: dict) -> None:
    path = getattr(config, "PRIMITIVES_JSON_PATH", str(config.HERE.parent / "user_primitives.json"))
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


_engine = None
_SessionLocal = None


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _json_loads(s: str | None, default: Any) -> Any:
    if not s:
        return default
    return json.loads(s)


def get_engine():
    global _engine, _SessionLocal
    if _engine is None:
        _engine = create_engine(
            f"sqlite:///{config.DB_PATH}",
            connect_args={"check_same_thread": False},
        )
        _SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)
    return _engine


def init_db() -> None:
    eng = get_engine()
    Base.metadata.create_all(bind=eng)

    # Ensure bandit_global exists so we can upsert cleanly later.
    Session = _SessionLocal
    assert Session is not None
    with Session() as session:
        row = session.execute(select(BanditGlobalRow)).scalar_one_or_none()
        if row is None:
            # Keep empty defaults; engine startup will already have proper values.
            # We'll overwrite the row on first persist.
            strategy_ids = set(getattr(config, "STRATEGY_ITEMS", {}) or {}).union(set(config.STRATEGY_NAMES))
            dummy_mu = {k: [0.0] * config.D for k in strategy_ids}
            dummy_sinv = {k: (np.eye(config.D) * 0.1).tolist() for k in strategy_ids}
            session.add(
                BanditGlobalRow(
                    id=1,
                    global_n=0,
                    global_mu_json=_json_dumps(dummy_mu),
                    global_sinv_json=_json_dumps(dummy_sinv),
                )
            )
            session.commit()


def log_conversation_message(
    *,
    user_id: str,
    pane: str,
    role: str,
    content: str,
    strategy: str | None = None,
    elapsed: float | None = None,
    widget: bool = False,
    ts: int | None = None,
) -> None:
    Session = _SessionLocal
    assert Session is not None
    now = int(ts if ts is not None else time.time())
    with Session() as session:
        session.add(
            ConversationLogRow(
                user_id=user_id,
                pane=str(pane),
                role=str(role),
                content=str(content or ""),
                strategy=str(strategy) if strategy else None,
                elapsed=str(elapsed) if elapsed is not None else None,
                widget=1 if widget else 0,
                ts=now,
            )
        )
        session.commit()


def get_recent_conversation_messages(*, user_id: str, pane: str, limit: int = 80) -> List[Dict[str, Any]]:
    Session = _SessionLocal
    assert Session is not None
    lim = max(1, min(int(limit), 400))
    with Session() as session:
        rows = (
            session.execute(
                select(ConversationLogRow)
                .where(ConversationLogRow.user_id == user_id)
                .where(ConversationLogRow.pane == pane)
                .order_by(ConversationLogRow.id.desc())
                .limit(lim)
            )
            .scalars()
            .all()
        )
        rows = list(reversed(rows))
        return [
            {
                "role": r.role,
                "content": r.content,
                "strategy": r.strategy,
                "elapsed": r.elapsed,
                "widget": bool(r.widget),
                "ts": r.ts,
            }
            for r in rows
        ]


def delete_conversation_logs(*, user_id: str, pane: str | None = None) -> None:
    Session = _SessionLocal
    assert Session is not None
    with Session() as session:
        q = ConversationLogRow.__table__.delete().where(ConversationLogRow.user_id == user_id)  # type: ignore[attr-defined]
        if pane:
            q = q.where(ConversationLogRow.pane == pane)
        session.execute(q)
        session.commit()


def aggregate_strategy_usage() -> Dict[str, Dict[str, Any]]:
    """
    Aggregate strategy usage stats for admin dashboards.

    Returns:
      {
        "<strategy_id>": {"wins": int, "trials": int, "last_ts": int | None},
        ...
      }
    """
    Session = _SessionLocal
    assert Session is not None

    by_id: Dict[str, Dict[str, Any]] = {}

    with Session() as session:
        user_rows = session.execute(select(BanditUserRow)).scalars().all()
        for row in user_rows:
            reward_log = _json_loads(row.reward_log_json, [])
            if not isinstance(reward_log, list):
                continue
            for entry in reward_log:
                if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                    continue
                sid = str(entry[0] or "").strip()
                if not sid:
                    continue
                try:
                    reward = float(entry[1])
                except Exception:
                    continue

                bucket = by_id.setdefault(sid, {"wins": 0, "trials": 0, "last_ts": None})
                bucket["trials"] += 1
                if reward >= 0.5:
                    bucket["wins"] += 1

        last_rows = session.execute(
            select(ConversationLogRow.strategy, func.max(ConversationLogRow.ts))
            .where(ConversationLogRow.pane == "adaptive")
            .where(ConversationLogRow.role == "assistant")
            .where(ConversationLogRow.strategy.isnot(None))
            .group_by(ConversationLogRow.strategy)
        ).all()

        for sid_raw, max_ts in last_rows:
            sid = str(sid_raw or "").strip()
            if not sid:
                continue
            bucket = by_id.setdefault(sid, {"wins": 0, "trials": 0, "last_ts": None})
            bucket["last_ts"] = int(max_ts) if max_ts is not None else None

    return by_id


def get_strategy_analytics(strategy_id: str, days: int = 30) -> Dict[str, Any]:
    """
    Return per-day usage series and summary for a strategy.
    """
    Session = _SessionLocal
    assert Session is not None

    sid = str(strategy_id or "").strip()
    days_n = max(7, min(int(days), 180))
    now = int(time.time())
    day_seconds = 86400
    start_ts = now - (days_n - 1) * day_seconds

    day_buckets = {}
    for i in range(days_n):
        ts = start_ts + i * day_seconds
        day_key = time.strftime("%Y-%m-%d", time.localtime(ts))
        day_buckets[day_key] = 0

    with Session() as session:
        rows = (
            session.execute(
                select(ConversationLogRow.ts)
                .where(ConversationLogRow.pane == "adaptive")
                .where(ConversationLogRow.role == "assistant")
                .where(ConversationLogRow.strategy == sid)
                .where(ConversationLogRow.ts >= start_ts)
            )
            .scalars()
            .all()
        )

        for ts in rows:
            if ts is None:
                continue
            key = time.strftime("%Y-%m-%d", time.localtime(int(ts)))
            if key in day_buckets:
                day_buckets[key] += 1

    labels = list(day_buckets.keys())
    usage_series = [int(day_buckets[k]) for k in labels]
    total_usage = int(sum(usage_series))

    usage_all = aggregate_strategy_usage().get(sid, {})
    wins = int(usage_all.get("wins") or 0)
    trials = int(usage_all.get("trials") or 0)

    return {
        "strategy_id": sid,
        "days": days_n,
        "labels": labels,
        "usage_series": usage_series,
        "summary": {
            "total_usage": total_usage,
            "wins": wins,
            "trials": trials,
        },
    }


def list_user_primitives(user_id: str) -> List[Dict[str, Any]]:
    with _PRIMITIVES_LOCK:
        data = _read_primitives_json()
        box = data.get(user_id) or {}
        items = box.get("items") if isinstance(box, dict) else None
        if not isinstance(items, list):
            return []
        out: List[Dict[str, Any]] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            prim_id = int(it.get("id") or 0)
            name = str(it.get("name") or "")
            inst = str(it.get("instruction") or "")
            if prim_id <= 0 or not name or not inst:
                continue
            out.append(
                {
                    "id": prim_id,
                    "name": name,
                    "instruction": inst,
                    "created_ts": int(it.get("created_ts") or 0),
                    "updated_ts": int(it.get("updated_ts") or 0),
                }
            )
        out.sort(key=lambda x: x["id"])
        # Migration fallback:
        # If the user has primitives in the legacy SQLite table, but JSON is empty,
        # automatically copy them into JSON so the UI can show everything.
        if out:
            return out

        try:
            Session = _SessionLocal
            if Session is None:
                # Ensure engine/session exists (init_db should have done this).
                get_engine()
                Session = _SessionLocal
            assert Session is not None
            with Session() as session:
                rows = (
                    session.execute(
                        select(UserPrimitiveRow)
                        .where(UserPrimitiveRow.user_id == user_id)
                        .order_by(UserPrimitiveRow.id.asc())
                    )
                    .scalars()
                    .all()
                )
                if not rows:
                    return []
                migrated: List[Dict[str, Any]] = []
                max_id = 0
                for r in rows:
                    prim_id = int(getattr(r, "id") or 0)
                    if prim_id <= 0:
                        continue
                    max_id = max(max_id, prim_id)
                    migrated.append(
                        {
                            "id": prim_id,
                            "name": str(getattr(r, "name") or ""),
                            "instruction": str(getattr(r, "instruction") or ""),
                            "created_ts": int(getattr(r, "created_ts") or 0),
                            "updated_ts": int(getattr(r, "updated_ts") or 0),
                        }
                    )

                if not migrated:
                    return []

                # Write into JSON.
                data = _read_primitives_json()
                data[user_id] = {"next_id": max_id + 1, "items": migrated}
                _atomic_write_primitives_json(data)
                return migrated
        except Exception:
            # If migration fails for any reason, return the JSON (empty) result.
            return []


def create_user_primitive(*, user_id: str, name: str, instruction: str) -> Dict[str, Any]:
    now = int(time.time())
    nm = str(name).strip()
    inst = str(instruction).strip()

    with _PRIMITIVES_LOCK:
        data = _read_primitives_json()
        box = data.get(user_id) or {"next_id": 1, "items": []}
        if not isinstance(box, dict):
            box = {"next_id": 1, "items": []}
        items = box.get("items") or []
        if not isinstance(items, list):
            items = []

        next_id = int(box.get("next_id") or 1)
        if items:
            max_id = max([int(it.get("id") or 0) for it in items if isinstance(it, dict)], default=0)
            next_id = max(next_id, max_id + 1)

        new_item = {"id": next_id, "name": nm, "instruction": inst, "created_ts": now, "updated_ts": now}
        items.append(new_item)
        box["items"] = items
        box["next_id"] = next_id + 1
        data[user_id] = box
        _atomic_write_primitives_json(data)
        return new_item


def update_user_primitive(*, user_id: str, prim_id: int, name: str, instruction: str) -> Dict[str, Any] | None:
    now = int(time.time())
    nm = str(name).strip()
    inst = str(instruction).strip()
    with _PRIMITIVES_LOCK:
        data = _read_primitives_json()
        box = data.get(user_id) or {}
        if not isinstance(box, dict):
            return None
        items = box.get("items") or []
        if not isinstance(items, list):
            return None

        found = False
        created_ts = 0
        for it in items:
            if not isinstance(it, dict):
                continue
            if int(it.get("id") or 0) != int(prim_id):
                continue
            it["name"] = nm
            it["instruction"] = inst
            it["updated_ts"] = now
            created_ts = int(it.get("created_ts") or 0)
            found = True
            break

        if not found:
            return None

        box["items"] = items
        data[user_id] = box
        _atomic_write_primitives_json(data)
        return {"id": int(prim_id), "name": nm, "instruction": inst, "created_ts": created_ts, "updated_ts": now}


def delete_user_primitive(*, user_id: str, prim_id: int) -> bool:
    with _PRIMITIVES_LOCK:
        data = _read_primitives_json()
        box = data.get(user_id)
        if not isinstance(box, dict):
            return False
        items = box.get("items") or []
        if not isinstance(items, list):
            return False
        before = len(items)
        items = [it for it in items if not (isinstance(it, dict) and int(it.get("id") or 0) == int(prim_id))]
        if len(items) == before:
            return False
        box["items"] = items
        data[user_id] = box
        _atomic_write_primitives_json(data)
        return True


def load_users_from_db() -> List[Dict[str, Any]]:
    Session = _SessionLocal
    assert Session is not None
    with Session() as session:
        rows = session.execute(select(UserRow)).scalars().all()
        return [
            {
                "user_id": r.user_id,
                "username": r.username,
                "email": r.email,
                "password_hash": r.password_hash,
            }
            for r in rows
        ]


def persist_user(user_rec: Any) -> None:
    """
    Upsert a user record into `users`.
    `user_rec` can be `backend.auth.UserRecord` or any object with
    {user_id, username, email, password_hash}.
    """
    Session = _SessionLocal
    assert Session is not None

    with Session() as session:
        existing = session.execute(select(UserRow).where(UserRow.user_id == user_rec.user_id)).scalar_one_or_none()
        if existing is None:
            session.add(
                UserRow(
                    user_id=user_rec.user_id,
                    username=user_rec.username,
                    email=user_rec.email,
                    password_hash=user_rec.password_hash,
                )
            )
        else:
            existing.username = user_rec.username
            existing.email = user_rec.email
            existing.password_hash = user_rec.password_hash
        session.commit()


def persist_global_state(bandit_engine: Any) -> None:
    Session = _SessionLocal
    assert Session is not None

    # Persist ALL keys present in the engine's posterior state (enabled + disabled).
    global_mu = {k: np.asarray(v).tolist() for k, v in (bandit_engine.global_mu or {}).items()}
    global_sinv = {k: np.asarray(v).tolist() for k, v in (bandit_engine.global_sinv or {}).items()}

    with Session() as session:
        row = session.execute(select(BanditGlobalRow).where(BanditGlobalRow.id == 1)).scalar_one_or_none()
        if row is None:
            row = BanditGlobalRow(id=1)
            session.add(row)
        row.global_n = int(bandit_engine.global_n)
        row.global_mu_json = _json_dumps(global_mu)
        row.global_sinv_json = _json_dumps(global_sinv)
        session.commit()


def load_global_state(bandit_engine: Any) -> None:
    Session = _SessionLocal
    assert Session is not None
    with Session() as session:
        row = session.execute(select(BanditGlobalRow).where(BanditGlobalRow.id == 1)).scalar_one_or_none()
        if not row:
            return

        global_mu = _json_loads(row.global_mu_json, {})
        global_sinv = _json_loads(row.global_sinv_json, {})
        bandit_engine.global_n = int(row.global_n or 0)
        desired_ids = set(getattr(config, "STRATEGY_ITEMS", {}) or {}).union(set(bandit_engine.global_mu.keys()))
        default_sinv = (np.eye(config.D) * 0.1).tolist()
        for k in desired_ids:
            bandit_engine.global_mu[k] = np.array(global_mu.get(k, [0.0] * config.D), dtype=float)
            bandit_engine.global_sinv[k] = np.array(global_sinv.get(k, default_sinv), dtype=float)


def _serialize_user(user: dict) -> Dict[str, Any]:
    # Persist ALL keys present in the user's posterior state (enabled + disabled).
    mu_dict = user.get("mu") or {}
    sinv_dict = user.get("sigma_inv") or {}
    mu = {k: np.asarray(v).tolist() for k, v in mu_dict.items()}
    sinv = {k: np.asarray(v).tolist() for k, v in sinv_dict.items()}

    reward_log = [[str(s), float(r)] for (s, r) in (user.get("reward_log") or [])]

    prefs = sorted(list(user.get("prefs") or []))
    last_x = user.get("last_x")

    return {
        "mu": mu,
        "sigma_inv": sinv,
        "history": user.get("history") or [],
        "reward_log": reward_log,
        "last_message": user.get("last_message") or "",
        "last_response": user.get("last_response") or "",
        "last_strategy": user.get("last_strategy"),
        "last_x": last_x,
        "msg_count": int(user.get("msg_count") or 0),
        "prefs": prefs,
        "locked_strategy": user.get("locked_strategy"),
        "pending_strategy": user.get("pending_strategy"),
    }


def _deserialize_user(user_id: str, payload: Dict[str, Any]) -> dict:
    desired_ids = set(getattr(config, "STRATEGY_ITEMS", {}) or {}).union(set(config.STRATEGY_NAMES))
    default_sinv = (np.eye(config.D) * 0.1).tolist()

    mu_payload = payload.get("mu") if isinstance(payload.get("mu"), dict) else {}
    sinv_payload = payload.get("sigma_inv") if isinstance(payload.get("sigma_inv"), dict) else {}

    mu = {}
    sinv = {}
    for k in desired_ids:
        mu[k] = np.array(mu_payload.get(k, [0.0] * config.D), dtype=float)
        sinv[k] = np.array(sinv_payload.get(k, default_sinv), dtype=float)

    user = {
        "mu": mu,
        "sigma_inv": sinv,
        "history": payload.get("history") or [],
        "reward_log": payload.get("reward_log") or [],
        "last_message": payload.get("last_message") or "",
        "last_response": payload.get("last_response") or "",
        "last_strategy": payload.get("last_strategy"),
        "last_x": payload.get("last_x"),
        "msg_count": int(payload.get("msg_count") or 0),
        "prefs": set(payload.get("prefs") or []),
        "locked_strategy": payload.get("locked_strategy"),
        "pending_strategy": payload.get("pending_strategy"),
    }
    return user


def persist_user_state(bandit_engine: Any, user_id: str) -> None:
    Session = _SessionLocal
    assert Session is not None

    user = bandit_engine.get_user(user_id)
    payload = _serialize_user(user)

    with Session() as session:
        row = session.execute(select(BanditUserRow).where(BanditUserRow.user_id == user_id)).scalar_one_or_none()
        if row is None:
            row = BanditUserRow(user_id=user_id)
            session.add(row)

        row.mu_json = _json_dumps(payload["mu"])
        row.sigma_inv_json = _json_dumps(payload["sigma_inv"])
        row.history_json = _json_dumps(payload["history"])
        row.reward_log_json = _json_dumps(payload["reward_log"])

        row.last_message = payload["last_message"]
        row.last_response = payload["last_response"]
        row.last_strategy = payload["last_strategy"]
        row.last_x_json = _json_dumps(payload["last_x"]) if payload["last_x"] is not None else None

        row.msg_count = payload["msg_count"]
        row.prefs_json = _json_dumps(payload["prefs"])
        row.locked_strategy = payload["locked_strategy"]
        row.pending_strategy = payload["pending_strategy"]

        session.commit()


def delete_user_state(user_id: str) -> None:
    Session = _SessionLocal
    assert Session is not None
    with Session() as session:
        session.execute(
            BanditUserRow.__table__.delete().where(BanditUserRow.user_id == user_id)  # type: ignore[attr-defined]
        )
        session.commit()


def load_user_states(bandit_engine: Any) -> None:
    """
    Load all persisted `bandit_users` into the in-memory engine.
    """
    Session = _SessionLocal
    assert Session is not None

    with Session() as session:
        rows = session.execute(select(BanditUserRow)).scalars().all()
        for r in rows:
            mu = _json_loads(r.mu_json, {})
            sinv = _json_loads(r.sigma_inv_json, {})
            history = _json_loads(r.history_json, [])
            reward_log = _json_loads(r.reward_log_json, [])
            last_x = _json_loads(r.last_x_json, None)
            prefs = _json_loads(r.prefs_json, [])

            payload = {
                "mu": mu,
                "sigma_inv": sinv,
                "history": history,
                "reward_log": reward_log,
                "last_message": r.last_message or "",
                "last_response": r.last_response or "",
                "last_strategy": r.last_strategy,
                "last_x": last_x,
                "msg_count": r.msg_count or 0,
                "prefs": prefs,
                "locked_strategy": r.locked_strategy,
                "pending_strategy": r.pending_strategy,
            }

            # Ensure engine creates structure, then overwrite fields.
            engine_user = bandit_engine.get_user(r.user_id)
            loaded = _deserialize_user(r.user_id, payload)

            engine_user.update(loaded)


