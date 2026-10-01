# -*- coding: utf-8 -*-
"""本地 SQLite：歌单分组和简谱。"""

import sqlite3
import os
import sys
import json
from datetime import datetime
from pathlib import Path

if getattr(sys, "frozen", False):
    # 安装到 Program Files 后目录通常不可写，用户数据放到当前用户目录。
    _user_data = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "Jianpu"
    DB_PATH = _user_data / "scores.db"
else:
    DB_PATH = Path(__file__).resolve().parent / "data" / "scores.db"
DEFAULT_GROUP = "我的歌单"


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS groups (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            key_sig TEXT,
            meter TEXT,
            bpm INTEGER,
            source TEXT NOT NULL,
            note_count INTEGER,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS settings (
            name TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    cols = {row[1] for row in conn.execute("PRAGMA table_info(scores)")}
    if "group_id" not in cols:
        conn.execute("ALTER TABLE scores ADD COLUMN group_id INTEGER")
    if "is_favorite" not in cols:
        conn.execute("ALTER TABLE scores ADD COLUMN is_favorite INTEGER NOT NULL DEFAULT 0")
    if "sort_order" not in cols:
        conn.execute("ALTER TABLE scores ADD COLUMN sort_order INTEGER")
        rows = conn.execute("SELECT id FROM scores ORDER BY updated_at DESC, id DESC").fetchall()
        for order, row in enumerate(rows):
            conn.execute("UPDATE scores SET sort_order = ? WHERE id = ?", (order, row["id"]))
    conn.execute(
        "INSERT OR IGNORE INTO groups (name, created_at) VALUES (?, ?)",
        (DEFAULT_GROUP, _now()),
    )
    default_id = conn.execute(
        "SELECT id FROM groups WHERE name = ?", (DEFAULT_GROUP,)
    ).fetchone()["id"]
    conn.execute(
        "UPDATE scores SET group_id = ? WHERE group_id IS NULL",
        (default_id,),
    )
    missing_order = conn.execute(
        "SELECT id FROM scores WHERE sort_order IS NULL ORDER BY updated_at DESC, id DESC"
    ).fetchall()
    next_order = conn.execute(
        "SELECT COALESCE(MIN(sort_order), 0) - 1 FROM scores"
    ).fetchone()[0]
    for row in missing_order:
        conn.execute("UPDATE scores SET sort_order = ? WHERE id = ?", (next_order, row["id"]))
        next_order -= 1
    conn.commit()
    return conn


def list_groups():
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, name FROM groups ORDER BY id"
        ).fetchall()
    return [dict(r) for r in rows]


def create_group(name: str):
    name = (name or "").strip()
    if not name:
        raise ValueError("歌单名不能为空")
    with connect() as conn:
        try:
            cur = conn.execute(
                "INSERT INTO groups (name, created_at) VALUES (?, ?)",
                (name, _now()),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError("已经有同名歌单") from exc
        return int(cur.lastrowid)


def rename_group(group_id: int, name: str) -> None:
    name = (name or "").strip()
    if not name:
        raise ValueError("歌单名不能为空")
    with connect() as conn:
        try:
            conn.execute("UPDATE groups SET name = ? WHERE id = ?", (name, group_id))
        except sqlite3.IntegrityError as exc:
            raise ValueError("已经有同名歌单") from exc


def delete_group(group_id: int) -> None:
    with connect() as conn:
        count = conn.execute("SELECT COUNT(*) AS n FROM groups").fetchone()["n"]
        if count <= 1:
            raise ValueError("至少保留一个歌单")
        fallback = conn.execute(
            "SELECT id FROM groups WHERE id != ? ORDER BY id LIMIT 1",
            (group_id,),
        ).fetchone()
        conn.execute(
            "UPDATE scores SET group_id = ? WHERE group_id = ?",
            (fallback["id"], group_id),
        )
        conn.execute("DELETE FROM groups WHERE id = ?", (group_id,))


def save_score(title, source, key_sig, meter, bpm, note_count, group_id) -> int:
    now = _now()
    title = (title or "未命名").strip() or "未命名"
    with connect() as conn:
        row = conn.execute(
            "SELECT id FROM scores WHERE title = ? AND group_id = ?",
            (title, group_id),
        ).fetchone()
        if row:
            top = conn.execute(
                "SELECT COALESCE(MIN(sort_order), 0) - 1 FROM scores WHERE group_id = ?",
                (group_id,),
            ).fetchone()[0]
            conn.execute(
                """
                UPDATE scores
                SET key_sig=?, meter=?, bpm=?, source=?, note_count=?, updated_at=?, sort_order=?
                WHERE id=?
                """,
                (key_sig, meter, bpm, source, note_count, now, top, row["id"]),
            )
            return int(row["id"])
        top = conn.execute(
            "SELECT COALESCE(MIN(sort_order), 0) - 1 FROM scores WHERE group_id = ?",
            (group_id,),
        ).fetchone()[0]
        cur = conn.execute(
            """
            INSERT INTO scores
            (title, key_sig, meter, bpm, source, note_count, group_id, created_at, updated_at, sort_order)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (title, key_sig, meter, bpm, source, note_count, group_id, now, now, top),
        )
        return int(cur.lastrowid)


def update_score(score_id, source, key_sig, meter, bpm, note_count, *, reorder=True) -> None:
    """更新已载入歌曲的谱面内容，保留用户编辑过的歌名和歌单。"""
    with connect() as conn:
        if reorder:
            group = conn.execute("SELECT group_id FROM scores WHERE id = ?", (score_id,)).fetchone()
            group_id = group[0] if group else None
            top = conn.execute(
                "SELECT COALESCE(MIN(sort_order), 0) - 1 FROM scores WHERE group_id = ?",
                (group_id,),
            ).fetchone()[0]
            conn.execute(
                """
                UPDATE scores
                SET key_sig=?, meter=?, bpm=?, source=?, note_count=?, updated_at=?, sort_order=?
                WHERE id=?
                """,
                (key_sig, meter, bpm, source, note_count, _now(), top, score_id),
            )
        else:
            conn.execute(
                """
                UPDATE scores
                SET key_sig=?, meter=?, bpm=?, source=?, note_count=?
                WHERE id=?
                """,
                (key_sig, meter, bpm, source, note_count, score_id),
            )


def list_scores(group_id=None):
    with connect() as conn:
        if group_id is None:
            rows = conn.execute(
                """
                SELECT id, title, key_sig, meter, bpm, note_count, updated_at, group_id, is_favorite, sort_order
                FROM scores ORDER BY sort_order ASC, updated_at DESC, id DESC
                """
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT id, title, key_sig, meter, bpm, note_count, updated_at, group_id, is_favorite, sort_order
                FROM scores WHERE group_id = ?
                ORDER BY sort_order ASC, updated_at DESC, id DESC
                """,
                (group_id,),
            ).fetchall()
    return [dict(r) for r in rows]


def set_favorite(score_id: int, value: bool) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE scores SET is_favorite = ? WHERE id = ?",
            (1 if value else 0, score_id),
        )


def get_setting(name: str, default=None):
    with connect() as conn:
        row = conn.execute("SELECT value FROM settings WHERE name = ?", (name,)).fetchone()
    return row["value"] if row else default


def set_setting(name: str, value: str) -> None:
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO settings (name, value, updated_at) VALUES (?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
            """,
            (name, value, _now()),
        )


def get_score(score_id: int):
    with connect() as conn:
        row = conn.execute("SELECT * FROM scores WHERE id = ?", (score_id,)).fetchone()
    return dict(row) if row else None


def find_score_by_source(source: str):
    """按谱面原文寻找最近编辑过的歌曲，用于恢复创作页编辑状态。"""
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM scores WHERE source = ? ORDER BY updated_at DESC, id DESC LIMIT 1",
            (source or "",),
        ).fetchone()
    return dict(row) if row else None


def rename_score(score_id: int, title: str) -> None:
    title = (title or "").strip()
    if not title:
        raise ValueError("歌名不能为空")
    with connect() as conn:
        conn.execute("UPDATE scores SET title = ? WHERE id = ?", (title, score_id))


def move_score(score_id: int, group_id: int) -> None:
    with connect() as conn:
        top = conn.execute(
            "SELECT COALESCE(MIN(sort_order), 0) - 1 FROM scores WHERE group_id = ?",
            (group_id,),
        ).fetchone()[0]
        conn.execute(
            "UPDATE scores SET group_id = ?, sort_order = ? WHERE id = ?",
            (group_id, top, score_id),
        )


def reorder_scores(group_id: int, ordered_ids) -> None:
    """保存歌单内的拖动顺序；播放不会调用此函数。"""
    with connect() as conn:
        for order, score_id in enumerate(ordered_ids):
            conn.execute(
                "UPDATE scores SET sort_order = ? WHERE id = ? AND group_id = ?",
                (order, int(score_id), int(group_id)),
            )


def delete_score(score_id: int) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM scores WHERE id = ?", (score_id,))


def delete_scores(ids):
    with connect() as conn:
        conn.executemany("DELETE FROM scores WHERE id = ?", [(int(i),) for i in ids])


def delete_groups(ids):
    """批量删除歌单，歌曲移入保留的歌单，事务内完成。"""
    ids = set(map(int, ids))
    with connect() as conn:
        remaining = [r[0] for r in conn.execute("SELECT id FROM groups ORDER BY id") if r[0] not in ids]
        if not remaining:
            raise ValueError("至少保留一个歌单；歌曲不会被删除。")
        for group_id in ids:
            conn.execute("UPDATE scores SET group_id=? WHERE group_id=?", (remaining[0], group_id))
            conn.execute("DELETE FROM groups WHERE id=?", (group_id,))


def export_playlists(group_ids=None):
    with connect() as conn:
        groups = []
        for row in conn.execute("SELECT * FROM groups ORDER BY id"):
            if group_ids is not None and row["id"] not in group_ids:
                continue
            songs = [dict(s) for s in conn.execute(
                "SELECT * FROM scores WHERE group_id=? ORDER BY sort_order, id", (row["id"],))]
            for song in songs:
                song.pop("id")
                song.pop("group_id")
            groups.append({"name": row["name"], "songs": songs})
    return {"format": "jianpu-playlists", "version": 1, "groups": groups}


def import_playlists(payload):
    """严格验证后一次提交；同名歌单新建副本，不覆盖本地歌曲。"""
    if not isinstance(payload, dict) or payload.get("format") != "jianpu-playlists" or payload.get("version") != 1:
        raise ValueError("不是支持的三角洲口琴简谱歌单文件（版本 1）。")
    groups = payload.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("文件没有歌单。")
    for group in groups:
        if not isinstance(group, dict) or not isinstance(group.get("name"), str) or not group["name"].strip() or not isinstance(group.get("songs"), list):
            raise ValueError("歌单格式无效。")
        for song in group["songs"]:
            if not isinstance(song, dict) or any(not isinstance(song.get(k), str) for k in ("title", "source")):
                raise ValueError("歌曲缺少歌名或谱面原文。")
            for key in ("key_sig", "meter", "created_at", "updated_at"):
                if song.get(key) is not None and not isinstance(song[key], str):
                    raise ValueError(f"歌曲字段 {key} 无效。")
            for key in ("bpm", "note_count"):
                if song.get(key) is not None and (type(song[key]) is not int or song[key] < 0):
                    raise ValueError(f"歌曲字段 {key} 无效。")
            if song.get("is_favorite", 0) not in (0, 1):
                raise ValueError("歌曲收藏标记无效。")
    count = 0
    with connect() as conn:
        for group in groups:
            base = group["name"].strip()
            name, suffix = base, 1
            while conn.execute("SELECT 1 FROM groups WHERE name=?", (name,)).fetchone():
                name = f"{base}（导入 {suffix}）"
                suffix += 1
            gid = conn.execute("INSERT INTO groups(name,created_at) VALUES (?,?)", (name, _now())).lastrowid
            for order, song in enumerate(group["songs"]):
                conn.execute("""INSERT INTO scores
                    (title,source,key_sig,meter,bpm,note_count,is_favorite,created_at,updated_at,group_id,sort_order)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?)""", (
                    song["title"], song["source"], song.get("key_sig"), song.get("meter"), song.get("bpm"),
                    song.get("note_count"), int(song.get("is_favorite", 0)),
                    song.get("created_at") or _now(), song.get("updated_at") or _now(), gid, order))
                count += 1
    return len(groups), count
