"""SQLite 历史记录（简单版，暂未接入命令行主流程，可单独调用）。

表 history:
    id / created_at / image_path / age / gender / height_cm / weight_kg / activity_level
    / total_weight_g / total_calories / foods(json) / bmr / tdee / llm_output
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..utils.config_loader import ensure_dir, get, resolve_path
from ..utils.logger import get_logger

log = get_logger("数据库")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS history (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at     TEXT NOT NULL,
    image_path     TEXT,
    age            INTEGER,
    gender         TEXT,
    height_cm      REAL,
    weight_kg      REAL,
    activity_level INTEGER,
    total_weight_g REAL,
    total_calories REAL,
    bmr            REAL,
    tdee           REAL,
    foods          TEXT,
    llm_output     TEXT
);
CREATE INDEX IF NOT EXISTS idx_history_created ON history(created_at);
"""

# 上位机新增字段：记录归属用户与餐次（老库通过 ALTER TABLE 平滑升级）
_MIGRATIONS = (
    ("user_name", "TEXT"),
    ("meal_type", "TEXT"),
)


class DBManager:
    """极简 SQLite 封装。"""

    def __init__(self, db_path: Optional[str | Path] = None):
        if db_path is None:
            p = resolve_path("database.path", "data/history.db")
        else:
            p = Path(db_path)
            if not p.is_absolute():
                p = resolve_path("database.path", "data/history.db").parent / p
        ensure_dir(p.parent)
        self.db_path = p
        self._init_db()

    # ------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(history)").fetchall()}
            for col, ctype in _MIGRATIONS:
                if col not in cols:
                    conn.execute(f"ALTER TABLE history ADD COLUMN {col} {ctype}")
                    log.info(f"数据库升级：history 新增字段 {col}")
        log.debug(f"数据库就绪：{self.db_path}")

    # ------------------------------------------------------------------
    def save_record(
        self,
        image_path: str,
        user_info: Dict[str, Any],
        foods: List[Dict[str, Any]],
        totals: Dict[str, Any],
        health: Optional[Dict[str, Any]] = None,
        llm_output: str = "",
        user_name: str = "",
        meal_type: str = "",
    ) -> int:
        """写入一条记录，返回自增 id。"""
        health = health or {}
        sql = """
        INSERT INTO history (created_at, image_path, age, gender, height_cm, weight_kg,
                             activity_level, total_weight_g, total_calories, bmr, tdee, foods, llm_output,
                             user_name, meal_type)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        values = (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            str(image_path),
            user_info.get("age"),
            user_info.get("gender"),
            user_info.get("height_cm"),
            user_info.get("weight_kg"),
            user_info.get("activity_level"),
            totals.get("weight_g"),
            totals.get("calories"),
            health.get("bmr"),
            health.get("tdee"),
            json.dumps(foods, ensure_ascii=False),
            llm_output or "",
            user_name or "",
            meal_type or "",
        )
        with self._connect() as conn:
            cur = conn.execute(sql, values)
            rid = int(cur.lastrowid)
        log.info(f"历史记录已保存：#{rid}")
        return rid

    # ------------------------------------------------------------------
    def get_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM history ORDER BY id DESC LIMIT ?", (int(limit),)
            ).fetchall()
        out: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            try:
                item["foods"] = json.loads(item.get("foods") or "[]")
            except json.JSONDecodeError:
                item["foods"] = []
            out.append(item)
        return out

    def get_record(self, record_id: int) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM history WHERE id = ?", (int(record_id),)).fetchone()
        if row is None:
            return None
        item = dict(row)
        try:
            item["foods"] = json.loads(item.get("foods") or "[]")
        except json.JSONDecodeError:
            item["foods"] = []
        return item

    def count(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM history").fetchone()[0])

    # ------------------------------------------------------------------
    def day_summary(self, user_name: str, day: Optional[str] = None,
                    exclude_id: Optional[int] = None) -> Dict[str, Any]:
        """某用户某天的摄入汇总（用于「本餐 + 今日已吃」的对比）。

        exclude_id：排除某条记录（通常是刚写入的本餐），避免把本餐算两遍。
        返回 {calories, protein_g, fat_g, carb_g, weight_g, meals: {早餐: kcal, ...}, count}
        """
        day = day or datetime.now().strftime("%Y-%m-%d")
        sql = ("SELECT meal_type, total_calories, foods FROM history "
               "WHERE substr(created_at, 1, 10) = ? AND (user_name = ? OR ? = '')")
        params: list = [day, user_name, user_name]
        if exclude_id:
            sql += " AND id != ?"
            params.append(int(exclude_id))
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        out = {"calories": 0.0, "protein_g": 0.0, "fat_g": 0.0, "carb_g": 0.0,
               "weight_g": 0.0, "meals": {}, "count": 0, "day": day}
        for row in rows:
            kcal = float(row["total_calories"] or 0)
            out["calories"] += kcal
            out["count"] += 1
            meal = row["meal_type"] or "未标注"
            out["meals"][meal] = round(out["meals"].get(meal, 0.0) + kcal, 1)
            try:
                for f in json.loads(row["foods"] or "[]"):
                    out["weight_g"] += float(f.get("weight_g", 0) or 0)
                    out["protein_g"] += float(f.get("protein_g", 0) or 0)
                    out["fat_g"] += float(f.get("fat_g", 0) or 0)
                    out["carb_g"] += float(f.get("carb_g", 0) or 0)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
        for k in ("calories", "protein_g", "fat_g", "carb_g", "weight_g"):
            out[k] = round(out[k], 1)
        return out

    def user_records(self, user_name: str, limit: int = 50) -> List[Dict[str, Any]]:
        """某用户的历史记录（新的在前）。"""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM history WHERE user_name = ? ORDER BY id DESC LIMIT ?",
                (user_name, int(limit)),
            ).fetchall()
        out: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            try:
                item["foods"] = json.loads(item.get("foods") or "[]")
            except json.JSONDecodeError:
                item["foods"] = []
            out.append(item)
        return out

    # ------------------------------------------------------------------
    def export_csv(self, csv_path: Optional[str | Path] = None, limit: Optional[int] = None) -> Path:
        """导出为 CSV（pandas 可选，缺失时用 csv 模块）。"""
        if csv_path is None:
            csv_path = self.db_path.parent / "history.csv"
        path = Path(csv_path)
        ensure_dir(path.parent)

        sql = "SELECT * FROM history ORDER BY id DESC"
        if limit:
            sql += f" LIMIT {int(limit)}"
        with self._connect() as conn:
            rows = [dict(r) for r in conn.execute(sql).fetchall()]

        try:
            import pandas as pd

            pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
        except ImportError:
            import csv

            if rows:
                with open(path, "w", encoding="utf-8-sig", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(rows)
        log.info(f"历史记录已导出：{path}（{len(rows)} 条）")
        return path


if __name__ == "__main__":
    db = DBManager()
    rid = db.save_record(
        image_path="data/images/test.jpg",
        user_info={"age": 25, "gender": "男", "height_cm": 175, "weight_kg": 70, "activity_level": 1},
        foods=[{"name": "rice", "weight_g": 180, "calories": 234}],
        totals={"weight_g": 180, "calories": 234},
        health={"bmr": 1673.8, "tdee": 2301.4},
        llm_output="（示例）",
    )
    print("写入 id:", rid, "总条数:", db.count())
    print(db.get_history(3))
    print("导出:", db.export_csv())
