# -*- coding: utf-8 -*-
"""用户档案的本地存储（JSON）。

设计要点：
  - 存在 data/users.json，与 SQLite 历史共用同一个 data 目录；
  - 结构 {"active": "姓名", "users": [ {...}, ... ]}；
  - 不需要每次重复填写身体信息，下次打开自动选中上次的用户；
  - 写盘用「先写临时文件再替换」的方式，避免异常退出把档案写坏。
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

USER_FIELDS = ("name", "age", "gender", "height_cm", "weight_kg", "activity_level")

ACTIVITY_NAMES = {1: "轻度（久坐/几乎不运动）", 2: "中度（每周 3-5 次运动）",
                  3: "重度（每周 6-7 次运动）", 4: "极重（体力劳动/运动员）"}


class UserStore:
    """用户档案的增删改查。"""

    def __init__(self, path: Optional[str | Path] = None):
        if path is None:
            root = Path(__file__).resolve().parents[1]
            path = root / "data" / "users.json"
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: Dict[str, Any] = {"active": "", "users": []}
        self.load()

    # ------------------------------------------------------------------ IO
    def load(self) -> Dict[str, Any]:
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(raw, dict) and isinstance(raw.get("users"), list):
                    self._data = {"active": str(raw.get("active", "")), "users": raw["users"]}
                elif isinstance(raw, list):          # 兼容早期纯列表格式
                    self._data = {"active": "", "users": raw}
            except (json.JSONDecodeError, OSError):
                # 文件损坏时保留空档案，不抛异常打断启动
                self._data = {"active": "", "users": []}
        return self._data

    def save(self) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    # ------------------------------------------------------------- 查询
    @property
    def users(self) -> List[Dict[str, Any]]:
        return list(self._data.get("users", []))

    def names(self) -> List[str]:
        return [str(u.get("name", "")) for u in self.users if u.get("name")]

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        for u in self.users:
            if str(u.get("name", "")) == str(name):
                return dict(u)
        return None

    @property
    def active_name(self) -> str:
        name = str(self._data.get("active", ""))
        return name if self.get(name) else (self.names()[0] if self.names() else "")

    def active(self) -> Optional[Dict[str, Any]]:
        name = self.active_name
        return self.get(name) if name else None

    def set_active(self, name: str) -> None:
        if self.get(name):
            self._data["active"] = str(name)
            self.save()

    # ------------------------------------------------------------- 写入
    def upsert(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """新增或更新（按姓名匹配）。返回保存后的档案。"""
        clean = self._validate(profile)
        for i, u in enumerate(self._data["users"]):
            if str(u.get("name", "")) == clean["name"]:
                clean["created_at"] = u.get("created_at") or datetime.now().isoformat(timespec="seconds")
                clean["updated_at"] = datetime.now().isoformat(timespec="seconds")
                self._data["users"][i] = clean
                break
        else:
            clean["created_at"] = datetime.now().isoformat(timespec="seconds")
            clean["updated_at"] = clean["created_at"]
            self._data["users"].append(clean)
        self._data["active"] = clean["name"]
        self.save()
        return clean

    def delete(self, name: str) -> bool:
        before = len(self._data["users"])
        self._data["users"] = [u for u in self._data["users"] if str(u.get("name")) != str(name)]
        if len(self._data["users"]) != before:
            if self._data.get("active") == name:
                self._data["active"] = self.names()[0] if self.names() else ""
            self.save()
            return True
        return False

    # ------------------------------------------------------------- 校验
    @staticmethod
    def _validate(profile: Dict[str, Any]) -> Dict[str, Any]:
        name = str(profile.get("name", "")).strip()
        if not name:
            raise ValueError("姓名不能为空")
        try:
            age = int(profile.get("age"))
            height = float(profile.get("height_cm"))
            weight = float(profile.get("weight_kg"))
            activity = int(profile.get("activity_level", 1))
        except (TypeError, ValueError):
            raise ValueError("年龄/身高/体重/活动量必须是数字")
        if not (1 <= age <= 129):
            raise ValueError("年龄需在 1-129 之间")
        if not (80 <= height <= 250):
            raise ValueError("身高需在 80-250 cm 之间")
        if not (10 <= weight <= 350):
            raise ValueError("体重需在 10-350 kg 之间")
        if activity not in (1, 2, 3, 4):
            raise ValueError("活动量只能是 1/2/3/4")
        gender = str(profile.get("gender", "")).strip()
        if gender not in ("男", "女"):
            raise ValueError("性别只能是 男 或 女")
        return {
            "name": name,
            "age": age,
            "gender": gender,
            "height_cm": round(height, 1),
            "weight_kg": round(weight, 1),
            "activity_level": activity,
        }
