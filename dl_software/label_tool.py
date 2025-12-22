#!/usr/bin/env python3
"""
PySide6 Labeling Tool (MVP)
- Zoom/pan, Select/Rect/Poly/Erase
- 중앙 annotations.json 하나만 사용(단일 파일 관리)
- 이미지 이동 전/저장 전/변형 시 -> 자동 커밋 & 자동 저장(디바운스)
- Rect 코너 리사이즈 핸들
- 가로/세로 크로스헤어(마우스 따라다님)
- ID 1..N 재번호(next_sid=N+1)
- 메타데이터(meta): classes(name,color), 각 이미지의 image_size
- YOLO 데이터셋 Export(Detection: Rect만)
- Train 버튼: ultralytics 자동 설치(선택) → 학습 자동 실행, 진행률/로그 UI 제공
- Inference 버튼: 모델/데이터 선택 → 현재 이미지/폴더 추론 결과를 캔버스에 직접 표시(오버레이)
- (NEW) Inference 결과 bbox + class name + confidence 라벨 표시
- (NEW) 왼쪽 기본 'object' 자동 생성 제거
- (NEW) 단일 inference 결과 캐시에 저장하여 재방문 시 자동 복원
- (NEW) 오른쪽 패널에 detection 리스트(class + conf) 표시
"""
from __future__ import annotations
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, QPointF, QRectF, Signal
from PySide6.QtGui import (
    QPixmap, QAction, QPen, QBrush, QColor, QIcon, QPainter, QFont
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QGraphicsItem, QGraphicsRectItem, QGraphicsPathItem, QGraphicsLineItem, QGraphicsTextItem,
    QGraphicsEllipseItem,
    QListWidget, QListWidgetItem, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QMessageBox, QToolBar, QDialog, QSlider,
    QSpinBox, QGridLayout, QTextEdit, QProgressBar, QRadioButton, QButtonGroup,
    QComboBox, QStackedLayout,
QStatusBar, QMenu
)
import copy
import math
import time
import json, sys, os, random, shutil, subprocess, re
import numpy as np
import datetime
from dataclasses import dataclass, field
try:
    from PIL import Image
except ImportError:
    Image = None
from collections import Counter
from pathlib import Path
try:
    from shiboken6 import shiboken6 as sb
except Exception:
    sb = None  # isValid 체크 불가 환경 대비

try:
    import albumentations as A
    _AUGMENT_LIB_AVAILABLE = Image is not None
except ImportError:
    A = None
    _AUGMENT_LIB_AVAILABLE = False

from .ui.widgets import Header, HSep, TitledGroup
from .ui.augment_dialog import AugmentDialog

# UI tweak: unify handle size for rect/poly edit points
# Increase this to make grabbing handles easier.
HANDLE_SIZE_PX: float = 10.0
# Rotation handle sizing/color
ROTATE_HANDLE_SIZE: float = max(18.0, HANDLE_SIZE_PX * 1.4)
ROTATE_STEM_LEN: float = ROTATE_HANDLE_SIZE * 1.4
ROTATE_COLOR: QColor = QColor(66, 133, 244)  # blue
# Reserved color band (hue degrees) we avoid for class colors
# because Smart preview uses a cyan/azure overlay around ~200°.
SMART_PREVIEW_FORBIDDEN_HUES: list[tuple[int, int]] = [(185, 215)]

@dataclass
class ExportAugmentationConfig:
    enabled: bool = False
    multiplier: int = 1
    techniques: list[str] = field(default_factory=list)
    details: dict | None = None


AUGMENTATION_TECHNIQUES: list[tuple[str, str]] = [
    ("flip_lr", "Flip horizontal"),
    ("flip_ud", "Flip vertical"),
    ("rotate90", "Random 90° rotate"),
    ("rotation", "Rotate ±15°"),
    ("crop", "Random crop & resize"),
    ("shear", "Shear"),
    ("grayscale", "Grayscale"),
    ("hue_saturation", "Hue/Saturation"),
    ("brightness", "Brightness/Exposure"),
    ("blur", "Blur"),
    ("noise", "Noise"),
]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECTS_ROOT = PROJECT_ROOT / "projects"
VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class ProjectMeta:
    name: str
    created_at: str
    last_opened: str | None = None
    last_opened_ts: float | None = None
    classes: list[dict[str, str]] = field(default_factory=list)
    augment_config: dict = field(default_factory=dict)
    export_ratios: dict[str, int] = field(default_factory=lambda: {"train": 80, "val": 10, "test": 10})

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "created_at": self.created_at,
            "last_opened": self.last_opened,
            "last_opened_ts": self.last_opened_ts,
            "classes": list(self.classes),
            "augment_config": dict(self.augment_config),
            "export_ratios": dict(self.export_ratios),
        }

    @staticmethod
    def from_dict(data: dict, *, name: str | None = None) -> "ProjectMeta":
        meta_name = name or data.get("name", "project")
        classes = data.get("classes") or []
        augment = data.get("augment_config") or {}
        ratios = data.get("export_ratios") or {"train": 80, "val": 10, "test": 10}
        created_at = data.get("created_at") or datetime.datetime.utcnow().isoformat()
        last_opened = data.get("last_opened")
        last_opened_ts = data.get("last_opened_ts")
        return ProjectMeta(
            name=meta_name,
            created_at=created_at,
            last_opened=last_opened,
            last_opened_ts=float(last_opened_ts) if last_opened_ts is not None else None,
            classes=classes,
            augment_config=augment,
            export_ratios=ratios,
        )


@dataclass
class Project:
    name: str
    root: Path
    images_dir: Path
    annotations_path: Path
    meta_path: Path
    meta: ProjectMeta

    def save_meta(self):
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)
        self.meta_path.write_text(json.dumps(self.meta.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


class ProjectManager:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def list_projects(self) -> list[Project]:
        projects: list[Project] = []
        for child in self.root.iterdir():
            if not child.is_dir():
                continue
            proj = self._build_project(child)
            if proj is not None:
                projects.append(proj)
        # 최근 열람 시각(last_opened) 기준으로 정렬; 없으면 mtime 사용
        def _sort_key(p: Project) -> float:
            if getattr(p.meta, "last_opened_ts", None) is not None:
                return float(p.meta.last_opened_ts)
            if p.meta.last_opened:
                try:
                    return datetime.datetime.fromisoformat(p.meta.last_opened).timestamp()
                except Exception:
                    pass
            candidates = [p.root]
            for sub in ("annotations.json", "meta.json"):
                candidates.append(p.root / sub)
            mt = 0.0
            for path in candidates:
                try:
                    mt = max(mt, path.stat().st_mtime)
                except Exception:
                    continue
            return mt
        projects.sort(key=_sort_key, reverse=True)
        return projects

    def _build_project(self, root: Path) -> Project | None:
        try:
            images_dir = root / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            annotations = root / "annotations.json"
            annotations.touch(exist_ok=True)
            meta_path = root / "meta.json"
            meta = self._load_meta(meta_path, root.name)
            return Project(root.name, root, images_dir, annotations, meta_path, meta)
        except Exception:
            return None

    def _load_meta(self, meta_path: Path, name: str) -> ProjectMeta:
        if meta_path.exists():
            try:
                data = json.loads(meta_path.read_text(encoding="utf-8"))
                return ProjectMeta.from_dict(data, name=name)
            except Exception:
                pass
        meta = ProjectMeta(name=name, created_at=datetime.datetime.utcnow().isoformat())
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return meta

    def create_project(self, name: str) -> Project:
        safe_name = name.strip()
        if not safe_name or "/" in safe_name or "\\" in safe_name:
            raise ValueError("Invalid project name")
        root = self.root / safe_name
        if root.exists():
            raise FileExistsError(f"Project '{safe_name}' already exists.")
        root.mkdir(parents=True, exist_ok=False)
        images_dir = root / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        annotations = root / "annotations.json"
        annotations.write_text(json.dumps({"images": {}, "meta": {"classes": []}}, ensure_ascii=False, indent=2), encoding="utf-8")
        meta_path = root / "meta.json"
        meta = ProjectMeta(name=safe_name, created_at=datetime.datetime.utcnow().isoformat())
        meta_path.write_text(json.dumps(meta.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return Project(safe_name, root, images_dir, annotations, meta_path, meta)

    def load_project(self, name: str) -> Project | None:
        path = self.root / name
        if not path.exists() or not path.is_dir():
            return None
        return self._build_project(path)

def _make_icon_layer(where: str) -> QIcon:
    """Small icon showing bring-to-front / send-to-back.
    where: 'front' or 'back'
    """
    d = 18
    pm = QPixmap(d, d); pm.fill(Qt.transparent)
    p = QPainter(pm); p.setRenderHint(QPainter.Antialiasing, True)
    edge_g = QPen(QColor(90, 90, 90, 200), 1)
    accent = QColor(255, 153, 0)  # orange accent similar to screenshot
    white = Qt.white

    # positions
    back_rect = QtCore.QRectF(3, 3, d-9, d-9)
    front_rect = QtCore.QRectF(6, 6, d-9, d-9)

    if where == 'front':
        # Draw back (white) then front (accent)
        p.setPen(edge_g); p.setBrush(white); p.drawRoundedRect(back_rect, 3, 3)
        p.setPen(QPen(accent, 2)); p.setBrush(accent.lighter(140))
        p.drawRoundedRect(front_rect, 3, 3)
    else:  # 'back'
        # Draw back (accent) then front (white)
        p.setPen(QPen(accent, 2)); p.setBrush(accent.lighter(140)); p.drawRoundedRect(back_rect, 3, 3)
        p.setPen(edge_g); p.setBrush(white); p.drawRoundedRect(front_rect, 3, 3)
    p.end()
    return QIcon(pm)


# ----------------------------- Annotation Store -----------------------------
class AnnotationStore:
    """
    파일 스키마
    {
      "images": {
        "<abs_image_path>": {
          "shapes": [ {type, points, class, id}, ... ],
          "image_size": [W, H]
        }
      },
      "meta": {
        "classes": [ {"name":"object","color":"#00FF00"}, ... ]
      }
    }
    """
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else None
        self._db: dict = {"images": {}, "meta": {"classes": []}}

    def set_path(self, p: Path):
        self.path = Path(p)

    def load(self):
        if not self.path: return
        try:
            if self.path.exists():
                self._db = json.loads(self.path.read_text(encoding="utf-8"))
            else:
                # New dataset folder without annotations.json yet → start fresh
                self._db = {"images": {}, "meta": {"classes": []}}
            if "images" not in self._db or not isinstance(self._db["images"], dict):
                self._db["images"] = {}
            if "meta" not in self._db or not isinstance(self._db["meta"], dict):
                self._db["meta"] = {}
            if "classes" not in self._db["meta"] or not isinstance(self._db["meta"]["classes"], list):
                self._db["meta"]["classes"] = []
        except Exception:
            self._db = {"images": {}, "meta": {"classes": []}}

    def save(self):
        if not self.path: return
        self.path.write_text(json.dumps(self._db, ensure_ascii=False, indent=2), encoding="utf-8")

    # image-level
    def get(self, image_path: Path) -> dict:
        return self._db["images"].get(str(Path(image_path).resolve()), {"shapes": [], "image_size": None})

    def put(self, image_path: Path, shapes: list[dict], img_size: tuple[int,int] | None = None):
        rec = {"shapes": shapes}
        if img_size is not None:
            rec["image_size"] = [int(img_size[0]), int(img_size[1])]
        self._db["images"][str(Path(image_path).resolve())] = rec

    # meta-level (classes)
    def list_classes(self) -> list[dict]:
        return list(self._db.get("meta", {}).get("classes", []))

    def set_classes(self, classes: list[dict]):
        self._db.setdefault("meta", {})["classes"] = classes

    def upsert_class(self, name: str, color_hex: str):
        classes = self.list_classes()
        for c in classes:
            if c.get("name") == name:
                c["color"] = color_hex
                self.set_classes(classes); return
        classes.append({"name": name, "color": color_hex})
        self.set_classes(classes)

    def remove_class(self, name: str):
        classes = [c for c in self.list_classes() if c.get("name") != name]
        self.set_classes(classes)


# ----------------------------- Data Model -----------------------------
class Shape:
    def __init__(self, shape_type: str, points: list[tuple[float, float]], klass: str, sid: int):
        self.type = shape_type
        self.points = points
        self.klass = klass
        self.id = sid

    def to_dict(self):
        return {"type": self.type, "points": self.points, "class": self.klass, "id": self.id}

    @staticmethod
    def from_dict(d):
        return Shape(d["type"], [tuple(p) for p in d["points"]], d["class"], d.get("id", -1))


# ----------------------------- Rect Resize Handles -----------------------------
class HandleItem(QGraphicsRectItem):
    TL, TR, BL, BR = range(4)

    def __init__(self, corner: int, size: float = HANDLE_SIZE_PX, parent: QGraphicsItem | None = None):
        super().__init__(-size/2, -size/2, size, size, parent)
        self.corner = corner
        self.setBrush(Qt.white)
        self.setPen(QPen(Qt.black, 1))
        self.setZValue(1e7)
        self.setAcceptedMouseButtons(Qt.LeftButton)
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, False)
        self._fixed_corner_pos = None
        self.setCursor(Qt.SizeFDiagCursor if corner in (HandleItem.TL, HandleItem.BR) else Qt.SizeBDiagCursor)

    def mousePressEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        rect_item: RectItem = self.parentItem()  # type: ignore
        r = rect_item.rect()
        self._fixed_corner_pos = {
            self.TL: r.bottomRight(),
            self.TR: r.bottomLeft(),
            self.BL: r.topRight(),
            self.BR: r.topLeft(),
        }[self.corner]
        lt = rect_item._label_tool()
        if lt is not None:
            lt._ensure_pre_move_snapshot()
        event.accept()

    def mouseMoveEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        rect_item: RectItem = self.parentItem()  # type: ignore
        if self._fixed_corner_pos is None:
            event.ignore(); return
        p1 = QPointF(self._fixed_corner_pos)
        p2 = rect_item.mapFromScene(event.scenePos())
        min_w, min_h = 3.0, 3.0
        nr = QRectF(p1, p2).normalized()
        if nr.width() < min_w:
            p2.setX(p1.x() + (min_w if p2.x() >= p1.x() else -min_w))
        if nr.height() < min_h:
            p2.setY(p1.y() + (min_h if p2.y() else -min_h))
        rect_item.setRect(QRectF(p1, p2).normalized())
        rect_item.update_handles_positions()
        rect_item.on_geometry_changed()
        lt = rect_item._label_tool()
        if lt is not None:
            lt._move_snapshot_dirty = True
        event.accept()

    def mouseReleaseEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        rect_item: RectItem = self.parentItem()  # type: ignore
        rect_item.on_geometry_changed()
        lt = rect_item._label_tool()
        if lt is not None:
            lt._finalize_move_snapshot()
            lt._commit_cur_to_store()
            lt._schedule_autosave()
            lt._schedule_history_push(0)
        self._fixed_corner_pos = None
        event.accept()


class RectItem(QGraphicsRectItem):
    """선택 시 4개 코너 핸들로 리사이즈 가능."""
    def __init__(self, rect: QRectF, color: QColor, sid: int, klass: str):
        super().__init__(rect)
        self.setFlags(
            QGraphicsItem.ItemIsSelectable |
            QGraphicsItem.ItemIsMovable |
            QGraphicsItem.ItemSendsGeometryChanges
        )
        self.setPen(QPen(color, 3))
        self.setBrush(Qt.NoBrush)
        self.sid = sid
        self.klass = klass
        self.handles: list[HandleItem] = [HandleItem(i, size=HANDLE_SIZE_PX, parent=self) for i in range(4)]
        for h in self.handles: h.setVisible(False)
        self.update_handles_positions()

    def _label_tool(self) -> "LabelTool | None":
        sc = self.scene()
        if sc is None:
            return None
        views = sc.views() if hasattr(sc, "views") else []
        if not views:
            return None
        widget = views[0]
        seen: set[int] = set()
        queue: list[QtWidgets.QWidget | None] = []
        if isinstance(widget, QtWidgets.QWidget):
            queue.append(widget)
            queue.append(widget.window())
        while queue:
            obj = queue.pop(0)
            if obj is None:
                continue
            oid = id(obj)
            if oid in seen:
                continue
            seen.add(oid)
            # 직접 LabelTool 인스턴스인지 확인
            if hasattr(obj, "_ensure_pre_move_snapshot") and hasattr(obj, "_finalize_move_snapshot"):
                return obj  # type: ignore[return-value]
            # LabelTab 처럼 내부에 tool/label_tool 속성으로 보관하는 경우 처리
            candidate = None
            if hasattr(obj, "tool"):
                candidate = getattr(obj, "tool")
            elif hasattr(obj, "label_tool"):
                candidate = getattr(obj, "label_tool")
            if candidate is not None and hasattr(candidate, "_ensure_pre_move_snapshot") and hasattr(candidate, "_finalize_move_snapshot"):
                return candidate  # type: ignore[return-value]
            if isinstance(obj, QtWidgets.QWidget):
                queue.append(obj.parentWidget())
        return None

    def mousePressEvent(self, ev: QtWidgets.QGraphicsSceneMouseEvent):
        lt = self._label_tool()
        if lt is not None:
            lt._ensure_pre_move_snapshot()
        sc = self.scene()
        if not (ev.modifiers() & (Qt.ControlModifier | Qt.ShiftModifier)):
            if sc:
                sc.clearSelection()
            self.setSelected(True)
        super().mousePressEvent(ev)

    def update_handles_positions(self):
        r = self.rect()
        pos = {
            HandleItem.TL: r.topLeft(),
            HandleItem.TR: r.topRight(),
            HandleItem.BL: r.bottomLeft(),
            HandleItem.BR: r.bottomRight(),
        }
        for h in self.handles: h.setPos(pos[h.corner])

    def setRect(self, rect: QRectF):  # type: ignore[override]
        super().setRect(rect.normalized())
        self.update_handles_positions()

    def itemChange(self, change, value):
        lt = self._label_tool()

        if change == QGraphicsItem.ItemPositionChange and lt is not None:
            lt._ensure_pre_move_snapshot()

        if change in (QGraphicsItem.ItemPositionHasChanged, QGraphicsItem.ItemTransformHasChanged):
            self.update_handles_positions()
            if lt is not None:
                lt._move_snapshot_dirty = True
        elif change == QGraphicsItem.ItemSelectedHasChanged:
            vis = bool(value)
            for h in self.handles: h.setVisible(vis)
        return super().itemChange(change, value)

    def mouseReleaseEvent(self, ev: QtWidgets.QGraphicsSceneMouseEvent):
        lt = self._label_tool()
        if lt is not None:
            lt._finalize_move_snapshot()
            lt._commit_cur_to_store()
            lt._schedule_autosave()
            lt._schedule_history_push(0)
        super().mouseReleaseEvent(ev)

    def on_geometry_changed(self):
        lt = self._label_tool()
        if lt is None:
            return
        lt._commit_cur_to_store()
        lt._schedule_autosave()
        try:
            lt._update_counts_ui()
        except Exception:
            pass

    def contextMenuEvent(self, event: QtWidgets.QGraphicsSceneContextMenuEvent):  # type: ignore[override]
        sc = self.scene()
        if sc is None or not self.isSelected():
            return super().contextMenuEvent(event)
        selected = [it for it in sc.selectedItems() if isinstance(it, RectItem)]
        if len(selected) != 1 or selected[0] is not self:
            return super().contextMenuEvent(event)

        menu = QMenu()
        act_front = menu.addAction(_make_icon_layer('front'), "Bring to Front")
        act_back = menu.addAction(_make_icon_layer('back'), "Send to Back")
        menu.addSeparator()
        act_sam_box = menu.addAction("Use as SAM Box Prompt")
        chosen = menu.exec(event.screenPos())
        if chosen is None:
            event.accept(); return

        lt = self._label_tool()
        if lt is not None:
            lt._ensure_pre_move_snapshot()

        if chosen == act_front:
            self._reorder_layer('front')
        elif chosen == act_back:
            self._reorder_layer('back')
        elif chosen == act_sam_box:
            # Apply this rectangle as SAM-ONNX box prompt (two corners with labels [2,3])
            if lt is not None:
                try:
                    r = self.rect()
                    tl = self.mapToScene(r.topLeft())
                    br = self.mapToScene(r.bottomRight())
                    lt._smart_set_box_prompt(float(tl.x()), float(tl.y()), float(br.x()), float(br.y()))
                except Exception:
                    pass

        # finalize
        if lt is not None:
            lt._finalize_move_snapshot()
            lt._commit_cur_to_store()
            lt._schedule_autosave()
            lt._schedule_history_push(0)
        event.accept()

    def _reorder_layer(self, where: str):
        sc = self.scene()
        lt = self._label_tool()
        if sc is None or lt is None:
            return
        shapes = [it for it in sc.items() if isinstance(it, (RectItem, PolyItem))]
        if len(shapes) <= 1:
            return
        if where == 'front':
            max_sid = max(getattr(it, 'sid', 0) for it in shapes)
            self.sid = max_sid + 1
        else:
            min_sid = min(getattr(it, 'sid', 0) for it in shapes)
            self.sid = min_sid - 1
        lt._normalize_shape_zvalues()
        try:
            lt._renumber_ids()
            lt._restack_shapes_by_sid()
        except Exception:
            pass
        try:
            sc.update()
        except Exception:
            pass


# ----------------------------- Polygon Item -----------------------------
class PolyHandleItem(QtWidgets.QGraphicsEllipseItem):
    def __init__(self, index: int, parent: QGraphicsItem | None = None, size: float = HANDLE_SIZE_PX):
        super().__init__(-size / 2, -size / 2, size, size, parent)
        self.index = index
        self.setBrush(Qt.white)
        self.setPen(QPen(Qt.black, 1))
        self.setZValue(1e7)
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, False)
        # accept both left (drag move) and right (context delete)
        self.setAcceptedMouseButtons(Qt.LeftButton | Qt.RightButton)
        self.setCursor(Qt.OpenHandCursor)

    def _poly_item(self) -> "PolyItem | None":
        parent = self.parentItem()
        return parent if isinstance(parent, PolyItem) else None

    def mousePressEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        poly = self._poly_item()
        if poly is None:
            event.ignore()
            return
        # Right-click → context menu: delete point
        if event.button() == Qt.RightButton:
            menu = QMenu()
            act_ins = menu.addAction("Insert Point Here")
            act_del = menu.addAction("Delete Point")
            chosen = menu.exec(event.screenPos())
            if chosen in (act_ins, act_del):
                lt = poly._label_tool()
                if lt is not None:
                    lt._ensure_pre_move_snapshot()
                if chosen == act_del:
                    ok = poly.remove_point(self.index)
                    if not ok and lt is not None:
                        try:
                            lt._show_status("Polygon needs at least 3 points")
                        except Exception:
                            pass
                else:  # insert
                    poly.insert_point_at(event.scenePos())
                if lt is not None:
                    lt._finalize_move_snapshot(); lt._commit_cur_to_store(); lt._schedule_autosave(); lt._schedule_history_push(0)
                event.accept(); return
        sc = poly.scene()
        if sc and not poly.isSelected():
            sc.clearSelection()
            poly.setSelected(True)
        lt = poly._label_tool()
        if lt is not None:
            lt._ensure_pre_move_snapshot()
        self.setCursor(Qt.ClosedHandCursor)
        event.accept()

    def mouseMoveEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        poly = self._poly_item()
        if poly is None:
            event.ignore()
            return
        local_pos = poly.mapFromScene(event.scenePos())
        poly.set_point(self.index, local_pos)
        lt = poly._label_tool()
        if lt is not None:
            lt._move_snapshot_dirty = True
        event.accept()

    def mouseReleaseEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        poly = self._poly_item()
        if poly is not None:
            poly.on_polygon_changed()
            lt = poly._label_tool()
            if lt is not None:
                lt._finalize_move_snapshot()
                lt._commit_cur_to_store()
                lt._schedule_autosave()
                lt._schedule_history_push(0)
            poly.update_handles_positions()
        self.setCursor(Qt.OpenHandCursor)
        event.accept()


class RotateHandleItem(QtWidgets.QGraphicsEllipseItem):
    def __init__(self, poly: "PolyItem", size: float = ROTATE_HANDLE_SIZE):
        super().__init__(-size / 2, -size / 2, size, size)
        self.setBrush(Qt.white)
        self.setPen(QPen(ROTATE_COLOR, 2))
        self.setZValue(1e7)
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, False)
        self.setAcceptedMouseButtons(Qt.LeftButton)
        self.setCursor(Qt.OpenHandCursor)
        self._poly_ref = poly
        self._center: QPointF | None = None
        self._start_angle: float | None = None
        self._base_points: list[QPointF] | None = None

    def _poly_item(self) -> "PolyItem | None":
        return self._poly_ref

    def mousePressEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        poly = self._poly_item()
        if poly is None or len(getattr(poly, '_points', [])) < 3:
            event.ignore(); return
        sc = poly.scene()
        if sc and not poly.isSelected():
            sc.clearSelection(); poly.setSelected(True)
        lt = poly._label_tool()
        if lt is not None:
            lt._ensure_pre_move_snapshot()
        # prepare base state
        c = poly._centroid()
        self._center = c
        local_pos = poly.mapFromScene(event.scenePos())
        self._start_angle = math.atan2(local_pos.y() - c.y(), local_pos.x() - c.x())
        self._base_points = [QPointF(p) for p in poly._points]
        self.setCursor(Qt.ClosedHandCursor)
        event.accept()

    def mouseMoveEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        poly = self._poly_item()
        if poly is None or self._center is None or self._start_angle is None or self._base_points is None:
            event.ignore(); return
        cur_local = poly.mapFromScene(event.scenePos())
        ang = math.atan2(cur_local.y() - self._center.y(), cur_local.x() - self._center.x())
        delta = ang - self._start_angle
        # rotate from base points to avoid incremental drift
        poly._set_points_from_rotated(self._base_points, self._center, delta)
        lt = poly._label_tool()
        if lt is not None:
            lt._move_snapshot_dirty = True
        event.accept()

    def mouseReleaseEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        poly = self._poly_item()
        if poly is not None:
            poly.on_polygon_changed()
            lt = poly._label_tool()
            if lt is not None:
                lt._finalize_move_snapshot()
                lt._commit_cur_to_store()
                lt._schedule_autosave()
                lt._schedule_history_push(0)
        self._center = None
        self._start_angle = None
        self._base_points = None
        self.setCursor(Qt.OpenHandCursor)
        event.accept()

    def paint(self, painter: QtGui.QPainter, option, widget=None):
        # Draw outer blue ring
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        r = self.rect()
        pen = QPen(ROTATE_COLOR, 2)
        painter.setPen(pen)
        painter.setBrush(Qt.white)
        painter.drawEllipse(r)
        # Draw circular arrow icon inside
        cx = r.center().x(); cy = r.center().y()
        radius = min(r.width(), r.height()) * 0.32
        # arc
        path = QtGui.QPainterPath()
        start_ang = -20.0 * math.pi/180.0
        end_ang = 260.0 * math.pi/180.0
        steps = 16
        for i in range(steps+1):
            t = start_ang + (end_ang - start_ang) * i/steps
            x = cx + radius * math.cos(t)
            y = cy + radius * math.sin(t)
            if i == 0: path.moveTo(x, y)
            else: path.lineTo(x, y)
        painter.drawPath(path)
        # arrow head at end
        ax = cx + radius * math.cos(end_ang)
        ay = cy + radius * math.sin(end_ang)
        ah = 4.0
        head = QtGui.QPolygonF([
            QtCore.QPointF(ax, ay),
            QtCore.QPointF(ax - ah, ay - ah*0.6),
            QtCore.QPointF(ax + ah*0.2, ay - ah*0.8)
        ])
        painter.setBrush(ROTATE_COLOR)
        painter.drawPolygon(head)

class PolyItem(QGraphicsPathItem):
    def __init__(self, points: list[QPointF], color: QColor, sid: int, klass: str):
        super().__init__()
        self.setFlags(QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemSendsGeometryChanges)
        self._points: list[QPointF] = [QPointF(p) for p in points]
        self._rebuild_path()
        self.setPen(QPen(color, 2))
        self.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 40)))
        self.sid = sid
        self.klass = klass
        self.handles: list[PolyHandleItem] = []
        self._create_handles()
        # rotation overlay (top-level, not a child)
        self.rot_handle: RotateHandleItem | None = None
        self.rot_stem: QGraphicsLineItem | None = None
        self._update_rotate_handle_position()

    def _label_tool(self) -> "LabelTool | None":
        sc = self.scene()
        if sc is None:
            return None
        views = sc.views() if hasattr(sc, "views") else []
        if not views:
            return None
        widget = views[0]
        seen: set[int] = set()
        queue: list[QtWidgets.QWidget | None] = []
        if isinstance(widget, QtWidgets.QWidget):
            queue.append(widget)
            queue.append(widget.window())
        while queue:
            obj = queue.pop(0)
            if obj is None:
                continue
            oid = id(obj)
            if oid in seen:
                continue
            seen.add(oid)
            if hasattr(obj, "_ensure_pre_move_snapshot") and hasattr(obj, "_finalize_move_snapshot"):
                return obj  # type: ignore[return-value]
            candidate = None
            if hasattr(obj, "tool"):
                candidate = getattr(obj, "tool")
            elif hasattr(obj, "label_tool"):
                candidate = getattr(obj, "label_tool")
            if candidate is not None and hasattr(candidate, "_ensure_pre_move_snapshot") and hasattr(candidate, "_finalize_move_snapshot"):
                return candidate  # type: ignore[return-value]
            if isinstance(obj, QtWidgets.QWidget):
                queue.append(obj.parentWidget())
        return None

    def _rebuild_path(self):
        path = QtGui.QPainterPath()
        if self._points:
            path.moveTo(self._points[0])
            for p in self._points[1:]:
                path.lineTo(p)
            path.closeSubpath()
        self.setPath(path)

    def _create_handles(self):
        # Clean up old handles completely (remove from scene to avoid ghosts)
        for h in self.handles:
            try:
                sc = h.scene() if hasattr(h, 'scene') else None
                if sc is not None:
                    sc.removeItem(h)
                else:
                    h.setParentItem(None)
            except Exception:
                pass
        self.handles = []
        for idx, pt in enumerate(self._points):
            handle = PolyHandleItem(idx, parent=self, size=HANDLE_SIZE_PX)
            # Keep handle visibility in sync with current selection state
            handle.setVisible(self.isSelected())
            handle.setPos(pt)
            self.handles.append(handle)

    def update_handles_positions(self):
        if len(self.handles) != len(self._points):
            self._create_handles()
            return
        for idx, handle in enumerate(self.handles):
            handle.index = idx
            handle.setPos(self._points[idx])
        self._update_rotate_handle_position()

    def _ensure_rotate_overlay(self):
        sc = self.scene()
        if sc is None:
            return
        if self.rot_handle is None:
            self.rot_handle = RotateHandleItem(self)
            self.rot_handle.setVisible(False)
            sc.addItem(self.rot_handle)
        if self.rot_stem is None:
            self.rot_stem = QGraphicsLineItem()
            self.rot_stem.setVisible(False)
            self.rot_stem.setZValue(1e7)
            self.rot_stem.setPen(QPen(ROTATE_COLOR, 1, Qt.DashLine))
            sc.addItem(self.rot_stem)

    def _update_rotate_handle_position(self):
        sc = self.scene()
        if sc is None:
            return
        self._ensure_rotate_overlay()
        if self.rot_handle is None or self.rot_stem is None:
            return
        # Compute scene-space bounding rect of the polygon (excluding overlay)
        local_br = super().boundingRect()
        br = self.mapRectToScene(local_br)
        cx = br.center().x()
        # place handle above the bounding rect
        pos = QPointF(cx, br.top() - (ROTATE_STEM_LEN + ROTATE_HANDLE_SIZE * 0.5))
        self.rot_handle.setRect(-ROTATE_HANDLE_SIZE/2, -ROTATE_HANDLE_SIZE/2, ROTATE_HANDLE_SIZE, ROTATE_HANDLE_SIZE)
        self.rot_handle.setPos(pos)
        y2 = pos.y() + ROTATE_HANDLE_SIZE * 0.5
        self.rot_stem.setLine(cx, br.top(), cx, y2)

    def set_point(self, idx: int, pos: QPointF):
        if not (0 <= idx < len(self._points)):
            return
        self.prepareGeometryChange()
        self._points[idx] = QPointF(pos)
        self._rebuild_path()
        self.update_handles_positions()
        self.on_polygon_changed()

    def remove_point(self, idx: int) -> bool:
        """Remove vertex at idx. Returns True if removed, False if blocked (needs >=3 pts)."""
        if not (0 <= idx < len(self._points)):
            return False
        if len(self._points) <= 3:
            # need at least 3 to keep a polygon
            return False
        self.prepareGeometryChange()
        del self._points[idx]
        self._rebuild_path()
        self._create_handles()
        self.update_handles_positions()
        self.on_polygon_changed()
        return True

    def insert_point_at(self, scene_pos: QPointF) -> int:
        """Insert a vertex at the closest point on an edge to scene_pos.
        Returns inserted index (position in the list)."""
        if len(self._points) < 2:
            return -1
        p = self.mapFromScene(scene_pos)
        px, py = p.x(), p.y()
        best_i = 0
        best_d2 = float('inf')
        best_q = QPointF(px, py)

        def proj(ax, ay, bx, by, px, py):
            vx, vy = bx - ax, by - ay
            wx, wy = px - ax, py - ay
            denom = vx*vx + vy*vy
            t = 0.0 if denom <= 1e-12 else max(0.0, min(1.0, (wx*vx + wy*vy) / denom))
            qx, qy = ax + t*vx, ay + t*vy
            dx, dy = px - qx, py - qy
            return qx, qy, dx*dx + dy*dy, t

        n = len(self._points)
        for i in range(n):
            a = self._points[i]
            b = self._points[(i+1) % n]
            qx, qy, d2, _ = proj(a.x(), a.y(), b.x(), b.y(), px, py)
            if d2 < best_d2:
                best_d2 = d2
                best_q = QPointF(qx, qy)
                best_i = i
        # insert after best_i (between best_i and best_i+1)
        self.prepareGeometryChange()
        self._points.insert(best_i + 1, best_q)
        self._rebuild_path()
        self._create_handles()
        self.update_handles_positions()
        self.on_polygon_changed()
        return best_i + 1

    def mousePressEvent(self, ev: QtWidgets.QGraphicsSceneMouseEvent):
        lt = self._label_tool()
        if lt is not None:
            lt._ensure_pre_move_snapshot()
        super().mousePressEvent(ev)

    def itemChange(self, change, value):
        lt = self._label_tool()

        if change == QGraphicsItem.ItemPositionChange and lt is not None:
            lt._ensure_pre_move_snapshot()

        if change in (QGraphicsItem.ItemPositionHasChanged, QGraphicsItem.ItemTransformHasChanged):
            # keep rotate overlay aligned while moving/scaling
            self._update_rotate_handle_position()
            if lt is not None:
                lt._move_snapshot_dirty = True
        elif change == QGraphicsItem.ItemSceneChange:
            # When leaving the scene (value is None), dispose overlays to prevent ghosts
            if not value:
                try:
                    if self.rot_handle and self.rot_handle.scene():
                        self.rot_handle.scene().removeItem(self.rot_handle)
                except Exception:
                    pass
                try:
                    if self.rot_stem and self.rot_stem.scene():
                        self.rot_stem.scene().removeItem(self.rot_stem)
                except Exception:
                    pass
                self.rot_handle = None
                self.rot_stem = None
        elif change == QGraphicsItem.ItemSelectedHasChanged:
            vis = bool(value)
            if vis:
                self.update_handles_positions()
            for h in self.handles:
                h.setVisible(vis)
            self._ensure_rotate_overlay()
            if self.rot_handle is not None:
                self.rot_handle.setVisible(vis)
            if self.rot_stem is not None:
                self.rot_stem.setVisible(vis)
            self._update_rotate_handle_position()
        return super().itemChange(change, value)

    def mouseReleaseEvent(self, ev: QtWidgets.QGraphicsSceneMouseEvent):
        lt = self._label_tool()
        if lt is not None:
            lt._finalize_move_snapshot()
            lt._commit_cur_to_store()
            lt._schedule_autosave()
            lt._schedule_history_push(0)
        super().mouseReleaseEvent(ev)

    def on_polygon_changed(self):
        lt = self._label_tool()
        if lt is None:
            return
        lt._commit_cur_to_store()
        lt._schedule_autosave()
        try:
            lt._update_counts_ui()
        except Exception:
            pass

    def _centroid(self) -> QPointF:
        if not self._points:
            return QPointF(0, 0)
        x = sum(p.x() for p in self._points) / len(self._points)
        y = sum(p.y() for p in self._points) / len(self._points)
        return QPointF(x, y)

    def _set_points_from_rotated(self, base_pts: list[QPointF], center: QPointF, delta_rad: float):
        if not base_pts:
            return
        cosd = math.cos(delta_rad)
        sind = math.sin(delta_rad)
        self.prepareGeometryChange()
        new_pts: list[QPointF] = []
        cx, cy = center.x(), center.y()
        for p in base_pts:
            dx = p.x() - cx
            dy = p.y() - cy
            rx = cx + (dx * cosd - dy * sind)
            ry = cy + (dx * sind + dy * cosd)
            new_pts.append(QPointF(rx, ry))
        self._points = new_pts
        self._rebuild_path()
        self.update_handles_positions()

    # ---- Context menu: Flip H/V ----
    def contextMenuEvent(self, event: QtWidgets.QGraphicsSceneContextMenuEvent):  # type: ignore[override]
        # Show only when exactly one PolyItem is selected
        sc = self.scene()
        if sc is None or not self.isSelected():
            return super().contextMenuEvent(event)
        selected = [it for it in sc.selectedItems() if isinstance(it, PolyItem)]
        if len(selected) != 1 or selected[0] is not self:
            return super().contextMenuEvent(event)

        menu = QMenu()
        act_ins_here = menu.addAction("Insert Point Here")
        act_dup = menu.addAction(self._icon_duplicate(), "Duplicate")
        menu.addSeparator()
        act_h = menu.addAction(self._icon_flip('h'), "Flip Horizontal")
        act_v = menu.addAction(self._icon_flip('v'), "Flip Vertical")
        menu.addSeparator()
        act_front = menu.addAction(_make_icon_layer('front'), "Bring to Front")
        act_back = menu.addAction(_make_icon_layer('back'), "Send to Back")
        chosen = menu.exec(event.screenPos())
        if chosen is None:
            event.accept(); return

        lt = self._label_tool()
        if lt is not None:
            lt._ensure_pre_move_snapshot()

        if chosen == act_ins_here:
            self.insert_point_at(event.scenePos())
        elif chosen == act_dup:
            # Use existing copy/paste logic to duplicate with offset
            if lt is not None:
                # ensure only this poly is part of selection
                sc.clearSelection(); self.setSelected(True)
                lt._copy_bboxes()
                lt._paste_bboxes()
        elif chosen == act_h:
            self._flip(axis='h')
        elif chosen == act_v:
            self._flip(axis='v')
        elif chosen == act_front:
            self._reorder_layer('front')
        elif chosen == act_back:
            self._reorder_layer('back')

        # finalize for flip actions only
        if chosen in (act_ins_here, act_h, act_v, act_front, act_back):
            self.on_polygon_changed()
            if lt is not None:
                lt._finalize_move_snapshot()
                lt._commit_cur_to_store()
                lt._schedule_autosave()
                lt._schedule_history_push(0)
        event.accept()

    def _flip(self, axis: str):
        if len(self._points) < 3:
            return
        br = super().boundingRect()
        cx, cy = br.center().x(), br.center().y()
        self.prepareGeometryChange()
        new_pts: list[QPointF] = []
        if axis == 'h':
            for p in self._points:
                nx = 2*cx - p.x()
                new_pts.append(QPointF(nx, p.y()))
        else:  # 'v'
            for p in self._points:
                ny = 2*cy - p.y()
                new_pts.append(QPointF(p.x(), ny))
        self._points = new_pts
        self._rebuild_path()
        self.update_handles_positions()

    def _reorder_layer(self, where: str):
        sc = self.scene()
        lt = self._label_tool()
        if sc is None or lt is None:
            return
        # sid 기반으로만 일관 처리
        shapes = [it for it in sc.items() if isinstance(it, (RectItem, PolyItem))]
        if len(shapes) <= 1:
            return
        if where == 'front':
            max_sid = max(getattr(it, 'sid', 0) for it in shapes)
            self.sid = max_sid + 1
        else:  # 'back'
            min_sid = min(getattr(it, 'sid', 0) for it in shapes)
            self.sid = min_sid - 1
        # zValue는 모두 0으로 정규화 → 순서만으로 스택 구성
        lt._normalize_shape_zvalues()
        # normalize IDs to keep order stable across saves/restores
        try:
            lt._renumber_ids()
            lt._restack_shapes_by_sid()
        except Exception:
            pass
        try:
            sc.update()
        except Exception:
            pass

    def _icon_flip(self, axis: str) -> QIcon:
        # Build a small painter icon resembling flip direction
        d = 18
        pm = QPixmap(d, d); pm.fill(Qt.transparent)
        p = QPainter(pm); p.setRenderHint(QPainter.Antialiasing, True)
        edge = QPen(QColor(60,60,60,180), 1)
        col = ROTATE_COLOR
        white = Qt.white
        if axis == 'h':
            # left triangle colored, right white, center divider
            left = QtGui.QPolygonF([
                QPointF(2, d/2), QPointF(d/2-2, 3), QPointF(d/2-2, d-3)
            ])
            right = QtGui.QPolygonF([
                QPointF(d-2, d/2), QPointF(d/2+2, d-3), QPointF(d/2+2, 3)
            ])
            p.setPen(edge); p.setBrush(col); p.drawPolygon(left)
            p.setPen(edge); p.setBrush(white); p.drawPolygon(right)
            p.setPen(QPen(QColor(120,120,120,180), 1))
            p.drawLine(d/2, 2, d/2, d-2)
        else:
            # top triangle colored, bottom white, center divider
            top = QtGui.QPolygonF([
                QPointF(d/2, 2), QPointF(d-3, d/2-2), QPointF(3, d/2-2)
            ])
            bottom = QtGui.QPolygonF([
                QPointF(d/2, d-2), QPointF(3, d/2+2), QPointF(d-3, d/2+2)
            ])
            p.setPen(edge); p.setBrush(col); p.drawPolygon(top)
            p.setPen(edge); p.setBrush(white); p.drawPolygon(bottom)
            p.setPen(QPen(QColor(120,120,120,180), 1))
            p.drawLine(2, d/2, d-2, d/2)
        p.end()
        return QIcon(pm)

    def _icon_duplicate(self) -> QIcon:
        d = 18
        pm = QPixmap(d, d); pm.fill(Qt.transparent)
        p = QPainter(pm); p.setRenderHint(QPainter.Antialiasing, True)
        edge = QPen(QColor(60,60,60,200), 1)
        fill = Qt.white
        accent = ROTATE_COLOR
        # back square
        p.setPen(edge); p.setBrush(fill)
        p.drawRoundedRect(4, 4, d-9, d-9, 3, 3)
        # front square with blue outline
        p.setPen(QPen(accent, 2)); p.setBrush(Qt.transparent)
        p.drawRoundedRect(6, 6, d-9, d-9, 3, 3)
        p.end()
        return QIcon(pm)


# ----------------------------- View -----------------------------
class GraphicsView(QGraphicsView):
    viewScaled = QtCore.Signal(float)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setDragMode(QGraphicsView.NoDrag)
        self._panning = False
        self._lastPos = None
        self.scaleFactor = 1.0
        self._space_down = False
        self.setFocusPolicy(Qt.StrongFocus)

        # >>> NEW: floating overlay support
        self._floating: QtWidgets.QWidget | None = None
        self._floating_margin = 12  # px

    def set_floating_palette(self, w: QtWidgets.QWidget | None):
        """Set a floating widget that should sit on the right-center inside the viewport."""
        # remove old
        if self._floating is not None:
            self._floating.setParent(None)
        self._floating = w
        if self._floating is not None:
            self._floating.setParent(self.viewport())
            self._reposition_floating()

    def _reposition_floating(self):
        if self._floating is None:
            return
        self._floating.adjustSize()
        w = self._floating.sizeHint().width()
        h = self._floating.sizeHint().height()
        x = self.viewport().width() - w - self._floating_margin
        y = (self.viewport().height() - h) // 2
        self._floating.move(max(0, x), max(0, y))

    def resizeEvent(self, e: QtGui.QResizeEvent):
        super().resizeEvent(e)
        # >>> keep overlay anchored
        self._reposition_floating()

    def wheelEvent(self, e: QtGui.QWheelEvent):
        # (원본과 동일)
        old_pos = self.mapToScene(e.position().toPoint())
        delta = e.angleDelta().y() if e.angleDelta().y() != 0 else e.pixelDelta().y()
        if delta == 0: return
        step = delta / 120.0
        factor = 1.15 ** step
        new_scale = max(0.05, min(50.0, self.scaleFactor * factor))
        factor = new_scale / self.scaleFactor
        self.scale(factor, factor)
        self.scaleFactor = new_scale
        new_pos = self.mapToScene(e.position().toPoint())
        self.translate(old_pos.x() - new_pos.x(), old_pos.y() - new_pos.y())
        self.viewScaled.emit(self.scaleFactor)
        self.viewport().update()
        e.accept()

    def mousePressEvent(self, e):
        # (원본과 동일)
        if e.button() == Qt.MiddleButton or (e.button() == Qt.LeftButton and self._space_down):
            self._panning = True
            self._lastPos = e.pos()
            self.setCursor(Qt.ClosedHandCursor)
            e.accept(); return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        # (원본과 동일)
        if self._panning and self._lastPos is not None:
            d = e.pos() - self._lastPos
            self._lastPos = e.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - d.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - d.y())
            self.viewport().update()
            e.accept(); return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        # (원본과 동일)
        if self._panning and e.button() in (Qt.LeftButton, Qt.MiddleButton):
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
            e.accept(); return
        super().mouseReleaseEvent(e)

    def keyPressEvent(self, e: QtGui.QKeyEvent):
        # (원본과 동일)
        k = e.key(); mods = e.modifiers()
        if k == Qt.Key_Space:
            self._space_down = True
            self.setCursor(Qt.ClosedHandCursor if QApplication.mouseButtons() & Qt.LeftButton else Qt.OpenHandCursor)
            super().keyPressEvent(e); return
        if k in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down) and (mods & (Qt.ShiftModifier | Qt.ControlModifier)):
            step = 40
            if mods & Qt.ShiftModifier: step *= 5
            elif mods & Qt.ControlModifier: step = 10
            if k == Qt.Key_Left:
                self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - step)
            elif k == Qt.Key_Right:
                self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + step)
            elif k == Qt.Key_Up:
                self.verticalScrollBar().setValue(self.verticalScrollBar().value() - step)
            elif k == Qt.Key_Down:
                self.verticalScrollBar().setValue(self.verticalScrollBar().value() + step)
            self.viewport().update(); return
        super().keyPressEvent(e)

    def keyReleaseEvent(self, e: QtGui.QKeyEvent):
        if e.key() == Qt.Key_Space:
            self._space_down = False
            if not self._panning: self.setCursor(Qt.ArrowCursor)
        super().keyReleaseEvent(e)
        
# ----------------------------- Export Ratio Dialog -----------------------------
class ExportDialog(QDialog):
    def __init__(self, parent=None, export_stats: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("New Dataset Export")
        self.resize(440, 320)
        self._export_stats = export_stats or {}
        self._aug_details: dict | None = None

        # Task (format) selection – at top
        self.cmb_task = QComboBox()
        self.cmb_task.addItem("Detect (bbox)", userData="detect")
        self.cmb_task.addItem("Segment (polygon)", userData="segment")
        try:
            # show all options in popup at once
            self.cmb_task.setMaxVisibleItems(self.cmb_task.count() or 10)
            self.cmb_task.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)
            v = QtWidgets.QListView(self.cmb_task)
            self.cmb_task.setView(v)
            rh = v.sizeHintForRow(0) or 22
            h = rh * max(1, self.cmb_task.count()) + 2 * v.frameWidth()
            v.setMinimumHeight(h); v.setMaximumHeight(h)
            v.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        except Exception:
            pass

        self.ed_dataset_name = QLineEdit(self._default_dataset_name())
        self.ed_dataset_name.setPlaceholderText("dataset name (폴더명)")

        # Dual-handle style split: boundary1=train end, boundary2=train+val end
        self.range_slider = RangeSlider(self)
        self.range_slider.setLowerValue(70)
        self.range_slider.setUpperValue(90)

        self.lbl_sum = QLabel("Sum: 100%"); self.lbl_sum.setStyleSheet("color:#0a0; font-weight:600")
        self.lbl_train_split = QLabel("")
        self.lbl_val_split = QLabel("")
        self.lbl_test_split = QLabel("")

        btn_cancel = QPushButton("❌ Cancel")
        btn_ok = QPushButton("✅ OK")
        btn_cancel.clicked.connect(self.reject)
        btn_ok.clicked.connect(self._on_ok)

        grid = QGridLayout(self)
        grid.addWidget(QLabel("Dataset name"), 0, 0); grid.addWidget(self.ed_dataset_name, 0, 1, 1, 2)
        # Task first row
        grid.addWidget(QLabel("Task"), 1, 0); grid.addWidget(self.cmb_task, 1, 1, 1, 2)
        # Ratios next (two-handle slider)
        split_wrap = QtWidgets.QWidget()
        split_lay = QtWidgets.QVBoxLayout(split_wrap); split_lay.setContentsMargins(0, 0, 0, 0); split_lay.setSpacing(6)
        split_lay.addWidget(self.range_slider)
        split_info = QtWidgets.QHBoxLayout(); split_info.setContentsMargins(0,0,0,0)
        split_info.addWidget(self.lbl_train_split)
        split_info.addWidget(self.lbl_val_split)
        split_info.addWidget(self.lbl_test_split)
        split_info.addStretch(1)
        split_lay.addLayout(split_info)
        grid.addWidget(QLabel("Split"), 2, 0)
        grid.addWidget(split_wrap, 2, 1, 1, 2)
        grid.addWidget(self.lbl_sum, 3, 0, 1, 3)
        self.chk_enable_aug = QtWidgets.QCheckBox("Enable dataset augmentation")
        self.spn_aug_multiplier = QtWidgets.QSpinBox()
        self.spn_aug_multiplier.setRange(1, 5)
        self.spn_aug_multiplier.setValue(2)

        aug_group = QtWidgets.QGroupBox("Dataset augmentation options")
        aug_layout = QtWidgets.QVBoxLayout(aug_group); aug_layout.setSpacing(0); aug_layout.setContentsMargins(8,8,8,8)
        aug_row = QtWidgets.QHBoxLayout()
        aug_row.addWidget(self.chk_enable_aug)
        aug_row.addStretch(1)
        aug_row.addWidget(QLabel("Multiplier"))
        aug_row.addWidget(self.spn_aug_multiplier)
        aug_layout.addLayout(aug_row)
        self.aug_panel = AugmentDialog(self, embed=True)
        self.aug_panel.setEnabled(False)
        aug_layout.addWidget(self.aug_panel)
        self.lbl_aug_warning = QLabel("")
        self.lbl_aug_warning.setStyleSheet("color:#a00;")
        aug_layout.addWidget(self.lbl_aug_warning)
        self.lbl_train_counts = QLabel("")
        self.lbl_train_counts.setStyleSheet("color:#444;")
        row_counts = QtWidgets.QHBoxLayout(); row_counts.setContentsMargins(0,0,0,0)
        row_counts.addWidget(self.aug_panel.btn_reset, 0, Qt.AlignLeft)
        row_counts.addStretch(1)
        row_counts.addWidget(self.lbl_train_counts, 0, Qt.AlignRight)
        aug_layout.addLayout(row_counts)
        grid.addWidget(aug_group, 9, 0, 1, 3)
        row = QHBoxLayout(); row.addStretch(1); row.addWidget(btn_cancel); row.addWidget(btn_ok)
        grid.addLayout(row, 10, 0, 1, 3)

        self.range_slider.lowerValueChanged.connect(self._refresh_sum)
        self.range_slider.upperValueChanged.connect(self._refresh_sum)
        self.range_slider.lowerValueChanged.connect(self._refresh_train_preview)
        self.range_slider.upperValueChanged.connect(self._refresh_train_preview)
        self.range_slider.lowerValueChanged.connect(self._refresh_split_counts)
        self.range_slider.upperValueChanged.connect(self._refresh_split_counts)

        self.chk_enable_aug.toggled.connect(self._update_aug_controls)
        self.spn_aug_multiplier.valueChanged.connect(self._refresh_train_preview)
        self._update_aug_controls()

        self._refresh_sum()
        self._refresh_split_counts()
        self._refresh_train_preview()

    def _refresh_sum(self):
        train, val, test, _, _ = self._get_split_pcts()
        s = train + val + test
        ok = (s == 100)
        total = int(self._export_stats.get("total_images") or 0)
        self.lbl_sum.setText(f"Sum: {s}% (total source images: {total})")
        self.lbl_sum.setStyleSheet("color:#0a0; font-weight:600" if ok else "color:#a00; font-weight:600")

    def _update_aug_controls(self):
        enabled = self.chk_enable_aug.isChecked() and _AUGMENT_LIB_AVAILABLE
        self.spn_aug_multiplier.setEnabled(enabled)
        self.aug_panel.setEnabled(enabled)
        warning = ""
        if self.chk_enable_aug.isChecked() and not _AUGMENT_LIB_AVAILABLE:
            warning = "albumentations + Pillow 설치가 필요합니다."
        self.lbl_aug_warning.setText(warning)
        self._refresh_train_preview()
        self._refresh_split_counts()

    def augmentation_config(self) -> ExportAugmentationConfig:
        enabled = self.chk_enable_aug.isChecked() and _AUGMENT_LIB_AVAILABLE
        multiplier = max(1, self.spn_aug_multiplier.value()) if enabled else 1
        self._aug_details = self.aug_panel.values() if enabled else None
        details = self._aug_details
        return ExportAugmentationConfig(enabled=enabled, multiplier=multiplier, techniques=[], details=details)

    def _on_ok(self):
        train, val, test, _, _ = self._get_split_pcts()
        s = train + val + test
        if s != 100:
            QMessageBox.warning(self, "Invalid ratio", "Train/Val/Test의 합이 100%가 되도록 맞춰주세요.")
            return
        self.accept()

    def ratios(self):
        return self._get_split_pcts()[:3]

    def task(self) -> str:
        return self.cmb_task.currentData() or "detect"

    def dataset_name(self) -> str:
        return self.ed_dataset_name.text().strip()

    def _default_dataset_name(self) -> str:
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"dataset_{now}"

    def _get_split_pcts(self) -> tuple[int, int, int, int, int]:
        b1 = int(self.range_slider.lowerValue())
        b2 = int(self.range_slider.upperValue())
        b1 = max(0, min(100, b1))
        b2 = max(b1, min(100, b2))
        train = b1
        val = b2 - b1
        test = 100 - b2
        return train, val, test, b1, b2
    def _refresh_train_preview(self):
        total = int(self._export_stats.get("total_images") or 0)
        rect_total = int(self._export_stats.get("rect_images") or 0)
        train_pct, _, _, _, _ = self._get_split_pcts()
        base_train = round(total * train_pct / 100.0)
        rect_train = round(rect_total * train_pct / 100.0)

        multiplier = 1
        if self.chk_enable_aug.isChecked() and _AUGMENT_LIB_AVAILABLE:
            self._aug_details = self.aug_panel.values()
            if self._aug_details:
                multiplier = max(1, self.spn_aug_multiplier.value())
        eff_train = base_train if total > 0 else 0
        if multiplier > 1 and rect_train > 0:
            eff_train = base_train + rect_train * (multiplier - 1)
        self.lbl_train_counts.setText(f"Augmentation copies (x{multiplier}): {eff_train}장")

    def _refresh_split_counts(self):
        total = int(self._export_stats.get("total_images") or 0)
        train_pct, val_pct, test_pct, _, _ = self._get_split_pcts()
        train = round(total * train_pct / 100.0)
        val = round(total * val_pct / 100.0)
        test = round(total * test_pct / 100.0)
        self.lbl_train_split.setText(f"Train {train} ({train_pct}%)")
        self.lbl_val_split.setText(f"Val {val} ({val_pct}%)")
        self.lbl_test_split.setText(f"Test {test} ({test_pct}%)")


class ProjectDialog(QDialog):
    def __init__(self, manager: ProjectManager, parent=None, allow_create: bool = True):
        super().__init__(parent)
        self.setWindowTitle("Select Project")
        self.manager = manager
        self.selected_project: Project | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        layout.addWidget(QLabel("Existing projects"))
        self.list_projects = QListWidget()
        self.list_projects.setSelectionMode(QListWidget.SingleSelection)
        layout.addWidget(self.list_projects, 1)

        self.ed_project_name = None
        if allow_create:
            form = QHBoxLayout()
            self.ed_project_name = QLineEdit()
            self.ed_project_name.setPlaceholderText("New project name")
            btn_create = QPushButton("Create Project")
            btn_create.clicked.connect(self._on_create)
            form.addWidget(self.ed_project_name)
            form.addWidget(btn_create)
            layout.addLayout(form)

        actions = QHBoxLayout()
        actions.addStretch(1)
        btn_open = QPushButton("Open Project")
        btn_cancel = QPushButton("Cancel")
        btn_open.clicked.connect(self._on_open)
        btn_cancel.clicked.connect(self.reject)
        actions.addWidget(btn_open)
        actions.addWidget(btn_cancel)
        layout.addLayout(actions)

        self.list_projects.itemDoubleClicked.connect(lambda _: self._on_open())
        self._refresh_projects()

    def _refresh_projects(self):
        self.list_projects.clear()
        for project in self.manager.list_projects():
            item = QListWidgetItem(f"{project.name} — {project.meta.created_at.split('T')[0]}")
            item.setData(Qt.UserRole, project)
            self.list_projects.addItem(item)

    def _on_open(self):
        item = self.list_projects.currentItem()
        if not item:
            return
        project = item.data(Qt.UserRole)
        if project is None:
            return
        self.selected_project = project
        self.accept()

    def _on_create(self):
        if self.ed_project_name is None:
            return
        name = self.ed_project_name.text().strip()
        if not name:
            QMessageBox.warning(self, "Invalid name", "프로젝트 이름을 입력하세요.")
            return
        try:
            project = self.manager.create_project(name)
        except Exception as e:
            QMessageBox.warning(self, "Create failed", str(e))
            return
        self.selected_project = project
        self.accept()

    def augmentation_config(self) -> ExportAugmentationConfig:
        enabled = self.chk_enable_aug.isChecked() and _AUGMENT_LIB_AVAILABLE
        multiplier = max(1, self.spn_aug_multiplier.value()) if enabled else 1
        techniques = [key for key, cb in self._aug_checks.items() if cb.isChecked()] if enabled else []
        return ExportAugmentationConfig(enabled=enabled, multiplier=multiplier, techniques=techniques)


# ----------------------------- Smart Simplify Dialog -----------------------------
class SmartSimplifyDialog(QDialog):
    """Minimal dialog that exposes a simplify slider and Finish/Cancel buttons.
    Emits value via callback for live preview updates managed by LabelTool.
    """
    def __init__(self, parent=None, initial: int = 80, on_change=None):
        super().__init__(parent)
        self.setWindowTitle("Smart Polygon")
        self.setModal(True)
        self.resize(320, 160)

        lbl_tip = QLabel("Click inside to refine; Slide to simplify.")
        self.sld = QSlider(Qt.Horizontal)
        self.sld.setRange(0, 100)
        self.sld.setValue(int(initial))
        lbl_l = QLabel("Simple"); lbl_r = QLabel("Complex")
        row = QHBoxLayout(); row.addWidget(lbl_l); row.addStretch(1); row.addWidget(lbl_r)

        btn_cancel = QPushButton("Delete")
        btn_ok = QPushButton("Finish (Enter)")
        btn_cancel.clicked.connect(self.reject)
        btn_ok.clicked.connect(self.accept)

        lay = QVBoxLayout(self)
        lay.addWidget(lbl_tip)
        lay.addWidget(QLabel("Simplify"))
        lay.addWidget(self.sld)
        lay.addLayout(row)
        actions = QHBoxLayout(); actions.addStretch(1); actions.addWidget(btn_cancel); actions.addWidget(btn_ok)
        lay.addLayout(actions)

        if callable(on_change):
            self.sld.valueChanged.connect(lambda _: on_change(self.value()))

        # Enter key as accept
        QtGui.QShortcut(QtGui.QKeySequence(Qt.Key_Return), self, self.accept)
        QtGui.QShortcut(QtGui.QKeySequence(Qt.Key_Enter), self, self.accept)

    def value(self) -> int:
        return int(self.sld.value())

# ----------------------------- Train Parameter Dialog -----------------------------
class TrainParamDialog(QDialog):
    def __init__(self, parent=None, data_yaml_path: str | None = None):
        super().__init__(parent)
        self.setWindowTitle("YOLO Train Parameters")
        self.resize(460, 360)

        self.ed_data = QLineEdit(data_yaml_path or ""); btn_browse = QPushButton("Browse")
        btn_browse.clicked.connect(self._pick_yaml)

        self.ed_model = QLineEdit("yolov8n.pt")
        self.sp_imgsz = QSpinBox(); self.sp_imgsz.setRange(64, 2048); self.sp_imgsz.setValue(640); self.sp_imgsz.setSingleStep(32)
        self.sp_epochs= QSpinBox(); self.sp_epochs.setRange(1, 1000); self.sp_epochs.setValue(100)
        self.sp_batch = QSpinBox(); self.sp_batch.setRange(1, 1024); self.sp_batch.setValue(16)
        self.sp_workers=QSpinBox(); self.sp_workers.setRange(0, 32); self.sp_workers.setValue(8)
        self.ed_device = QLineEdit("0")
        self.ed_name   = QLineEdit("train_run")
        self.chk_cache = QtWidgets.QCheckBox("cache"); self.chk_cache.setChecked(False)
        self.chk_aug   = QtWidgets.QCheckBox("augment"); self.chk_aug.setChecked(True)
        self.chk_resume= QtWidgets.QCheckBox("resume"); self.chk_resume.setChecked(False)

        grid = QGridLayout(self)
        grid.addWidget(QLabel("data.yaml"), 0, 0); grid.addWidget(self.ed_data, 0, 1); grid.addWidget(btn_browse, 0, 2)
        grid.addWidget(QLabel("model"),     1, 0); grid.addWidget(self.ed_model, 1, 1, 1, 2)
        grid.addWidget(QLabel("imgsz"),     2, 0); grid.addWidget(self.sp_imgsz, 2, 1, 1, 2)
        grid.addWidget(QLabel("epochs"),    3, 0); grid.addWidget(self.sp_epochs,3, 1, 1, 2)
        grid.addWidget(QLabel("batch"),     4, 0); grid.addWidget(self.sp_batch, 4, 1, 1, 2)
        grid.addWidget(QLabel("workers"),   5, 0); grid.addWidget(self.sp_workers,5, 1, 1, 2)
        grid.addWidget(QLabel("device"),    6, 0); grid.addWidget(self.ed_device,6, 1, 1, 2)
        grid.addWidget(QLabel("name"),      7, 0); grid.addWidget(self.ed_name,  7, 1, 1, 2)
        grid.addWidget(self.chk_cache,      8, 1)
        grid.addWidget(self.chk_aug,        8, 2)
        grid.addWidget(self.chk_resume,     9, 1)

        btn_cancel = QPushButton("Cancel"); btn_ok = QPushButton("Train")
        btn_cancel.clicked.connect(self.reject); btn_ok.clicked.connect(self.accept)
        row = QHBoxLayout(); row.addStretch(1); row.addWidget(btn_cancel); row.addWidget(btn_ok)
        grid.addLayout(row, 10, 0, 1, 3)

    def _pick_yaml(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select data.yaml", "", "YAML (*.yaml *.yml)")
        if p: self.ed_data.setText(p)

    def params(self) -> dict:
        return dict(
            data=self.ed_data.text().strip(),
            model=self.ed_model.text().strip(),
            imgsz=int(self.sp_imgsz.value()),
            epochs=int(self.sp_epochs.value()),
            batch=int(self.sp_batch.value()),
            workers=int(self.sp_workers.value()),
            device=self.ed_device.text().strip(),
            name=self.ed_name.text().strip(),
            cache=self.chk_cache.isChecked(),
            augment=self.chk_aug.isChecked(),
            resume=self.chk_resume.isChecked(),
            project="runs/detect",
        )


# ----------------------------- Inference Parameter Dialog -----------------------------
class InferenceDialog(QDialog):
    def __init__(self, parent=None, default_model: str = ""):
        super().__init__(parent)
        self.setWindowTitle("YOLO Inference")
        self.resize(480, 260)

        # Model
        self.ed_model = QLineEdit(default_model)
        btn_model = QPushButton("Browse")
        btn_model.clicked.connect(self._pick_model)

        # Source: current / folder
        self.rad_current = QRadioButton("Current Image")
        self.rad_folder  = QRadioButton("Folder")
        self.rad_current.setChecked(True)
        self.grp = QButtonGroup(self)
        self.grp.addButton(self.rad_current); self.grp.addButton(self.rad_folder)

        self.ed_folder = QLineEdit("")
        self.ed_folder.setEnabled(False)
        btn_folder = QPushButton("Select")
        btn_folder.setEnabled(False)

        def on_toggle():
            use_folder = self.rad_folder.isChecked()
            self.ed_folder.setEnabled(use_folder); btn_folder.setEnabled(use_folder)
        self.rad_folder.toggled.connect(on_toggle)

        def pick_folder():
            d = QFileDialog.getExistingDirectory(self, "Select images folder")
            if d: self.ed_folder.setText(d)
        btn_folder.clicked.connect(pick_folder)

        # Params
        self.sp_imgsz = QSpinBox(); self.sp_imgsz.setRange(64, 2048); self.sp_imgsz.setValue(640); self.sp_imgsz.setSingleStep(32)
        self.sp_conf  = QtWidgets.QDoubleSpinBox(); self.sp_conf.setRange(0.01, 0.99); self.sp_conf.setSingleStep(0.01); self.sp_conf.setValue(0.25)
        self.ed_device = QLineEdit("0")

        # Buttons
        btn_cancel = QPushButton("Cancel"); btn_ok = QPushButton("Run")
        btn_cancel.clicked.connect(self.reject); btn_ok.clicked.connect(self.accept)

        grid = QGridLayout(self)
        grid.addWidget(QLabel("model"), 0, 0); grid.addWidget(self.ed_model, 0, 1); grid.addWidget(btn_model, 0, 2)
        grid.addWidget(self.rad_current, 1, 0)
        grid.addWidget(self.rad_folder,  1, 1)
        grid.addWidget(self.ed_folder,   2, 1); grid.addWidget(btn_folder, 2, 2)
        grid.addWidget(QLabel("imgsz"),  3, 0); grid.addWidget(self.sp_imgsz, 3, 1, 1, 2)
        grid.addWidget(QLabel("conf"),   4, 0); grid.addWidget(self.sp_conf,  4, 1, 1, 2)
        grid.addWidget(QLabel("device"), 5, 0); grid.addWidget(self.ed_device,5, 1, 1, 2)

        row = QHBoxLayout(); row.addStretch(1); row.addWidget(btn_cancel); row.addWidget(btn_ok)
        grid.addLayout(row, 6, 0, 1, 3)

    def _pick_model(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select YOLO weights", "", "Weights (*.pt *.onnx *.engine *.torchscript *.tflite *.xml *.bin);;All (*.*)")
        if p: self.ed_model.setText(p)

    def params(self) -> dict:
        return dict(
            model=self.ed_model.text().strip(),
            use_folder=self.rad_folder.isChecked(),
            folder=self.ed_folder.text().strip(),
            imgsz=int(self.sp_imgsz.value()),
            conf=float(self.sp_conf.value()),
            device=self.ed_device.text().strip() or "0",
        )


# ----------------------------- Train/Infer Workers & Dialogs -----------------------------
class TrainWorker(QtCore.QThread):
    sig_log = QtCore.Signal(str)
    sig_prog = QtCore.Signal(int, int)  # cur, total
    sig_done = QtCore.Signal(bool, str)
    # PID of the process that actually performs training (this process or a child)
    sig_pid = QtCore.Signal(int)

    def __init__(self, params: dict):
        super().__init__()
        self.params = params
        self._stop_requested = False
        self._proc = None
        self._custom_aug_prev = None
        self._custom_aug_prev_set = False

    def _split_aug_overrides(self) -> tuple[dict, dict | None, bool]:
        overrides = self.params.get("augment_overrides")
        if not overrides:
            # If user turned augment off entirely, enforce zero-augmentation defaults
            if not bool(self.params.get("augment", True)):
                train_kwargs = {
                    "mosaic": 0.0,
                    "mixup": 0.0,
                    "copy_paste": 0.0,
                    "flipud": 0.0,
                    "fliplr": 0.0,
                    "hsv_h": 0.0,
                    "hsv_s": 0.0,
                    "hsv_v": 0.0,
                    "degrees": 0.0,
                    "translate": 0.0,
                    "scale": 0.0,
                    "shear": 0.0,
                    "perspective": 0.0,
                }
                alb_cfg = {
                    "rotate90_p": 0.0,
                    "grayscale_p": 0.0,
                    "exposure_p": 0.0,
                    "blur_p": 0.0,
                    "noise_p": 0.0,
                }
                return train_kwargs, alb_cfg, True
            return {}, None, False
        train_kwargs = dict(overrides.get("train_kwargs") or {})
        alb_cfg = dict(overrides.get("albumentations") or {})
        return train_kwargs, alb_cfg, True

    def _sync_custom_aug_module(self, cfg: dict | None):
        try:
            from ultralytics.yolo.data import augment as aug_mod
            setter = getattr(aug_mod, "set_custom_aug", None)
            if callable(setter):
                setter(cfg)
        except Exception:
            pass

    def _set_custom_aug_env(self, cfg: dict | None):
        if not self._custom_aug_prev_set:
            self._custom_aug_prev = os.environ.get("YOLO_CUSTOM_AUG")
            self._custom_aug_prev_set = True
        if cfg is None:
            os.environ.pop("YOLO_CUSTOM_AUG", None)
        else:
            try:
                payload = json.dumps(cfg)
            except Exception:
                payload = json.dumps({})
                cfg = {}
            os.environ["YOLO_CUSTOM_AUG"] = payload
        self._sync_custom_aug_module(cfg)

    def _restore_custom_aug_env(self):
        if not self._custom_aug_prev_set:
            return
        prev = self._custom_aug_prev
        if prev is None:
            os.environ.pop("YOLO_CUSTOM_AUG", None)
            cfg = None
        else:
            os.environ["YOLO_CUSTOM_AUG"] = prev
            try:
                cfg = json.loads(prev)
            except Exception:
                cfg = None
        self._sync_custom_aug_module(cfg)
        self._custom_aug_prev = None
        self._custom_aug_prev_set = False

    @staticmethod
    def _format_cli_value(value):
        if isinstance(value, bool):
            return str(value)
        if isinstance(value, float):
            text = f"{value:.6f}"
            text = text.rstrip("0").rstrip(".")
            return text if text else "0"
        return str(value)

    def _python_api_train(self) -> None:
        from ultralytics import YOLO
        train_kwargs, alb_cfg, has_overrides = self._split_aug_overrides()
        self._set_custom_aug_env(alb_cfg if has_overrides else None)
        try:
            model = YOLO(self.params["model"])
            total_epochs = int(self.params["epochs"])

            def on_fit_epoch_end(trainer):
                cur = int(getattr(trainer, "epoch", 0)) + 1
                self.sig_prog.emit(cur, total_epochs)
                self.sig_log.emit(f"Epoch {cur}/{total_epochs} finished.")
                if self._stop_requested:
                    raise RuntimeError("stopped")

            add_cb = getattr(model, "add_callback", None)
            if callable(add_cb):
                for k in ("on_fit_epoch_end", "on_train_epoch_end"):
                    try:
                        add_cb(k, on_fit_epoch_end)
                    except Exception:
                        pass

            model.train(
                data=self.params["data"],
                imgsz=self.params["imgsz"],
                epochs=total_epochs,
                batch=self.params["batch"],
                workers=self.params["workers"],
                device=self.params["device"],
                project=self.params["project"],
                name=self.params["name"],
                cache=self.params["cache"],
                augment=self.params["augment"],
                resume=self.params["resume"],
                verbose=True,
                **train_kwargs,
            )
            self.sig_prog.emit(total_epochs, total_epochs)
        finally:
            self._restore_custom_aug_env()

    def _cli_fallback(self) -> None:
        train_kwargs, alb_cfg, has_overrides = self._split_aug_overrides()
        base_cmd = self._cli_base_cmd()
        args = [
            *base_cmd, "train",
            f"model={self.params['model']}",
            f"data={self.params['data']}",
            f"imgsz={self.params['imgsz']}",
            f"epochs={self.params['epochs']}",
            f"batch={self.params['batch']}",
            f"workers={self.params['workers']}",
            f"device={self.params['device']}",
            f"project={self.params['project']}",
            f"name={self.params['name']}",
            f"cache={str(self.params['cache'])}",
            f"augment={str(self.params['augment'])}",
            f"resume={str(self.params['resume'])}",
        ]
        for key, value in train_kwargs.items():
            args.append(f"{key}={self._format_cli_value(value)}")
        self._set_custom_aug_env(alb_cfg if has_overrides else None)
        env = os.environ.copy()
        self.sig_log.emit("Falling back to CLI: " + " ".join(args))
        proc = None
        try:
            proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
            self._proc = proc
            try:
                self.sig_pid.emit(int(proc.pid))
            except Exception:
                pass
            total = int(self.params["epochs"])
            epoch_re = re.compile(r"(?:Epoch[:\s]+|epoch[:\s]+|)\b(\d+)\s*/\s*(\d+)")
            stream = proc.stdout
            if stream is None:
                raise RuntimeError("CLI training failed: no stdout pipe")
            for line in stream:
                if self._stop_requested:
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                    raise RuntimeError("stopped")
                self.sig_log.emit(line.rstrip())
                m = epoch_re.search(line)
                if m:
                    try:
                        cur = int(m.group(1)); total = int(m.group(2))
                        self.sig_prog.emit(cur, total)
                    except Exception:
                        pass
            proc.wait()
            if proc.returncode != 0:
                if self._stop_requested:
                    raise RuntimeError("stopped")
                raise RuntimeError(f"CLI training failed (code {proc.returncode})")
            self.sig_prog.emit(total, total)
        finally:
            self._restore_custom_aug_env()

    def _cli_base_cmd(self) -> list[str]:
        """Return a CLI command prefix compatible with installed ultralytics."""
        import importlib.util
        try:
            if importlib.util.find_spec("ultralytics.__main__") is not None:
                return [sys.executable, "-m", "ultralytics"]
        except Exception:
            pass
        # Fallback to yolo entry point in current venv
        exe_dir = Path(sys.executable).parent
        for cand in ("yolo", "yolo.exe"):
            yolo_path = exe_dir / cand
            if yolo_path.exists():
                return [str(yolo_path)]
        return ["yolo"]

    def run(self):
        try:
            self.sig_log.emit("Training start...")
            try:
                # Python API → training happens in this process
                try:
                    import os as _os
                    self.sig_pid.emit(int(_os.getpid()))
                except Exception:
                    pass
                import ultralytics  # noqa
                self._python_api_train()
            except Exception as e_py:
                self.sig_log.emit(f"Python API failed: {e_py}\n→ Try CLI fallback")
                self._cli_fallback()
            self.sig_done.emit(True, "학습 완료")
        except Exception as e:
            # Normalize stop reason message
            msg = "학습 중지됨" if str(e).lower().startswith("stopped") else f"학습 중 오류: {e}"
            self.sig_done.emit(False, msg)

    def stop(self):
        self._stop_requested = True
        try:
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()
        except Exception:
            pass


class TrainProgressDialog(QDialog):
    def __init__(self, parent=None, params: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("Training...")
        self.resize(640, 420)
        self.txt = QTextEdit(); self.txt.setReadOnly(True)
        self.prg = QProgressBar(); self.prg.setRange(0, 100); self.prg.setValue(0)
        self.btn_close = QPushButton("Close"); self.btn_close.setEnabled(False)
        self.btn_close.clicked.connect(self.accept)

        lay = QVBoxLayout(self)
        lay.addWidget(self.prg); lay.addWidget(self.txt); lay.addWidget(self.btn_close)

        self.worker: TrainWorker | None = None
        if params:
            self.start(params)

    def start(self, params: dict):
        self.worker = TrainWorker(params)
        self.worker.sig_log.connect(self._append)
        self.worker.sig_prog.connect(self._progress)
        self.worker.sig_done.connect(self._done)
        self.worker.start()

    def _append(self, s: str):
        self.txt.append(s)

    def _progress(self, cur: int, total: int):
        pct = int(cur * 100 / max(1, total))
        self.prg.setValue(pct)

    def _done(self, ok: bool, msg: str):
        self._append(msg)
        self.btn_close.setEnabled(True)


class InferWorker(QtCore.QThread):
    sig_log = QtCore.Signal(str)
    sig_done_single = QtCore.Signal(dict)              # result dict
    sig_each = QtCore.Signal(str, dict)                # path, result dict (folder mode)
    sig_done_folder = QtCore.Signal(int)               # processed count
    sig_video_frame = QtCore.Signal(object, dict)      # frame (BGR np.ndarray), result dict

    def __init__(self, params: dict, image_path: str | None):
        super().__init__()
        self.params = params
        self.image_path = image_path  # 현재 이미지 경로 (use_folder=False 일 때만 사용)
        self._stop_requested = False

    def _boxes_from_result(self, r, model):
        boxes = []
        def _tolist(x):
            try:
                return x.cpu().tolist()
            except Exception:
                try:
                    import numpy as _np  # type: ignore
                    return _np.asarray(x).tolist()
                except Exception:
                    try:
                        return x.tolist()  # type: ignore
                    except Exception:
                        try:
                            return list(x)
                        except Exception:
                            return []
        try:
            if hasattr(r, "boxes") and r.boxes is not None:
                xyxy = _tolist(getattr(r.boxes, "xyxy", []))
                cls = _tolist(getattr(r.boxes, "cls", [])) if getattr(r.boxes, "cls", None) is not None else [0]*len(xyxy)
                conf = _tolist(getattr(r.boxes, "conf", [])) if getattr(r.boxes, "conf", None) is not None else [1.0]*len(xyxy)

                names = []
                try:
                    names = [model.names[int(i)] for i in range(len(model.names))]
                except Exception:
                    names = []
                for (x1,y1,x2,y2),c,cf in zip(xyxy, cls, conf):
                    try:
                        ci = int(c)
                    except Exception:
                        try:
                            ci = int(float(c))
                        except Exception:
                            ci = 0
                    cname = names[ci] if names and 0 <= ci < len(names) else str(ci)
                    try:
                        cfv = float(cf)
                    except Exception:
                        try:
                            cfv = float(cf[0])
                        except Exception:
                            cfv = 0.0
                    boxes.append({"xyxy": (float(x1),float(y1),float(x2),float(y2)), "cls": ci, "conf": cfv, "name": cname})
        except Exception as e:
            self.sig_log.emit(f"parse error: {e}")
        names = []
        try:
            names = [model.names[int(i)] for i in range(len(model.names))]
        except Exception:
            names = []
        return boxes, names

    def _masks_from_result(self, r):
        try:
            if getattr(r, "masks", None) is None or getattr(r.masks, "data", None) is None:
                return None
            data = r.masks.data
            try:
                return data.cpu().numpy()
            except Exception:
                return None
        except Exception:
            return None

    def stop(self):
        self._stop_requested = True

    def run(self):
        try:
            from ultralytics import YOLO
            model_path = self.params["model"]
            ext = str(model_path).lower().split('.')[-1]
            # Backend hints for ONNX/OpenVINO
            dev = self.params.get("device", "cpu")
            if ext == "onnx":
                try:
                    import onnxruntime as ort  # type: ignore
                    prov = set(ort.get_available_providers())
                    if "CUDAExecutionProvider" not in prov:
                        dev = "cpu"  # ensure CPU when onnxruntime-gpu not available
                        self.sig_log.emit("ONNXRuntime CUDA provider not available → using CPU")
                except Exception as e:
                    self.sig_log.emit(f"ERROR: onnxruntime not available: {e}")
                    return
            if ext == "xml":
                # OpenVINO IR requires openvino runtime
                try:
                    import openvino  # noqa: F401
                except Exception as e:
                    self.sig_log.emit(f"ERROR: OpenVINO not available: {e}")
                    return
                # Force CPU device for OpenVINO inference
                dev = "cpu"
            model = YOLO(model_path)
        except Exception as e:
            self.sig_log.emit(f"ultralytics/model error: {e}")
            return

        task = self.params.get("task", "detect")
        if self.params.get("use_video"):
            video_path = self.params.get("video_path", "")
            if not video_path or not Path(video_path).exists():
                self.sig_log.emit("Invalid video source")
                return
            try:
                import cv2
            except Exception as e:
                self.sig_log.emit(f"cv2 not available: {e}")
                return
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.sig_log.emit("Failed to open video.")
                return
            self.sig_log.emit(f"Predict video: {Path(video_path).name}")
            frame_idx = 0
            imgsz = self.params.get("imgsz", None)
            while not self._stop_requested:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                frame_idx += 1
                try:
                    pred_kwargs = dict(
                        source=frame,
                        conf=self.params["conf"],
                        device=dev,
                        save=False,
                        verbose=False,
                    )
                    if isinstance(imgsz, (int, float)) and imgsz > 0:
                        pred_kwargs["imgsz"] = imgsz
                    res = model.predict(**pred_kwargs)
                    if not res:
                        continue
                    r0 = res[0]
                    if task.startswith("instance") or task.startswith("segment"):
                        masks_np = self._masks_from_result(r0)
                        names = []
                        try:
                            names = [model.names[int(i)] for i in range(len(model.names))]
                        except Exception:
                            names = []
                        boxes, _ = self._boxes_from_result(r0, model)
                        payload = {"masks": masks_np, "names": names, "boxes": boxes}
                        self.sig_video_frame.emit(frame.copy(), payload)
                    else:
                        boxes, names = self._boxes_from_result(r0, model)
                        self.sig_video_frame.emit(frame.copy(), {"boxes": boxes, "names": names})
                except Exception as e:
                    self.sig_log.emit(f"Inference error (frame {frame_idx}): {e}")
                    continue
            try:
                cap.release()
            except Exception:
                pass
            self.sig_log.emit("Video inference finished.")
        elif self.params["use_folder"]:
            src = self.params["folder"]
            if not src or not Path(src).exists():
                self.sig_log.emit("Invalid folder source")
                return
            exts = {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}
            # Exclude directories that happen to end with image-like suffixes
            imgs = [p.resolve() for p in sorted(Path(src).glob("**/*")) if p.is_file() and p.suffix.lower() in exts]
            if not imgs:
                self.sig_log.emit("No images in folder")
                return
            total = len(imgs)
            self.sig_log.emit(f"Predict {total} images (no save)")
            for i, img_path in enumerate(imgs, 1):
                if self._stop_requested:
                    break
                try:
                    imgsz = self.params.get("imgsz", None)
                    pred_kwargs = dict(
                        source=str(img_path),
                        conf=self.params["conf"],
                        device=dev,
                        save=False,
                        verbose=False,
                    )
                    if isinstance(imgsz, (int, float)) and imgsz > 0:
                        pred_kwargs["imgsz"] = imgsz
                    res = model.predict(**pred_kwargs)
                    if not res:
                        self.sig_log.emit(f"No result for {img_path.name}")
                        continue
                    r0 = res[0]
                    if task.startswith("instance") or task.startswith("segment"):
                        masks_np = self._masks_from_result(r0)
                        names = []
                        try:
                            names = [model.names[int(i)] for i in range(len(model.names))]
                        except Exception:
                            names = []
                        boxes, _ = self._boxes_from_result(r0, model)
                        result = {"masks": masks_np, "names": names, "boxes": boxes}
                        self.sig_each.emit(str(img_path), result)
                    else:
                        boxes, names = self._boxes_from_result(r0, model)
                        self.sig_each.emit(str(img_path), {"boxes": boxes, "names": names})
                except Exception as e:
                    self.sig_log.emit(f"Inference error ({img_path.name}): {e}")
                if i % 5 == 0 or i == total:
                    self.sig_log.emit(f"{i}/{total} done")
            self.sig_done_folder.emit(total)
        else:
            if not self.image_path or not Path(self.image_path).exists():
                self.sig_log.emit("No current image to infer.")
                return
            try:
                imgsz = self.params.get("imgsz", None)
                pred_kwargs = dict(
                    source=self.image_path,
                    conf=self.params["conf"],
                    device=dev,
                    save=False,
                    verbose=False,
                )
                if isinstance(imgsz, (int, float)) and imgsz > 0:
                    pred_kwargs["imgsz"] = imgsz
                res = model.predict(**pred_kwargs)
                if not res:
                    self.sig_log.emit("No result")
                    return
                r0 = res[0]
                if task.startswith("instance") or task.startswith("segment"):
                    masks_np = self._masks_from_result(r0)
                    names = []
                    try:
                        names = [model.names[int(i)] for i in range(len(model.names))]
                    except Exception:
                        names = []
                    boxes, _ = self._boxes_from_result(r0, model)
                    self.sig_done_single.emit({"masks": masks_np, "names": names, "boxes": boxes})
                else:
                    boxes, names = self._boxes_from_result(r0, model)
                    self.sig_done_single.emit({"boxes": boxes, "names": names})
            except Exception as e:
                self.sig_log.emit(f"Inference error: {e}")


class InferProgressDialog(QDialog):
    def __init__(self, parent=None, on_single_boxes=None, on_folder_box=None):
        super().__init__(parent)
        self.setWindowTitle("Running Inference...")
        self.resize(620, 380)
        self.txt = QTextEdit(); self.txt.setReadOnly(True)
        self.btn_close = QPushButton("Close"); self.btn_close.setEnabled(False)
        self.btn_close.clicked.connect(self.accept)
        lay = QVBoxLayout(self); lay.addWidget(self.txt); lay.addWidget(self.btn_close)
        self.worker: InferWorker | None = None
        self.on_single_boxes = on_single_boxes
        self.on_folder_box = on_folder_box

    def start(self, params: dict, cur_image_path: str | None):
        self.worker = InferWorker(params, cur_image_path)
        self.worker.sig_log.connect(lambda s: self.txt.append(s))
        self.worker.sig_done_single.connect(self._done_single)
        self.worker.sig_each.connect(self._each_folder)
        self.worker.sig_done_folder.connect(self._done_folder)
        self.worker.start()

    def _done_single(self, result: dict):
        self.txt.append("Single-image inference done.")
        if callable(self.on_single_boxes):
            self.on_single_boxes(result)
        self.btn_close.setEnabled(True)

    def _each_folder(self, path: str, result: dict):
        if callable(self.on_folder_box):
            self.on_folder_box(path, result)
        self.txt.append(f"{path} done")

    def _done_folder(self, count: int):
        self.txt.append(f"Folder inference finished. {count} images.")
        self.btn_close.setEnabled(True)


# ----------------------------- Main Widget -----------------------------
class LabelTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DL Software - PySide6")
        self.resize(1400, 900)
        self._status_bar_proxy: QtWidgets.QStatusBar | None = self.statusBar()

        # ----- State -----
        self.image_paths: list[Path] = []
        self._visible_paths: list[Path] = []
        self.cur_idx = -1
        self.active_class = ""  # 기본 'object' 제거
        self.class_colors: dict[str, QColor] = {}
        self.class_edit: QLineEdit | None = None
        self.view: GraphicsView | None = None
        self.next_sid = 1
        self.drawing_mode: str | None = None
        self.poly_points: list[QPointF] = []
        self.store = AnnotationStore()
        self._save_timer: QtCore.QTimer | None = None
        self.last_export_root: Path | None = None
        self.dataset_root: Path | None = None
        self._pred_overlays: list[QGraphicsItem] = []
        self._infer_cache: dict[str, tuple[list, list]] = {}  # abs_img_path -> (boxes, names)
        self.project_manager = ProjectManager(PROJECTS_ROOT)
        self.project: Project | None = None
        self.project_meta: ProjectMeta | None = None
        # Images excluded from the current working list (in-memory only)
        self._excluded_paths: set[Path] = set()
        # ---- Smart Polygon (SAM-ONNX) preview state ----
        self._smart_state = {
            "enabled": False,
            "points": [],             # list[(x,y)] in scene/image coords
            "labels": [],             # list[int] 1=positive,0=negative
            "overlay": None,          # QGraphicsPixmapItem mask preview
            "point_items": [],        # small point markers
            "image_key": None,        # current image path string
            "embedding": None,        # numpy embedding from SamPredictor
            "predictor": None,        # SamPredictor instance (torch)
            "onnx_session": None,     # onnxruntime.InferenceSession
            "mask_threshold": 0.0,    # threshold from predictor.model.mask_threshold if available
            "hover": None,            # last hover point (x,y) for live preview
            "prompt_mode": "point",   # 'point' or 'box'
        }
        # temp box drawing state for SAM prompt
        self._smart_box_anchor: QPointF | None = None
        self._smart_box_item: QGraphicsRectItem | None = None
        self._smart_hover_timer = QtCore.QTimer(self)
        self._smart_hover_timer.setSingleShot(True)
        self._smart_hover_timer.setInterval(80)  # debounce ~80ms
        self._smart_hover_timer.timeout.connect(self._smart_run_hover_preview)
        # ---- Clipboard for copy/paste ----
        self._clip_rects: list[dict] = []  # [{"points":[(x1,y1),(x2,y2)], "class": str}]
        self._clip_polys: list[dict] = []  # [{"points":[(x,y), ...], "class": str}]
        self._clip_offset_dx: float = 10.0
        self._clip_offset_dy: float = 10.0
        self._clip_bump: float = 10.0
        # ---- Undo/Redo history (per image) ----
        self._history: dict[str, list[list[dict]]] = {}
        self._hist_pos: dict[str, int] = {}
        self._hist_timer: QtCore.QTimer | None = None
        self._history_block = False
        self._move_snapshot_active = False
        self._move_snapshot_dirty = False
        self._force_snapshot_once = False
        self._move_snapshot_before: list[dict] | None = None

        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)
        self.setCentralWidget(central)

        # ----- Left: Dataset & Classes -----
        left = QWidget()
        left.setFixedWidth(360)
        lL = QVBoxLayout(left)
        lL.setContentsMargins(0, 0, 0, 0)
        lL.addWidget(Header("Dataset Browser", "Open a folder to populate thumbnails"))

        ds_box = TitledGroup("Images")
        ds_lay = QVBoxLayout(ds_box)
        top_row = QHBoxLayout()
        self.btn_open_folder = QPushButton("Import Images")
        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.setEnabled(False)
        self.btn_open_folder.clicked.connect(self.open_folder)
        self.btn_refresh.clicked.connect(self.refresh_dataset)
        top_row.addWidget(self.btn_open_folder)
        top_row.addWidget(self.btn_refresh)
        ds_lay.addLayout(top_row)

        fil_row = QtWidgets.QVBoxLayout()
        lbl_filter = QtWidgets.QLabel("Filter")
        fil_row.addWidget(lbl_filter)
        filter_line = QtWidgets.QHBoxLayout()
        self.rad_filter_all = QtWidgets.QRadioButton("All")
        self.rad_filter_labeled = QtWidgets.QRadioButton("Labeled")
        self.rad_filter_unlabeled = QtWidgets.QRadioButton("Unlabeled")
        self.rad_filter_all.setChecked(True)
        self.filter_group = QtWidgets.QButtonGroup(self)
        for btn in (self.rad_filter_all, self.rad_filter_labeled, self.rad_filter_unlabeled):
            self.filter_group.addButton(btn)
            btn.toggled.connect(lambda _: self._populate_image_list())
            filter_line.addWidget(btn)
        filter_line.addStretch(1)
        fil_row.addLayout(filter_line)
        self.lbl_filter_count = QtWidgets.QLabel("(0)")
        self.lbl_filter_count.setStyleSheet("color:#555;font-size:11px;")
        fil_row.addWidget(self.lbl_filter_count)
        ds_lay.addLayout(fil_row)

        self.list_images = QListWidget()
        self.list_images.setViewMode(QtWidgets.QListView.ListMode)
        self.list_images.setUniformItemSizes(True)
        self.list_images.setSpacing(3)
        # Extended selection to mimic filesystem list behavior (Shift+click range, Ctrl multi).
        self.list_images.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.list_images.setStyleSheet("QListWidget::item { padding:4px 8px; }")
        self.list_images.currentRowChanged.connect(self._on_image_row_changed)
        # Context menu to remove/restore items from the working list
        self.list_images.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list_images.customContextMenuRequested.connect(self._on_list_images_menu)
        # Keyboard handling (Delete to remove) via event filter
        self.list_images.installEventFilter(self)
        ds_lay.addWidget(self.list_images)
        lL.addWidget(ds_box, 1)

        # (Classes UI moved to right panel)

        root.addWidget(left, 0)

        # ----- Center: Canvas & Status -----
        center = QWidget()
        cL = QVBoxLayout(center)
        cL.setContentsMargins(0, 0, 0, 0)

        self.image_info_label = QLabel("Image: None (0/0)")
        self.image_info_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.image_info_label.setProperty("role", "header-title")
        cL.addWidget(self.image_info_label)

        self.scene = QGraphicsScene()
        self.view = GraphicsView(self.scene)
        self.img_item: QGraphicsPixmapItem | None = None
        self.h_guide: QGraphicsLineItem | None = None
        self.v_guide: QGraphicsLineItem | None = None
        self.view.setBackgroundBrush(QtGui.QColor("#FFFFFF"))
        cL.addWidget(self.view, 1)

        # Floating tool palette
        self._init_floating_palette()

        status = QHBoxLayout()
        self.lbl_img = QLabel("Image: None (0/0)")
        self.lbl_project = QLabel("Project: -")
        self.lbl_mode = QLabel("Mode: select")
        self.lbl_mode.setProperty("hint", "subtle")
        status.addWidget(self.lbl_img)
        status.addWidget(self.lbl_project)
        status.addStretch(1)
        status.addWidget(self.lbl_mode)
        cL.addLayout(status)

        self.mode_label = self.lbl_mode  # backward compatibility
        root.addWidget(center, 1)

        # ----- Right: Tooling & Counts -----
        right = QWidget()
        right.setFixedWidth(360)
        rL = QVBoxLayout(right)
        rL.setContentsMargins(0, 0, 0, 0)

        workflow_box = TitledGroup("Workflow")
        wL = QVBoxLayout(workflow_box)
        self.btn_export = QPushButton("New Dataset Export")
        self.btn_train = QPushButton("Train")
        self.btn_infer = QPushButton("Inference")
        self.btn_export.clicked.connect(self.open_export_dialog)
        # Train button switches to Train tab via main window wiring; no dialog here
        self.btn_infer.clicked.connect(self.open_infer_dialog)
        wL.addWidget(self.btn_export)
        wL.addWidget(self.btn_train)
        wL.addWidget(self.btn_infer)
        rL.addWidget(workflow_box)

        # Classes management (moved from left)
        classes_box = TitledGroup("Classes")
        cl = QVBoxLayout(classes_box)
        header = QtWidgets.QLabel("<span style='font-weight:600;'>Class</span> <span style='color:#888;font-size:11px;'>(dataset total)</span>")
        header.setStyleSheet("padding:0 2px 4px;")
        cl.addWidget(header)
        self.class_list = QListWidget()
        self.class_list.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.class_list.itemSelectionChanged.connect(self.on_class_selected)
        cl.addWidget(self.class_list)

        row = QHBoxLayout()
        self.class_edit = QLineEdit()
        self.class_edit.setPlaceholderText("Add class")
        self.class_edit.returnPressed.connect(self.add_class)
        self.class_edit.installEventFilter(self)
        btn_add = QPushButton("+")
        btn_del = QPushButton("-")
        btn_add.clicked.connect(self.add_class)
        btn_del.clicked.connect(self.del_class)
        row.addWidget(self.class_edit)
        row.addWidget(btn_add)
        row.addWidget(btn_del)
        cl.addLayout(row)
        rL.addWidget(classes_box)

        # Tools group omitted on Label tab; use floating palette + shortcuts instead

        counts_box = TitledGroup("Current Image Classes")
        ccounts = QVBoxLayout(counts_box)
        self.lbl_total = QLabel("Total: 0")
        ccounts.addWidget(self.lbl_total)
        self.count_list = QListWidget()
        self.count_list.setFixedHeight(150)
        ccounts.addWidget(self.count_list)
        rL.addWidget(counts_box)

        # Detections (Current Image) — removed from Label tab
        self.lbl_det_total = None
        self.det_list = None

        rL.addStretch(1)
        root.addWidget(right, 0)

        # ----- Toolbar & Shortcuts -----
        tb = QToolBar("Main")
        self.addToolBar(tb)
        tb.addActions([
            QAction("Open Folder (O)", self, triggered=self.open_folder),
            QAction("Save (S)", self, triggered=self.save_cur),
            QAction("Prev (A)", self, triggered=lambda: self.shift_image(-1)),
            QAction("Next (D)", self, triggered=lambda: self.shift_image(+1)),
        ])
        # Ensure shortcuts work when embedded: bind to central widget with Application context
        self._shortcut_parent = central
        self._shortcuts: list[QtGui.QShortcut] = []
        def bind(seq, handler, context=Qt.ApplicationShortcut):
            sc = QtGui.QShortcut(QtGui.QKeySequence(seq), self._shortcut_parent)
            sc.activated.connect(handler)
            sc.setContext(context)
            self._shortcuts.append(sc)
        def _handle_return_key():
            if getattr(self, "class_edit", None) and self.class_edit.hasFocus():
                self.add_class()
                return
            self._confirm_polygon_finish()
        bind("O", self.open_folder)
        bind("S", self.save_cur)
        bind("A", lambda: self.shift_image(-1))
        bind("D", lambda: self.shift_image(+1))
        bind(Qt.Key_Left,  lambda: self.shift_image(-1))
        bind(Qt.Key_Right, lambda: self.shift_image(+1))
        bind("R", lambda: self.set_mode('rect'))
        bind("P", lambda: self.set_mode('poly'))
        bind("V", lambda: self.set_mode(None))
        bind("E", lambda: self.set_mode('erase'))
        bind("Ctrl+Z", self.undo)
        bind("Ctrl+Shift+Z", self.redo)
        bind("Ctrl+C", self._copy_bboxes)
        bind("Ctrl+V", self._paste_bboxes)
        # Use selected rectangle as SAM box prompt
        bind("B", self._smart_use_selected_rect_as_prompt)
        bind(Qt.Key_Escape, self.cancel_drawing)
        bind(Qt.Key_Delete, self.delete_selected)
        bind(Qt.Key_Return, _handle_return_key)
        bind(Qt.Key_Enter, _handle_return_key)

        # Mouse tracking for crosshair guides
        self.view.setMouseTracking(True)
        self.view.viewport().installEventFilter(self)

        # (NEW) detection item click → bbox highlight (guarded)
        if getattr(self, 'det_list', None) is not None and (sb is None or sb.isValid(self.det_list)):
            self.det_list.itemClicked.connect(self._on_det_item_clicked)
        # detection bbox rects (order-matched with list)
        self._det_rects: list[QGraphicsRectItem] = []

    # ---- Public getters for sharing preview ----
    def current_pixmap(self) -> QPixmap | None:
        try:
            return self.img_item.pixmap() if self.img_item else None
        except Exception:
            return None

    def current_image_path(self) -> str | None:
        if 0 <= self.cur_idx < len(self.image_paths):
            try:
                return str(self.image_paths[self.cur_idx].resolve())
            except Exception:
                return None
        return None

    def _init_floating_palette(self):
        palette = QtWidgets.QFrame(self.view.viewport())
        palette.setObjectName("floatingPalette")
        palette.setAttribute(Qt.WA_StyledBackground, True)
        layout = QVBoxLayout(palette)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        # Keep references to mode buttons to show active/checked state
        self._mode_buttons: dict[str | None, QtWidgets.QToolButton] = {}
        self._mode_btn_group = QtWidgets.QButtonGroup(palette)
        self._mode_btn_group.setExclusive(True)

        def tool(text: str, mode_key: str | None, tip: str | None = None):
            btn = QtWidgets.QToolButton()
            btn.setText(text)
            btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setFixedWidth(40)
            btn.setCheckable(True)
            if tip:
                btn.setToolTip(tip)
                btn.setStatusTip(tip)
                btn.setAccessibleName(tip)
            btn.clicked.connect(lambda: self.set_mode(mode_key))
            self._mode_buttons[mode_key] = btn
            self._mode_btn_group.addButton(btn)
            return btn

        layout.addWidget(tool("✋", None, "Select (V)"))
        layout.addWidget(tool("▢", 'rect', "Rectangle (R)"))
        layout.addWidget(tool("⭔", 'poly', "Polygon (P)"))
        btn_smart = tool("🧠", 'smart', "Smart Polygon (SAM) – Right‑click to choose prompt")
        # Right-click context menu on smart button to choose prompt type
        try:
            btn_smart.setContextMenuPolicy(Qt.CustomContextMenu)
            btn_smart.customContextMenuRequested.connect(self._show_smart_prompt_menu)
        except Exception:
            pass
        layout.addWidget(btn_smart)
        layout.addWidget(tool("🧹", 'erase', "Erase (E)"))
        layout.addWidget(self._palette_separator())
        # Non-mode utility buttons (not checkable)
        def util(text: str, handler=None, tip: str | None = None):
            btn = QtWidgets.QToolButton()
            btn.setText(text)
            btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setFixedWidth(40)
            if tip:
                btn.setToolTip(tip); btn.setStatusTip(tip); btn.setAccessibleName(tip)
            if handler:
                btn.clicked.connect(handler)
            return btn

        layout.addWidget(util("↶", self.undo, "Undo (Ctrl+Z)"))
        layout.addWidget(util("↷", self.redo, "Redo (Ctrl+Shift+Z)"))
        layout.addWidget(self._palette_separator())
        layout.addWidget(util("◀", lambda: self.shift_image(-1), "Prev (A)"))
        layout.addWidget(util("▶", lambda: self.shift_image(+1), "Next (D)"))
        layout.addWidget(self._palette_separator())
        layout.addWidget(util("💾", self.save_cur))
        layout.addStretch(1)

        shadow = QtWidgets.QGraphicsDropShadowEffect(palette)
        shadow.setBlurRadius(22)
        shadow.setOffset(0, 4)
        shadow.setColor(QtGui.QColor(0, 0, 0, 160))
        palette.setGraphicsEffect(shadow)
        self.view.set_floating_palette(palette)
        # Initialize checked state for current mode
        self._update_mode_button_checks()

    def _update_mode_button_checks(self):
        cur = self.drawing_mode
        for key, btn in getattr(self, '_mode_buttons', {}).items():
            try:
                btn.blockSignals(True)
                btn.setChecked(key == cur)
            finally:
                btn.blockSignals(False)

    # ---- Smart prompt mode selection ----
    def _show_smart_prompt_menu(self, pos: QtCore.QPoint):
        btn = self._mode_buttons.get('smart')
        if btn is None:
            return
        global_pos = btn.mapToGlobal(pos)
        menu = QMenu(self)
        act_point = menu.addAction("Use Point Prompt")
        act_box = menu.addAction("Use Box Prompt")
        cur = self._smart_state.get("prompt_mode", "point")
        act_point.setCheckable(True); act_box.setCheckable(True)
        if cur == 'point':
            act_point.setChecked(True)
        else:
            act_box.setChecked(True)
        chosen = menu.exec(global_pos)
        if chosen == act_point:
            self._smart_set_prompt_mode('point')
        elif chosen == act_box:
            self._smart_set_prompt_mode('box')

    def _smart_set_prompt_mode(self, mode: str):
        if mode not in ('point', 'box'):
            return
        self._smart_state['prompt_mode'] = mode
        tip = "Smart Polygon (SAM) – Prompt: " + ("Point" if mode=='point' else "Box")
        try:
            btn = self._mode_buttons.get('smart')
            if btn is not None:
                btn.setToolTip(tip); btn.setStatusTip(tip); btn.setAccessibleName(tip)
        except Exception:
            pass
        self._show_status(f"Smart prompt: {mode}")

    def _palette_separator(self):
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setStyleSheet("color: rgba(255,255,255,90);")
        return line

    def _smart_release_resources(self):
        """Free SAM predictor/ONNX session (GPU/CPU memory) when smart mode is left."""
        st = self._smart_state
        # Clear overlays/points first
        self._smart_clear()
        st["image_key"] = None
        st["embedding"] = None
        st["hover"] = None
        st["mask_threshold"] = 0.0
        # Drop predictor (torch model) and session
        predictor = st.pop("predictor", None)
        if predictor is not None:
            try:
                model = getattr(predictor, "model", None)
                del predictor
                if model is not None:
                    del model
            except Exception:
                pass
        sess = st.pop("onnx_session", None)
        try:
            if sess is not None:
                del sess
        except Exception:
            pass
        # Try to release GPU cache if torch is available
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        st["enabled"] = False

    # ================= SMART POLYGON (SAM-ONNX) =================
    def _smart_defaults(self) -> tuple[Path, Path]:
        base = Path(__file__).resolve().parent / "models"
        onnx = base / "sam_onnx_example.onnx"
        pth = base / "sam_vit_h_4b8939.pth"
        return onnx, pth

    def _smart_clear(self):
        st = self._smart_state
        st["enabled"] = False
        st["points"] = []
        st["labels"] = []
        # clear temp box preview
        try:
            if getattr(self, "_smart_box_item", None) is not None:
                it = self._smart_box_item
                self._smart_box_item = None
                if it is not None:
                    sc = it.scene()
                    if sc:
                        sc.removeItem(it)
        except Exception:
            pass
        self._smart_box_anchor = None
        # remove overlay
        ov = st.get("overlay")
        try:
            if isinstance(ov, QGraphicsPixmapItem):
                if sb is not None and not sb.isValid(ov):
                    pass
                else:
                    sc = ov.scene()
                    if sc:
                        sc.removeItem(ov)
        except Exception:
            pass
        st["overlay"] = None
        # remove point markers
        for it in list(st.get("point_items") or []):
            try:
                if sb is not None and not sb.isValid(it):
                    continue
                sc = it.scene()
                if sc:
                    sc.removeItem(it)
            except Exception:
                pass
        st["point_items"] = []

    def _smart_on_new_image(self):
        st = self._smart_state
        st["image_key"] = None
        st["embedding"] = None
        st["hover"] = None
        self._smart_clear()
        # keep prompt mode as is; ensure overlays cleared

    def _smart_on_hover(self, scene_pos: QPointF):
        st = self._smart_state
        if not st.get("enabled"):
            # enable preview silently when in smart mode
            st["enabled"] = True
        st["hover"] = (scene_pos.x(), scene_pos.y())
        # debounce: if embedding ready request preview, else trigger embedding and then preview
        if not self._smart_ensure_embedding():
            return
        # restart timer
        try:
            self._smart_hover_timer.stop()
        except Exception:
            pass
        self._smart_hover_timer.start()

    def _smart_run_hover_preview(self):
        st = self._smart_state
        if st.get("hover") is None:
            return
        # Run prediction with existing clicked points + current hover as positive
        hover_pt = st["hover"]
        base_pts = list(st.get("points") or [])
        base_lbs = list(st.get("labels") or [])
        pts = base_pts + [hover_pt]
        lbs = base_lbs + [1]
        # Temporarily swap and run
        saved_pts, saved_lbs = st["points"], st["labels"]
        st["points"], st["labels"] = pts, lbs
        try:
            self._smart_predict_and_draw()
        finally:
            st["points"], st["labels"] = saved_pts, saved_lbs

    def _smart_ensure_ready(self) -> bool:
        st = self._smart_state
        # ONNX session
        if st.get("onnx_session") is None:
            try:
                import onnxruntime as ort  # type: ignore
            except Exception as e:
                QMessageBox.critical(self, "ONNX Runtime missing", f"onnxruntime를 가져올 수 없습니다: {e}")
                return False
            onnx_path, _ = self._smart_defaults()
            if not onnx_path.exists():
                QMessageBox.warning(self, "ONNX not found", f"ONNX 모델이 없습니다:\n{onnx_path}")
                return False
            providers = []
            try:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                sess = ort.InferenceSession(str(onnx_path), providers=providers)
            except Exception:
                # fallback cpu only
                sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])  # type: ignore
            st["onnx_session"] = sess

        # Torch predictor for embedding and transforms
        if st.get("predictor") is None:
            try:
                from segment_anything import sam_model_registry, SamPredictor  # type: ignore
                import torch  # noqa: F401
            except Exception as e:
                QMessageBox.critical(self, "SAM (torch) missing", f"segment-anything/Torch 불러오기 실패: {e}")
                return False
            _, pth = self._smart_defaults()
            if not pth.exists():
                QMessageBox.warning(self, "Checkpoint not found", f"SAM 가중치(.pth)가 없습니다:\n{pth}")
                return False
            try:
                sam = sam_model_registry["vit_h"](checkpoint=str(pth))
                # device auto
                try:
                    sam.to(device='cuda')
                except Exception:
                    pass
                predictor = SamPredictor(sam)
                st["predictor"] = predictor
                try:
                    st["mask_threshold"] = float(getattr(predictor.model, "mask_threshold", 0.0))
                except Exception:
                    st["mask_threshold"] = 0.0
            except Exception as e:
                QMessageBox.critical(self, "SAM init failed", f"SAM 초기화 실패: {e}")
                return False
        return True

    def _smart_ensure_embedding(self) -> bool:
        if not self._smart_ensure_ready():
            return False
        st = self._smart_state
        img_path = self.current_image_path()
        if not img_path:
            return False
        if st.get("image_key") == img_path and st.get("embedding") is not None:
            return True
        # compute embedding
        try:
            import cv2  # type: ignore
        except Exception as e:
            QMessageBox.critical(self, "OpenCV missing", f"opencv-python 필요: {e}")
            return False
        try:
            bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if bgr is None:
                QMessageBox.warning(self, "Image read error", f"이미지를 읽을 수 없습니다:\n{img_path}")
                return False
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            t0 = time.perf_counter()
            predictor = st["predictor"]
            predictor.set_image(rgb)
            emb = predictor.get_image_embedding().cpu().numpy()
            dt = time.perf_counter() - t0
            print(f"[Label Tool] SAM embedding: {dt*1000:.1f} ms")  # 로그/표시
            st["embedding"] = emb
            st["image_key"] = img_path
            return True
        except Exception as e:
            QMessageBox.critical(self, "Embedding failed", f"임베딩 계산 실패: {e}")
            return False

    def _smart_add_point(self, scene_pos: QPointF, label: int):
        # Prepare model + embedding lazily
        if not self._smart_ensure_embedding():
            return
        st = self._smart_state
        st["enabled"] = True
        st["points"].append((scene_pos.x(), scene_pos.y()))
        st["labels"].append(int(1 if label else 0))
        # draw point marker
        c = Qt.green if label == 1 else Qt.red
        r = 4
        dot = QGraphicsEllipseItem(-r, -r, 2*r, 2*r)
        dot.setPos(scene_pos)
        dot.setBrush(QBrush(c)); dot.setPen(QPen(Qt.white, 1))
        dot.setZValue(5_000)
        self.scene.addItem(dot)
        st["point_items"].append(dot)
        # run prediction and update overlay
        self._smart_predict_and_draw()

    def _smart_set_box_prompt(self, x0: float, y0: float, x1: float, y1: float):
        """Use a rectangle as SAM prompt by encoding it as two corner points
        with labels [2, 3]. This follows SAM-ONNX's box prompt format.
        """
        if not self._smart_ensure_embedding():
            return
        st = self._smart_state
        st["enabled"] = True
        # Clear any explicit point markers since box prompts don't use them
        for it in list(st.get("point_items") or []):
            try:
                if sb is not None and not sb.isValid(it):
                    continue
                sc = it.scene()
                if sc:
                    sc.removeItem(it)
            except Exception:
                pass
        st["point_items"] = []
        st["points"] = [(float(x0), float(y0)), (float(x1), float(y1))]
        st["labels"] = [2.0, 3.0]
        try:
            self._smart_predict_and_draw()
            self._show_status("Applied SAM box prompt")
        except Exception:
            pass

    def _smart_predict_and_draw(self):
        t0 = time.perf_counter()
        st = self._smart_state
        if st.get("embedding") is None:
            return
        predictor = st.get("predictor")
        sess = st.get("onnx_session")
        if predictor is None or sess is None:
            return
        try:
            if not self.img_item or (sb is not None and not sb.isValid(self.img_item)):
                return
            pts = np.array(st["points"], dtype=np.float32)
            lbs = np.array(st["labels"], dtype=np.float32)
            if pts.size == 0:
                self._smart_remove_overlay(); return
            size = self.img_item.pixmap().size()
            im_h, im_w = size.height(), size.width()
            onnx_coord = predictor.transform.apply_coords(pts[None, :, :], (im_h, im_w))
            onnx_label = lbs[None, :].astype(np.float32)
            onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
            onnx_has_mask_input = np.zeros(1, dtype=np.float32)
            H = float(im_h)
            W = float(im_w)
            ort_inputs = {
                "image_embeddings": st["embedding"],
                "point_coords": onnx_coord.astype(np.float32),
                "point_labels": onnx_label,
                "mask_input": onnx_mask_input,
                "has_mask_input": onnx_has_mask_input,
                "orig_im_size": np.array([H, W], dtype=np.float32),
            }
            outs = sess.run(None, ort_inputs)
            dt = time.perf_counter() - t0
            # print(f"[Label Tool] SAM infer: {dt*1000:.1f} ms")
            masks = outs[0]
            # choose best mask if multiple
            if masks.ndim == 4:
                m = masks[0]
                if m.shape[0] > 1:
                    # score is outs[1]; pick argmax
                    try:
                        scores = outs[1][0]
                        idx = int(np.argmax(scores))
                    except Exception:
                        idx = 0
                    m = m[idx]
                else:
                    m = m[0]
            else:
                m = masks
            thr = float(st.get("mask_threshold", 0.0))
            mask = (m > thr).astype(np.uint8)
            self._smart_draw_mask(mask)
        except Exception as e:
            QMessageBox.critical(self, "SAM predict failed", f"예측 실패: {e}")

    # ---------- Box Prompt Drawing (Smart mode) ----------
    def _smart_start_box(self, p: QPointF):
        # begin temporary rectangle for SAM prompt
        if not self.active_class or self.active_class not in self.class_colors:
            QMessageBox.information(self, "Select class", "그리기 전에 클래스를 추가/선택하세요.")
            return
        self._smart_box_anchor = p
        if self._smart_box_item is None:
            color = QColor(255, 230, 0)
            it = QGraphicsRectItem(QRectF(p, p))
            it.setZValue(9000)
            it.setPen(QPen(color, 2, Qt.DashLine))
            it.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 40)))
            self.scene.addItem(it)
            self._smart_box_item = it

    def _smart_update_box(self, p: QPointF):
        if self._smart_box_item is None or self._smart_box_anchor is None:
            return
        self._smart_box_item.setRect(QRectF(self._smart_box_anchor, p).normalized())

    def _smart_finish_box(self):
        it = self._smart_box_item
        self._smart_box_item = None
        anchor = self._smart_box_anchor
        self._smart_box_anchor = None
        if it is None:
            return
        rect = it.rect().normalized()
        try:
            sc = it.scene()
            if sc:
                sc.removeItem(it)
        except Exception:
            pass
        if rect.width() < 3 or rect.height() < 3:
            return
        # Ensure SAM ready
        if not self._smart_ensure_embedding():
            return
        # Set box prompt
        tl = rect.topLeft(); br = rect.bottomRight()
        self._smart_set_box_prompt(float(tl.x()), float(tl.y()), float(br.x()), float(br.y()))
        # Build mask from box prompt
        mask = self._smart_mask_from_current(include_hover=False)
        if mask is None:
            return
        prefer = (float((tl.x()+br.x())/2.0), float((tl.y()+br.y())/2.0))
        cnt = self._smart_mask_to_contour(mask, prefer)
        if cnt is None or len(cnt) < 3:
            return
        base_cnt = cnt.reshape(-1, 2).astype(np.float32)

        def simplify(val: int) -> np.ndarray:
            diag = (mask.shape[0]**2 + mask.shape[1]**2) ** 0.5
            t = max(0, min(100, int(val))) / 100.0
            min_eps = 0.0
            max_eps = 0.02 * diag
            eps = min_eps + (max_eps - min_eps) * (1.0 - t) ** 2
            try:
                import cv2
                approx = cv2.approxPolyDP(base_cnt, eps, True)
                xy = approx.reshape(-1, 2)
            except Exception:
                xy = base_cnt
            return xy

        dlg = SmartSimplifyDialog(self, initial=80, on_change=lambda v: self._smart_show_poly_preview(simplify(v)))
        self._smart_show_poly_preview(simplify(80))
        if dlg.exec() == QDialog.Accepted:
            xy = simplify(dlg.value())
            color = self.class_colors.get(self.active_class, QColor(0,255,0))
            pts_q = [QPointF(float(x), float(y)) for x, y in xy]
            item = PolyItem(pts_q, color, self.next_sid, self.active_class)
            self.scene.addItem(item)
            self.next_sid += 1
            self._smart_hide_poly_preview()
            self._smart_clear()
            self._update_counts_ui()
            self._commit_cur_to_store(); self._schedule_autosave(); self._schedule_history_push(0)
        else:
            self._smart_hide_poly_preview()
            self._smart_clear()

    def _smart_mask_from_current(self, include_hover: bool = True) -> np.ndarray | None:
        """Return current predicted mask using base points + optional hover point."""
        st = self._smart_state
        if st.get("embedding") is None:
            return None
        predictor = st.get("predictor")
        sess = st.get("onnx_session")
        if predictor is None or sess is None or not self.img_item:
            return None
        size = self.img_item.pixmap().size()
        im_h, im_w = size.height(), size.width()
        pts = list(st.get("points") or [])
        lbs = list(st.get("labels") or [])
        if include_hover and st.get("hover") is not None:
            pts.append(st["hover"])
            lbs.append(1)
        if len(pts) == 0:
            return None
        onnx_coord = predictor.transform.apply_coords(np.array(pts, np.float32)[None, :, :], (im_h, im_w))
        onnx_label = np.array(lbs, np.float32)[None, :]
        onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        onnx_has_mask_input = np.zeros(1, dtype=np.float32)
        ort_inputs = {
            "image_embeddings": st["embedding"],
            "point_coords": onnx_coord.astype(np.float32),
            "point_labels": onnx_label,
            "mask_input": onnx_mask_input,
            "has_mask_input": onnx_has_mask_input,
            "orig_im_size": np.array([float(im_h), float(im_w)], dtype=np.float32),
        }
        outs = sess.run(None, ort_inputs)
        masks = outs[0]
        if masks.ndim == 4:
            m = masks[0]
            if m.shape[0] > 1:
                try:
                    scores = outs[1][0]
                    idx = int(np.argmax(scores))
                except Exception:
                    idx = 0
                m = m[idx]
            else:
                m = m[0]
        else:
            m = masks
        thr = float(st.get("mask_threshold", 0.0))
        mask = (m > thr).astype(np.uint8)
        # resize if needed
        H, W = mask.shape[:2]
        if (H, W) != (im_h, im_w):
            try:
                import cv2
                mask = cv2.resize(mask, (im_w, im_h), interpolation=cv2.INTER_NEAREST)
            except Exception:
                pass
        return mask

    def _smart_use_selected_rect_as_prompt(self):
        """Use the single selected RectItem as a SAM box prompt (shortcut 'B')."""
        if self.scene is None:
            return
        selected = [it for it in self.scene.selectedItems() if isinstance(it, RectItem)]
        if len(selected) != 1:
            self._show_status("Select one rectangle first")
            return
        it: RectItem = selected[0]
        r = it.rect()
        tl = it.mapToScene(r.topLeft())
        br = it.mapToScene(r.bottomRight())
        self.set_mode('smart')
        self._smart_set_box_prompt(float(tl.x()), float(tl.y()), float(br.x()), float(br.y()))

    def _smart_mask_to_contour(self, mask: np.ndarray, prefer_pt: tuple[float, float] | None = None) -> np.ndarray | None:
        try:
            import cv2
        except Exception:
            return None
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            return None
        if prefer_pt is not None and len(cnts) > 1:
            px, py = prefer_pt
            best = None; best_val = None
            for c in cnts:
                val = cv2.pointPolygonTest(c, (float(px), float(py)), False)
                if val >= 0:  # inside or on edge
                    area = abs(cv2.contourArea(c))
                    if best is None or area > best_val:
                        best, best_val = c, area
            if best is not None:
                return best
        # fallback: largest area
        areas = [abs(cv2.contourArea(c)) for c in cnts]
        return cnts[int(np.argmax(areas))]

    def _smart_show_poly_preview(self, pts_xy: np.ndarray):
        # Remove previous
        try:
            it = getattr(self, "_smart_poly_preview", None)
            if it is not None and it.scene():
                self.scene.removeItem(it)
        except Exception:
            pass
        path = QtGui.QPainterPath()
        if len(pts_xy) >= 2:
            path.moveTo(float(pts_xy[0,0]), float(pts_xy[0,1]))
            for i in range(1, len(pts_xy)):
                path.lineTo(float(pts_xy[i,0]), float(pts_xy[i,1]))
            path.closeSubpath()
        it = QGraphicsPathItem(path)
        it.setZValue(4000)
        it.setPen(QPen(QColor(255, 230, 0), 2))
        it.setBrush(QBrush(QColor(255, 230, 0, 40)))
        self.scene.addItem(it)
        self._smart_poly_preview = it

    def _smart_hide_poly_preview(self):
        try:
            it = getattr(self, "_smart_poly_preview", None)
            if it is not None and it.scene():
                self.scene.removeItem(it)
        except Exception:
            pass
        self._smart_poly_preview = None

    def _smart_finalize_from_hover(self):
        # Require active class
        if not self.active_class or self.active_class not in self.class_colors:
            QMessageBox.information(self, "Select class", "그리기 전에 클래스를 추가/선택하세요.")
            return
        # Build mask using hover
        if not self._smart_ensure_embedding():
            return
        mask = self._smart_mask_from_current(include_hover=True)
        if mask is None:
            return
        hv = self._smart_state.get("hover")
        cnt = self._smart_mask_to_contour(mask, hv)
        if cnt is None or len(cnt) < 3:
            return
        base_cnt = cnt.reshape(-1, 2).astype(np.float32)

        # Simplification function
        def simplify(val: int) -> np.ndarray:
            # val: 0(simple) .. 100(complex). Smaller epsilon for higher complexity.
            diag = (mask.shape[0]**2 + mask.shape[1]**2) ** 0.5
            t = max(0, min(100, int(val))) / 100.0  # 0..1
            # Map to epsilon in [0.0, 0.02*diag] with bias to small eps near complex.
            min_eps = 0.0
            max_eps = 0.02 * diag
            eps = min_eps + (max_eps - min_eps) * (1.0 - t) ** 2
            try:
                import cv2
                approx = cv2.approxPolyDP(base_cnt, eps, True)
                xy = approx.reshape(-1, 2)
            except Exception:
                xy = base_cnt
            return xy

        # Live preview while sliding
        dlg = SmartSimplifyDialog(self, initial=80, on_change=lambda v: self._smart_show_poly_preview(simplify(v)))
        self._smart_show_poly_preview(simplify(80))
        if dlg.exec() == QDialog.Accepted:
            xy = simplify(dlg.value())
            color = self.class_colors.get(self.active_class, QColor(0,255,0))
            pts_q = [QPointF(float(x), float(y)) for x, y in xy]
            item = PolyItem(pts_q, color, self.next_sid, self.active_class)
            self.scene.addItem(item)
            self.next_sid += 1
            self._smart_hide_poly_preview()
            # clear smart session (points/overlay) for next object
            self._smart_clear()
            self._update_counts_ui()
            self._commit_cur_to_store(); self._schedule_autosave(); self._schedule_history_push(0)
        else:
            self._smart_hide_poly_preview()
            self._smart_clear()

    def _smart_remove_overlay(self):
        st = self._smart_state
        ov = st.get("overlay")
        try:
            if isinstance(ov, QGraphicsPixmapItem):
                if sb is not None and not sb.isValid(ov):
                    pass
                else:
                    sc = ov.scene()
                    if sc:
                        sc.removeItem(ov)
        except Exception:
            pass
        st["overlay"] = None

    def _smart_draw_mask(self, mask: np.ndarray):
        # mask expected HxW in image coords
        self._smart_remove_overlay()
        # ensure to image size
        target_W = int(self.img_item.pixmap().width()) if self.img_item else mask.shape[1]
        target_H = int(self.img_item.pixmap().height()) if self.img_item else mask.shape[0]
        H, W = int(mask.shape[0]), int(mask.shape[1])
        if (H, W) != (target_H, target_W):
            try:
                import cv2  # type: ignore
                mask = cv2.resize(mask.astype(np.uint8), (target_W, target_H), interpolation=cv2.INTER_NEAREST)
                H, W = target_H, target_W
            except Exception:
                pass
        # build RGBA overlay
        rgba = np.zeros((H, W, 4), dtype=np.uint8)
        # Use cyan/azure to avoid class-color conflict
        rgba[..., 0] = (mask > 0) * 10   # R
        rgba[..., 1] = (mask > 0) * 190  # G
        rgba[..., 2] = (mask > 0) * 255  # B
        rgba[..., 3] = (mask > 0) * 80   # alpha
        qimg = QtGui.QImage(rgba.data, W, H, W * 4, QtGui.QImage.Format_RGBA8888)
        # keep numpy buffer alive by storing it on the item
        pm = QtGui.QPixmap.fromImage(qimg.copy())
        item = QGraphicsPixmapItem(pm)
        item.setZValue(-10)  # above image(-100), below shapes(0)
        self.scene.addItem(item)
        self._smart_state["overlay"] = item

    def refresh_dataset(self):
        if self.dataset_root is None:
            return
        # confirm destructive refresh
        try:
            has_any = False
            if self.store is not None:
                images_db = self.store._db.get("images", {})
                for rec in images_db.values():
                    if rec.get("shapes"):
                        has_any = True; break
                if not has_any:
                    meta = self.store._db.get("meta", {})
                    if meta.get("classes"):
                        has_any = True
            # Ask only when there is something to clear; but it's fine to ask always
            if has_any:
                ret = QMessageBox.question(
                    self,
                    "Refresh 확인",
                    "이 작업은 되돌릴 수 없습니다.\n\n정말 Refresh하여 모든 라벨(현재 이미지 및 annotations.json의 모든 레코드)과 클래스 정보를 삭제하시겠습니까?",
                    QMessageBox.Yes | QMessageBox.Cancel,
                    QMessageBox.Cancel,
                )
                if ret != QMessageBox.Yes:
                    # user canceled refresh
                    return
        except Exception:
            pass
        # purge all annotations and class metadata
        cleared_any = False
        images_db = self.store._db.get("images", {}) if self.store else {}
        for rec in images_db.values():
            shapes = rec.get("shapes", [])
            if shapes:
                rec["shapes"] = []
                cleared_any = True
        if self.store and self.store._db.get("meta", {}).get("classes"):
            self.store._db.setdefault("meta", {})["classes"] = []
            cleared_any = True
        if cleared_any and self.store:
            try:
                self.store.save()
            except Exception:
                pass

        self.class_list.clear()
        self.class_colors.clear()
        self.active_class = ""
        self._adjust_class_list_height()
        self._clear_shape_items()
        self.next_sid = 1
        if cleared_any:
            self._commit_cur_to_store()
            self._schedule_autosave()
            self._history.clear(); self._hist_pos.clear()
        self._update_counts_ui()
        self._load_project_images()

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        # Only include regular files; exclude folders that look like files (e.g., "inference test.jpg")
        paths = [p for p in sorted(self.dataset_root.glob("**/*")) if p.is_file() and p.suffix.lower() in exts]
        if not paths:
            QMessageBox.warning(self, "No images", "No images found in folder.")
            return
        self.image_paths = paths
        self._populate_image_list()
        if self.cur_idx >= len(self.image_paths):
            self.cur_idx = len(self.image_paths) - 1
        self._switch_to_index(self.cur_idx if self.cur_idx >= 0 else 0)

    def _filtered_paths(self) -> list[Path]:
        """Return paths according to labeled/unlabeled checkboxes and annotations.json."""
        base = list(self.image_paths)
        # Exclude any paths removed from the working list
        if self._excluded_paths:
            try:
                base = [p for p in base if p.resolve() not in self._excluded_paths]
            except Exception:
                base = [p for p in base if p not in self._excluded_paths]
        if self.rad_filter_all.isChecked():
            return base
        show_labeled = self.rad_filter_labeled.isChecked()
        paths: list[Path] = []
        for p in base:
            rec = self.store.get(p)
            has_label = bool(rec.get("shapes"))
            if show_labeled and has_label:
                paths.append(p)
            elif not show_labeled and not has_label:
                paths.append(p)
        return paths

    def _populate_image_list(self):
        self.list_images.blockSignals(True)
        self.list_images.clear()
        visible_raw = self._filtered_paths()
        visible = [Path(p).resolve() for p in visible_raw]
        self._visible_paths = visible
        if hasattr(self, 'lbl_filter_count'):
            self.lbl_filter_count.setText(f"({len(visible)})")
        cur_path = None
        if 0 <= self.cur_idx < len(self.image_paths):
            try:
                cur_path = self.image_paths[self.cur_idx].resolve()
            except Exception:
                cur_path = self.image_paths[self.cur_idx]
        selected_row = -1
        for idx, path in enumerate(visible):
            item = QListWidgetItem(path.name)
            item.setData(Qt.UserRole, str(path.resolve()))
            item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            self.list_images.addItem(item)
            if cur_path is not None and path == cur_path:
                selected_row = idx
        self.list_images.blockSignals(False)
        # Select current image if it is visible; otherwise select first
        if self.list_images.count() > 0:
            self.list_images.setCurrentRow(selected_row if selected_row >= 0 else 0)
        else:
            # No visible items; clear canvas state
            self.cur_idx = -1

    def _on_image_row_changed(self, row: int):
        if row < 0 or row >= self.list_images.count():
            return
        item = self.list_images.item(row)
        if item is None:
            return
        path_str = item.data(Qt.UserRole)
        try:
            target = Path(path_str)
        except Exception:
            return
        # Find the index in the full list; fall back to 0
        try:
            idx = self.image_paths.index(target)
        except ValueError:
            idx = 0
        if idx == self.cur_idx:
            return
        self._switch_to_index(idx, sync_selection=False)

    def _switch_to_index(self, idx: int, *, sync_selection: bool = True):
        if not self.image_paths:
            return
        idx = max(0, min(idx, len(self.image_paths) - 1))
        if idx == self.cur_idx:
            return
        if 0 <= self.cur_idx < len(self.image_paths):
            self._finalize_partial_polygon_if_any()
            self._renumber_ids()
            self._commit_cur_to_store()
            self.store.save()
        self.cur_idx = idx
        self.load_cur()
        if sync_selection:
            sel_rows = self.list_images.selectionModel().selectedRows() if self.list_images.selectionModel() else []
            # CTRL/Shift 다중 선택 시 selection을 깨지 않도록 현재 행 설정은 단일 선택일 때만 수행
            if len(sel_rows) > 1:
                return
            self.list_images.blockSignals(True)
            row = -1
            if getattr(self, "_visible_paths", None):
                try:
                    cur_path = self.image_paths[self.cur_idx].resolve()
                except Exception:
                    cur_path = self.image_paths[self.cur_idx]
                try:
                    row = self._visible_paths.index(cur_path)
                except ValueError:
                    row = -1
            if row >= 0 and row < self.list_images.count():
                self.list_images.setCurrentRow(row)
            elif 0 <= self.cur_idx < self.list_images.count():
                self.list_images.setCurrentRow(self.cur_idx)
            self.list_images.blockSignals(False)

    def _switch_to_path(self, path: Path | str):
        try:
            target = Path(path).resolve()
        except Exception:
            try:
                target = Path(path)
            except Exception:
                return
        for idx, img in enumerate(self.image_paths):
            try:
                img_resolved = img.resolve()
            except Exception:
                img_resolved = img
            if img_resolved == target:
                self._switch_to_index(idx)
                return

    def _reset_dataset_state(self):
        """Clear dataset-related UI/state before loading a new project."""
        self.image_paths = []
        self._visible_paths = []
        self.cur_idx = -1
        self._excluded_paths.clear()
        self._infer_cache.clear()
        self._history.clear()
        self._hist_pos.clear()
        self.class_list.clear()
        self.class_colors.clear()
        self.active_class = ""
        self._adjust_class_list_height()
        self.list_images.blockSignals(True)
        self.list_images.clear()
        self.list_images.blockSignals(False)
        self.image_info_label.setText("Image: None (0/0)")
        self.lbl_img.setText("Image: None (0/0)")
        try:
            self.scene.clear()
        except Exception:
            pass
        self.img_item = None
        # clear counts UI
        if getattr(self, "count_list", None):
            self.count_list.clear()
        if getattr(self, "lbl_total", None):
            self.lbl_total.setText("Total: 0")

    def _remove_selected_images(self):
        """Remove selected images from the working list (non-destructive)."""
        items = self.list_images.selectedItems()
        if not items:
            return
        removed: set[Path] = set()
        for it in items:
            path_str = it.data(Qt.UserRole)
            try:
                p = Path(path_str).resolve()
            except Exception:
                p = Path(path_str)
            removed.add(p)
        # Confirm destructive delete from project
        msg = "선택한 이미지를 프로젝트에서 삭제합니다.\n(실제 파일도 프로젝트 폴더에서 제거됩니다.)"
        ret = QMessageBox.question(self, "Delete images", msg, QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel)
        if ret != QMessageBox.Yes:
            return

        self._delete_images(list(removed))

        cur_path = None
        if 0 <= self.cur_idx < len(self.image_paths):
            try:
                cur_path = self.image_paths[self.cur_idx].resolve()
            except Exception:
                cur_path = self.image_paths[self.cur_idx]

        self._populate_image_list()
        vis = self._visible_paths if getattr(self, "_visible_paths", None) else []
        if not vis:
            self.cur_idx = -1
            self.image_info_label.setText("Image: None (0/0)")
            self.lbl_img.setText("Image: None (0/0)")
            self.scene.clear()
            self.img_item = None
            return
        # If current image got removed, jump to the first visible
        if cur_path is None or cur_path in removed or cur_path not in vis:
            self._switch_to_path(vis[0])

    def _delete_images(self, paths: list[Path]):
        """Physically remove images from project and annotations."""
        if not self.project:
            return
        removed_set = set()
        for p in paths:
            try:
                pr = p.resolve()
            except Exception:
                pr = p
            removed_set.add(pr)
            try:
                if pr.exists():
                    pr.unlink()
            except Exception:
                pass
            # remove annotations
            try:
                self.store._db.get("images", {}).pop(str(pr), None)
            except Exception:
                pass
        # prune in-memory lists
        new_paths = []
        for img in self.image_paths:
            try:
                res = img.resolve()
            except Exception:
                res = img
            if res not in removed_set:
                new_paths.append(img)
        self.image_paths = new_paths
        self._excluded_paths.difference_update(removed_set)
        # Persist removal
        try:
            self.store.save()
        except Exception:
            pass

    # ---- List widget context menu: remove/restore images ----
    def _on_list_images_menu(self, pos: QtCore.QPoint):
        it = self.list_images.itemAt(pos)
        menu = QMenu(self)
        act_remove = None
        if it is not None:
            act_remove = menu.addAction("Remove selected")
        act_restore = menu.addAction("Restore all removed")
        chosen = menu.exec(self.list_images.mapToGlobal(pos))
        if chosen is None:
            return
        if chosen == act_remove and it is not None:
            self._remove_selected_images()
            return
        if chosen == act_restore:
            self._excluded_paths.clear()
            self._populate_image_list()
            # If nothing visible, clear canvas info
            if not self._visible_paths:
                self.cur_idx = -1
                self.image_info_label.setText("Image: None (0/0)")
                self.lbl_img.setText("Image: None (0/0)")
    # ------------------------- Class list + meta -------------------------
    def _make_color_icon(self, color: QColor, d: int = 12) -> QIcon:
        pm = QPixmap(d, d); pm.fill(Qt.transparent)
        p = QPainter(pm); p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(QPen(QColor(60, 60, 60, 160), 1))
        p.setBrush(color); p.drawEllipse(1, 1, d-2, d-2)
        p.end()
        return QIcon(pm)

    def _color_to_hex(self, c: QColor) -> str:
        return "#{:02X}{:02X}{:02X}".format(c.red(), c.green(), c.blue())

    def _hex_to_color(self, s: str) -> QColor:
        try:
            return QColor(s) if s else QColor(0,255,0)
        except Exception:
            return QColor(0,255,0)

    def add_class(self):
        name = self.class_edit.text().strip()
        if not name: return
        self.add_class_text(name)
        self.class_edit.clear()
        col = self.class_colors.get(name, QColor(0,255,0))
        self.store.upsert_class(name, self._color_to_hex(col))
        self._schedule_autosave()
        self._update_counts_ui()

    def add_class_text(self, name: str, color: QColor | None = None):
        if self._class_item_by_name(name) is not None:
            self._update_class_totals()
            return
        if color is None:
            color = self._next_color(len(self.class_colors))
        self.class_colors[name] = color
        item = QListWidgetItem()
        item.setData(Qt.UserRole, name)
        item.setIcon(self._make_color_icon(color))
        item.setForeground(QColor(20, 20, 20))
        self.class_list.addItem(item)
        self._refresh_class_item_label(item)
        if self.class_list.count()==1:
            self.class_list.setCurrentItem(item)
        self._persist_project_meta_classes()
        self._update_class_totals()
        self._adjust_class_list_height()

    def del_class(self):
        it = self.class_list.currentItem()
        if not it: return
        name = it.data(Qt.UserRole) or it.text()
        # Remove shapes of this class from current scene
        removed_scene = False
        for item in list(self.scene.items()):
            if isinstance(item, (RectItem, PolyItem)) and getattr(item, "klass", None) == name:
                self.scene.removeItem(item)
                removed_scene = True

        if removed_scene:
            self._commit_cur_to_store()
            self._schedule_history_push()

        # Remove annotations from store across all images
        removed_store = False
        images_db = self.store._db.get("images", {})
        for key, rec in images_db.items():
            shapes = rec.get("shapes", [])
            new_shapes = [s for s in shapes if s.get("class") != name]
            if len(new_shapes) != len(shapes):
                rec["shapes"] = new_shapes
                removed_store = True

        if removed_store or removed_scene:
            self._schedule_autosave()

        self.class_colors.pop(name, None)
        self.class_list.takeItem(self.class_list.row(it))
        self.store.remove_class(name)
        if self.class_list.count()>0:
            self.class_list.setCurrentRow(0)
        else:
            self.active_class = ""
        self._persist_project_meta_classes()
        self._update_counts_ui()
        self._populate_image_list()
        self._adjust_class_list_height()

    def on_class_selected(self):
        it = self.class_list.currentItem()
        if it:
            self.active_class = it.data(Qt.UserRole) or it.text()
            # 선택된 도형이 있으면 해당 클래스로 즉시 변경
            try:
                self._apply_class_to_selection(self.active_class)
            except Exception:
                pass
        # 같은 클래스를 연속 적용할 수 있도록 선택 상태를 바로 해제한다.
        if getattr(self, "class_list", None):
            prev = self.class_list.blockSignals(True)
            self.class_list.clearSelection()
            self.class_list.setCurrentRow(-1)
            self.class_list.blockSignals(prev)

    def _apply_class_to_selection(self, to_class: str | None = None):
        if to_class is None:
            to_class = self.active_class
        if not to_class:
            return
        # 대상 아이템 수집
        sel = [it for it in self.scene.selectedItems() if isinstance(it, (RectItem, PolyItem))]
        if not sel:
            return
        # 색상 확보 (없으면 생성)
        color = self.class_colors.get(to_class)
        if color is None:
            color = self._next_color(len(self.class_colors))
            self.add_class_text(to_class, color)
            self.store.upsert_class(to_class, self._color_to_hex(color))

        pen = QPen(color, 2)

        # 변경 적용
        for it in sel:
            it.klass = to_class
            it.setPen(pen)
            if isinstance(it, RectItem):
                it.setBrush(Qt.NoBrush)
            else:
                fill = QBrush(QColor(color.red(), color.green(), color.blue(), 40))
                it.setBrush(fill)

        # 커밋/히스토리/카운트
        self._commit_cur_to_store()
        self._schedule_autosave()
        self._schedule_history_push(0)
        self._update_counts_ui()

    # ------------------------- Folder / IO -------------------------
    def open_folder(self):
        if self.project is None:
            QMessageBox.warning(self, "No project", "프로젝트를 먼저 선택하거나 생성하세요.")
            return
        filters = "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff);;All Files (*)"
        paths, _ = QFileDialog.getOpenFileNames(self, "Import images", "", filters)
        if not paths:
            return
        files = [Path(p) for p in paths if Path(p).suffix.lower() in VALID_IMAGE_EXTS]
        if not files:
            QMessageBox.information(self, "No images", "선택한 파일에 이미지가 없습니다.")
            return
        added = self._copy_images(files)
        if added == 0:
            QMessageBox.information(self, "No images", "새로 추가된 이미지가 없습니다.")
            return
        self._show_status(f"{added}개 이미지가 프로젝트에 추가되었습니다.")
        self._load_project_images()

    def _load_project(self, project: Project):
        self._reset_dataset_state()
        self.project = project
        self.project_meta = project.meta
        try:
            # 업데이트된 최근 열람 시각 기록
            now = datetime.datetime.utcnow()
            self.project.meta.last_opened = now.isoformat(timespec="microseconds")
            self.project.meta.last_opened_ts = now.timestamp()
            self.project.save_meta()
        except Exception:
            pass
        self.dataset_root = project.root
        self._excluded_paths.clear()
        self.store = AnnotationStore(project.annotations_path)
        self.store.load()
        self._apply_project_classes()
        self._update_project_label(project.name)
        self._load_project_images()
        self.btn_refresh.setEnabled(True)

    def _update_project_label(self, name: str | None):
        if name:
            self.lbl_project.setText(f"Project: {name}")
            self.setWindowTitle(f"DL Software - {name}")
        else:
            self.lbl_project.setText("Project: -")
            self.setWindowTitle("DL Software - PySide6")

    def _apply_project_classes(self):
        self.class_list.clear()
        self.class_colors.clear()
        self._adjust_class_list_height()
        classes = {}
        if self.project_meta:
            for entry in self.project_meta.classes:
                name = entry.get("name")
                if not name:
                    continue
                classes[name] = entry.get("color", "#00FF00")
        for entry in self.store.list_classes():
            name = entry.get("name")
            if not name:
                continue
            if name not in classes:
                classes[name] = entry.get("color", "#00FF00")

        if not classes:
            seen = []
            for rec in self.store._db.get("images", {}).values():
                for sh in rec.get("shapes", []):
                    cname = sh.get("class")
                    if cname and cname not in seen:
                        seen.append(cname)
            for i, name in enumerate(seen):
                col = self._next_color(i)
                self.add_class_text(name, col)
                self.store.upsert_class(name, self._color_to_hex(col))
            self.store.save()
        else:
            for i, (name, color_hex) in enumerate(sorted(classes.items())):
                color = self._hex_to_color(color_hex)
                self.add_class_text(name, color)
        self._persist_project_meta_classes()

    def _load_project_images(self):
        if self.project is None:
            return
        self.image_paths = [p for p in sorted(self.project.images_dir.glob("*")) if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS]
        self._visible_paths = list(self.image_paths)
        self._populate_image_list()
        self._update_class_totals()
        if self.image_paths:
            self.cur_idx = 0
            self.list_images.blockSignals(True)
            if self.list_images.count() > 0:
                self.list_images.setCurrentRow(self.cur_idx)
            self.list_images.blockSignals(False)
            self.load_cur()
        else:
            self.cur_idx = -1
            self.image_info_label.setText("Image: None (0/0)")
            self.lbl_img.setText("Image: None (0/0)")

    def _persist_project_meta_classes(self):
        if not self.project:
            return
        class_entries = []
        for name, color in self.class_colors.items():
            class_entries.append({"name": name, "color": self._color_to_hex(color)})
        self.project.meta.classes = class_entries
        self.project.save_meta()

    def _copy_images(self, sources: list[Path]) -> int:
        if self.project is None:
            return 0
        dest_dir = self.project.images_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        copied = 0
        for src in sources:
            dest = dest_dir / src.name
            if dest.exists():
                dest = self._resolve_conflict_name(dest)
            try:
                shutil.copy2(src, dest)
                copied += 1
            except Exception:
                continue
        return copied

    def _resolve_conflict_name(self, path: Path) -> Path:
        base = path.stem
        suffix = path.suffix
        parent = path.parent
        idx = 1
        candidate = parent / f"{base}_{idx}{suffix}"
        while candidate.exists():
            idx += 1
            candidate = parent / f"{base}_{idx}{suffix}"
        return candidate

    def ann_path_for(self, img_path: Path) -> Path:
        return img_path.with_suffix(".json")  # (레거시 읽기용만)

    def load_cur(self):
        if self.cur_idx < 0 or self.cur_idx >= len(self.image_paths): return
        img_path = self.image_paths[self.cur_idx]
        label_text = f"Image: {img_path.name} ({self.cur_idx+1}/{len(self.image_paths)})"
        self.image_info_label.setText(label_text)
        self.lbl_img.setText(label_text)

        pix = QPixmap(str(img_path))
        if pix.isNull():
            QMessageBox.warning(self, "Load error", f"Failed to load image: {img_path}"); return

        # scene & view 초기화
        # 1) 먼저 오버레이를 안전하게 제거 (Qt 경고 방지)
        self._clear_pred_overlays()
        # 2) 이후 씬 비우기
        self.scene.clear()
        # make sure the graphics view is bound to the scene in embedded/tabbed mode
        if self.view.scene() is not self.scene:
            self.view.setScene(self.scene)
        self.img_item = QGraphicsPixmapItem(pix); self.img_item.setZValue(-100)
        self.scene.addItem(self.img_item)
        self.scene.setSceneRect(QRectF(pix.rect()))
        try:
            self.view.setSceneRect(QRectF(pix.rect()))
        except Exception:
            pass
        self.view.resetTransform()
        # delay fit to ensure the view has its final size inside tabs
        def _fit_center():
            try:
                self.view.fitInView(self.img_item, Qt.KeepAspectRatio)
                self.view.centerOn(self.img_item)
            except Exception:
                pass
        QtCore.QTimer.singleShot(0, _fit_center)
        self.view.viewport().update()
        self.view.scaleFactor = float(self.view.transform().m11())

        # guides
        self._setup_guides()

        # reset smart mask preview state for new image
        try:
            self._smart_on_new_image()
        except Exception:
            pass

        # drawing state
        self.next_sid = 1
        self.poly_points.clear()
        self._clear_poly_preview()

        # restore: 중앙 store
        data = self.store.get(img_path)
        shapes_data = data.get("shapes", [])
        for sd in shapes_data:
            sh = Shape.from_dict(sd)
            self.next_sid = max(self.next_sid, sh.id + 1)
            self._add_shape_item(sh)

        # 이미지 크기 정보 최신화 저장
        self.store.put(img_path, shapes_data, (pix.width(), pix.height()))

        # id 정규화
        self._renumber_ids()
        self._update_counts_ui()

        # 오른쪽 detection 리스트 초기화 (guarded)
        if getattr(self, 'det_list', None) is not None and (sb is None or sb.isValid(self.det_list)):
            self.det_list.clear()
        if getattr(self, 'lbl_det_total', None) is not None and (sb is None or sb.isValid(self.lbl_det_total)):
            self.lbl_det_total.setText("Detections: 0")

        # push initial history snapshot for this image
        self._push_history()

        # 추론 캐시가 있으면 즉시 오버레이
        key = str(img_path.resolve())
        if key in self._infer_cache:
            boxes, names = self._infer_cache[key]
            self._show_pred_overlays(boxes, names)

    # ---- save / autosave / commit ----
    def _commit_cur_to_store(self):
        if self.cur_idx < 0 or not self.image_paths: return
        img_path = self.image_paths[self.cur_idx]
        self._renumber_ids()
        shapes = self._extract_shapes_from_scene()
        W, H = (self.img_item.pixmap().width(), self.img_item.pixmap().height()) if self.img_item else (0, 0)
        self.store.put(img_path, shapes, (W, H))

    def _schedule_autosave(self, ms: int = 400):
        if self._save_timer is None:
            self._save_timer = QtCore.QTimer(self)
            self._save_timer.setSingleShot(True)
            self._save_timer.timeout.connect(self.store.save)
        self._save_timer.start(ms)

    def save_cur(self):
        if self.cur_idx < 0: return
        self._finalize_partial_polygon_if_any()
        self._renumber_ids()
        self._commit_cur_to_store()
        self.store.save()
        self._show_status("Saved (annotations.json)")
        self._update_counts_ui()

    def shift_image(self, delta: int):
        visible = self._visible_paths if getattr(self, "_visible_paths", None) else []
        if not visible:
            visible = [Path(p).resolve() for p in self._filtered_paths()]
        if not visible:
            return
        cur_path = None
        if 0 <= self.cur_idx < len(self.image_paths):
            try:
                cur_path = self.image_paths[self.cur_idx].resolve()
            except Exception:
                cur_path = self.image_paths[self.cur_idx]
        if cur_path in visible:
            pos = visible.index(cur_path)
            pos = (pos + delta) % len(visible)
        else:
            pos = 0 if delta >= 0 else len(visible) - 1
        target_path = visible[pos]
        self._switch_to_path(target_path)

    def set_status_bar_proxy(self, bar: QtWidgets.QStatusBar | None):
        """Allow embedding contexts to supply their own status bar."""
        self._status_bar_proxy = bar

    def _show_status(self, message: str, timeout: int = 1500):
        if self._status_bar_proxy is not None:
            self._status_bar_proxy.showMessage(message, timeout)

    # ------------------------- Drawing -------------------------
    def set_mode(self, mode: str|None):
        self.drawing_mode = mode
        self.poly_points.clear()
        self._clear_poly_preview()
        # Clear any selection/handles from previous mode to avoid lingering points
        try:
            if self.scene is not None:
                self.scene.clearSelection()
        except Exception:
            pass
        # smart mode preview state reset
        if mode != 'smart':
            self._smart_release_resources()
            # Also hide any smart poly preview that may be showing
            try:
                self._smart_hide_poly_preview()
            except Exception:
                pass
        else:
            self._smart_state["enabled"] = True
            # ensure model lazy-load takes place on first click
        self.mode_label.setText(f"Mode: {mode or 'select'}")
        # reflect in floating palette buttons
        try:
            self._update_mode_button_checks()
        except Exception:
            pass

    def eventFilter(self, obj, event):
        if obj is getattr(self, "class_edit", None) and event.type() == QtCore.QEvent.KeyPress:
            if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                self.add_class()
                return True
        if obj is self.list_images and event.type() == QtCore.QEvent.KeyPress:
            if event.matches(QtGui.QKeySequence.SelectAll):
                self.list_images.selectAll()
                return True
            if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
                self._remove_selected_images()
                return True
        if getattr(self, "view", None) is not None and obj is self.view.viewport() and isinstance(event, QtGui.QMouseEvent):
            pos = self.view.mapToScene(event.position().toPoint())
            if event.type() == QtCore.QEvent.MouseButtonPress and event.button()==Qt.LeftButton:
                if self.drawing_mode == 'rect':
                    # class 없는 그리기 방지
                    if not self.active_class or self.active_class not in self.class_colors:
                        QMessageBox.information(self, "Select class", "그리기 전에 클래스를 추가/선택하세요.")
                        return True
                    self._start_rect(pos); return True
                elif self.drawing_mode == 'poly':
                    if not self.active_class or self.active_class not in self.class_colors:
                        QMessageBox.information(self, "Select class", "그리기 전에 클래스를 추가/선택하세요.")
                        return True
                    self._poly_click(pos); return True
                elif self.drawing_mode == 'erase':
                    self.delete_at(pos); return True
                elif self.drawing_mode == 'smart':
                    # In smart mode, behavior depends on prompt mode
                    if self._smart_state.get("prompt_mode", "point") == 'box':
                        self._smart_start_box(pos); return True
                    else:
                        # Finalize hover preview into polygon with dialog
                        self._smart_finalize_from_hover(); return True
            if event.type() == QtCore.QEvent.MouseButtonPress and event.button()==Qt.RightButton:
                if self.drawing_mode == 'smart' and self._smart_state.get("prompt_mode", "point") == 'point':
                    self._smart_add_point(pos, 0); return True
                elif self.drawing_mode is None:
                    # If clicking on any interactive overlay/handles, let the scene deliver the event
                    under = self.scene.items(pos)
                    if any(isinstance(it, (RectItem, PolyItem, RotateHandleItem, PolyHandleItem, HandleItem)) for it in under):
                        return False
                    # Otherwise, click on empty canvas: clear selection
                    self.scene.clearSelection(); return True
            elif event.type() == QtCore.QEvent.MouseMove:
                if self.drawing_mode == 'rect' and hasattr(self, '_rect_anchor'):
                    self._update_rect(pos); return True
                if self.drawing_mode == 'poly' and len(self.poly_points) > 0:
                    self._update_poly_preview(pos)
                if self.drawing_mode == 'smart':
                    if self._smart_state.get("prompt_mode", "point") == 'box' and self._smart_box_anchor is not None:
                        self._smart_update_box(pos)
                    else:
                        self._smart_on_hover(pos)
                self._update_guides(pos)
            elif event.type() == QtCore.QEvent.MouseButtonDblClick and self.drawing_mode=='poly':
                self._finish_poly(); return True
            elif event.type() == QtCore.QEvent.MouseButtonRelease and self.drawing_mode=='rect' and event.button()==Qt.LeftButton:
                self._finish_rect(); return True
            elif event.type() == QtCore.QEvent.MouseButtonRelease and self.drawing_mode=='smart' and event.button()==Qt.LeftButton:
                if self._smart_state.get("prompt_mode", "point") == 'box' and self._smart_box_anchor is not None:
                    self._smart_finish_box(); return True
            elif event.type() in (QtCore.QEvent.Leave, QtCore.QEvent.HoverLeave):
                self._hide_guides()
        return super().eventFilter(obj, event)

    def _start_rect(self, p: QPointF):
        self.scene.clearSelection()
        self._hide_guides()
        self._rect_anchor = p
        color = self.class_colors.get(self.active_class, QColor(0,255,0))
        r = QRectF(p, p)
        self._rect_item = RectItem(r, color, self.next_sid, self.active_class)
        self.scene.addItem(self._rect_item)

    def _update_rect(self, p: QPointF):
        self._rect_item.setRect(QRectF(self._rect_anchor, p).normalized())

    def _finish_rect(self):
        if hasattr(self, '_rect_item'):
            if self._rect_item.rect().width()<3 or self._rect_item.rect().height()<3:
                self.scene.removeItem(self._rect_item)
            else:
                self.next_sid += 1
                self.scene.clearSelection()
                self._rect_item.setSelected(True)
            del self._rect_item
            del self._rect_anchor
            self._update_counts_ui()
            self._commit_cur_to_store(); self._schedule_autosave(); self._schedule_history_push(0)
            self._refresh_guides_now()

    def _poly_click(self, p: QPointF):
        self.poly_points.append(p)
        color = self.class_colors.get(self.active_class, QColor(0,255,0))
        if not hasattr(self, '_poly_preview') or self._poly_preview is None:
            self._poly_preview = QGraphicsPathItem()
            self._poly_preview.setPen(QPen(color, 1, Qt.DashLine))
            self._poly_preview.setZValue(9999)
            self.scene.addItem(self._poly_preview)
        # 첫 포인트를 찍고 나면 가이드라인 숨김
        if len(self.poly_points) == 1:
            self._hide_guides()

    def _finish_poly(self):
        if len(self.poly_points) < 3:
            self.poly_points.clear()
            self._clear_poly_preview()
            # 3점 미만으로 종료될 때도 가이드 복원
            self._refresh_guides_now()
            return
        color = self.class_colors.get(self.active_class, QColor(0,255,0))
        item = PolyItem(self.poly_points, color, self.next_sid, self.active_class)
        self.scene.addItem(item)
        self.next_sid += 1
        self.poly_points.clear()
        self._clear_poly_preview()
        # 완료 후 가이드를 즉시 복원(다음 인스턴스 첫 포인트 전까지 표시)
        self._refresh_guides_now()
        self._update_counts_ui()
        self._commit_cur_to_store(); self._schedule_autosave(); self._schedule_history_push(0)

    def _confirm_polygon_finish(self):
        if self.drawing_mode != 'poly':
            return
        if len(self.poly_points) < 3:
            return
        self._finish_poly()

    def delete_selected(self):
        sel = self.scene.selectedItems()
        if sel:
            self.scene.removeItem(sel[0])
            self._renumber_ids()
            self._update_counts_ui()
            self._commit_cur_to_store(); self._schedule_autosave(); self._schedule_history_push(0)
            return
        pos = self.view.mapToScene(self.view.mapFromGlobal(QtGui.QCursor.pos()))
        self.delete_at(pos)

    # --------- Copy / Paste (BBox + Polygon) ----------
    def _copy_bboxes(self):
        sel_rects = [it for it in self.scene.selectedItems() if isinstance(it, RectItem)]
        sel_polys = [it for it in self.scene.selectedItems() if isinstance(it, PolyItem)]
        if not sel_rects and not sel_polys:
            self._show_status("선택된 항목이 없습니다.")
            return
        rects: list[dict] = []
        for it in sel_rects:
            r_scene = it.mapRectToScene(it.rect()).normalized()
            pts = [(r_scene.left(), r_scene.top()), (r_scene.right(), r_scene.bottom())]
            rects.append({"points": pts, "class": getattr(it, "klass", self.active_class) or self.active_class})
        polys: list[dict] = []
        for it in sel_polys:
            try:
                poly_local = QtGui.QPolygonF(it._points)
                poly_scene = it.mapToScene(poly_local)
                pts = [(p.x(), p.y()) for p in poly_scene]
            except Exception:
                # fallback using path
                poly_local = it.path().toFillPolygon()
                poly_scene = it.mapToScene(poly_local)
                pts = [(p.x(), p.y()) for p in poly_scene]
            polys.append({"points": pts, "class": getattr(it, "klass", self.active_class) or self.active_class})
        self._clip_rects = rects
        self._clip_polys = polys
        # reset paste offset for a new copy cycle
        self._clip_offset_dx = 10.0
        self._clip_offset_dy = 10.0
        msg_parts = []
        if rects: msg_parts.append(f"bbox {len(rects)}")
        if polys: msg_parts.append(f"poly {len(polys)}")
        self._show_status("Copied " + ", ".join(msg_parts))

    def _paste_bboxes(self):
        if not self._clip_rects and not self._clip_polys:
            self._show_status("클립보드가 비어 있습니다.")
            return
        new_items: list[QGraphicsItem] = []
        for rec in self._clip_rects:
            (x1, y1), (x2, y2) = rec.get("points", ((0, 0), (0, 0)))
            klass = rec.get("class", self.active_class)
            # ensure class color exists
            color = self.class_colors.get(klass)
            if color is None and klass:
                color = self._next_color(len(self.class_colors))
                self.add_class_text(klass, color)
                self.store.upsert_class(klass, self._color_to_hex(color))
            if color is None:
                color = QColor(0, 255, 0)
            r = QRectF(QPointF(x1 + self._clip_offset_dx, y1 + self._clip_offset_dy),
                       QPointF(x2 + self._clip_offset_dx, y2 + self._clip_offset_dy)).normalized()
            item = RectItem(r, color, self.next_sid, klass or self.active_class)
            self.scene.addItem(item)
            new_items.append(item)
            self.next_sid += 1
        # paste polys
        for rec in self._clip_polys:
            pts = rec.get("points", [])
            klass = rec.get("class", self.active_class)
            color = self.class_colors.get(klass)
            if color is None and klass:
                color = self._next_color(len(self.class_colors))
                self.add_class_text(klass, color)
                self.store.upsert_class(klass, self._color_to_hex(color))
            if color is None:
                color = QColor(0, 255, 0)
            pts_off = [QPointF(x + self._clip_offset_dx, y + self._clip_offset_dy) for (x, y) in pts]
            item = PolyItem(pts_off, color, self.next_sid, klass or self.active_class)
            self.scene.addItem(item)
            new_items.append(item)
            self.next_sid += 1
        # select new items
        self.scene.clearSelection()
        for it in new_items:
            it.setSelected(True)
        # bump offset for repeated paste
        self._clip_offset_dx += self._clip_bump
        self._clip_offset_dy += self._clip_bump
        # commit and update
        self._renumber_ids()
        self._update_counts_ui()
        self._commit_cur_to_store(); self._schedule_autosave(); self._schedule_history_push(0)
        self._show_status(f"Pasted {len(new_items)} item(s)")

    # ------------------------- Helpers -------------------------
    def _next_color(self, i: int) -> QColor:
        golden_angle = 137.508
        hue = int((i * golden_angle) % 360.0)
        # Avoid reserved cyan/azure band used by Smart preview
        def in_forbidden(h: int) -> bool:
            for lo, hi in SMART_PREVIEW_FORBIDDEN_HUES:
                # inclusive of lo, exclusive of hi
                if lo <= h < hi:
                    return True
            return False
        bump = 30
        safety = 0
        while in_forbidden(hue) and safety < 12:
            hue = (hue + bump) % 360
            safety += 1
        sat = 255
        val = 255
        return QColor.fromHsv(hue, sat, val)

    # --- guides ---
    def _setup_guides(self):
        self._dispose_guide('h_guide'); self._dispose_guide('v_guide')
        rect = self.scene.sceneRect()
        # 좀 더 잘 보이도록 두께 2px의 흰색 점선 + 그림자 효과
        pen = QPen(QColor(255, 255, 255, 255), 2, Qt.DashLine)
        pen.setDashPattern([6, 4])
        self.h_guide = QGraphicsLineItem(rect.left(), rect.center().y(), rect.right(), rect.center().y())
        self.v_guide = QGraphicsLineItem(rect.center().x(), rect.top(), rect.center().x(), rect.bottom())
        for g in (self.h_guide, self.v_guide):
            g.setPen(pen)
            g.setZValue(9999)
            g.setAcceptedMouseButtons(Qt.NoButton)
            g.setVisible(False)
            self.scene.addItem(g)

    def _dispose_guide(self, attr: str):
        g = getattr(self, attr, None)
        if g is not None:
            try: self.scene.removeItem(g)
            except Exception: pass
        setattr(self, attr, None)

    # ---- Undo/Redo helpers ----
    def _hist_key(self) -> str | None:
        if not (0 <= self.cur_idx < len(self.image_paths)):
            return None
        return str(self.image_paths[self.cur_idx].resolve())

    def _schedule_history_push(self, ms: int = 250):
        if getattr(self, "_history_block", False):
            return
        if self._hist_timer is None:
            self._hist_timer = QtCore.QTimer(self)
            self._hist_timer.setSingleShot(True)
            self._hist_timer.timeout.connect(self._push_history)
        if ms <= 0:
            if self._hist_timer.isActive():
                self._hist_timer.stop()
            self._push_history()
            return
        self._hist_timer.start(ms)

    def _push_history(self):
        if getattr(self, "_history_block", False):
            return
        key = self._hist_key()
        if key is None:
            return
        snap = self._extract_shapes_from_scene()
        stk = self._history.setdefault(key, [])
        pos = self._hist_pos.get(key, -1)
        # truncate future states
        if pos < len(stk) - 1:
            del stk[pos+1:]
        # avoid duplicate consecutive snapshots
        force = getattr(self, "_force_snapshot_once", False)
        if stk and stk[-1] == snap and not force:
            self._hist_pos[key] = len(stk) - 1
            self._force_snapshot_once = False
            return
        stk.append(snap)
        self._force_snapshot_once = False
        # cap length
        if len(stk) > 50:
            overflow = len(stk) - 50
            del stk[:overflow]
        self._hist_pos[key] = len(stk) - 1

    def _append_history_snapshot(self, snap: list[dict], *, force: bool = False):
        if getattr(self, "_history_block", False):
            return
        key = self._hist_key()
        if key is None:
            return
        stk = self._history.setdefault(key, [])
        pos = self._hist_pos.get(key, -1)
        if pos < len(stk) - 1:
            del stk[pos+1:]
        if stk and stk[-1] == snap and not force:
            self._hist_pos[key] = len(stk) - 1
            return
        stk.append(copy.deepcopy(snap))
        if len(stk) > 50:
            overflow = len(stk) - 50
            del stk[:overflow]
        self._hist_pos[key] = len(stk) - 1

    def _ensure_pre_move_snapshot(self):
        if self._move_snapshot_active:
            return
        print("[debug] ensure_pre_move_snapshot 호출", flush=True)
        self._move_snapshot_before = copy.deepcopy(self._extract_shapes_from_scene())
        self._move_snapshot_active = True
        self._move_snapshot_dirty = False

    def _finalize_move_snapshot(self):
        if not self._move_snapshot_active:
            print("[debug] finalize 호출됐지만 active 아님 → return", flush=True)
            return
        print(f"[debug] finalize 시작 dirty={self._move_snapshot_dirty}", flush=True)
        if self._move_snapshot_dirty and self._move_snapshot_before is not None:
            ...
            print("[debug] finalize → before/after 히스토리에 push", flush=True)
        self._move_snapshot_active = False
        self._move_snapshot_dirty = False
        self._move_snapshot_before = None
        print("[debug] finalize 종료", flush=True)

    def _clear_shape_items(self):
        for it in list(self.scene.items()):
            if isinstance(it, (RectItem, PolyItem)):
                try:
                    self.scene.removeItem(it)
                except Exception:
                    pass

    def _restore_snapshot(self, snap: list[dict]):
        self._history_block = True
        try:
            self._clear_shape_items()
            self.next_sid = 1
            for sd in snap:
                sh = Shape.from_dict(sd)
                self.next_sid = max(self.next_sid, sh.id + 1)
                self._add_shape_item(sh)
            self._renumber_ids()
            self._update_counts_ui()
            # commit to store
            if 0 <= self.cur_idx < len(self.image_paths):
                img_path = self.image_paths[self.cur_idx]
                self.store.put(img_path, snap, (self.img_item.pixmap().width(), self.img_item.pixmap().height()) if self.img_item else (0,0))
                self._schedule_autosave()
        finally:
            self._history_block = False

    def undo(self):
        key = self._hist_key()
        if key is None:
            return
        pos = self._hist_pos.get(key, -1)
        stk = self._history.get(key, [])
        if pos <= 0 or not stk:
            return
        pos -= 1
        self._hist_pos[key] = pos
        self._restore_snapshot(stk[pos])

    def redo(self):
        key = self._hist_key()
        if key is None:
            return
        pos = self._hist_pos.get(key, -1)
        stk = self._history.get(key, [])
        if pos < 0 or pos >= len(stk) - 1:
            return
        pos += 1
        self._hist_pos[key] = pos
        self._restore_snapshot(stk[pos])

    def _is_valid_item(self, item):
        if item is None: return False
        if sb is None:   return True
        try: return sb.isValid(item)
        except Exception: return True

    def _update_guides(self, pos: QPointF):
        if getattr(self, "_rect_anchor", None) is not None:
            self._hide_guides()
            return
        if not (self.img_item and self._is_valid_item(self.h_guide) and self._is_valid_item(self.v_guide)):
            return
        # 폴리곤 모드에서 첫 포인트 이후에는 가이드라인 비활성화
        if getattr(self, 'drawing_mode', None) == 'poly' and len(self.poly_points) > 0:
            self._hide_guides(); return
        srect = self.scene.sceneRect()
        if srect.contains(pos):
            # 마우스 위치 사용
            self.h_guide.setLine(srect.left(), pos.y(), srect.right(), pos.y())
            self.v_guide.setLine(pos.x(), srect.top(), pos.x(), srect.bottom())
            self.h_guide.setVisible(True)
            self.v_guide.setVisible(True)
        else:
            self._hide_guides()

    def _hide_guides(self):
        if self._is_valid_item(self.h_guide): self.h_guide.setVisible(False)
        if self._is_valid_item(self.v_guide): self.v_guide.setVisible(False)

    def _refresh_guides_now(self):
        """현재 마우스 위치 기준으로 가이드를 즉시 갱신/표시"""
        try:
            if not hasattr(self, 'view') or self.view is None:
                return
            gp = QtGui.QCursor.pos()
            vp = self.view.mapFromGlobal(gp)
            sp = self.view.mapToScene(vp)
            self._update_guides(sp)
        except Exception:
            pass

    # --- poly preview ---
    def _update_poly_preview(self, cur: QPointF):
        if not hasattr(self, '_poly_preview') or self._poly_preview is None: return
        if len(self.poly_points) == 0:
            self._poly_preview.setPath(QtGui.QPainterPath()); return
        path = QtGui.QPainterPath()
        path.moveTo(self.poly_points[0])
        for pt in self.poly_points[1:]: path.lineTo(pt)
        path.lineTo(cur)
        self._poly_preview.setPath(path)

    def _clear_poly_preview(self):
        item = getattr(self, '_poly_preview', None)
        if not item:
            return
        if self._is_valid_item(item):
            scn = item.scene() if hasattr(item, 'scene') else None
            if scn is None and hasattr(self, 'scene'):
                scn = self.scene
            if scn is not None:
                try:
                    scn.removeItem(item)
                except Exception:
                    pass
        self._poly_preview = None

    def cancel_drawing(self):
        if hasattr(self, '_rect_item'):
            try: self.scene.removeItem(self._rect_item)
            except Exception: pass
            del self._rect_item
        self.poly_points.clear()
        self._clear_poly_preview()
        self.set_mode(None); self._update_counts_ui()
        self._commit_cur_to_store(); self._schedule_autosave()
        # 취소 후 가이드를 즉시 복원
        self._refresh_guides_now()

    def delete_at(self, pos: QPointF):
        for it in self.scene.items(pos):
            if isinstance(it, (RectItem, PolyItem)):
                self.scene.removeItem(it); break
        self._renumber_ids()
        self._update_counts_ui()
        self._commit_cur_to_store(); self._schedule_autosave(); self._schedule_history_push(0)

    # --------- ID Renumbering ----------
    def _iter_shape_items(self):
        items = []
        for it in self.scene.items():
            if isinstance(it, (RectItem, PolyItem)):
                items.append(it)
        items.sort(key=lambda x: (getattr(x, "sid", 1_000_000_000), id(x)))
        return items

    def _renumber_ids(self):
        items = self._iter_shape_items()
        sid = 1
        for it in items:
            it.sid = sid; sid += 1
        self.next_sid = sid

    def _restack_shapes_by_sid(self):
        """Restack RectItem/PolyItem to match ascending sid order by reinsertion.
        This guarantees immediate visual update regardless of stackBefore quirks.
        """
        sc = getattr(self, 'scene', None)
        if sc is None:
            return
        items = self._iter_shape_items()
        if len(items) <= 1:
            return
        # Preserve selection set
        selected_ids = {id(it) for it in items if it.isSelected()}
        # Temporarily block history and remove items in current order
        try:
            for it in items:
                try:
                    sc.removeItem(it)
                except Exception:
                    pass
            # Re-add in sid ascending order (bottom -> top)
            for it in items:
                try:
                    sc.addItem(it)
                except Exception:
                    pass
                # restore selection
                try:
                    it.setSelected(id(it) in selected_ids)
                except Exception:
                    pass
            # Force a redraw
            try:
                sc.invalidate(sc.sceneRect(), QtWidgets.QGraphicsScene.AllLayers)
            except Exception:
                try:
                    sc.update()
                except Exception:
                    pass
            # Also nudge view to repaint
            try:
                if hasattr(self, 'view') and self.view is not None:
                    self.view.viewport().update()
            except Exception:
                pass
        except Exception:
            pass

    def _normalize_shape_zvalues(self):
        """Ensure Rect/Poly shapes share a common z so stacking is driven by order only."""
        try:
            for it in self._iter_shape_items():
                try:
                    it.setZValue(0.0)
                except Exception:
                    pass
            if hasattr(self, 'scene') and self.scene is not None:
                try:
                    self.scene.update()
                except Exception:
                    pass
        except Exception:
            pass

    # --------- Scene <-> Model ----------
    def _extract_shapes_from_scene(self) -> list[dict]:
        """장면(scene) 좌표계로 변환해서 저장 (이동/리사이즈 반영)"""
        shapes=[]
        for it in self._iter_shape_items():
            if isinstance(it, RectItem):
                # <<<<<<<<< 핵심 수정: 로컬 rect → scene rect 변환 >>>>>>>>>>
                r_scene = it.mapRectToScene(it.rect())
                pts=[(r_scene.left(), r_scene.top()), (r_scene.right(), r_scene.bottom())]
                shapes.append(Shape('rect', pts, it.klass, it.sid).to_dict())
            elif isinstance(it, PolyItem):
                # PolyItem의 원본 버텍스(_points)를 scene 좌표로 변환해 저장
                # toFillPolygon()은 시작/끝 점을 중복 포함할 수 있어 핸들 중복을 유발할 수 있음
                if hasattr(it, '_points'):
                    poly_local = QtGui.QPolygonF(it._points)
                    poly_scene = it.mapToScene(poly_local)
                    pts=[(p.x(), p.y()) for p in poly_scene]
                else:
                    # 안전장치: 기존 방식으로 fallback
                    poly_local = it.path().toFillPolygon()
                    poly_scene = it.mapToScene(poly_local)
                    pts=[(p.x(), p.y()) for p in poly_scene]
                # 정규화: 마지막 점이 첫 점과 거의 같은 경우 제거
                if len(pts) >= 2:
                    x0,y0 = pts[0]
                    xL,yL = pts[-1]
                    if abs(x0 - xL) + abs(y0 - yL) < 1e-6:
                        pts = pts[:-1]
                shapes.append(Shape('poly', pts, it.klass, it.sid).to_dict())
        return shapes

    def _add_shape_item(self, sh: Shape):
        color=self.class_colors.get(sh.klass, QColor(0,255,0))
        if sh.type == 'rect':
            (x1,y1),(x2,y2) = sh.points
            r = QRectF(QPointF(x1,y1), QPointF(x2,y2)).normalized()
            item = RectItem(r, color, sh.id, sh.klass)
        else:
            # sanitize: remove duplicated closing vertex if present
            raw_pts=[QPointF(x,y) for x,y in sh.points]
            if len(raw_pts) >= 2:
                p0, pL = raw_pts[0], raw_pts[-1]
                if abs(p0.x()-pL.x()) + abs(p0.y()-pL.y()) < 1e-6:
                    raw_pts = raw_pts[:-1]
            item = PolyItem(raw_pts, color, sh.id, sh.klass)
        self.scene.addItem(item)

    def _finalize_partial_polygon_if_any(self):
        if getattr(self,"drawing_mode",None) == 'poly' and len(self.poly_points)>0:
            if len(self.poly_points)>=3:
                color=self.class_colors.get(self.active_class, QColor(0,255,0))
                item=PolyItem(self.poly_points, color, self.next_sid, self.active_class)
                self.scene.addItem(item); self.next_sid+=1
        self.poly_points.clear()
        self._clear_poly_preview()
        self._update_counts_ui()
        self._commit_cur_to_store(); self._schedule_autosave(); self._schedule_history_push(0)

    # --------- Counts ----------
    def _compute_counts_from_scene(self):
        total=0; per={}
        for it in self.scene.items():
            if isinstance(it,(RectItem,PolyItem)):
                total+=1; per[it.klass]=per.get(it.klass,0)+1
        return total, per

    def _update_counts_ui(self):
        if not hasattr(self, "lbl_total"):
            return
        total, per = self._compute_counts_from_scene()
        self.lbl_total.setText(f"Total: {total}")
        self.count_list.clear()

        for name in sorted(per.keys()):
            n = per[name]
            if n <= 0:
                continue
            item = QListWidgetItem(f"{name}: {n}")
            col = self.class_colors.get(name, QColor(0, 255, 0))
            item.setIcon(self._make_color_icon(col))
            item.setForeground(QColor(20, 20, 20))
            self.count_list.addItem(item)
        self._update_class_totals()

    def _dataset_class_totals(self) -> Counter[str]:
        counts: Counter[str] = Counter()
        for rec in self.store._db.get("images", {}).values():
            for sh in rec.get("shapes", []):
                cname = sh.get("class")
                if cname:
                    counts[cname] += 1
        return counts

    def _adjust_class_list_height(self):
        """Grow/shrink the class list height to fit all items (no scrollbar)."""
        if not getattr(self, "class_list", None):
            return
        count = self.class_list.count()
        row_h = self.class_list.sizeHintForRow(0)
        if row_h <= 0:
            row_h = self.class_list.fontMetrics().height() + 8
        spacing = self.class_list.spacing() if hasattr(self.class_list, "spacing") else 0
        frame = self.class_list.frameWidth() * 2
        total = row_h * max(1, count) + spacing * max(0, count - 1) + frame + 4
        self.class_list.setFixedHeight(total)

    def _class_item_by_name(self, name: str) -> QListWidgetItem | None:
        for i in range(self.class_list.count()):
            item = self.class_list.item(i)
            if (item.data(Qt.UserRole) or item.text()) == name:
                return item
        return None

    def _refresh_class_item_label(self, item: QListWidgetItem, counts: Counter[str] | None = None):
        if item is None:
            return
        name = item.data(Qt.UserRole) or item.text()
        if counts is None:
            counts = self._dataset_class_totals()
        count = counts.get(name, 0)
        item.setText(f"{name} ({count})")

    def _update_class_totals(self):
        counts = self._dataset_class_totals()
        for i in range(self.class_list.count()):
            item = self.class_list.item(i)
            self._refresh_class_item_label(item, counts)

    def _compute_export_stats(self) -> dict[str, int]:
        """Collect basic export stats for preview labels."""
        stats = {"total_images": 0, "rect_images": 0}
        try:
            excluded = {str(p.resolve()) for p in self._excluded_paths}
        except Exception:
            excluded = {str(p) for p in self._excluded_paths}
        imgs = [p for p in list(self.store._db.get("images", {}).keys()) if p not in excluded]
        if not imgs:
            base_paths = []
            try:
                base_paths = [p for p in self.image_paths if (p.resolve() not in self._excluded_paths)]
            except Exception:
                base_paths = [p for p in self.image_paths if (p not in self._excluded_paths)]
            imgs = [str(p.resolve()) for p in base_paths]
            for p in base_paths:
                # ensure records exist for preview counts
                pm = QPixmap(str(p)); self.store.put(p, [], (pm.width(), pm.height()))
        stats["total_images"] = len(imgs)
        rect_images = 0
        for ipath in imgs:
            rec = self.store.get(Path(ipath))
            if any((sh.get("type") == "rect" and len(sh.get("points", [])) >= 2) for sh in rec.get("shapes", [])):
                rect_images += 1
        stats["rect_images"] = rect_images
        return stats

    # ====================== YOLO EXPORT ======================
    def open_export_dialog(self):
        if not self.image_paths:
            QMessageBox.warning(self, "No images", "먼저 이미지 폴더를 여세요.")
            return
        stats = self._compute_export_stats()
        d = ExportDialog(self, export_stats=stats)
        if d.exec() != QDialog.Accepted:
            return
        tr, va, te = d.ratios()
        task = d.task()
        ds_name = d.dataset_name()
        if not ds_name or "/" in ds_name or "\\" in ds_name:
            QMessageBox.warning(self, "Dataset name", "유효한 dataset 이름을 입력하세요.")
            return
        out_root = PROJECT_ROOT / "datasets" / ds_name
        if out_root.exists():
            QMessageBox.warning(self, "Dataset name", "동일한 이름의 폴더가 이미 존재합니다.")
            return
        aug_cfg = d.augmentation_config()
        # --- pre-check: detect vs segment annotation types ---
        has_rect, has_poly = self._scan_annotation_kinds()
        if task == 'detect' and has_poly:
            QMessageBox.warning(self, "Export warning",
                                "Detect 포맷을 선택했지만 polygon 주석이 존재합니다.\npolygon은 내보내기에서 무시됩니다.")
        if task == 'segment' and has_rect:
            QMessageBox.warning(self, "Export blocked",
                                "Segment 포맷으로 내보내려면 polygon 주석만 있어야 합니다.\nbbox가 포함되어 있어 내보내기를 중단합니다.")
            return
        out_root.mkdir(parents=True, exist_ok=True)
        # Progress dialog
        prog = QtWidgets.QProgressDialog("Exporting dataset…", None, 0, 100, self)
        prog.setWindowTitle("Exporting")
        prog.setWindowModality(Qt.ApplicationModal)
        prog.setAutoClose(False)
        prog.setAutoReset(False)
        prog.show()
        prog.setValue(0)
        QApplication.processEvents()
        def _update_progress(done: int, total: int):
            if total <= 0:
                return
            val = int(done * 100 / max(1, total))
            prog.setValue(min(100, max(0, val)))
            QApplication.processEvents()
        try:
            self._export_yolo(out_root, tr, va, te, task, augment_config=aug_cfg, progress_cb=_update_progress)
            self.last_export_root = out_root
            prog.setValue(100)
            QApplication.processEvents()
            QMessageBox.information(self, "Export", f"완료: {out_root}")
        except Exception as e:
            QMessageBox.critical(self, "Export failed", f"{e}")
        finally:
            prog.close()

    def _scan_annotation_kinds(self) -> tuple[bool, bool]:
        """Return (has_rect, has_poly). Stops early when both are found."""
        has_rect = False; has_poly = False
        try:
            for rec in self.store._db.get("images", {}).values():
                for sh in rec.get("shapes", []):
                    t = (sh.get("type") or "").lower()
                    if t == 'rect':
                        has_rect = True
                    elif t == 'poly':
                        has_poly = True
                    if has_rect and has_poly:
                        return True, True
        except Exception:
            pass
        return has_rect, has_poly

    def _export_yolo(
        self,
        out_root: Path,
        train_p: int,
        val_p: int,
        test_p: int,
        task: str = "detect",
        *,
        augment_config: ExportAugmentationConfig | None = None,
        progress_cb=None
    ):
        out_root.mkdir(parents=True, exist_ok=True)
        is_segment = (task == "segment")
        # Detect: train|val|test each has images/labels. Segment: images/labels each has train|val|test.
        split_paths: dict[str, tuple[Path, Path]] = {}
        if is_segment:
            for split in ("train", "val", "test"):
                split_paths[split] = (out_root / "images" / split, out_root / "labels" / split)
        else:
            for split in ("train", "val", "test"):
                split_paths[split] = (out_root / split / "images", out_root / split / "labels")
        for split in ("train", "val", "test"):
            img_dir, lbl_dir = split_paths[split]
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)

        # 메타 → 없으면 실제 주석에서 사용된 클래스 수집
        classes = [c["name"] for c in self.store.list_classes()]
        if not classes:
            used = []
            for rec in self.store._db.get("images", {}).values():
                for sh in rec.get("shapes", []):
                    cname = sh.get("class")
                    if cname and cname not in used:
                        used.append(cname)
            classes = used

        if not classes:
            QMessageBox.warning(self, "Export aborted",
                                "사용할 클래스가 없습니다. 클래스를 추가하거나 주석에 class를 지정하세요.")
            return

        cls_to_id = {c:i for i,c in enumerate(classes)}
        aug_cfg = augment_config or ExportAugmentationConfig()

        # Build export image list, honoring session-level exclusions
        excluded = set()
        try:
            excluded = {str(p.resolve()) for p in self._excluded_paths}
        except Exception:
            excluded = {str(p) for p in self._excluded_paths}
        imgs = [p for p in list(self.store._db.get("images", {}).keys()) if p not in excluded]
        if not imgs:
            base_paths = []
            try:
                base_paths = [p for p in self.image_paths if (p.resolve() not in self._excluded_paths)]
            except Exception:
                base_paths = [p for p in self.image_paths if (p not in self._excluded_paths)]
            imgs = [str(p.resolve()) for p in base_paths]
            for p in base_paths:
                pm = QPixmap(str(p)); self.store.put(p, [], (pm.width(), pm.height()))

        random.shuffle(imgs)
        n = len(imgs)
        n_train = round(n * train_p / 100.0)
        n_val   = round(n * val_p   / 100.0)
        n_test  = n - n_train - n_val

        splits = [("train", imgs[:n_train]),
                  ("val",   imgs[n_train:n_train+n_val]),
                  ("test",  imgs[n_train+n_val:])]

        # pre-compute total operations (copies + augmentations + yaml)
        aug_enabled = (
            isinstance(aug_cfg, ExportAugmentationConfig)
            and aug_cfg.enabled
            and aug_cfg.multiplier > 1
            and _AUGMENT_LIB_AVAILABLE
            and (aug_cfg.techniques or aug_cfg.details)
        )
        total_ops = 1  # yaml write
        for ipath in imgs:
            total_ops += 1  # base copy/label
            if not aug_enabled:
                continue
            rec = self.store.get(Path(ipath))
            has_rect = any((sh.get("type") == "rect" and len(sh.get("points", [])) >= 2) for sh in rec.get("shapes", []))
            if has_rect:
                total_ops += max(0, aug_cfg.multiplier - 1)
        done_ops = 0

        def _tick(delta: int = 1):
            nonlocal done_ops
            done_ops += max(0, delta)
            if progress_cb:
                try:
                    progress_cb(done_ops, total_ops)
                except Exception:
                    pass

        for split, paths in splits:
            for ipath in paths:
                ipath = Path(ipath)
                rec = self.store.get(ipath)
                W,H = rec.get("image_size") or [None, None]
                if not W or not H:
                    pm = QPixmap(str(ipath))
                    W,H = pm.width(), pm.height()

                dst_img = split_paths[split][0] / ipath.name
                try:
                    shutil.copy2(ipath, dst_img)
                except Exception:
                    pass

                txt_lines=[]
                for sh in rec.get("shapes", []):
                    cname = sh.get("class")
                    if not cname or cname not in cls_to_id:
                        continue
                    cid = cls_to_id[cname]
                    pts = sh.get("points", [])
                    if task == 'detect':
                        if sh.get("type") != "rect" or len(pts) < 2:
                            continue
                        (x1,y1),(x2,y2) = pts
                        cx,cy,ww,hh = self._norm_box(x1,y1,x2,y2,float(W),float(H))
                        txt_lines.append(f"{cid} {cx:.6f} {cy:.6f} {ww:.6f} {hh:.6f}")
                    else:  # segment format: class followed by normalized polygon points (polygons only)
                        if sh.get("type") == 'poly' and len(pts) >= 3:
                            flat = []
                            for (x,y) in pts:
                                flat.extend([x/float(W), y/float(H)])
                            txt_lines.append(f"{cid} " + " ".join(f"{v:.6f}" for v in flat))

                dst_lbl = split_paths[split][1] / (ipath.stem + ".txt")
                dst_lbl.write_text("\n".join(txt_lines), encoding="utf-8")
                if split == "train":
                    created = self._write_augmented_versions(
                        ipath, rec, out_root, cls_to_id, aug_cfg, progress_cb, done_ops, total_ops, task=task
                    )
                    # created includes its own progress ticks
                    done_ops = created if isinstance(created, int) else done_ops
                _tick(1)

        if task == "segment":
            train_rel = "images/train"
            val_rel = "images/val"
            test_rel = "images/test"
            yaml_name = f"{out_root.name}.yaml"
        else:
            train_rel = "train/images"
            val_rel = "val/images"
            test_rel = "test/images"
            yaml_name = "data.yaml"
        yaml = [
            f"path: {str(out_root.resolve())}",
            f"train: {train_rel}",
            f"val: {val_rel}",
            f"test: {test_rel}",
            "names:"
        ] + [f"  {i}: {name}" for i, name in enumerate(classes)]
        (out_root / yaml_name).write_text("\n".join(yaml), encoding="utf-8")
        _tick(1)

    @staticmethod
    def _norm_box(x1, y1, x2, y2, W, H):
        if W == 0 or H == 0:
            return 0.0, 0.0, 0.0, 0.0
        cx = ((x1 + x2) / 2.0) / W
        cy = ((y1 + y2) / 2.0) / H
        ww = abs(x2 - x1) / W
        hh = abs(y2 - y1) / H
        return cx, cy, ww, hh

    def _build_aug_pipeline(self, width: int, height: int, aug_cfg: ExportAugmentationConfig):
        if not _AUGMENT_LIB_AVAILABLE or width <= 0 or height <= 0:
            return None
        transforms = []
        h = max(1, int(height))
        w = max(1, int(width))
        details = getattr(aug_cfg, "details", None) if isinstance(aug_cfg, ExportAugmentationConfig) else None
        techniques = aug_cfg.techniques if isinstance(aug_cfg, ExportAugmentationConfig) else []

        if details:
            tk = details.get("train_kwargs") or {}
            alb = details.get("albumentations") or {}
            if tk.get("fliplr", 0) > 0:
                transforms.append(A.HorizontalFlip(p=1.0))
            if tk.get("flipud", 0) > 0:
                transforms.append(A.VerticalFlip(p=1.0))
            deg = max(0.0, float(tk.get("degrees", 0.0) or 0.0))
            scale_strength = max(0.0, float(tk.get("scale", 0.0) or 0.0))
            shear_deg = max(0.0, float(tk.get("shear", 0.0) or 0.0))
            if deg > 0 or scale_strength > 0:
                scale_low = max(0.0, 1.0 - scale_strength)
                scale_high = 1.0 + scale_strength
                transforms.append(A.ShiftScaleRotate(
                    shift_limit=0.03, scale_limit=(scale_low, scale_high), rotate_limit=deg, border_mode=0, p=0.7))
            if shear_deg > 0:
                transforms.append(A.Affine(shear=(-shear_deg, shear_deg), p=0.5))
            hsv_h = float(tk.get("hsv_h", 0.0) or 0.0)
            hsv_s = float(tk.get("hsv_s", 0.0) or 0.0)
            hsv_v = float(tk.get("hsv_v", 0.0) or 0.0)
            if hsv_h or hsv_s or hsv_v:
                transforms.append(A.HueSaturationValue(
                    hue_shift_limit=hsv_h * 50.0,
                    sat_shift_limit=hsv_s * 50.0,
                    val_shift_limit=hsv_v * 50.0,
                    p=0.6
                ))
            rot90_p = float(alb.get("rotate90_p", 0.0) or 0.0)
            if rot90_p > 0:
                transforms.append(A.RandomRotate90(p=min(1.0, rot90_p)))
            gs_p = float(alb.get("grayscale_p", 0.0) or 0.0)
            if gs_p > 0:
                transforms.append(A.ToGray(p=min(1.0, gs_p)))
            exp = float(alb.get("exposure_p", 0.0) or 0.0)
            if exp > 0:
                transforms.append(A.RandomBrightnessContrast(
                    brightness_limit=exp, contrast_limit=exp / 2.0, p=0.6))
            blur_p = float(alb.get("blur_p", 0.0) or 0.0)
            if blur_p > 0:
                lim = max(3, int(round(blur_p * 6)) | 1)
                transforms.append(A.GaussianBlur(blur_limit=(3, lim), p=0.5))
            noise_p = float(alb.get("noise_p", 0.0) or 0.0)
            if noise_p > 0:
                transforms.append(A.GaussNoise(var_limit=(10.0, max(10.0, noise_p * 60.0)), p=0.5))
        else:
            if "flip_lr" in techniques:
                transforms.append(A.HorizontalFlip(p=0.7))
            if "flip_ud" in techniques:
                transforms.append(A.VerticalFlip(p=0.4))
            if "rotate90" in techniques:
                transforms.append(A.RandomRotate90(p=0.6))
            if "rotation" in techniques:
                transforms.append(A.ShiftScaleRotate(
                    shift_limit=0.03, scale_limit=0.1, rotate_limit=18, border_mode=0, p=0.7))
            if "crop" in techniques:
                transforms.append(A.RandomResizedCrop(
                    size=(h, w), scale=(0.7, 1.0), ratio=(0.8, 1.25), p=0.6))
            if "shear" in techniques:
                transforms.append(A.Affine(shear=(-16, 16), p=0.5))
            if "grayscale" in techniques:
                transforms.append(A.ToGray(p=0.4))
            if "hue_saturation" in techniques:
                transforms.append(A.HueSaturationValue(
                    hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.6))
            if "brightness" in techniques:
                transforms.append(A.RandomBrightnessContrast(
                    brightness_limit=0.3, contrast_limit=0.2, p=0.6))
            if "blur" in techniques:
                transforms.append(A.GaussianBlur(blur_limit=(3, 7), p=0.5))
            if "noise" in techniques:
                transforms.append(A.GaussNoise(var_limit=(10.0, 60.0), p=0.5))

        if not transforms:
            return None
        return A.Compose(
            transforms,
            bbox_params=A.BboxParams(
                format="pascal_voc",
                label_fields=["class_labels"],
                min_visibility=0.1,
                min_area=1
            )
        )

    def _write_augmented_versions(
        self,
        ipath: Path,
        rec: dict,
        out_root: Path,
        cls_to_id: dict[str, int],
        augment_config: ExportAugmentationConfig,
        progress_cb=None,
        done_ops: int = 0,
        total_ops: int = 0,
        *,
        task: str = "detect",
    ):
        if not augment_config.enabled or augment_config.multiplier <= 1:
            return done_ops
        if (not augment_config.techniques and not getattr(augment_config, "details", None)) or not _AUGMENT_LIB_AVAILABLE:
            return done_ops
        boxes = []
        labels = []
        for sh in rec.get("shapes", []):
            if sh.get("type") != "rect":
                continue
            pts = sh.get("points", [])
            if len(pts) < 2:
                continue
            (x1, y1), (x2, y2) = pts[:2]
            boxes.append([
                min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
            ])
            labels.append(sh.get("class"))
        if not boxes:
            return
        size = rec.get("image_size") or [None, None]
        W = size[0] or 0
        H = size[1] or 0
        if not W or not H:
            pm = QPixmap(str(ipath))
            W, H = pm.width(), pm.height()
        pipeline = self._build_aug_pipeline(W, H, augment_config)
        if pipeline is None:
            return done_ops
        try:
            base_img = Image.open(ipath).convert("RGB")
        except Exception:
            return done_ops
        base_arr = np.array(base_img)
        for idx in range(1, augment_config.multiplier):
            try:
                result = pipeline(image=base_arr, bboxes=[list(b) for b in boxes], class_labels=list(labels))
            except Exception:
                continue
            aug_boxes = result.get("bboxes") or []
            aug_labels = result.get("class_labels") or []
            aug_img = result.get("image")
            if not aug_boxes or aug_img is None:
                continue
            suffix = ipath.suffix or ".jpg"
            sample_name = f"{ipath.stem}_aug{idx}{suffix}"
            if task == "segment":
                img_train = out_root / "images" / "train"
                lbl_train = out_root / "labels" / "train"
            else:
                img_train = out_root / "train" / "images"
                lbl_train = out_root / "train" / "labels"
            img_train.mkdir(parents=True, exist_ok=True)
            lbl_train.mkdir(parents=True, exist_ok=True)
            dst_img = img_train / sample_name
            dst_lbl = lbl_train / f"{Path(sample_name).stem}.txt"
            try:
                Image.fromarray(aug_img).save(dst_img)
            except Exception:
                continue
            h, w = aug_img.shape[:2]
            txt_lines = []
            for bbox, label in zip(aug_boxes, aug_labels):
                if len(bbox) != 4 or label is None:
                    continue
                cid = cls_to_id.get(label)
                if cid is None:
                    continue
                cx, cy, ww, hh = self._norm_box(bbox[0], bbox[1], bbox[2], bbox[3], float(w), float(h))
                txt_lines.append(f"{cid} {cx:.6f} {cy:.6f} {ww:.6f} {hh:.6f}")
            if not txt_lines:
                try:
                    dst_img.unlink()
                except Exception:
                    pass
                continue
            dst_lbl.write_text("\n".join(txt_lines), encoding="utf-8")
            done_ops += 1
            if progress_cb:
                try:
                    progress_cb(done_ops, total_ops)
                except Exception:
                    pass
        return done_ops

    # ====================== TRAIN ======================
    def open_train_dialog(self):
        # Backward-compatible dialog; Train tab uses start_training_with_params()
        try:
            import ultralytics  # noqa
        except ImportError:
            reply = QMessageBox.question(
                self, "Install ultralytics?",
                "ultralytics 패키지가 설치되어 있지 않습니다.\n지금 설치하시겠습니까?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
                    QMessageBox.information(self, "Installed", "설치 완료")
                except Exception as e:
                    QMessageBox.critical(self, "Install failed", f"{e}")
                    return
            else:
                return

        data_yaml = ""
        if self.last_export_root and (self.last_export_root / "data.yaml").exists():
            data_yaml = str((self.last_export_root / "data.yaml").resolve())

        d = TrainParamDialog(self, data_yaml_path=data_yaml)
        if d.exec() != QDialog.Accepted:
            return
        params = d.params()

        if not params["data"] or not Path(params["data"]).exists():
            QMessageBox.warning(self, "data.yaml", "유효한 data.yaml 경로를 선택하세요.")
            return

        prog = TrainProgressDialog(self, params)
        prog.exec()

    # Public API for Train tab to start training with embedded parameters
    def start_training_with_params(self, params: dict):
        try:
            import ultralytics  # noqa: F401
        except ImportError:
            reply = QMessageBox.question(
                self, "Install ultralytics?",
                "ultralytics 패키지가 설치되어 있지 않습니다.\n지금 설치하시겠습니까?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
                    QMessageBox.information(self, "Installed", "설치 완료")
                except Exception as e:
                    QMessageBox.critical(self, "Install failed", f"{e}")
                    return
            else:
                return

        if not params.get("data") or not Path(params["data"]).exists():
            QMessageBox.warning(self, "data.yaml", "유효한 data.yaml 경로를 선택하세요.")
            return

        prog = TrainProgressDialog(self, params)
        prog.exec()

    # ====================== INFERENCE ======================
    def open_infer_dialog(self):
        try:
            import ultralytics  # noqa
        except ImportError:
            reply = QMessageBox.question(
                self, "Install ultralytics?",
                "ultralytics 패키지가 설치되어 있지 않습니다.\n지금 설치하시겠습니까?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
                    QMessageBox.information(self, "Installed", "설치 완료")
                except Exception as e:
                    QMessageBox.critical(self, "Install failed", f"{e}")
                    return
            else:
                return

        default_w = ""
        runs = Path("runs/detect")
        if runs.exists():
            cand = sorted(runs.glob("*/weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
            if cand:
                default_w = str(cand[0])

        d = InferenceDialog(self, default_model=default_w)
        if d.exec() != QDialog.Accepted:
            return
        params = d.params()

        if not params["model"] or not Path(params["model"]).exists():
            QMessageBox.warning(self, "weights", "유효한 모델 가중치 파일을 선택하세요.")
            return
        if params["use_folder"]:
            if not params["folder"] or not Path(params["folder"]).exists():
                QMessageBox.warning(self, "source", "유효한 폴더를 선택하세요.")
                return

        cur_img = None
        if not params["use_folder"]:
            if 0 <= self.cur_idx < len(self.image_paths):
                cur_img = str(self.image_paths[self.cur_idx].resolve())
            else:
                QMessageBox.warning(self, "No image", "현재 로드된 이미지가 없습니다.")
                return

        prog = InferProgressDialog(
            self,
            on_single_boxes=self._on_single_boxes,       # 캐시 + 표시
            on_folder_box=self._cache_and_maybe_show
        )
        prog.start(params, cur_img)
        prog.exec()

    # 단일 inference 결과 캐시에 저장 후 표시
    def _on_single_boxes(self, boxes: list, names: list):
        if 0 <= self.cur_idx < len(self.image_paths):
            key = str(self.image_paths[self.cur_idx].resolve())
            self._infer_cache[key] = (boxes, names)
        self._show_pred_overlays(boxes, names)

    def _cache_and_maybe_show(self, img_path: str, boxes: list, names: list):
        key = str(Path(img_path).resolve())
        self._infer_cache[key] = (boxes, names)
        if 0 <= self.cur_idx < len(self.image_paths):
            cur = str(self.image_paths[self.cur_idx].resolve())
            if cur == key:
                self._show_pred_overlays(boxes, names)

    def _clear_pred_overlays(self):
        for it in list(self._pred_overlays):
            try:
                if it is None:
                    continue
                # skip invalid or already-detached items to avoid Qt warnings
                if sb is not None:
                    try:
                        if not sb.isValid(it):
                            continue
                    except Exception:
                        pass
                scn = getattr(it, 'scene', None)
                if callable(scn):
                    sobj = scn()
                    if sobj is not None:
                        sobj.removeItem(it)
            except Exception:
                pass
        self._pred_overlays.clear()
        self._det_rects = []  # reset mapping

        # 오른쪽 리스트도 초기화 (guarded)
        if getattr(self, 'det_list', None) is not None:
            try:
                if sb is None or sb.isValid(self.det_list):
                    self.det_list.clear()
            except Exception:
                pass
        if getattr(self, 'lbl_det_total', None) is not None:
            try:
                if sb is None or sb.isValid(self.lbl_det_total):
                    self.lbl_det_total.setText("Detections: 0")
            except Exception:
                pass

    def _show_pred_overlays(self, boxes: list, names: list):
        """boxes: [{'xyxy':(x1,y1,x2,y2),'cls':int,'conf':float,'name':str}, ...]"""
        self._clear_pred_overlays()
        pen_default = QPen(QColor(0,120,255), 2, Qt.DashLine)
        brush = QBrush(QColor(0,120,255, 40))

        label_font = QFont()
        label_font.setPointSize(10)
        label_font.setBold(True)

        scene_rect = self.scene.sceneRect()

        self._det_rects = []

        for b in boxes:
            x1,y1,x2,y2 = b["xyxy"]
            r = QRectF(QPointF(x1,y1), QPointF(x2,y2)).normalized()

            # --- BBox ---
            rect_item = QGraphicsRectItem(r)
            rect_item.setPen(pen_default); rect_item.setBrush(brush); rect_item.setZValue(9e6)
            rect_item.setFlag(QGraphicsItem.ItemIsSelectable, False)
            rect_item.setFlag(QGraphicsItem.ItemIsMovable, False)
            self.scene.addItem(rect_item)
            self._pred_overlays.append(rect_item)
            self._det_rects.append(rect_item)  # order matters

            # --- Label text (class + conf) ---
            cname = b.get("name", str(b.get("cls", "")))
            conf = float(b.get("conf", 0.0)) * 100.0
            label_text = f"{cname} {conf:.1f}%"

            txt_item = QGraphicsTextItem(label_text)
            txt_item.setFont(label_font)
            txt_item.setDefaultTextColor(Qt.white)
            txt_item.setZValue(9e6 + 1)
            txt_item.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)

            # Background for text
            pad = 3
            txt_item.setPos(0, 0)
            self.scene.addItem(txt_item)
            br = txt_item.boundingRect()
            bg_item = QGraphicsRectItem(0, 0, br.width() + 2*pad, br.height() + 2*pad)
            bg_item.setBrush(QBrush(QColor(0,120,255, 220)))
            bg_item.setPen(Qt.NoPen)
            bg_item.setZValue(9e6)
            bg_item.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
            self.scene.addItem(bg_item)

            # Place at top-left of bbox (clamped to scene)
            px = r.left()
            py = r.top() - (br.height() + 2*pad + 2)
            if py < scene_rect.top():
                py = r.top() + 2
            if px < scene_rect.left():
                px = scene_rect.left() + 2
            if px + br.width() + 2*pad > scene_rect.right():
                px = scene_rect.right() - (br.width() + 2*pad) - 2

            bg_item.setPos(px, py)
            txt_item.setPos(px + pad, py + pad)

            self._pred_overlays.extend([bg_item, txt_item])

        # ---- 오른쪽 리스트: detections 채우기 (정렬 X, 원본 순서 유지) ----
        if getattr(self, 'det_list', None) is not None and (sb is None or sb.isValid(self.det_list)):
            self.det_list.clear()
            for idx, b in enumerate(boxes):
                cname = b.get("name") or str(b.get("cls", ""))
                conf = float(b.get("conf", 0.0)) * 100.0
                item = QListWidgetItem(f"{cname}: {conf:.1f}%")
                item.setData(Qt.UserRole, idx)  # rect 인덱스와 1:1 매칭
                col = self.class_colors.get(cname, QColor(0,120,255))
                item.setIcon(self._make_color_icon(col))
                item.setForeground(QColor(20,20,20))
                self.det_list.addItem(item)
            if getattr(self, 'lbl_det_total', None) is not None and (sb is None or sb.isValid(self.lbl_det_total)):
                self.lbl_det_total.setText(f"Detections: {len(boxes)}")

    def _on_det_item_clicked(self, item: QListWidgetItem):
        idx = item.data(Qt.UserRole)
        if not isinstance(idx, int): 
            return
        if idx < 0 or idx >= len(self._det_rects):
            return

        pen_default = QPen(QColor(0,120,255), 2, Qt.DashLine)
        pen_highlight = QPen(QColor(255, 0, 0), 2.5, Qt.SolidLine)

        for i, rect in enumerate(self._det_rects):
            if not isinstance(rect, QGraphicsRectItem): 
                continue
            rect.setPen(pen_highlight if i == idx else pen_default)

        target_rect: QGraphicsRectItem = self._det_rects[idx]
        self.view.ensureVisible(target_rect, 40, 40)  # 패딩


# ----------------------------- Run -----------------------------
def main():
    app = QApplication(sys.argv)
    w = LabelTool()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
# Simple two-handle range slider for train/val/test split
class RangeSlider(QtWidgets.QSlider):
    lowerValueChanged = Signal(int)
    upperValueChanged = Signal(int)

    def __init__(self, parent=None):
        super().__init__(Qt.Horizontal, parent)
        self._lower = 70
        self._upper = 90
        self.setRange(0, 100)
        self.setTickPosition(QtWidgets.QSlider.NoTicks)
        self._active = None
        self.setFixedHeight(26)

    def lowerValue(self) -> int:
        return self._lower

    def upperValue(self) -> int:
        return self._upper

    def setLowerValue(self, v: int):
        v = max(self.minimum(), min(v, self._upper))
        if v != self._lower:
            self._lower = v
            self.lowerValueChanged.emit(v)
            self.update()

    def setUpperValue(self, v: int):
        v = min(self.maximum(), max(v, self._lower))
        if v != self._upper:
            self._upper = v
            self.upperValueChanged.emit(v)
            self.update()

    def _pixel_pos_to_value(self, x: int) -> int:
        opt = QtWidgets.QStyleOptionSlider()
        self.initStyleOption(opt)
        slider_len = self.style().pixelMetric(QtWidgets.QStyle.PM_SliderLength, opt, self)
        slider_min = slider_len // 2
        slider_max = self.width() - slider_len // 2
        x = max(slider_min, min(x, slider_max))
        span = slider_max - slider_min
        return QtWidgets.QStyle.sliderValueFromPosition(self.minimum(), self.maximum(), x - slider_min, span, False)

    def mousePressEvent(self, e: QtGui.QMouseEvent):
        if e.button() != Qt.LeftButton:
            return super().mousePressEvent(e)
        pos = e.position().toPoint().x()
        low_pos = self._value_to_pixel(self._lower)
        up_pos = self._value_to_pixel(self._upper)
        if abs(pos - low_pos) <= abs(pos - up_pos):
            self._active = "lower"
            self.setLowerValue(self._pixel_pos_to_value(pos))
        else:
            self._active = "upper"
            self.setUpperValue(self._pixel_pos_to_value(pos))
        e.accept()

    def mouseMoveEvent(self, e: QtGui.QMouseEvent):
        if self._active is None:
            return super().mouseMoveEvent(e)
        pos = e.position().toPoint().x()
        val = self._pixel_pos_to_value(pos)
        if self._active == "lower":
            self.setLowerValue(val)
        else:
            self.setUpperValue(val)
        e.accept()

    def mouseReleaseEvent(self, e: QtGui.QMouseEvent):
        self._active = None
        return super().mouseReleaseEvent(e)

    def _value_to_pixel(self, v: int) -> int:
        opt = QtWidgets.QStyleOptionSlider()
        self.initStyleOption(opt)
        slider_len = self.style().pixelMetric(QtWidgets.QStyle.PM_SliderLength, opt, self)
        slider_min = slider_len // 2
        slider_max = self.width() - slider_len // 2
        span = slider_max - slider_min
        return slider_min + int(span * (v - self.minimum()) / max(1, self.maximum() - self.minimum()))

    def paintEvent(self, event: QtGui.QPaintEvent):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        groove_rect = QtCore.QRect(8, self.height() // 2 - 3, self.width() - 16, 6)
        grad = QtGui.QLinearGradient(groove_rect.left(), groove_rect.center().y(), groove_rect.right(), groove_rect.center().y())
        grad.setColorAt(0.0, QtGui.QColor("#7b61ff"))
        grad.setColorAt(0.7, QtGui.QColor("#5cd0ff"))
        grad.setColorAt(1.0, QtGui.QColor("#f2a900"))
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(grad)
        p.drawRoundedRect(groove_rect, 4, 4)

        for v, col in [(self._lower, QtGui.QColor("#4a3be2")), (self._upper, QtGui.QColor("#c88200"))]:
            cx = self._value_to_pixel(v)
            handle_rect = QtCore.QRectF(cx - 8, self.height() / 2 - 9, 16, 18)
            p.setBrush(QtGui.QColor("#ffffff"))
            pen = QtGui.QPen(col)
            pen.setWidth(2)
            p.setPen(pen)
            p.drawRoundedRect(handle_rect, 8, 8)
        p.end()
