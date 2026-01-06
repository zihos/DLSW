"""Train tab UI skeleton."""

from __future__ import annotations

from PySide6 import QtWidgets, QtGui, QtCore
import re, os, time, csv, subprocess, shutil, yaml, math, colorsys
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from collections import Counter
from pathlib import Path
from PySide6.QtCore import Qt, QTimer
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from ..widgets import Header, TitledGroup, configure_combo, wrap_expanding
from ...label_tool import TrainWorker


class SimpleAugListDialog(QtWidgets.QDialog):
    """Lightweight augmentation hyperparameter editor (no previews)."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Augmentation Settings")
        self.resize(460, 560)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(10)

        defaults = {
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "degrees": 0.0,
            "translate": 0.1,
            "scale": 0.5,
            "shear": 0.0,
            "perspective": 0.0,
            "flipud": 0.0,
            "fliplr": 0.5,
            "mosaic": 1.0,
            "mixup": 0.0,
            "copy_paste": 0.0,
            "erasing": 0.4,
            "crop_fraction": 1.0,
            "auto_augment": "randaugment",
        }

        self._fields: dict[str, QtWidgets.QWidget] = {}
        form = QtWidgets.QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(6)

        def add_spin(key: str, label: str, minimum: float, maximum: float, step: float, decimals: int = 3):
            sp = QtWidgets.QDoubleSpinBox()
            sp.setRange(minimum, maximum)
            sp.setSingleStep(step)
            sp.setDecimals(decimals)
            sp.setValue(float(defaults[key]))
            sp.setAlignment(Qt.AlignRight)
            self._fields[key] = sp
            form.addRow(label, sp)

        add_spin("hsv_h", "hsv_h", 0.0, 1.0, 0.005, 3)
        add_spin("hsv_s", "hsv_s", 0.0, 2.0, 0.05, 3)
        add_spin("hsv_v", "hsv_v", 0.0, 2.0, 0.05, 3)
        add_spin("degrees", "degrees", 0.0, 45.0, 1.0, 1)
        add_spin("translate", "translate", 0.0, 1.0, 0.05, 3)
        add_spin("scale", "scale", 0.0, 2.0, 0.05, 3)
        add_spin("shear", "shear", 0.0, 45.0, 1.0, 1)
        add_spin("perspective", "perspective", 0.0, 0.2, 0.01, 3)
        add_spin("flipud", "flipud", 0.0, 1.0, 0.05, 3)
        add_spin("fliplr", "fliplr", 0.0, 1.0, 0.05, 3)
        add_spin("mosaic", "mosaic", 0.0, 1.0, 0.1, 2)
        add_spin("mixup", "mixup", 0.0, 1.0, 0.1, 2)
        add_spin("copy_paste", "copy_paste", 0.0, 1.0, 0.1, 2)
        add_spin("erasing", "erasing", 0.0, 1.0, 0.05, 2)
        add_spin("crop_fraction", "crop_fraction", 0.0, 1.5, 0.05, 2)

        self.ed_auto_aug = QtWidgets.QLineEdit(str(defaults["auto_augment"]))
        self._fields["auto_augment"] = self.ed_auto_aug
        form.addRow("auto_augment", self.ed_auto_aug)

        lay.addLayout(form)
        lay.addStretch(1)
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

    def values(self) -> dict:
        """Return overrides in the same shape TrainWorker expects."""
        tk: dict[str, float | str] = {}
        for key, w in self._fields.items():
            if isinstance(w, QtWidgets.QDoubleSpinBox):
                tk[key] = float(w.value())
            elif isinstance(w, QtWidgets.QLineEdit):
                tk[key] = w.text().strip()
        return {"train_kwargs": tk, "albumentations": {}}
    """Lightweight augmentation hyperparameter editor (no previews)."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Augmentation Settings")
        self.resize(460, 560)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(10)

        defaults = {
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "degrees": 0.0,
            "translate": 0.1,
            "scale": 0.5,
            "shear": 0.0,
            "perspective": 0.0,
            "flipud": 0.0,
            "fliplr": 0.5,
            "mosaic": 1.0,
            "mixup": 0.0,
            "copy_paste": 0.0,
            "erasing": 0.4,
            "crop_fraction": 1.0,
            "auto_augment": "randaugment",
        }

        self._fields: dict[str, QtWidgets.QWidget] = {}
        form = QtWidgets.QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(6)

        def add_spin(key: str, label: str, minimum: float, maximum: float, step: float, decimals: int = 3):
            sp = QtWidgets.QDoubleSpinBox()
            sp.setRange(minimum, maximum)
            sp.setSingleStep(step)
            sp.setDecimals(decimals)
            sp.setValue(float(defaults[key]))
            sp.setAlignment(Qt.AlignRight)
            self._fields[key] = sp
            form.addRow(label, sp)

        add_spin("hsv_h", "hsv_h", 0.0, 1.0, 0.005, 3)
        add_spin("hsv_s", "hsv_s", 0.0, 2.0, 0.05, 3)
        add_spin("hsv_v", "hsv_v", 0.0, 2.0, 0.05, 3)
        add_spin("degrees", "degrees", 0.0, 45.0, 1.0, 1)
        add_spin("translate", "translate", 0.0, 1.0, 0.05, 3)
        add_spin("scale", "scale", 0.0, 2.0, 0.05, 3)
        add_spin("shear", "shear", 0.0, 45.0, 1.0, 1)
        add_spin("perspective", "perspective", 0.0, 0.2, 0.01, 3)
        add_spin("flipud", "flipud", 0.0, 1.0, 0.05, 3)
        add_spin("fliplr", "fliplr", 0.0, 1.0, 0.05, 3)
        add_spin("mosaic", "mosaic", 0.0, 1.0, 0.1, 2)
        add_spin("mixup", "mixup", 0.0, 1.0, 0.1, 2)
        add_spin("copy_paste", "copy_paste", 0.0, 1.0, 0.1, 2)
        add_spin("erasing", "erasing", 0.0, 1.0, 0.05, 2)
        add_spin("crop_fraction", "crop_fraction", 0.0, 1.5, 0.05, 2)

        self.ed_auto_aug = QtWidgets.QLineEdit(str(defaults["auto_augment"]))
        self._fields["auto_augment"] = self.ed_auto_aug
        form.addRow("auto_augment", self.ed_auto_aug)

        lay.addLayout(form)
        lay.addStretch(1)
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

    def values(self) -> dict:
        """Return overrides in the same shape TrainWorker expects."""
        tk: dict[str, float | str] = {}
        for key, w in self._fields.items():
            if isinstance(w, QtWidgets.QDoubleSpinBox):
                tk[key] = float(w.value())
            elif isinstance(w, QtWidgets.QLineEdit):
                tk[key] = w.text().strip()
        return {"train_kwargs": tk, "albumentations": {}}

    def _outlined_icon(self, icon: QtGui.QIcon, *, margin: int = 1, color: QtGui.QColor | Qt.GlobalColor = QtCore.Qt.white) -> QtGui.QIcon:
        base = icon.pixmap(20, 20)
        if base.isNull():
            return icon
        size = base.size() + QtCore.QSize(margin * 2, margin * 2)
        outline_pm = QtGui.QPixmap(size)
        outline_pm.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(outline_pm)
        painter.setCompositionMode(QtGui.QPainter.CompositionMode_Source)
        # draw outline by offsetting the pixmap around
        for dx in range(-margin, margin + 1):
            for dy in range(-margin, margin + 1):
                if dx == 0 and dy == 0:
                    continue
                painter.drawPixmap(QtCore.QPoint(margin + dx, margin + dy), base)
        painter.setCompositionMode(QtGui.QPainter.CompositionMode_SourceIn)
        painter.fillRect(outline_pm.rect(), QtGui.QColor(color))
        painter.end()
        # draw original on top, centered
        final_pm = QtGui.QPixmap(size)
        final_pm.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(final_pm)
        painter.drawPixmap(0, 0, outline_pm)
        painter.drawPixmap(margin, margin, base)
        painter.end()
        return QtGui.QIcon(final_pm)

    def _make_spin(self, minimum: float, maximum: float, step: float, value: float, *, decimals: int = 2) -> QtWidgets.QDoubleSpinBox:
        spin = QtWidgets.QDoubleSpinBox()
        spin.setDecimals(decimals)
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setValue(value)
        spin.setAlignment(Qt.AlignRight)
        return spin

    def _update_aug_controls(self):
        enabled = bool(self.chk_custom_aug.isChecked())
        for w in self._aug_override_controls:
            w.setEnabled(enabled)
        for lbl in self._aug_override_labels:
            lbl.setEnabled(enabled)

    def values(self) -> dict | None:
        """Return overrides in the same shape TrainWorker expects."""
        tk: dict[str, float | str] = {}
        for key, w in self._fields.items():
            if isinstance(w, QtWidgets.QDoubleSpinBox):
                tk[key] = float(w.value())
            elif isinstance(w, QtWidgets.QLineEdit):
                tk[key] = w.text().strip()
        return {"train_kwargs": tk, "albumentations": {}}

    # ------ preview helpers ------
    def _connect_preview_updates(self):
        # spin boxes
        for sp in (
            self.sp_rotation_deg,
            self.sp_scale_strength,
            self.sp_shear_deg,
            self.sp_grayscale_prob,
            self.sp_hsv_h,
            self.sp_hsv_s,
            self.sp_hsv_v,
            self.sp_exposure_strength,
            self.sp_blur_strength,
            self.sp_noise_strength,
        ):
            sp.valueChanged.connect(self._schedule_refresh)
        # checkboxes
        for chk in (
            self.chk_flip_lr,
            self.chk_flip_ud,
            self.chk_rot90_cw,
            self.chk_rot90_ccw,
        ):
            chk.toggled.connect(self._schedule_refresh)

    def _apply_defaults(self):
        self._refresh_timer.stop()
        bvals = [
            (self.chk_custom_aug, self._defaults["custom"]),
            (self.chk_flip_lr, self._defaults["flip_lr"]),
            (self.chk_flip_ud, self._defaults["flip_ud"]),
            (self.chk_rot90_cw, self._defaults["rot90_cw"]),
            (self.chk_rot90_ccw, self._defaults["rot90_ccw"]),
        ]
        for w, val in bvals:
            w.blockSignals(True)
            w.setChecked(val)
            w.blockSignals(False)
        svals = [
            (self.sp_rotation_deg, self._defaults["rotation"]),
            (self.sp_scale_strength, self._defaults["scale"]),
            (self.sp_shear_deg, self._defaults["shear"]),
            (self.sp_grayscale_prob, self._defaults["grayscale"]),
            (self.sp_hsv_h, self._defaults["hsv_h"]),
            (self.sp_hsv_s, self._defaults["hsv_s"]),
            (self.sp_hsv_v, self._defaults["hsv_v"]),
            (self.sp_exposure_strength, self._defaults["exposure"]),
            (self.sp_blur_strength, self._defaults["blur"]),
            (self.sp_noise_strength, self._defaults["noise"]),
        ]
        for w, val in svals:
            w.blockSignals(True)
            w.setValue(val)
            w.blockSignals(False)
        self._refresh_previews()

    def _on_reset(self):
        self._apply_defaults()

    def _schedule_refresh(self):
        # throttle refreshes to avoid repeated heavy work
        self._refresh_timer.start(30)

    def _refresh_previews(self):
        if self._base_image is None:
            return
        try:
            for lbl, name in self._image_labels:
                aug_img = self._render_aug_example(name)
                if aug_img is None:
                    pm = QtGui.QPixmap()
                else:
                    qimg = QtGui.QImage(
                        aug_img.data, aug_img.shape[1], aug_img.shape[0], QtGui.QImage.Format_RGBA8888
                    )
                    pm = QtGui.QPixmap.fromImage(qimg).scaled(lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                lbl.setPixmap(pm)
        except Exception:
            pass

    def _render_aug_example(self, key: str) -> Image.Image | None:
        if self._base_image is None:
            return None
        img = self._base_image.copy()
        try:
            if key == "fliplr.png":
                if self.chk_flip_lr.isChecked():
                    img = cv2.flip(img, 1)
                if self.chk_flip_ud.isChecked():
                    img = cv2.flip(img, 0)
            elif key == "rot_90.png":
                if self.chk_rot90_cw.isChecked():
                    img = np.rot90(img, k=3)
                elif self.chk_rot90_ccw.isChecked():
                    img = np.rot90(img, k=1)
            elif key == "rot_10.png":
                deg = float(self.sp_rotation_deg.value())
                if deg:
                    img = self._rotate_deg(img, deg)
            elif key == "scale_1.5.png":
                scale = float(self.sp_scale_strength.value())
                if scale != 1.0:
                    img = self._scale_img(img, scale)
            elif key == "shear_15.png":
                shear = float(self.sp_shear_deg.value())
                if shear:
                    img = self._shear_img(img, shear)
            elif key == "grayscale.png":
                p = float(self.sp_grayscale_prob.value())
                if p > 0:
                    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                    img = cv2.addWeighted(img, max(0.0, 1.0 - p), gray3, min(1.0, p), 0)
            elif key == "hue.png":
                delta = float(self.sp_hsv_h.value())
                if delta:
                    img = self._adjust_hsv_np(img, h=delta, s=0, v=0)
            elif key == "saturation.png":
                delta = float(self.sp_hsv_s.value())
                if delta:
                    img = self._adjust_hsv_np(img, h=0, s=delta, v=0)
            elif key == "value.png":
                delta = float(self.sp_hsv_v.value())
                if delta:
                    img = self._adjust_hsv_np(img, h=0, s=0, v=delta)
            elif key == "rot_350.png":
                p = float(self.sp_exposure_strength.value())
                if p > 0:
                    img = cv2.convertScaleAbs(img, alpha=1.0 + p, beta=0)
            elif key == "blur.png":
                p = float(self.sp_blur_strength.value())
                if p > 0:
                    radius = max(0.0, p)
                    k = max(1, int(radius * 2) | 1)
                    img = cv2.GaussianBlur(img, (k, k), sigmaX=radius)
            elif key == "noise.png":
                p = float(self.sp_noise_strength.value())
                if p > 0:
                    img = self._add_noise_np(img, p)
            img = cv2.resize(img, (160, 160), interpolation=cv2.INTER_AREA)
            return cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
        except Exception:
            return None
    def _add_noise_np(self, img: np.ndarray, strength: float) -> np.ndarray:
        sigma = max(1.0, 25.0 * strength)
        noise = np.random.normal(0, sigma, img.shape).astype("float32")
        out = img.astype("float32") + noise
        return np.clip(out, 0, 255).astype("uint8")

    def _rotate_deg(self, img: np.ndarray, deg: float) -> np.ndarray:
        img_sq = self._pad_to_square(img)
        h, w = img_sq.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, deg, 1.0)
        return cv2.warpAffine(img_sq, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    def _scale_img(self, img: np.ndarray, scale: float) -> np.ndarray:
        base = self._pad_to_square(img)
        size = base.shape[0]
        new_size = max(1, int(size * scale))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        scaled = cv2.resize(base, (new_size, new_size), interpolation=interp)
        if new_size > size:
            # zoom-in: center crop
            start = (new_size - size) // 2
            end = start + size
            return scaled[start:end, start:end]
        if new_size < size:
            # zoom-out: center pad with black
            pad_total = size - new_size
            pad_before = pad_total // 2
            pad_after = pad_total - pad_before
            return cv2.copyMakeBorder(
                scaled, pad_before, pad_after, pad_before, pad_after, cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )
        return scaled

    def _shear_img(self, img: np.ndarray, shear_deg: float) -> np.ndarray:
        img_sq = self._pad_to_square(img)
        shear_rad = math.radians(shear_deg)
        M = np.float32([[1, math.tan(shear_rad), 0], [0, 1, 0]])
        h, w = img_sq.shape[:2]
        return cv2.warpAffine(img_sq, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    def _adjust_hsv_np(self, img: np.ndarray, h: float, s: float, v: float) -> np.ndarray:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype("float32")
        hsv[..., 0] = (hsv[..., 0] + h * 179) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] + s * 255, 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] + v * 255, 0, 255)
        hsv = np.clip(hsv, 0, 255).astype("uint8")
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    def _pad_to_square(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if h == w:
            return img
        size = max(h, w)
        top = (size - h) // 2
        bottom = size - h - top
        left = (size - w) // 2
        right = size - w - left
        return cv2.copyMakeBorder(img, top, bottom, left, right, borderType=cv2.BORDER_REFLECT)


class TrainTab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        left = QtWidgets.QWidget()
        left.setFixedWidth(360)
        lL = QtWidgets.QVBoxLayout(left)
        lL.addWidget(Header("Training Runs", "Pick data.yaml and model preset"))

        run_box = TitledGroup("Load Dataset")
        rl = QtWidgets.QFormLayout(run_box)
        self.ed_data_yaml = QtWidgets.QLineEdit()
        self.btn_browse_data = QtWidgets.QPushButton("Browse")
        row_data = QtWidgets.QHBoxLayout(); row_data.setContentsMargins(0,0,0,0)
        row_data.addWidget(self.ed_data_yaml)
        row_data.addWidget(self.btn_browse_data)
        data_wrap = QtWidgets.QWidget(); dlay = QtWidgets.QHBoxLayout(data_wrap); dlay.setContentsMargins(0,0,0,0); dlay.addLayout(row_data)
        self.cmb_preset = QtWidgets.QComboBox()
        rl.addRow("data.yaml", data_wrap)
        rl.addRow("Preset", wrap_expanding(self.cmb_preset))
        lL.addWidget(run_box)

        stats_box = TitledGroup("Dataset Preview")
        sl = QtWidgets.QVBoxLayout(stats_box)
        sl.setContentsMargins(12, 8, 12, 12)
        sl.setSpacing(6)

        lbl_classes_title = QtWidgets.QLabel("<b>Classes</b>")
        sl.addWidget(lbl_classes_title)
        self.lbl_classes = QtWidgets.QLabel("-")
        self.lbl_classes.setWordWrap(True)
        sl.addWidget(self.lbl_classes)

        sl.addWidget(QtWidgets.QLabel("<b>Images</b>"))
        self.layout_img_bars = QtWidgets.QVBoxLayout()
        self.layout_img_bars.setSpacing(4)
        self.lbl_total_images = QtWidgets.QLabel("Total Images: -")
        self.lbl_total_images.setAlignment(Qt.AlignCenter)
        self.lbl_total_images.setStyleSheet(
            "color:#1f3c88;font-size:13px;font-weight:600;"
            "padding:6px;border:1px solid #c1c7d0;border-radius:4px;background:#eef2ff;"
        )
        sl.addLayout(self.layout_img_bars)
        sl.addWidget(self.lbl_total_images)

        sl.addWidget(QtWidgets.QLabel("<b>Labels / Class</b>"))
        self.layout_label_bars = QtWidgets.QVBoxLayout()
        self.layout_label_bars.setSpacing(4)
        sl.addLayout(self.layout_label_bars)
        lL.addWidget(stats_box)
        lL.addStretch(1)

        center = QtWidgets.QWidget()
        cL = QtWidgets.QVBoxLayout(center)
        cL.addWidget(Header("Progress"))
        self.prg = QtWidgets.QProgressBar()
        self.prg.setRange(0, 100)
        self.prg.setValue(0)
        cL.addWidget(self.prg)
        # Table removed per request; keep attribute as None for code guards
        self.tbl = None
        self.txt = QtWidgets.QTextEdit()
        self.txt.setReadOnly(True)
        # Improve readability for console-like logs
        self.txt.setLineWrapMode(QtWidgets.QTextEdit.NoWrap)
        self.txt.setPlaceholderText("Training logs will appear here...")
        self._log_hint_active = False
        self._show_log_hint()
        try:
            f = self.txt.font(); f.setFamily("JetBrains Mono, Consolas, Menlo, monospace"); self.txt.setFont(f)
        except Exception:
            pass
        cL.addWidget(self.txt, 1)

        right = QtWidgets.QWidget()
        right.setFixedWidth(360)
        rL = QtWidgets.QVBoxLayout(right)

        params = TitledGroup("Training Parameters")
        pf_wrap = QtWidgets.QVBoxLayout(params)
        # add horizontal padding so fields don't hug the border
        pf_wrap.setContentsMargins(12, 8, 12, 12)
        pf_wrap.setSpacing(8)

        form_top = QtWidgets.QFormLayout()
        form_top.setContentsMargins(0, 0, 0, 0)
        form_top.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)

        self.cmb_task = configure_combo(QtWidgets.QComboBox(), show_all=True)
        self.cmb_task.addItems(["Object Detection", "Instance Segmentation"])
        task_view = self.cmb_task.view()
        if isinstance(task_view, QtWidgets.QListView):
            row_height = task_view.sizeHintForRow(0)
            if row_height > 0:
                task_view.setMinimumHeight(row_height * self.cmb_task.count() + task_view.frameWidth() * 2)
        form_top.addRow("task", wrap_expanding(self.cmb_task))
        pf_wrap.addLayout(form_top)

        self.stack = QtWidgets.QStackedWidget()
        self.stack.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)

        page_det = QtWidgets.QWidget()
        page_det.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        detL = QtWidgets.QFormLayout(page_det)
        detL.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        detL.setContentsMargins(0, 0, 0, 0)
        # Dialog-equivalent controls
        self.ed_model = QtWidgets.QLineEdit("yolov8n.pt")
        self.btn_browse_model = QtWidgets.QPushButton("Browse")
        mrow = QtWidgets.QHBoxLayout(); mrow.setContentsMargins(0,0,0,0)
        mrow.addWidget(self.ed_model)
        mrow.addWidget(self.btn_browse_model)
        mwrap = QtWidgets.QWidget(); mlay = QtWidgets.QHBoxLayout(mwrap); mlay.setContentsMargins(0,0,0,0); mlay.addLayout(mrow)

        self.det_imgsz = wrap_expanding(QtWidgets.QSpinBox()); self.det_imgsz.widget().setRange(64,2048) if hasattr(self.det_imgsz,'widget') else None
        self.sp_imgsz = QtWidgets.QSpinBox(); self.sp_imgsz.setRange(64,2048); self.sp_imgsz.setValue(640); self.sp_imgsz.setSingleStep(32)
        self.sp_epochs = QtWidgets.QSpinBox(); self.sp_epochs.setRange(1,1000); self.sp_epochs.setValue(100)
        self.sp_batch = QtWidgets.QSpinBox(); self.sp_batch.setRange(1,1024); self.sp_batch.setValue(8)
        self.sp_workers = QtWidgets.QSpinBox(); self.sp_workers.setRange(0,32); self.sp_workers.setValue(8)
        self.ed_device = QtWidgets.QLineEdit("0")
        self.ed_name = QtWidgets.QLineEdit("train_run")
        self.chk_cache = QtWidgets.QCheckBox("cache")
        self.chk_augment = QtWidgets.QCheckBox("augment"); self.chk_augment.setChecked(True)
        self.chk_resume = QtWidgets.QCheckBox("resume")

        detL.addRow("model", mwrap)
        detL.addRow("imgsz", self.sp_imgsz)
        detL.addRow("epochs", self.sp_epochs)
        detL.addRow("batch", self.sp_batch)
        detL.addRow("workers", self.sp_workers)
        detL.addRow("device", self.ed_device)
        detL.addRow("name", self.ed_name)
        detL.addRow(self.chk_cache)
        detL.addRow(self.chk_augment)
        detL.addRow(self.chk_resume)

        page_seg = QtWidgets.QWidget()
        page_seg.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        segL = QtWidgets.QFormLayout(page_seg)
        segL.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        segL.setContentsMargins(0, 0, 0, 0)
        self.seg_model_combo = configure_combo(QtWidgets.QComboBox())
        for size in ("n", "s", "m", "l", "x"):
            self.seg_model_combo.addItem(size)
        self.seg_model_combo.setCurrentText("m")
        self.seg_model = wrap_expanding(self.seg_model_combo)
        self.seg_imgsz_spin = QtWidgets.QSpinBox(); self.seg_imgsz_spin.setRange(64, 4096); self.seg_imgsz_spin.setValue(1024); self.seg_imgsz_spin.setSingleStep(32)
        self.seg_imgsz = wrap_expanding(self.seg_imgsz_spin)
        self.seg_epochs_spin = QtWidgets.QSpinBox(); self.seg_epochs_spin.setRange(1, 2000); self.seg_epochs_spin.setValue(100)
        self.seg_epochs = wrap_expanding(self.seg_epochs_spin)
        self.seg_batch_spin = QtWidgets.QSpinBox(); self.seg_batch_spin.setRange(1, 1024); self.seg_batch_spin.setValue(4)
        self.seg_batch = wrap_expanding(self.seg_batch_spin)
        self.seg_device_edit = QtWidgets.QLineEdit("0")
        self.seg_device = wrap_expanding(self.seg_device_edit)
        self.seg_name_edit = QtWidgets.QLineEdit("segment_run")
        self.seg_name = wrap_expanding(self.seg_name_edit)
        segL.addRow("model size", self.seg_model)
        segL.addRow("imgsz", self.seg_imgsz)
        segL.addRow("epochs", self.seg_epochs)
        segL.addRow("batch", self.seg_batch)
        segL.addRow("device", self.seg_device)
        segL.addRow("name", self.seg_name)

        self.stack.addWidget(page_det)
        self.stack.addWidget(page_seg)
        pf_wrap.addWidget(self.stack)

        self.cmb_task.currentIndexChanged.connect(self.stack.setCurrentIndex)

        # Wire browse buttons
        self.btn_browse_data.clicked.connect(self._pick_yaml)
        self.btn_browse_model.clicked.connect(self._pick_model)
        self.ed_data_yaml.editingFinished.connect(self._update_dataset_preview)

        rL.addWidget(params)

        # Model Training section (Train/Stop buttons)
        train_box = TitledGroup("Model Training")
        tr = QtWidgets.QHBoxLayout(train_box)
        tr.setContentsMargins(12, 0, 12, 12)  # left, top, right, bottom (위쪽을 절반쯤)
        self.btn_start = QtWidgets.QPushButton("Train Start")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        tr.addWidget(self.btn_start)
        tr.addStretch(1)
        tr.addWidget(self.btn_stop)
        rL.addWidget(train_box)

        # Training Information 
        info_box = TitledGroup("Training Information")
        infoL = QtWidgets.QFormLayout(info_box)
        infoL.setContentsMargins(12, 0, 12, 12)
        infoL.setSpacing(6)
        self.lbl_info_epoch = QtWidgets.QLabel("-")
        self.lbl_info_loss = QtWidgets.QLabel("-")
        self.lbl_info_seg_loss = QtWidgets.QLabel("-")
        self.lbl_info_map = QtWidgets.QLabel("-")
        self.lbl_info_best_epoch = QtWidgets.QLabel("-")
        self.lbl_info_best_map = QtWidgets.QLabel("-")
        for v in (self.lbl_info_epoch, self.lbl_info_loss, self.lbl_info_seg_loss, self.lbl_info_map, self.lbl_info_best_epoch, self.lbl_info_best_map):
            v.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        infoL.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        infoL.addRow("Current epoch", self.lbl_info_epoch)
        infoL.addRow("Current loss", self.lbl_info_loss)
        infoL.addRow("Current seg loss", self.lbl_info_seg_loss)
        infoL.addRow("Current mAP (fitness)", self.lbl_info_map)
        infoL.addRow("Best epoch", self.lbl_info_best_epoch)
        infoL.addRow("Best mAP (fitness)", self.lbl_info_best_map)

        # show chart
        btn_chart = QtWidgets.QPushButton("Show Chart")
        btn_chart.setFixedWidth(110)
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.addWidget(btn_chart)
        infoL.addRow(btn_row)
        self.btn_show_chart = btn_chart
        rL.addWidget(info_box)

        # Export Model section
        export_box = TitledGroup("Export Model")
        ex = QtWidgets.QHBoxLayout(export_box)
        ex.setContentsMargins(12, 0, 12, 12)  # left, top, right, bottom (위쪽을 절반쯤)
        self.btn_export_model = QtWidgets.QPushButton("Export Model")
        ex.addWidget(self.btn_export_model)
        ex.addStretch(1)
        rL.addWidget(export_box)
        rL.addStretch(1)

        root.addWidget(left, 0)
        root.addWidget(center, 1)
        root.addWidget(right, 0)

        # runtime
        self.worker: TrainWorker | None = None
        self._best_fitness: float | None = None
        self._best_epoch: str | None = None
        self._chart_epochs: list[int] = []
        self._chart_loss: list[float] = []
        self._chart_fitness: list[float] = []
        self._chart_max_epoch: int | None = None
        self._current_task: str = "detect"
        self._last_run_name: str | None = None
        self._chart_dialog: _MetricsChart | None = None

        # wire buttons for local training
        self.btn_start.clicked.connect(self.start_training)
        self.btn_stop.clicked.connect(self.stop_training)
        self.btn_export_model.clicked.connect(self._on_export_model)
        self.btn_show_chart.clicked.connect(self._on_show_chart)
        # initial preview state
        self._update_dataset_preview()

        # regex for parsing ultralytics logs
        self._re_epoch = re.compile(r"^(\d+)/(\d+)\s+([0-9.]+)G\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)")
        self._re_metrics = re.compile(r"^\s*all\s+\d+\s+\d+\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)")
        # Fallback for our custom log: "Epoch X/T finished."
        self._re_epoch_simple = re.compile(r"^Epoch\s+(\d+)/(\d+)\s+finished\.?")
        # CLI-style: "epoch: 1/100, gpu_mem : ..."
        self._re_epoch_cli = re.compile(r"epoch[:\s]+(\d+)\s*/\s*(\d+)", re.IGNORECASE)
        self._epoch_row = {}
        self._run_dir: Path | None = None
        self._train_started_ts: float | None = None
        self._emitted_epochs: set[str] = set()
        self._retry_counts: dict[str, int] = {}

    def _show_log_hint(self):
        """Display a faint hint so the empty log area has a visible purpose."""
        self._log_hint_active = True
        self.txt.setHtml(
            '<div style="color:#9aa0a6; text-align:center; padding:12px 0;">'
            "Training logs will appear here..."
            "</div>"
        )

    def _clear_log_hint(self):
        if self._log_hint_active:
            self.txt.clear()
            self._log_hint_active = False

    def _append_log(self, text: str):
        """Append a log line and keep the cursor pinned to the bottom."""
        self._clear_log_hint()
        self.txt.append(text)
        try:
            cursor = self.txt.textCursor()
            cursor.movePosition(QtGui.QTextCursor.End)
            self.txt.setTextCursor(cursor)
            self.txt.ensureCursorVisible()
        except Exception:
            pass

    # ------ training control ------
    def start_training(self):
        if self.worker is not None:
            return
        params = self.gather_params()
        self._current_task = params.get("task") or "detect"
        self._last_run_name = params.get("name") or self.ed_name.text().strip()
        # basic validation; let backend handle the rest
        if not params.get("data"):
            self._append_log("Select data.yaml first.")
            return
        self._reset_training_info()
        aug_overrides = None
        if params.get("augment"):
            dlg = SimpleAugListDialog(self)
            if dlg.exec() != QtWidgets.QDialog.Accepted:
                self._append_log("Training cancelled: augmentation settings dialog closed.")
                return
            aug_overrides = dlg.values()
            if aug_overrides:
                params["augment_overrides"] = aug_overrides
        self.txt.clear()
        self._show_log_hint()
        self.prg.setValue(0)
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        # no export button in Train tab
        # Track possible run directory roots to discover active run
        try:
            self._train_started_ts = time.time()
        except Exception:
            self._train_started_ts = None
        self._run_dir = None
        self.worker = TrainWorker(params)
        self.worker.sig_log.connect(self._on_log)
        self.worker.sig_prog.connect(self._on_prog)
        self.worker.sig_done.connect(self._on_done)
        # Track training PID(s) to compute per-process GPU memory
        self._train_pids: set[int] = set()
        try:
            import os as _os
            self._train_pids.add(int(_os.getpid()))  # Python API uses same process
        except Exception:
            pass
        if hasattr(self.worker, 'sig_pid'):
            try:
                self.worker.sig_pid.connect(self._on_train_pid)
            except Exception:
                pass
        self.worker.start()

    def stop_training(self):
        if self.worker is not None:
            self.worker.stop()
            self.btn_stop.setEnabled(False)

    def _on_log(self, s: str):
        # Filter: suppress raw epoch-finished and compact epoch lines; we'll emit our own
        try:
            s_clean = s.strip()
            suppress_raw = bool(self._re_epoch_simple.match(s_clean) or self._re_epoch.match(s_clean))
            if not suppress_raw:
                self._append_log(s)
        except Exception:
            pass
        # Parse and reflect summary metrics
        s_clean = s.strip()
        m = self._re_epoch.match(s_clean)
        if m:
            ep, total, gpu, box, cls, dfl = m.groups()
            # emit once using CSV (and fall back to parsed losses)
            self._emit_epoch_line(ep, total, prefer_csv=True, fallback_losses=(box, cls, dfl), retries=0, task=self._current_task)
            return
        if (self._current_task or "").lower().startswith("segment"):
            m_cli = self._re_epoch_cli.search(s_clean)
            if m_cli:
                ep, total = m_cli.groups()
                self._emit_epoch_line(ep, total, prefer_csv=True, fallback_losses=None, retries=4, task=self._current_task)
        m_simple = self._re_epoch_simple.match(s_clean)
        if m_simple:
            ep, total = m_simple.groups()
            # Delay a bit to let results.csv flush, then emit once
            self._emit_epoch_line(ep, total, prefer_csv=True, fallback_losses=None, retries=6, task=self._current_task)
            return
        m2 = self._re_metrics.match(s_clean)
        if m2 and self._epoch_row:
            # update latest row with P/R/mAPs (table hidden; keep logic guarded)
            P, R, m50, m5095 = m2.groups()
            # last inserted row index
            row = max(self._epoch_row.values()) if self._epoch_row else 0
            if self.tbl:
                for col, val in zip((5,6,7,8), (P,R,m50,m5095)):
                    self.tbl.setItem(row, col, QtWidgets.QTableWidgetItem(val))

    def _emit_epoch_line(self, ep: str, total: str, prefer_csv: bool, fallback_losses: tuple[str,str,str] | None, retries: int, task: str | None = None):
        if ep in self._emitted_epochs:
            return
        # Try gathering from CSV
        task_name = task or self._current_task or "detect"
        metrics = self._update_epoch_from_csv(ep, total, task_name, exact_only=True)
        # If CSV lacks losses and we have fallback from parsed log, fill in
        if fallback_losses is not None:
            b, c, d = fallback_losses
            if metrics.get("box", "") == "":
                metrics["box"] = b
            if metrics.get("cls", "") == "":
                metrics["cls"] = c
            if metrics.get("dfl", "") == "":
                metrics["dfl"] = d
        # Determine if we have enough data to print; require at least box or dfl
        have_core = bool(metrics.get("box") or metrics.get("dfl"))
        if not have_core and retries > 0:
            # Retry after 500ms to wait for CSV flush
            self._retry_counts[ep] = retries
            QTimer.singleShot(500, lambda: self._emit_epoch_line(ep, total, prefer_csv, fallback_losses, retries-1, task_name))
            return
        gpu_gb = None
        try:
            g = self._query_gpu_mem_gb()
            gpu_gb = f"{g:.2f}" if g is not None else None
        except Exception:
            gpu_gb = None
        self._update_training_info(
            ep,
            total,
            box=metrics.get("box"),
            cls=metrics.get("cls"),
            dfl=metrics.get("dfl"),
            seg=metrics.get("seg"),
            m50b=metrics.get("m50b"),
            m5095b=metrics.get("m5095b"),
            m50m=metrics.get("m50m"),
            m5095m=metrics.get("m5095m"),
        )
        line = self._format_epoch_line(ep, total, metrics, gpu_gb)
        if line:
            self._append_log(line)
            self._emitted_epochs.add(ep)

    def _reset_training_info(self):
        self._best_fitness = None
        self._best_epoch = None
        for lbl in (self.lbl_info_epoch, self.lbl_info_loss, self.lbl_info_seg_loss, self.lbl_info_map, self.lbl_info_best_epoch, self.lbl_info_best_map):
            lbl.setText("-")
        self._chart_epochs.clear()
        self._chart_loss.clear()
        self._chart_fitness.clear()
        self._chart_max_epoch = None

    def _update_training_info(
        self,
        ep: str,
        total: str,
        *,
        box: str | float | None = None,
        cls: str | float | None = None,
        dfl: str | float | None = None,
        seg: str | float | None = None,
        m50b: str | float | None = None,
        m5095b: str | float | None = None,
        m50m: str | float | None = None,
        m5095m: str | float | None = None,
    ):
        self.lbl_info_epoch.setText(f"{ep}/{total}")
        def _safe_float(v):
            try:
                return float(v)
            except Exception:
                return 0.0
        cur_m50b = _safe_float(m50b)
        cur_m5095b = _safe_float(m5095b)
        cur_m50m = _safe_float(m50m)
        cur_m5095m = _safe_float(m5095m)
        fit_box = 0.1 * cur_m50b + 0.9 * cur_m5095b if (m50b not in (None, "") or m5095b not in (None, "")) else None
        fit_mask = 0.1 * cur_m50m + 0.9 * cur_m5095m if (m50m not in (None, "") or m5095m not in (None, "")) else None
        cur_fit = None
        if fit_box is not None or fit_mask is not None:
            cur_fit = (fit_box or 0.0) + (fit_mask or 0.0)
            self.lbl_info_map.setText(f"{cur_fit:.4f}")
        else:
            self.lbl_info_map.setText("-")
        try:
            max_ep = int(total)
            self._chart_max_epoch = max_ep
        except Exception:
            max_ep = None
        if cur_fit is not None:
            if self._best_fitness is None or cur_fit > self._best_fitness:
                self._best_fitness = cur_fit
                self._best_epoch = ep
        if self._best_epoch is not None and self._best_fitness is not None:
            self.lbl_info_best_epoch.setText(str(self._best_epoch))
            self.lbl_info_best_map.setText(f"{self._best_fitness:.4f}")
        else:
            self.lbl_info_best_epoch.setText("-")
            self.lbl_info_best_map.setText("-")
        # current total loss (box+cls+dfl)
        tot_loss = _safe_float(box) + _safe_float(cls) + _safe_float(dfl) + _safe_float(seg)
        # if all zero and no inputs, show "-"
        if (box in (None, "") and cls in (None, "") and dfl in (None, "") and seg in (None, "")):
            self.lbl_info_loss.setText("-")
        else:
            self.lbl_info_loss.setText(f"{tot_loss:.4f}")
        self.lbl_info_seg_loss.setText(f"{_safe_float(seg):.4f}" if seg not in (None, "") else "-")
        try:
            ep_int = int(ep)
        except Exception:
            ep_int = None
        if ep_int is not None:
            self._chart_epochs.append(ep_int)
            self._chart_loss.append(tot_loss)
            self._chart_fitness.append(cur_fit if cur_fit is not None else 0.0)
            if self._chart_dialog is not None:
                curr_task = (self._current_task or "").lower()
                map_ylim = (0.0, 2.0) if curr_task.startswith("segment") else (0.0, 1.0)
                self._chart_dialog.update_data(
                    self._chart_epochs,
                    self._chart_loss,
                    self._chart_fitness,
                    max_epoch=self._chart_max_epoch,
                    map_ylim=map_ylim,
                )

    # ------ CSV + GPU helpers ------
    def _on_train_pid(self, pid: int):
        try:
            self._train_pids.add(int(pid))
        except Exception:
            pass

    def _on_show_chart(self):
        if self._chart_dialog is None:
            self._chart_dialog = _MetricsChart(self)
            self._chart_dialog.finished.connect(lambda _=None: setattr(self, "_chart_dialog", None))
        map_ylim = (0.0, 2.0) if (self._current_task or "").lower().startswith("segment") else (0.0, 1.0)
        self._chart_dialog.update_data(self._chart_epochs, self._chart_loss, self._chart_fitness, max_epoch=self._chart_max_epoch, map_ylim=map_ylim)
        self._chart_dialog.show()
        self._chart_dialog.raise_()

    def _candidate_run_roots(self) -> list[Path]:
        """Return possible base directories that may contain the active run."""
        roots: list[Path] = []
        # project path relative to CWD
        roots.append(Path("runs/detect").resolve())
        roots.append(Path("runs/segment").resolve())
        # alongside the pyside6 package
        try:
            here = Path(__file__).resolve()
            roots.append(here.parents[3] / "runs/detect")  # repo/pyside6/runs/detect
            roots.append(here.parents[3] / "runs/segment")  # repo/pyside6/runs/segment
        except Exception:
            pass
        # Deduplicate while preserving order
        uniq: list[Path] = []
        seen = set()
        for r in roots:
            rp = r.resolve()
            if rp not in seen:
                uniq.append(rp); seen.add(rp)
        return [r for r in uniq if r.exists()]

    def _discover_run_dir(self) -> Path | None:
        """Guess the current run directory by name prefix and mtime."""
        name_prefix = (self._last_run_name or "").strip() or self.ed_name.text().strip() or "train"
        candidates: list[tuple[float, Path]] = []
        for root in self._candidate_run_roots():
            try:
                for p in root.iterdir():
                    if not p.is_dir():
                        continue
                    if not p.name.startswith(name_prefix):
                        continue
                    # Prefer runs created/modified after training started
                    try:
                        mt = p.stat().st_mtime
                    except Exception:
                        mt = 0.0
                    if self._train_started_ts is None or mt >= self._train_started_ts - 5:
                        candidates.append((mt, p))
            except Exception:
                pass
        if not candidates:
            return None
        candidates.sort(key=lambda t: t[0], reverse=True)
        return candidates[0][1]

    def _results_csv(self) -> Path | None:
        if self._run_dir is None or not self._run_dir.exists():
            rd = self._discover_run_dir()
            if rd is not None:
                self._run_dir = rd
        if self._run_dir is None:
            return None
        csv_path = self._run_dir / "results.csv"
        return csv_path if csv_path.exists() else None

    @staticmethod
    def _fmt4(v: str) -> str:
        try:
            return f"{float(v):.4f}"
        except Exception:
            return v

    def _update_epoch_from_csv(self, ep_str: str, total_str: str, task: str = "detect", *, exact_only: bool = False) -> dict:
        """Read latest metrics for the epoch from results.csv and fill table row.
        Returns a dict with keys: box, cls, dfl, P, R, m50b, m5095b and segment-only keys seg, m50m, m5095m.
        """
        csv_path = self._results_csv()
        if csv_path is None:
            try:
                print(f"[train tab] results.csv not found for task={task}, epoch={ep_str}, run_dir={self._run_dir}")
            except Exception:
                pass
            return {}
        # else:
        #     try:
        #         print(f"[train tab] reading metrics from {csv_path}")
        #     except Exception:
        #         pass
        try:
            # Find the last row for the given epoch (or the last row)
            with csv_path.open("r", encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = list(reader)
            if not rows:
                return {}
            header = rows[0]
            # Build name->index map
            idx = {name: i for i, name in enumerate(header)}
            # prefer matching epoch
            target = None
            for r in reversed(rows[1:]):
                if not r or len(r) <= 1:
                    continue
                try:
                    if r[idx.get("epoch", 0)] == ep_str:
                        target = r; break
                except Exception:
                    pass
            if target is None:
                if exact_only:
                    return {}
                target = rows[-1]
            def get(k: str) -> str:
                j = idx.get(k)
                return target[j] if (j is not None and j < len(target)) else ""
            box = self._fmt4(get("train/box_loss"))
            seg = self._fmt4(get("train/seg_loss")) if task == "segment" else ""
            cls = self._fmt4(get("train/cls_loss"))
            dfl = self._fmt4(get("train/dfl_loss"))
            P = self._fmt4(get("metrics/precision(B)"))
            R = self._fmt4(get("metrics/recall(B)"))
            m50b = self._fmt4(get("metrics/mAP50(B)"))
            m5095b = self._fmt4(get("metrics/mAP50-95(B)"))
            m50m = self._fmt4(get("metrics/mAP50(M)")) if task == "segment" else ""
            m5095m = self._fmt4(get("metrics/mAP50-95(M)")) if task == "segment" else ""
            row = self._epoch_row.get(ep_str)
            if row is None:
                row = len(self._epoch_row)
                self._epoch_row[ep_str] = row
                if self.tbl:
                    self.tbl.setItem(row, 0, QtWidgets.QTableWidgetItem(ep_str+"/"+total_str))
            # Fill metrics if not yet set
            vals = [box, cls, dfl, P, R, m50b, m5095b]
            if self.tbl:
                for col, v in zip(range(2, 9), vals):
                    if v:
                        self.tbl.setItem(row, col, QtWidgets.QTableWidgetItem(v))
            return {"box": box, "seg": seg, "cls": cls, "dfl": dfl, "P": P, "R": R, "m50b": m50b, "m5095b": m5095b, "m50m": m50m, "m5095m": m5095m}
        except Exception:
            # Ignore CSV read errors quietly (file being written)
            return {}

    def _query_gpu_mem_gb(self) -> float | None:
        """Return GPU memory used by the training process in GB (float), if available."""
        try:
            dev_text = self.ed_device.text().strip()
            if not dev_text or dev_text.lower() == "cpu":
                return None
            sel = []
            for t in dev_text.split(','):
                t = t.strip()
                if t.isdigit():
                    sel.append(int(t))
            if not sel:
                return None
            # Prefer querying per-process via nvidia-smi (compute apps)
            used_mb: float | None = None
            pids = list(getattr(self, "_train_pids", []) or [])
            try:
                print(f"[train tab] GPU mem check; tracking PIDs={pids}")
            except Exception:
                pass
            # nvidia-smi mapping from uuid->index
            try:
                map_out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
                uuid_by_index = {}
                for line in map_out.strip().splitlines():
                    if not line.strip():
                        continue
                    idx_str, uuid = [x.strip() for x in line.split(',')[:2]]
                    if idx_str.isdigit():
                        uuid_by_index[int(idx_str)] = uuid
                if uuid_by_index:
                    q = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits"], text=True)
                    total = 0
                    for line in q.strip().splitlines():
                        parts = [x.strip() for x in line.split(',')]
                        if len(parts) < 3:
                            continue
                        try:
                            pid = int(parts[0])
                            uuid = parts[1]
                            mem = int(parts[2])  # MB
                        except Exception:
                            continue
                        if pids and pid not in pids:
                            continue
                        # Map uuid back to index to filter by selected GPUs
                        try:
                            gpu_index = next((i for i,u in uuid_by_index.items() if u == uuid), None)
                        except Exception:
                            gpu_index = None
                        if gpu_index is None or gpu_index not in sel:
                            continue
                        total += mem
                    used_mb = float(total)
            except Exception:
                used_mb = None
            if used_mb is None:
                # Fallback using torch memory allocated by this process
                try:
                    import torch
                    if torch.cuda.is_available():
                        total_bytes = 0
                        for i in sel:
                            try:
                                total_bytes += int(torch.cuda.memory_allocated(i))
                            except Exception:
                                pass
                        if total_bytes > 0:
                            used_mb = total_bytes / (1024 * 1024)
                except Exception:
                    used_mb = None
            if used_mb is None:
                return None
            return used_mb / 1024.0
        except Exception:
            return None

    def _update_gpu_mem(self, row_index: int) -> str | None:
        gb = self._query_gpu_mem_gb()
        if gb is None:
            return None
        s = f"{gb:.2f}"
        if self.tbl:
            try:
                self.tbl.setItem(row_index, 1, QtWidgets.QTableWidgetItem(s))
            except Exception:
                pass
        return s

    def _format_epoch_line(self, ep: str, total: str, metrics: dict, gpu_gb: str | None) -> str:
        task = (self._current_task or "").lower()
        box = metrics.get("box", "") if metrics else ""
        seg = metrics.get("seg", "") if metrics else ""
        cls = metrics.get("cls", "") if metrics else ""
        dfl = metrics.get("dfl", "") if metrics else ""
        P = metrics.get("P", "") if metrics else ""
        R = metrics.get("R", "") if metrics else ""
        m50b = metrics.get("m50b", "") if metrics else ""
        m5095b = metrics.get("m5095b", "") if metrics else ""
        m50m = metrics.get("m50m", "") if metrics else ""
        m5095m = metrics.get("m5095m", "") if metrics else ""
        gtxt = (gpu_gb + "G") if gpu_gb else ""
        seg_part = f" seg_loss:{seg}," if task.startswith("segment") else ""
        if task.startswith("segment"):
            # print(f"[DEBUG] _format_epoch_line task: segment")
            map_part = f"mAP50(B): {m50b}, mAP50-95(B): {m5095b}, mAP50(M): {m50m}, mAP50-95(M): {m5095m}"
        else:
            map_part = f"mAP50: {m50b}, mAP50-95: {m5095b}"
        return (
            f"epoch: {ep}/{total}, gpu_mem : {gtxt}, "
            f"box_loss:{box},{seg_part} cls_loss:{cls}, dfl_loss: {dfl}, "
            f"Precision:{P}, Recall: {R}, {map_part}"
        )

    # ------ Export model ------
    def _locate_trained_weight(self) -> Path | None:
        run_csv = self._results_csv()
        if run_csv is None:
            return None
        rd = run_csv.parent
        cands = [rd/"weights"/"best.pt", rd/"weights"/"last.pt", rd/"best.pt", rd/"last.pt"]
        for p in cands:
            if p.exists():
                return p
        return None

    def _show_export_dialog(self, source_path: Path, allow_custom_source: bool) -> dict | None:
        class ExportDialog(QtWidgets.QDialog):
            def __init__(self, parent, src: Path, allow_src_browse: bool):
                super().__init__(parent)
                self.setWindowTitle("Export Model")
                self.resize(460, 220)
                lay = QtWidgets.QVBoxLayout(self)
                form = QtWidgets.QFormLayout()
                self.ed_source = QtWidgets.QLineEdit(str(src))
                self.ed_source.setReadOnly(True)
                self.btn_src = QtWidgets.QPushButton("Browse")
                if allow_src_browse:
                    src_wrap = QtWidgets.QWidget()
                    src_layout = QtWidgets.QHBoxLayout(src_wrap)
                    src_layout.setContentsMargins(0,0,0,0)
                    src_layout.addWidget(self.ed_source, 1)
                    src_layout.addWidget(self.btn_src, 0)
                    form.addRow("Source", src_wrap)
                else:
                    self.btn_src.setVisible(False)
                    form.addRow("Source", self.ed_source)

                self.cmb_format = QtWidgets.QComboBox()
                self.cmb_format.addItem("Copy as .pt", "pt")
                self.cmb_format.addItem("ONNX", "onnx")
                self.cmb_format.addItem("OpenVINO", "openvino")
                form.addRow("Format", self.cmb_format)
                # Show full dropdown list without tiny popup height
                try:
                    self.cmb_format.setMaxVisibleItems(max(10, self.cmb_format.count()))
                    view = self.cmb_format.view()
                    if isinstance(view, QtWidgets.QListView):
                        row_h = view.sizeHintForRow(0)
                        if row_h > 0:
                            view.setMinimumHeight(row_h * self.cmb_format.count() + view.frameWidth() * 2)
                except Exception:
                    pass

                self.ed_dest = QtWidgets.QLineEdit()
                self.ed_dest.setReadOnly(True)
                self.btn_dest = QtWidgets.QPushButton("Browse")
                # ONNX opset (visible only when ONNX selected)
                self.sp_opset = QtWidgets.QSpinBox()
                self.sp_opset.setRange(7, 18)
                self.sp_opset.setValue(12)
                self.sp_opset.setToolTip("ONNX opset version (default 12)")
                self._opset_row_label = QtWidgets.QLabel("ONNX opset")
                form.addRow(self._opset_row_label, self.sp_opset)

                dest_row = QtWidgets.QHBoxLayout()
                dest_row.setContentsMargins(0,0,0,0)
                dest_row.addWidget(self.ed_dest, 1)
                dest_row.addWidget(self.btn_dest)
                dest_wrap = QtWidgets.QWidget()
                dest_wrap.setLayout(dest_row)
                form.addRow("Output", dest_wrap)
                lay.addLayout(form)

                self.btn_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
                lay.addWidget(self.btn_box)

                self._allow_src_browse = allow_src_browse
                self._result: dict | None = None
                self._user_changed_dest = False
                self.btn_box.accepted.connect(self.accept)
                self.btn_box.rejected.connect(self.reject)
                self.cmb_format.currentIndexChanged.connect(self._on_format_change)
                self.btn_dest.clicked.connect(self._browse_dest)
                if allow_src_browse:
                    self.btn_src.clicked.connect(self._browse_src)
                self._update_placeholder()
                self._update_opset_visibility()

            def _browse_src(self):
                p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select weights", "", "Weights (*.pt);;All (*.*)")
                if p:
                    self.ed_source.setText(p)
                    self._user_changed_dest = False
                    self._update_placeholder()

            def _browse_dest(self):
                fmt = self.cmb_format.currentData()
                if fmt == "pt":
                    suggested = Path(self.ed_source.text()).name
                    p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save weights as", suggested, "Weights (*.pt);;All (*.*)")
                elif fmt == "onnx":
                    suggested = Path(self.ed_source.text()).with_suffix(".onnx")
                    p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save ONNX as", str(suggested), "ONNX (*.onnx);;All (*.*)")
                else:
                    p = QtWidgets.QFileDialog.getExistingDirectory(self, "Select export folder")
                if p:
                    self.ed_dest.setText(p)
                    self._user_changed_dest = True

            def _on_format_change(self):
                self._user_changed_dest = False
                self._update_placeholder()
                self._update_opset_visibility()

            def _update_placeholder(self):
                if not self.ed_source.text().strip():
                    return
                src_path = Path(self.ed_source.text().strip())
                fmt = self.cmb_format.currentData()
                if fmt == "pt":
                    default_path = src_path.with_name(src_path.stem + "_export.pt")
                    self.ed_dest.setPlaceholderText(str(default_path))
                    if not self._user_changed_dest or not self.ed_dest.text().strip():
                        self.ed_dest.setText(str(default_path))
                elif fmt == "onnx":
                    default_path = src_path.with_suffix(".onnx")
                    self.ed_dest.setPlaceholderText(str(default_path))
                    if not self._user_changed_dest or not self.ed_dest.text().strip():
                        self.ed_dest.setText(str(default_path))
                else:
                    default_dir = src_path.parent / f"{src_path.stem}_openvino"
                    self.ed_dest.setPlaceholderText(str(default_dir))
                    if not self._user_changed_dest or not self.ed_dest.text().strip():
                        self.ed_dest.setText(str(default_dir))

            def _update_opset_visibility(self):
                is_onnx = (self.cmb_format.currentData() == "onnx")
                self.sp_opset.setVisible(is_onnx)
                self._opset_row_label.setVisible(is_onnx)

            def _gather(self) -> dict | None:
                src = Path(self.ed_source.text().strip())
                dest = self.ed_dest.text().strip()
                if not src.exists():
                    QtWidgets.QMessageBox.warning(self, "Export Model", "Source weights not found.")
                    return None
                if not dest:
                    QtWidgets.QMessageBox.warning(self, "Export Model", "Select an output location.")
                    return None
                data = {"format": self.cmb_format.currentData(), "source": src, "destination": dest}
                if data["format"] == "onnx":
                    data["opset"] = int(self.sp_opset.value())
                return data

            def accept(self):
                data = self._gather()
                if data is None:
                    return
                self._result = data
                super().accept()

            def result_data(self) -> dict | None:
                return self._result

        dlg = ExportDialog(self, source_path, allow_custom_source)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return None
        return dlg.result_data()

    def _on_export_model(self):
        try:
            src_path = self._locate_trained_weight()
            if src_path is None:
                QtWidgets.QMessageBox.information(self, "Export Model", "No trained weights detected. Select a .pt file to export.")
                initial, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select trained weights", "", "Weights (*.pt);;All (*.*)")
                if not initial:
                    return
                src_path = Path(initial)

            # Always allow browsing the source model in the dialog
            opts = self._show_export_dialog(src_path, True)
            if not opts:
                return
            fmt = opts["format"]
            src = opts["source"]
            dest = opts["destination"]
            opset = int(opts.get("opset") or 12)

            if fmt == "pt":
                dest_path = Path(dest)
                if dest_path.suffix.lower() != ".pt":
                    dest_path = dest_path.with_suffix(".pt")
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copyfile(str(src), str(dest_path))
                    self._append_log(f"Exported .pt → {dest_path}")
                except Exception as e:
                    QtWidgets.QMessageBox.warning(self, "Export Model", f"Failed to save: {e}")
                return

            try:
                from ultralytics import YOLO
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Export Model", f"ultralytics not available: {e}")
                return

            imgsz = int(self.sp_imgsz.value()) if hasattr(self, 'sp_imgsz') else 640
            fmt_text = "ONNX" if fmt == "onnx" else "OpenVINO"
            self._append_log(f"Exporting to {fmt_text}...")
            try:
                model = YOLO(str(src))
                if fmt == "onnx":
                    dest_path = Path(dest)
                    if dest_path.is_dir() or dest_path.suffix == "":
                        dest_path = dest_path / (src.stem + ".onnx")
                    elif dest_path.suffix.lower() != ".onnx":
                        dest_path = dest_path.with_suffix(".onnx")
                    dest_path = dest_path.resolve()
                    project_dir = dest_path.parent
                    project_dir.mkdir(parents=True, exist_ok=True)
                    export_name = dest_path.stem
                    result = model.export(format="onnx", project=str(project_dir), name=export_name, imgsz=imgsz, opset=opset)
                    produced: Path | None = None
                    if isinstance(result, (list, tuple)) and result:
                        produced = Path(result[0])
                    elif isinstance(result, (str, Path)):
                        produced = Path(result)
                    else:
                        candidates = list((project_dir / export_name).rglob("*.onnx"))
                        if candidates:
                            produced = Path(candidates[0])
                    if produced is None or not produced.exists():
                        QtWidgets.QMessageBox.warning(self, "Export Model", "ONNX file not produced.")
                        return
                    produced = produced.resolve()
                    if produced != dest_path:
                        dest_path.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            shutil.move(str(produced), str(dest_path))
                        except Exception:
                            shutil.copyfile(str(produced), str(dest_path))
                    # Clean up temporary export folder if it exists and differs
                    try:
                        export_dir = project_dir / export_name
                        if export_dir.exists() and export_dir.resolve() != dest_path.parent.resolve():
                            shutil.rmtree(export_dir)
                    except Exception:
                        pass
                    self._append_log(f"Exported ONNX → {dest_path}")
                    return

                # OpenVINO export
                dest_root = Path(dest)
                if dest_root.suffix:
                    dest_root = dest_root.with_suffix('')
                dest_root = dest_root.resolve()
                dest_root.parent.mkdir(parents=True, exist_ok=True)
                export_name = dest_root.name
                result = model.export(format="openvino", project=str(dest_root.parent), name=export_name, imgsz=imgsz)
                produced_dir: Path | None = None
                if isinstance(result, (list, tuple)) and result:
                    produced_dir = Path(result[0])
                elif isinstance(result, (str, Path)):
                    produced_dir = Path(result)
                else:
                    produced_dir = dest_root.parent / export_name
                if produced_dir.is_file():
                    produced_dir = produced_dir.parent
                produced_dir = produced_dir.resolve()
                if produced_dir != dest_root:
                    try:
                        if dest_root.exists():
                            shutil.rmtree(dest_root)
                    except Exception:
                        pass
                    try:
                        shutil.copytree(str(produced_dir), str(dest_root), dirs_exist_ok=True)
                    except TypeError:
                        # dirs_exist_ok not available
                        shutil.copytree(str(produced_dir), str(dest_root))
                xmls = list(dest_root.rglob("*.xml"))
                if xmls:
                    self._append_log("Exported OpenVINO → " + ", ".join(str(x) for x in xmls))
                else:
                    self._append_log(f"Exported OpenVINO → {dest_root}")
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Export Model", f"Export failed: {e}")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Export Model", f"Unexpected error: {e}")

    def _on_prog(self, cur: int, total: int):
        pct = int(cur * 100 / max(1, total))
        self.prg.setValue(pct)

    def _on_done(self, ok: bool, msg: str):
        # Append completion and where the model is saved
        self._on_log(msg)
        if ok:
            run_csv = self._results_csv()
            if run_csv is not None:
                rd = run_csv.parent
                candidates = [rd/"weights"/"best.pt", rd/"weights"/"last.pt", rd/"best.pt", rd/"last.pt"]
                found = [(str(p.resolve()), p.exists()) for p in candidates]
                existing = [p for p, ex in found if ex]
                if existing:
                    self._append_log("학습 완료: 모델 저장 위치")
                    for p in existing:
                        self._append_log(p)
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        # no export button in Train tab
        self.worker = None

    # ------ helpers ------
    def _pick_yaml(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select data.yaml", "", "YAML (*.yaml *.yml)")
        if p:
            self.ed_data_yaml.setText(p)
            self._update_dataset_preview()

    def _pick_model(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select YOLO weights", "", "Weights (*.pt *.onnx *.engine *.torchscript *.tflite *.xml *.bin);;All (*.*)")
        if p:
            self.ed_model.setText(p)

    def gather_params(self) -> dict:
        task = self.cmb_task.currentText().strip().lower()
        if task == "instance segmentation":
            size = (self.seg_model_combo.currentText() or "m").strip()
            model_yaml = f"yolov8{size}-seg.yaml"
            return dict(
                task="segment",
                model_size=size,
                model=model_yaml,
                data=self.ed_data_yaml.text().strip(),
                imgsz=int(self.seg_imgsz_spin.value()),
                epochs=int(self.seg_epochs_spin.value()),
                batch=int(self.seg_batch_spin.value()),
                workers=int(self.sp_workers.value()),
                device=self.seg_device_edit.text().strip() or "0",
                name=self.seg_name_edit.text().strip(),
                cache=self.chk_cache.isChecked(),
                augment=self.chk_augment.isChecked(),
                resume=self.chk_resume.isChecked(),
                project="runs/segment",
            )
        else:
            return dict(
                task="detect",
                data=self.ed_data_yaml.text().strip(),
                model=self.ed_model.text().strip(),
                imgsz=int(self.sp_imgsz.value()),
                epochs=int(self.sp_epochs.value()),
                batch=int(self.sp_batch.value()),
                workers=int(self.sp_workers.value()),
                device=self.ed_device.text().strip(),
                name=self.ed_name.text().strip(),
                cache=self.chk_cache.isChecked(),
                augment=self.chk_augment.isChecked(),
                resume=self.chk_resume.isChecked(),
                project="runs/detect",
            )

    def _update_dataset_preview(self):
        path = self.ed_data_yaml.text().strip()
        if not path:
            self._set_dataset_preview("-", [], 0, Counter(), {})
            return
        try:
            yaml_path = Path(path)
            if not yaml_path.is_absolute():
                yaml_path = yaml_path.resolve()
            data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        except Exception as e:
            self._set_dataset_preview(f"Failed to load: {e}", [], 0, Counter(), {})
            return

        base_dir = yaml_path.parent

        def _resolve_paths(val):
            if val is None:
                return []
            if isinstance(val, (list, tuple)):
                items = list(val)
            else:
                items = [val]
            resolved = []
            for item in items:
                try:
                    raw = str(item).strip()
                    if not raw:
                        continue
                    raw_path = Path(raw)
                    if raw_path.is_absolute():
                        candidate = raw_path.resolve()
                    else:
                        candidate = (base_dir / raw_path).resolve()

                    if not candidate.exists():
                        # try removing leading ./ or ../ segments relative to data root
                        cleaned = raw.replace("\\", "/")
                        cleaned = cleaned.lstrip("./")
                        while cleaned.startswith("../"):
                            cleaned = cleaned[3:]
                        if cleaned:
                            alt = (base_dir / cleaned).resolve()
                            if alt.exists():
                                candidate = alt
                            else:
                                # also try relative to parent of data root
                                parent_alt = (base_dir.parent / cleaned).resolve()
                                if parent_alt.exists():
                                    candidate = parent_alt

                    resolved.append(candidate)
                except Exception:
                    continue
            return resolved

        IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        LABEL_EXTS = {".txt"}

        def _count_files(paths, exts):
            total = 0
            for p in paths:
                if not isinstance(p, Path):
                    continue
                if p.is_dir():
                    try:
                        total += sum(1 for fp in p.rglob("*") if fp.suffix.lower() in exts)
                    except Exception:
                        pass
                elif p.is_file() and p.suffix.lower() in exts:
                    total += 1
            return total

        def _labels_for(images_path: Path) -> Path | None:
            ref = images_path
            if images_path.is_file():
                ref = images_path.parent
            parts = list(ref.parts)
            if "images" in parts:
                idx = parts.index("images")
                return Path(*parts[:idx], "labels", *parts[idx+1:])
            return ref.parent / "labels"

        split_totals: list[tuple[str, int]] = []
        class_counter: Counter[int] = Counter()
        for key in ("train", "val", "test"):
            paths = _resolve_paths(data.get(key))
            if not paths:
                continue
            img_count = _count_files(paths, IMG_EXTS)
            split_totals.append((key, img_count))
            for path_obj in paths:
                label_dir = _labels_for(path_obj)
                if label_dir and label_dir.exists():
                    for txt in label_dir.rglob("*.txt"):
                        try:
                            with txt.open("r", encoding="utf-8") as fh:
                                for line in fh:
                                    line = line.strip()
                                    if not line:
                                        continue
                                    try:
                                        cid = int(float(line.split()[0]))
                                        class_counter[cid] += 1
                                    except Exception:
                                        continue
                        except Exception:
                            continue

        total_images_overall = sum(count for _, count in split_totals)
        # classes info
        classes_text = "-"
        class_names: dict[int, str] = {}
        names = data.get("names")
        if isinstance(names, dict):
            try:
                ordered = [names[k] for k in sorted(names, key=lambda x: int(x))]
                class_names = {int(k): names[k] for k in names}
            except Exception:
                ordered = list(names.values())
                class_names = {i: n for i, n in enumerate(ordered)}
            classes_text = f"{len(ordered)} → {', '.join(map(str, ordered[:6]))}{'…' if len(ordered) > 6 else ''}"
        elif isinstance(names, list):
            class_names = {i: n for i, n in enumerate(names)}
            classes_text = f"{len(names)} → {', '.join(map(str, names[:6]))}{'…' if len(names) > 6 else ''}"
        elif "nc" in data:
            classes_text = f"{data['nc']} classes"

        self._set_dataset_preview(classes_text, split_totals, total_images_overall, class_counter, class_names)

    def _set_dataset_preview(self, classes_text: str, splits: list[tuple[str, int]], total_images: int,
                              class_counts: Counter[int], class_names: dict[int, str]):
        class_lines: list[str] = []
        if class_counts:
            for cid, _ in sorted(class_counts.items(), key=lambda kv: (-kv[1], kv[0])):
                raw_name = class_names.get(cid, str(cid))
                class_lines.append(str(raw_name))
        elif class_names:
            for _, name in sorted(class_names.items(), key=lambda kv: kv[0]):
                class_lines.append(str(name))
        elif classes_text:
            class_lines.append(classes_text)
        class_display = "\n".join(class_lines) if class_lines else "-"
        self.lbl_classes.setText(class_display)
        self._populate_image_bars(splits, total_images)
        self._populate_class_bars(class_counts, class_names)

    def _clear_layout(self, layout: QtWidgets.QLayout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()
            if widget is not None:
                widget.deleteLater()
            elif child_layout is not None:
                self._clear_layout(child_layout)

    def _add_placeholder(self, layout: QtWidgets.QLayout, text: str):
        lbl = QtWidgets.QLabel(text)
        lbl.setStyleSheet("color: #666666;")
        layout.addWidget(lbl)

    def _populate_image_bars(self, splits: list[tuple[str, int]], total_images: int):
        self._clear_layout(self.layout_img_bars)
        total_text = f"Total Images: {total_images}" if total_images else "Total Images: 0"
        self.lbl_total_images.setText(total_text)
        if not splits:
            self._add_placeholder(self.layout_img_bars, "-")
            return
        max_value = total_images if total_images else max((count for _, count in splits), default=1)
        max_value = max_value or 1
        for key, count in splits:
            bar = QtWidgets.QProgressBar()
            bar.setRange(0, max_value)
            bar.setValue(count)
            percent = (count / total_images * 100) if total_images else 0
            bar.setFormat(f"{key} ({percent:.0f}%): {count}")
            bar.setStyleSheet(
                "QProgressBar {border:1px solid #c1c7d0;border-radius:3px;background:#f1f3f5;height:16px;"
                "text-align:center;color:#112347;font-weight:500;}"
                "QProgressBar::chunk {background:#7ca8ff;border-radius:3px;margin:0px;}"
            )
            bar.setTextVisible(True)
            bar.setMaximumHeight(18)
            bar.setMinimumHeight(16)
            self.layout_img_bars.addWidget(bar)

    def _populate_class_bars(self, class_counts: Counter[int], class_names: dict[int, str]):
        self._clear_layout(self.layout_label_bars)
        if not class_counts:
            self._add_placeholder(self.layout_label_bars, "-")
            return
        items = sorted(class_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        max_count = max((cnt for _, cnt in items), default=1) or 1
        for cid, cnt in items:
            name = class_names.get(cid, str(cid))
            bar = QtWidgets.QProgressBar()
            bar.setRange(0, max_count)
            bar.setValue(cnt)
            bar.setFormat(f"{name}: {cnt}")
            bar.setStyleSheet(
                "QProgressBar {border:1px solid #c1c7d0;border-radius:3px;background:#f1f3f5;height:16px;"
                "text-align:center;color:#0f2a1b;font-weight:500;}"
                "QProgressBar::chunk {background:#74d28d;border-radius:3px;margin:0px;}"
            )
            bar.setTextVisible(True)
            bar.setMaximumHeight(18)
            bar.setMinimumHeight(16)
            self.layout_label_bars.addWidget(bar)


class _MetricsChart(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Training Charts")
        lay = QtWidgets.QVBoxLayout(self)
        fig = Figure(figsize=(12, 5))
        fig.patch.set_facecolor("#111318")
        self.ax_loss = fig.add_subplot(1, 2, 1)
        self.ax_map = fig.add_subplot(1, 2, 2)
        self.canvas = FigureCanvas(fig)
        lay.addWidget(self.canvas)
        self._last_epochs: list[int] = []
        self._last_loss: list[float] = []
        self._last_map: list[float] = []
        self._max_epoch: int | None = None
        # simple theme palette
        self._bg = "#111318"
        self._panel = "#171a1f"
        self._grid = "#2d323a"
        self._fg = "#d5d8de"
        self._accent_loss = "#ff6b4a"
        self._accent_fit = "#2dd6a7"

    def update_data(self, epochs: list[int], losses: list[float], maps: list[float], *, max_epoch: int | None = None, map_ylim: tuple[float, float] | None = None):
        if max_epoch is not None:
            self._max_epoch = max_epoch
        self._last_epochs = list(epochs)
        self._last_loss = list(losses)
        self._last_map = list(maps)
        for ax in (self.ax_loss, self.ax_map):
            ax.clear()
            ax.set_facecolor(self._panel)
            for spine in ax.spines.values():
                spine.set_color(self._grid)
            ax.tick_params(colors=self._fg, labelcolor=self._fg)
            ax.grid(True, linestyle="--", linewidth=0.6, color=self._grid, alpha=0.7)
        max_ep = self._max_epoch or (max(epochs) if epochs else None)
        if epochs:
            self.ax_loss.fill_between(epochs, losses, color=self._accent_loss, alpha=0.12, step="mid")
            self.ax_loss.plot(epochs, losses, marker="o", color=self._accent_loss, linewidth=1.8, markersize=4, label="Loss")
            self.ax_loss.set_xlabel("Epoch", color=self._fg)
            self.ax_loss.set_ylabel("Loss", color=self._fg)
            self.ax_loss.set_title("Loss", color=self._fg)
            self.ax_loss.legend(facecolor=self._panel, edgecolor=self._grid, labelcolor=self._fg)

            self.ax_map.fill_between(epochs, maps, color=self._accent_fit, alpha=0.12, step="mid")
            self.ax_map.plot(epochs, maps, marker="o", color=self._accent_fit, linewidth=1.8, markersize=4, label="mAP (fitness)")
            self.ax_map.set_xlabel("Epoch", color=self._fg)
            self.ax_map.set_ylabel("mAP (fitness)", color=self._fg)
            self.ax_map.set_title("mAP (fitness)", color=self._fg)
            self.ax_map.legend(facecolor=self._panel, edgecolor=self._grid, labelcolor=self._fg)
        if max_ep is not None:
            self.ax_loss.set_xlim(1, max_ep)
            self.ax_map.set_xlim(1, max_ep)
        # clamp mAP axis
        if map_ylim:
            self.ax_map.set_ylim(*map_ylim)
        else:
            self.ax_map.set_ylim(0.0, 1.0)
        # keep y tick colors
        self.ax_loss.tick_params(colors=self._fg)
        self.ax_map.tick_params(colors=self._fg)
        self.ax_loss.yaxis.label.set_color(self._fg)
        self.ax_map.yaxis.label.set_color(self._fg)
        self.canvas.draw_idle()
