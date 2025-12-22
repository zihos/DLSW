"""Shared Augmentation Details dialog for training/export."""

from __future__ import annotations

from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtCore import Qt
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


class AugmentDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, *, embed: bool = False):
        super().__init__(parent)
        self._embed = bool(embed)
        if not self._embed:
            self.setWindowTitle("Augmentation Details")
            self.setModal(True)
        self.resize(1100, 820 if not self._embed else 720)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        self._image_labels: list[tuple[QtWidgets.QLabel, str]] = []
        self._refresh_timer = QtCore.QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.timeout.connect(self._refresh_previews)
        self._last_rot90: str | None = None  # "cw" or "ccw"

        top_row = QtWidgets.QHBoxLayout()
        lbl_title = QtWidgets.QLabel("<b>Augmentation Details</b>")
        top_row.addWidget(lbl_title)
        top_row.addStretch(1)
        if not self._embed:
            self.chk_custom_aug = QtWidgets.QCheckBox("Enable custom overrides")
            top_row.addWidget(self.chk_custom_aug)
        else:
            self.chk_custom_aug = None
        layout.addLayout(top_row)

        grid = QtWidgets.QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        layout.addLayout(grid, 1)

        self._aug_override_controls: list[QtWidgets.QWidget] = []
        self._aug_override_labels: list[QtWidgets.QWidget] = []
        # assets live under dl_software/assets
        assets_root = Path(__file__).resolve().parents[1] / "assets"

        def add_card(row: int, col: int, title: str, img_name: str, control: QtWidgets.QWidget, tooltip: str | None = None):
            frame = QtWidgets.QFrame()
            frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
            frame.setStyleSheet("QFrame{border:1px solid #d0d5dd;border-radius:8px;}")
            frame.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
            v = QtWidgets.QVBoxLayout(frame)
            v.setContentsMargins(8, 8, 8, 8)
            v.setSpacing(6)
            lbl_img = QtWidgets.QLabel()
            lbl_img.setFixedSize(130, 130)
            lbl_img.setAlignment(Qt.AlignCenter)
            lbl_img.setObjectName(f"img_{title}")
            self._image_labels.append((lbl_img, img_name))
            lbl_title = QtWidgets.QLabel(title)
            lbl_title.setAlignment(Qt.AlignCenter)
            lbl_title.setStyleSheet("font-weight:600;")
            if tooltip:
                lbl_title.setToolTip(tooltip)
                control.setToolTip(tooltip)
            control.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            control.setMinimumWidth(lbl_img.width())
            control.setMaximumWidth(lbl_img.width())
            v.addWidget(lbl_img)
            v.addWidget(lbl_title)
            v.addWidget(control)
            grid.addWidget(frame, row, col)
            self._aug_override_controls.append(control)
            self._aug_override_labels.append(lbl_title)

        # probability/strength controls
        flip_box = QtWidgets.QWidget()
        flip_layout = QtWidgets.QVBoxLayout(flip_box)
        flip_layout.setContentsMargins(0, 0, 0, 0)
        flip_layout.setSpacing(4)
        self.chk_flip_lr = QtWidgets.QCheckBox("Horizontal")
        self.chk_flip_lr.setChecked(True)
        self.chk_flip_ud = QtWidgets.QCheckBox("Vertical")
        flip_layout.addWidget(self.chk_flip_lr)
        flip_layout.addWidget(self.chk_flip_ud)
        add_card(0, 0, "Flip", "fliplr.png", flip_box, "Enable horizontal/vertical flip")

        rot90_box = QtWidgets.QWidget()
        rot90_layout = QtWidgets.QVBoxLayout(rot90_box)
        rot90_layout.setContentsMargins(0, 0, 0, 0)
        rot90_layout.setSpacing(4)
        self.chk_rot90_cw = QtWidgets.QCheckBox("Clockwise")
        self.chk_rot90_ccw = QtWidgets.QCheckBox("Counter-Clockwise")
        rot90_layout.addWidget(self.chk_rot90_cw)
        rot90_layout.addWidget(self.chk_rot90_ccw)
        add_card(0, 1, "90° Rotate", "rot_90.png", rot90_box, "Apply 90° rotations")

        self.sp_rotation_deg = self._make_spin(0.0, 90.0, 5.0, 0.0, decimals=1)
        add_card(0, 2, "Rotation ±deg", "rot_10.png", self.sp_rotation_deg, "General rotation range (±deg)")

        # scale_strength now 0.0~1.0 meaning random scale in [1-strength, 1+strength]
        self.sp_scale_strength = self._make_spin(0.0, 1.0, 0.05, 0.2)
        add_card(0, 3, "Scale (crop)", "scale_1.5.png", self.sp_scale_strength, "Scale/crop strength (0-1 → random [1-s, 1+s])")

        self.sp_shear_deg = self._make_spin(0.0, 45.0, 1.0, 0.0, decimals=1)
        add_card(1, 0, "Shear ±deg", "shear_15.png", self.sp_shear_deg, "Shear angle (±deg)")

        self.sp_grayscale_prob = self._make_spin(0.0, 1.0, 0.05, 0.01)
        add_card(1, 1, "Grayscale prob", "grayscale.png", self.sp_grayscale_prob, "Probability to convert to grayscale")

        self.sp_hsv_h = self._make_spin(0.0, 1.0, 0.005, 0.015, decimals=3)
        add_card(1, 2, "Hue", "hue.png", self.sp_hsv_h, "Hue augmentation strength")

        self.sp_hsv_s = self._make_spin(0.0, 1.0, 0.05, 0.7)
        add_card(1, 3, "Saturation", "saturation.png", self.sp_hsv_s, "Saturation augmentation strength")

        self.sp_hsv_v = self._make_spin(0.0, 1.0, 0.05, 0.4)
        add_card(2, 0, "Brightness", "value.png", self.sp_hsv_v, "Value/Brightness augmentation strength")

        self.sp_exposure_strength = self._make_spin(0.0, 2.0, 0.05, 0.0)
        add_card(2, 1, "Exposure", "rot_350.png", self.sp_exposure_strength, "Exposure/contrast strength")

        self.sp_blur_strength = self._make_spin(0.0, 5.0, 0.1, 0.5)
        add_card(2, 2, "Blur", "blur.png", self.sp_blur_strength, "Gaussian blur strength (px radius)")

        self.sp_noise_strength = self._make_spin(0.0, 2.0, 0.1, 0.0)
        add_card(2, 3, "Noise", "noise.png", self.sp_noise_strength, "Gaussian noise strength")

        self._defaults = {
            "custom": False,
            "flip_lr": True,
            "flip_ud": False,
            "rot90_cw": False,
            "rot90_ccw": False,
            "rotation": 0.0,
            "scale": 0.2,
            "shear": 0.0,
            "grayscale": 0.01,
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "exposure": 0.0,
            "blur": 0.5,
            "noise": 0.0,
        }

        layout.addStretch(1)

        self.btn_reset = QtWidgets.QPushButton("Reset")
        self.btn_reset.setIcon(self._outlined_icon(self.style().standardIcon(QtWidgets.QStyle.SP_BrowserReload), color=QtCore.Qt.black))
        self.btn_reset.clicked.connect(self._on_reset)

        if not self._embed:
            btns = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
            )
            btns.addButton(self.btn_reset, QtWidgets.QDialogButtonBox.ResetRole)
            btns.accepted.connect(self.accept)
            btns.rejected.connect(self.reject)
            layout.addWidget(btns)

        # load base/original and refresh previews
        self._base_image = None
        base_path = assets_root / "original.png"
        if base_path.exists():
            try:
                if cv2 is not None:
                    self._base_image = cv2.cvtColor(cv2.imread(str(base_path)), cv2.COLOR_BGR2RGB)
                else:
                    pil = Image.open(str(base_path)).convert("RGB")
                    self._base_image = np.array(pil)
            except Exception:
                self._base_image = None
        self._apply_defaults()
        self._connect_preview_updates()
        self._refresh_previews()

    # ------ ui helpers ------
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
        enabled = True if self._embed else bool(self.chk_custom_aug.isChecked())
        for w in self._aug_override_controls:
            w.setEnabled(enabled)
        for lbl in self._aug_override_labels:
            lbl.setEnabled(enabled)

    def values(self) -> dict | None:
        if (self.chk_custom_aug is not None) and (not self.chk_custom_aug.isChecked()):
            return None
        train_kwargs = {
            "fliplr": 1.0 if self.chk_flip_lr.isChecked() else 0.0,
            "flipud": 1.0 if self.chk_flip_ud.isChecked() else 0.0,
            "degrees": float(self.sp_rotation_deg.value()),
            "scale": float(self.sp_scale_strength.value()),
            "shear": float(self.sp_shear_deg.value()),
            "hsv_h": float(self.sp_hsv_h.value()),
            "hsv_s": float(self.sp_hsv_s.value()),
            "hsv_v": float(self.sp_hsv_v.value()),
        }
        alb_cfg = {
            "rotate90_p": 1.0 if (self.chk_rot90_cw.isChecked() or self.chk_rot90_ccw.isChecked()) else 0.0,
            "grayscale_p": float(self.sp_grayscale_prob.value()),
            "exposure_p": float(self.sp_exposure_strength.value()),
            "blur_p": float(self.sp_blur_strength.value()),
            "noise_p": float(self.sp_noise_strength.value()),
        }
        return {"train_kwargs": train_kwargs, "albumentations": alb_cfg}

    def set_values(self, values: dict | None):
        """Populate controls from a previously saved override dict."""
        if not values:
            self._apply_defaults()
            self.chk_custom_aug.setChecked(False)
            self._update_aug_controls()
            return
        tk = values.get("train_kwargs") if isinstance(values, dict) else None
        alb = values.get("albumentations") if isinstance(values, dict) else None
        # defaults as fallback
        self.chk_custom_aug.setChecked(True)
        bvals = [
            (self.chk_flip_lr, (tk or {}).get("fliplr", self._defaults["flip_lr"]) >= 0.5),
            (self.chk_flip_ud, (tk or {}).get("flipud", self._defaults["flip_ud"]) >= 0.5),
            (self.chk_rot90_cw, (alb or {}).get("rotate90_p", 0.0) >= 0.5),
            (self.chk_rot90_ccw, False),
        ]
        for w, val in bvals:
            w.blockSignals(True)
            w.setChecked(bool(val))
            w.blockSignals(False)
        svals = [
            (self.sp_rotation_deg, (tk or {}).get("degrees", self._defaults["rotation"])),
            (self.sp_scale_strength, (tk or {}).get("scale", self._defaults["scale"])),
            (self.sp_shear_deg, (tk or {}).get("shear", self._defaults["shear"])),
            (self.sp_grayscale_prob, (alb or {}).get("grayscale_p", self._defaults["grayscale"])),
            (self.sp_hsv_h, (tk or {}).get("hsv_h", self._defaults["hsv_h"])),
            (self.sp_hsv_s, (tk or {}).get("hsv_s", self._defaults["hsv_s"])),
            (self.sp_hsv_v, (tk or {}).get("hsv_v", self._defaults["hsv_v"])),
            (self.sp_exposure_strength, (alb or {}).get("exposure_p", self._defaults["exposure"])),
            (self.sp_blur_strength, (alb or {}).get("blur_p", self._defaults["blur"])),
            (self.sp_noise_strength, (alb or {}).get("noise_p", self._defaults["noise"])),
        ]
        for w, val in svals:
            w.blockSignals(True)
            try:
                w.setValue(float(val))
            except Exception:
                pass
            w.blockSignals(False)
        self._update_aug_controls()
        self._refresh_previews()

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
        self.chk_rot90_cw.toggled.connect(lambda _: self._on_rot90_toggled("cw"))
        self.chk_rot90_ccw.toggled.connect(lambda _: self._on_rot90_toggled("ccw"))
        if self.chk_custom_aug is not None:
            self.chk_custom_aug.toggled.connect(self._update_aug_controls)

    def _apply_defaults(self):
        self._refresh_timer.stop()
        bvals = []
        if self.chk_custom_aug is not None:
            bvals.append((self.chk_custom_aug, self._defaults["custom"]))
        bvals.extend([
            (self.chk_flip_lr, self._defaults["flip_lr"]),
            (self.chk_flip_ud, self._defaults["flip_ud"]),
            (self.chk_rot90_cw, self._defaults["rot90_cw"]),
            (self.chk_rot90_ccw, self._defaults["rot90_ccw"]),
        ])
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

    def _on_rot90_toggled(self, flag: str):
        # track last toggled direction to decide preview when both are checked
        self._last_rot90 = flag

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
                    h, w = aug_img.shape[:2]
                    # make a copy so QImage owns the memory
                    qimg = QtGui.QImage(aug_img.tobytes(), w, h, QtGui.QImage.Format_RGBA8888)
                    pm = QtGui.QPixmap.fromImage(qimg).scaled(lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                lbl.setPixmap(pm)
        except Exception:
            pass

    def _render_aug_example(self, key: str) -> np.ndarray | None:
        if self._base_image is None:
            return None
        img = np.array(self._base_image, copy=True)
        if key == "fliplr.png":
            if self.chk_flip_lr.isChecked():
                img = np.flip(img, axis=1)
            if self.chk_flip_ud.isChecked():
                img = np.flip(img, axis=0)
        elif key == "rot_90.png":
            base = np.array(self._base_image, copy=True)
            cw = self.chk_rot90_cw.isChecked()
            ccw = self.chk_rot90_ccw.isChecked()
            if cw and ccw:
                if self._last_rot90 == "ccw":
                    img = np.rot90(base, 1)
                else:
                    img = np.rot90(base, -1)
            elif cw:
                img = np.rot90(base, -1)
            elif ccw:
                img = np.rot90(base, 1)
            else:
                img = base
        elif key == "rot_10.png":
            deg = self.sp_rotation_deg.value()
            if deg > 0:
                img = self._rotate_image(img, deg)
        elif key == "scale_1.5.png":
            strength = self.sp_scale_strength.value()
            factor = 1.0 + strength
            if abs(factor - 1.0) > 1e-6:
                img = self._scale_crop(img, factor)
        elif key == "shear_15.png":
            shear = self.sp_shear_deg.value()
            if shear != 0:
                img = self._shear(img, shear)
        elif key == "grayscale.png":
            p = self.sp_grayscale_prob.value()
            if p > 0.001:
                img = ImageOps.grayscale(Image.fromarray(img)).convert("RGB")
                img = np.array(img)
        elif key == "hue.png":
            h = self.sp_hsv_h.value()
            img = self._hsv_shift(img, h, 0.0, 0.0)
        elif key == "saturation.png":
            s = self.sp_hsv_s.value()
            img = self._hsv_shift(img, 0.0, s, 0.0)
        elif key == "value.png":
            v = self.sp_hsv_v.value()
            img = self._hsv_shift(img, 0.0, 0.0, v)
        elif key == "rot_350.png":
            exp = self.sp_exposure_strength.value()
            if exp != 0:
                pil = Image.fromarray(img)
                pil = ImageEnhance.Brightness(pil).enhance(1 + exp)
                pil = ImageEnhance.Contrast(pil).enhance(1 + exp / 2.0)
                img = np.array(pil)
        elif key == "blur.png":
            blur = self.sp_blur_strength.value()
            if blur > 0:
                pil = Image.fromarray(img)
                pil = pil.filter(ImageFilter.GaussianBlur(radius=blur))
                img = np.array(pil)
        elif key == "noise.png":
            noise = self.sp_noise_strength.value()
            if noise > 0:
                sigma = noise * 25.0
                noise_arr = np.random.normal(0, sigma, img.shape).astype(np.float32)
                img = np.clip(img.astype(np.float32) + noise_arr, 0, 255).astype(np.uint8)
        # Ensure RGB numpy
        if isinstance(img, Image.Image):
            img = np.array(img.convert("RGB"))
        if img.ndim == 2:  # grayscale -> rgb
            img = np.stack([img]*3, axis=-1)
        img_rgba = np.concatenate([img, np.full((*img.shape[:2], 1), 255, dtype=np.uint8)], axis=2)
        return img_rgba

    def _rotate_image(self, img: np.ndarray, deg: float) -> np.ndarray:
        if deg == 0:
            return img
        if cv2 is None:
            pil = Image.fromarray(img)
            pil = pil.rotate(deg, resample=Image.BILINEAR)
            return np.array(pil)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))

    def _scale_crop(self, img: np.ndarray, factor: float) -> np.ndarray:
        h, w = img.shape[:2]
        if factor <= 0:
            return img
        new_w = int(w * factor)
        new_h = int(h * factor)
        if new_w <= 0 or new_h <= 0:
            return img
        if cv2 is None:
            resized = np.array(Image.fromarray(img).resize((new_w, new_h), resample=Image.BILINEAR))
        else:
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        # center crop/pad to original size
        top = max(0, (new_h - h) // 2)
        left = max(0, (new_w - w) // 2)
        bottom = top + h
        right = left + w
        if resized.shape[0] < bottom or resized.shape[1] < right:
            pad_h = max(0, bottom - resized.shape[0])
            pad_w = max(0, right - resized.shape[1])
            resized = np.pad(resized, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=0)
        return resized[top:bottom, left:right]

    def _shear(self, img: np.ndarray, shear_deg: float) -> np.ndarray:
        if shear_deg == 0:
            return img
        if cv2 is None:
            pil = Image.fromarray(img)
            rad = np.deg2rad(shear_deg)
            pil = pil.transform(pil.size, Image.AFFINE, (1, np.tan(rad), 0, 0, 1, 0), resample=Image.BILINEAR)
            return np.array(pil)
        h, w = img.shape[:2]
        shear_rad = np.deg2rad(shear_deg)
        M = np.array([[1, np.tan(shear_rad), 0], [0, 1, 0]], dtype=np.float32)
        return cv2.warpAffine(img, M, (w, h), borderValue=(0, 0, 0))

    def _hsv_shift(self, img: np.ndarray, h: float, s: float, v: float) -> np.ndarray:
        if cv2 is None:
            pil = Image.fromarray(img).convert("HSV")
            h_ch, s_ch, v_ch = pil.split()
            h_arr = np.array(h_ch, dtype=np.float32)
            s_arr = np.array(s_ch, dtype=np.float32)
            v_arr = np.array(v_ch, dtype=np.float32)
            h_arr = (h_arr + h * 180) % 180
            s_arr = np.clip(s_arr * (1 + s), 0, 255)
            v_arr = np.clip(v_arr * (1 + v), 0, 255)
            merged = Image.merge("HSV", [Image.fromarray(a.astype(np.uint8)) for a in (h_arr, s_arr, v_arr)])
            return np.array(merged.convert("RGB"))
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        img_hsv[..., 0] = (img_hsv[..., 0] + h * 180) % 180
        img_hsv[..., 1] = np.clip(img_hsv[..., 1] * (1 + s), 0, 255)
        img_hsv[..., 2] = np.clip(img_hsv[..., 2] * (1 + v), 0, 255)
        img_hsv = img_hsv.astype(np.uint8)
        return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
