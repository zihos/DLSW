"""Infer tab UI skeleton."""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets
import os
import time
import numpy as np
import cv2
from PySide6.QtCore import Qt
from ...label_tool import InferWorker

from ..widgets import Header, TitledGroup, configure_combo


class ZoomGraphicsView(QtWidgets.QGraphicsView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)
        self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        self.scaleFactor = 1.0

    def wheelEvent(self, e: QtGui.QWheelEvent):
        if self.scene() is None:
            return super().wheelEvent(e)
        old_pos = self.mapToScene(e.position().toPoint())
        delta = e.angleDelta().y() if e.angleDelta().y() != 0 else e.pixelDelta().y()
        if delta == 0:
            return
        step = delta / 120.0
        factor = 1.15 ** step
        new_scale = max(0.05, min(50.0, self.scaleFactor * factor))
        factor = new_scale / self.scaleFactor
        self.scale(factor, factor)
        self.scaleFactor = new_scale
        new_pos = self.mapToScene(e.position().toPoint())
        self.translate(old_pos.x() - new_pos.x(), old_pos.y() - new_pos.y())
        self.viewport().update()
        e.accept()


class InferTab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        left = QtWidgets.QWidget()
        left.setFixedWidth(360)
        lL = QtWidgets.QVBoxLayout(left)
        lL.addWidget(Header("Model & Source"))

        mod_box = TitledGroup("Model")
        ml = QtWidgets.QFormLayout(mod_box)
        self.ed_weights = QtWidgets.QLineEdit()
        self.btn_browse_weights = QtWidgets.QPushButton("Browse")
        wrow = QtWidgets.QHBoxLayout(); wrow.setContentsMargins(0,0,0,0)
        wrow.addWidget(self.ed_weights)
        wrow.addWidget(self.btn_browse_weights)
        wwrap = QtWidgets.QWidget(); wlay = QtWidgets.QHBoxLayout(wwrap); wlay.setContentsMargins(0,0,0,0); wlay.addLayout(wrow)
        self.cmb_type = configure_combo(QtWidgets.QComboBox(), show_all=True)
        type_options = ["Object Detection", "Instance Segmentation"]
        self.cmb_type.addItem("")
        self.cmb_type.addItems(type_options)
        type_view = self.cmb_type.view()
        if isinstance(type_view, QtWidgets.QListView) and self.cmb_type.count():
            row_index = 1 if self.cmb_type.count() > 1 else 0
            row_height = type_view.sizeHintForRow(row_index)
            if row_height <= 0:
                row_height = type_view.sizeHintForRow(0)
            if row_height > 0:
                type_view.setMinimumHeight(row_height * self.cmb_type.count() + type_view.frameWidth() * 2)
        self.cmb_type.setMaxVisibleItems(self.cmb_type.count())
        self.cmb_type.setCurrentIndex(0)
        ml.addRow("weights", wwrap)
        ml.addRow("task", self.cmb_type)
        lL.addWidget(mod_box)

        src_box = TitledGroup("Source")
        sl = QtWidgets.QFormLayout(src_box)
        sl.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        self.cmb_mode = configure_combo(QtWidgets.QComboBox(), show_all=True)
        mode_options = ["Current Image", "Single Image", "Folder", "Video", "Webcam"]
        self.cmb_mode.addItem("")
        self.cmb_mode.addItems(mode_options)
        mode_view = self.cmb_mode.view()
        if isinstance(mode_view, QtWidgets.QListView):
            count = self.cmb_mode.count()
            if count:
                row_index = 1 if count > 1 else 0
                row_height = mode_view.sizeHintForRow(row_index)
                if row_height <= 0:
                    row_height = mode_view.sizeHintForRow(0)
                if row_height > 0:
                    mode_view.setMinimumHeight(row_height * count + mode_view.frameWidth() * 2)
                self.cmb_mode.setMaxVisibleItems(count)
        self.cmb_mode.setCurrentIndex(0)
        self.ed_path = QtWidgets.QLineEdit()
        self.btn_browse = QtWidgets.QPushButton("Browse")
        prow = QtWidgets.QHBoxLayout(); prow.setContentsMargins(0,0,0,0)
        prow.addWidget(self.ed_path)
        prow.addWidget(self.btn_browse)
        path_wrap = QtWidgets.QWidget(); pwl = QtWidgets.QHBoxLayout(path_wrap); pwl.setContentsMargins(0,0,0,0); pwl.addLayout(prow)
        sl.addRow("mode", self.cmb_mode)
        sl.addRow("path", path_wrap)
        # folder mode: show loaded image files
        self.list_images = QtWidgets.QListWidget()
        # Grow with the panel vertically instead of fixed height
        self.list_images.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        self.list_images.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        sl.addRow("files", self.list_images)
        # Let the Source box take vertical space proportionally
        lL.addWidget(src_box, 1)

        # Run button (moved here under Source)
        run_row = QtWidgets.QHBoxLayout()
        self.btn_save_overlay = QtWidgets.QPushButton("Run Inference")
        self.btn_save_overlay.setMinimumWidth(160)
        self.btn_save_overlay.setToolTip("Run inference (Ctrl+R)")
        run_row.addStretch(1)
        run_row.addWidget(self.btn_save_overlay)
        lL.addLayout(run_row)
        # Make Source box take most of the left column vertically
        try:
            lL.setStretch(0, 0)  # header
            lL.setStretch(1, 0)  # model box
            lL.setStretch(2, 1)  # source box grows
            lL.setStretch(3, 0)  # run row
        except Exception:
            pass

        center = QtWidgets.QWidget()
        cL = QtWidgets.QVBoxLayout(center)

        # Title row: center pager only (filename will be below image)
        self.title_bar = QtWidgets.QWidget()
        tlay = QtWidgets.QHBoxLayout(self.title_bar)
        tlay.setContentsMargins(0, 0, 4, 8)
        tlay.setSpacing(8)
        tlay.addStretch(1)
        # pager in center
        self.pager = QtWidgets.QWidget(); pgl = QtWidgets.QHBoxLayout(self.pager)
        pgl.setContentsMargins(10,2,10,2); pgl.setSpacing(10)
        self.btn_page_prev = QtWidgets.QToolButton(); self.btn_page_prev.setText("◀"); self.btn_page_prev.setToolTip("Prev")
        self.btn_page_prev.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn_page_prev.clicked.connect(self._prev_image)
        self.lbl_page = QtWidgets.QLabel("0 / 0"); self.lbl_page.setAlignment(QtCore.Qt.AlignCenter)
        # Make page indicator larger and bold
        self.lbl_page.setStyleSheet("font-weight:700; font-size:14px;")
        self.btn_page_next = QtWidgets.QToolButton(); self.btn_page_next.setText("▶"); self.btn_page_next.setToolTip("Next")
        self.btn_page_next.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn_page_next.clicked.connect(self._next_image)
        pgl.addWidget(self.btn_page_prev); pgl.addWidget(self.lbl_page); pgl.addWidget(self.btn_page_next)
        tlay.addWidget(self.pager, 0, QtCore.Qt.AlignCenter)
        tlay.addStretch(1)
        cL.addWidget(self.title_bar)

        self.canvas = ZoomGraphicsView()
        scn = QtWidgets.QGraphicsScene()
        self.canvas.setScene(scn)
        self.canvas.setBackgroundBrush(QtGui.QColor("#FFFFFF"))
        self._placeholder_rect = scn.addRect(0, 0, 960, 600, QtGui.QPen(QtGui.QColor("#C6CDD6"), 1, QtCore.Qt.DashLine))
        cL.addWidget(self.canvas, 1)
        # Filename under image (left aligned)
        fn_row = QtWidgets.QHBoxLayout(); fn_row.setContentsMargins(0,4,0,8)
        self.lbl_img_filename = QtWidgets.QLabel("-")
        self.lbl_img_filename.setProperty("hint", "subtle")
        fn_row.addWidget(self.lbl_img_filename)
        fn_row.addStretch(1)
        self.lbl_fps = QtWidgets.QLabel("")
        self.lbl_fps.setProperty("hint", "subtle")
        fn_row.addWidget(self.lbl_fps)
        cL.addLayout(fn_row)
        self._pm_item: QtWidgets.QGraphicsPixmapItem | None = None
        self._overlay_items: list[QtWidgets.QGraphicsItem] = []
        self._bbox_items: list[QtWidgets.QGraphicsRectItem] = []
        self._bbox_base_pen: list[QtGui.QPen] = []
        self._mask_highlight: list[QtWidgets.QGraphicsItem] = []
        self._class_colors: dict[str, QtGui.QColor] = {}
        self._class_color_cycle = [
            QtGui.QColor(79, 110, 219),   # indigo
            QtGui.QColor(80, 166, 111),   # green
            QtGui.QColor(238, 111, 146),  # pink
            QtGui.QColor(130, 96, 206),   # purple
            QtGui.QColor(244, 183, 65),   # amber
            QtGui.QColor(77, 208, 225),   # teal
            QtGui.QColor(153, 119, 73),   # brown
            QtGui.QColor(95, 164, 255),   # light blue
        ]
        self._class_color_index = 0
        self._active_task: str = "object detection"

        right = QtWidgets.QWidget()
        right.setFixedWidth(360)
        rL = QtWidgets.QVBoxLayout(right)

        det_box = TitledGroup("Detections / Instances")
        dt = QtWidgets.QTableWidget(0, 6)
        dt.setHorizontalHeaderLabels(["👁️", "ID", "Class", "Conf", "x1,y1", "x2,y2"])
        dt.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        dt.horizontalHeader().setStretchLastSection(True)
        dt.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        dt.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        v = QtWidgets.QVBoxLayout(det_box)
        v.addWidget(dt)
        self.tbl_detections = dt
        rL.addWidget(det_box, 1)

        # Display controls (moved from top): score threshold and mask alpha
        disp_box = TitledGroup("Display")
        dl = QtWidgets.QFormLayout(disp_box)
        self.sld_score = QtWidgets.QSlider(Qt.Horizontal)
        self.sld_score.setRange(0, 100)   # 0.00 ~ 1.00 mapped
        self.sld_score.setValue(25)
        self.ed_score_val = QtWidgets.QLineEdit("0.25")
        self.ed_score_val.setValidator(QtGui.QDoubleValidator(0.0, 1.0, 2, self))
        self.ed_score_val.setFixedWidth(56)
        self.ed_score_val.setAlignment(QtCore.Qt.AlignRight)
        row_score = QtWidgets.QWidget(); row_score_l = QtWidgets.QHBoxLayout(row_score)
        row_score_l.setContentsMargins(0,0,0,0); row_score_l.setSpacing(6)
        row_score_l.addWidget(self.sld_score, 1)
        row_score_l.addWidget(self.ed_score_val, 0)

        self.sld_alpha = QtWidgets.QSlider(Qt.Horizontal)
        self.sld_alpha.setRange(0, 100)
        self.sld_alpha.setValue(40)
        self.ed_alpha_val = QtWidgets.QLineEdit("40")
        self.ed_alpha_val.setValidator(QtGui.QIntValidator(0, 100, self))
        self.ed_alpha_val.setFixedWidth(56)
        self.ed_alpha_val.setAlignment(QtCore.Qt.AlignRight)
        row_alpha = QtWidgets.QWidget(); row_alpha_l = QtWidgets.QHBoxLayout(row_alpha)
        row_alpha_l.setContentsMargins(0,0,0,0); row_alpha_l.setSpacing(6)
        row_alpha_l.addWidget(self.sld_alpha, 1)
        row_alpha_l.addWidget(self.ed_alpha_val, 0)

        dl.addRow("score", row_score)
        dl.addRow("mask α", row_alpha)
        rL.addWidget(disp_box)

        exp_box = TitledGroup("Export Results")
        ex = QtWidgets.QHBoxLayout(exp_box)
        self.btn_export_json = QtWidgets.QPushButton("Export .json")
        self.btn_export_txt = QtWidgets.QPushButton("Export .txt")
        self.btn_export_masks = QtWidgets.QPushButton("Export masks")
        self.btn_export_masks.setEnabled(False)
        self.btn_export_masks.clicked.connect(self._export_masks)
        ex.addWidget(self.btn_export_json)
        ex.addWidget(self.btn_export_txt)
        ex.addWidget(self.btn_export_masks)
        ex.addStretch(1)
        rL.addWidget(exp_box)
        rL.addStretch(1)

        root.addWidget(left, 0)
        root.addWidget(center, 1)
        root.addWidget(right, 0)

        # wiring
        self.btn_browse.clicked.connect(self._pick_image)
        self.btn_browse_weights.clicked.connect(self._pick_weights)
        self.cmb_type.currentTextChanged.connect(self._on_task_changed)
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._cleanup_worker)
        self.tbl_detections.itemSelectionChanged.connect(self._on_table_select)
        self.list_images.itemClicked.connect(self._on_list_clicked)
        # Arrow shortcuts for navigating files in Folder mode
        self._shortcuts: list[QtGui.QShortcut] = []
        def bind(seq, fn):
            sc = QtGui.QShortcut(QtGui.QKeySequence(seq), self)
            sc.setContext(Qt.ApplicationShortcut)
            sc.activated.connect(fn)
            self._shortcuts.append(sc)
        bind(Qt.Key_Left, lambda: self._arrow_nav(-1))
        bind(Qt.Key_Right, lambda: self._arrow_nav(+1))
        bind("Ctrl+R", self.run_inference)
        # Export buttons
        self.btn_export_txt.clicked.connect(self._export_txt)
        self.btn_export_json.clicked.connect(self._export_json)
        self._update_export_buttons()
        # Update input display + re-render on display control changes
        self.sld_score.valueChanged.connect(self._on_display_changed)
        self.sld_alpha.valueChanged.connect(self._on_display_changed)
        # Per-row visibility toggles
        self.tbl_detections.itemChanged.connect(self._on_table_item_changed)
        # Allow manual input
        self.ed_score_val.editingFinished.connect(self._on_score_edited)
        self.ed_alpha_val.editingFinished.connect(self._on_alpha_edited)

        # state
        self._label_tool = None
        self._folder_images: list[str] = []
        self._folder_idx: int = -1
        self._folder_root: str | None = None
        self.worker: InferWorker | None = None
        self._current_view_path: str | None = None
        self._result_cache: dict[str, dict] = {}
        self._last_run_path: str | None = None
        self._last_model_path: str | None = None
        self._mode_user_selected: bool = False
        self._active_task: str = ""
        self._notified_no_detections: bool = False
        self._table_mode: str = ""
        self._table_block: bool = False
        self._video_source: str | None = None
        self._video_last_ts: float | None = None

        # Detect explicit user selection of mode
        try:
            self.cmb_mode.currentIndexChanged.connect(self._on_mode_changed)
        except Exception:
            pass

    def _on_mode_changed(self, *_):
        # Mark that user explicitly selected a mode
        self._mode_user_selected = True
        mode = self.cmb_mode.currentText()
        # Stop any running worker when switching modes
        if self.worker is not None:
            try:
                if hasattr(self.worker, "stop"):
                    self.worker.stop()
            except Exception:
                pass
            try:
                self.worker.wait(100)
            except Exception:
                pass
            self.worker = None
        # Clear overlays and detection table to avoid stale visuals
        self._clear_overlays()
        try:
            self.tbl_detections.setRowCount(0)
        except Exception:
            pass
        if mode != "Video":
            self._video_source = None
            self._video_last_ts = None
            try:
                self.lbl_fps.setText("")
            except Exception:
                pass
        if mode == "Single Image":
            # Reset folder state and file list when leaving Folder mode
            self._folder_images = []
            self._folder_root = None
            self._folder_idx = -1
            try:
                self.list_images.clear()
            except Exception:
                pass
            # Reset pager since we're not in a folder anymore
            try:
                self.lbl_page.setText("0 / 0")
            except Exception:
                pass
        elif mode == "Folder":
            # Prepare for folder mode; clear current single-image preview path
            self._current_view_path = None
            try:
                self.lbl_img_filename.setText("-")
                self.lbl_page.setText("0 / 0")
            except Exception:
                pass

    def set_label_tool(self, tool):
        self._label_tool = tool

    # ----- helpers -----
    def set_preview_from_pixmap(self, pm: QtGui.QPixmap | None):
        # Reset overlays and table when switching the preview image to
        # avoid dangling QGraphicsItem references from previous results.
        self._clear_overlays()
        try:
            self.tbl_detections.blockSignals(True)
            self.tbl_detections.setRowCount(0)
        finally:
            self.tbl_detections.blockSignals(False)

        scn = self.canvas.scene()
        scn.clear()
        if pm is None or pm.isNull():
            self._placeholder_rect = scn.addRect(0, 0, 960, 600, QtGui.QPen(QtGui.QColor("#C6CDD6"), 1, QtCore.Qt.DashLine))
            return
        self._pm_item = scn.addPixmap(pm)
        scn.setSceneRect(pm.rect())
        QtCore.QTimer.singleShot(0, lambda: self.canvas.fitInView(self._pm_item, QtCore.Qt.KeepAspectRatio))

    def _set_preview_from_frame(self, frame_bgr):
        """Set preview directly from a BGR numpy array (video frame)."""
        self._clear_overlays()
        try:
            self.tbl_detections.blockSignals(True)
            self.tbl_detections.setRowCount(0)
        finally:
            self.tbl_detections.blockSignals(False)
        try:
            import numpy as _np
            import cv2 as _cv2
            if frame_bgr is None:
                return
            arr = _np.asarray(frame_bgr)
            if arr.ndim != 3 or arr.shape[2] < 3:
                return
            h, w = arr.shape[:2]
            rgb = _cv2.cvtColor(arr, _cv2.COLOR_BGR2RGB)
            qimg = QtGui.QImage(rgb.data, w, h, rgb.strides[0], QtGui.QImage.Format_RGB888).copy()
            pm = QtGui.QPixmap.fromImage(qimg)
        except Exception:
            return
        scn = self.canvas.scene()
        scn.clear()
        self._pm_item = scn.addPixmap(pm)
        scn.setSceneRect(pm.rect())
        QtCore.QTimer.singleShot(0, lambda: self.canvas.fitInView(self._pm_item, QtCore.Qt.KeepAspectRatio))

    def _clear_overlays(self):
        scn = self.canvas.scene()
        for it in self._overlay_items:
            try:
                scn.removeItem(it)
            except Exception:
                pass
        self._overlay_items.clear()
        self._bbox_items.clear()
        self._bbox_base_pen.clear()
        for it in self._mask_highlight:
            try:
                scn.removeItem(it)
            except Exception:
                pass
        self._mask_highlight.clear()

    def _color_for_class(self, name: str) -> QtGui.QColor:
        key = name or "_"
        color = self._class_colors.get(key)
        if color is None:
            palette = self._class_color_cycle
            color = palette[self._class_color_index % len(palette)]
            self._class_color_index += 1
            self._class_colors[key] = color
        return QtGui.QColor(color)
        self._bbox_items.clear()

    def _compute_class_masks(self, masks: np.ndarray | None, boxes: list, names: list[str], w: int, h: int, thr: float):
        """Merge instance masks by class id after resizing to (w,h)."""
        try:
            masks_np = np.asarray(masks)
        except Exception:
            return []
        if masks_np.ndim < 3 or masks_np.shape[0] == 0:
            return []
        n = masks_np.shape[0]
        class_to_mask: dict[int, np.ndarray] = {}
        class_to_name: dict[int, str] = {}
        for idx in range(n):
            # filter by conf if possible
            try:
                conf_val = float(boxes[idx].get("conf", 1.0)) if boxes and idx < len(boxes) else 1.0
            except Exception:
                conf_val = 1.0
            if conf_val < thr:
                continue
            try:
                m_small = masks_np[idx]
            except Exception:
                continue
            try:
                m_bin = (m_small > 0.5).astype(np.uint8)
                m_up = cv2.resize(m_bin, (w, h), interpolation=cv2.INTER_NEAREST)
            except Exception:
                continue
            if m_up.max() == 0:
                continue
            # class id / name
            cls_id = None
            try:
                if boxes and idx < len(boxes):
                    cls_id = boxes[idx].get("cls")
            except Exception:
                cls_id = None
            try:
                cid = int(cls_id) if cls_id is not None else idx
            except Exception:
                cid = idx
            cname = None
            try:
                if names and 0 <= cid < len(names):
                    cname = names[cid]
            except Exception:
                cname = None
            if cname is None and boxes and idx < len(boxes):
                try:
                    cname = str(boxes[idx].get("name", ""))
                except Exception:
                    cname = None
            if not cname:
                cname = str(cid)
            mask_bool = m_up.astype(bool)
            if cid not in class_to_mask:
                class_to_mask[cid] = mask_bool.copy()
                class_to_name[cid] = cname
            else:
                class_to_mask[cid] |= mask_bool
        class_items = []
        for cid in sorted(class_to_mask.keys()):
            class_items.append({"cid": cid, "name": class_to_name.get(cid, str(cid)), "mask": class_to_mask[cid]})
        return class_items

    def _render_class_masks(self, class_items: list[dict], visibility: list[bool] | None = None):
        if self._pm_item is None:
            return
        self._clear_overlays()
        scn = self.canvas.scene()
        fill_alpha = int(self.sld_alpha.value() / 100.0 * 160)
        pm = self._pm_item.pixmap()
        w = pm.width()
        h = pm.height()
        vis_list = visibility if isinstance(visibility, (list, tuple)) else None
        if vis_list is None or len(vis_list) != len(class_items):
            vis_list = [True] * len(class_items)
        for idx, item in enumerate(class_items):
            if idx < len(vis_list) and not vis_list[idx]:
                continue
            mask_bool = item.get("mask")
            if mask_bool is None or not np.any(mask_bool):
                continue
            color = self._color_for_class(item.get("name", str(item.get("cid", idx))))
            color.setAlpha(max(45, fill_alpha))
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[..., 0] = color.red()
            rgba[..., 1] = color.green()
            rgba[..., 2] = color.blue()
            rgba[..., 3] = color.alpha()
            rgba[~mask_bool] = 0
            hbytes, wbytes = rgba.shape[:2]
            qimg = QtGui.QImage(rgba.data, wbytes, hbytes, QtGui.QImage.Format_RGBA8888)
            pm_overlay = QtGui.QPixmap.fromImage(qimg)
            pit = scn.addPixmap(pm_overlay)
            pit.setZValue(10 + idx)
            self._overlay_items.append(pit)

    def _render_mask_outline(self, mask_bool: np.ndarray | None):
        """Draw a red outline for the selected mask (class-merged)."""
        for it in self._mask_highlight:
            try:
                self.canvas.scene().removeItem(it)
            except Exception:
                pass
        self._mask_highlight.clear()
        if mask_bool is None or not np.any(mask_bool):
            return
        try:
            m = (mask_bool.astype(np.uint8) * 255)
        except Exception:
            return
        try:
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        except Exception:
            return
        if not contours:
            return
        outline = np.zeros_like(m)
        try:
            cv2.drawContours(outline, contours, -1, 255, 2)
        except Exception:
            return
        pm = self._pm_item.pixmap()
        h, w = outline.shape[:2]
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[..., 0] = 235  # red
        rgba[..., 3] = np.where(outline > 0, 255, 0)
        qimg = QtGui.QImage(rgba.data, w, h, QtGui.QImage.Format_RGBA8888)
        pm_overlay = QtGui.QPixmap.fromImage(qimg)
        pit = self.canvas.scene().addPixmap(pm_overlay)
        pit.setZValue(100)
        self._mask_highlight.append(pit)

    def _pick_image(self):
        mode = self.cmb_mode.currentText()
        if not mode:
            QtWidgets.QMessageBox.warning(self, "Mode Required", "Please choose a <b>source mode</b> before browsing for a path.")
            return
        if mode == "Folder":
            d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select folder")
            if not d:
                return
            self.ed_path.setText(d)
            self._folder_root = d
            self._load_folder(d)
            return
        if mode == "Video":
            p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select video", "", "Videos (*.mp4 *.avi *.mov *.mkv *.wmv *.m4v);;All (*.*)")
        else:
            p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if not p:
            return
        self.ed_path.setText(p)
        if mode == "Video":
            self.cmb_mode.setCurrentText("Video")
        else:
            self.cmb_mode.setCurrentText("Single Image")
            self.set_preview_from_path(p)

    def set_preview_from_path(self, path: str):
        # Normalize to an absolute, resolved path to keep comparisons stable
        try:
            from pathlib import Path as _P
            npath = str(_P(path).resolve())
        except Exception:
            npath = path
        self._current_view_path = npath
        self.set_preview_from_pixmap(QtGui.QPixmap(npath))
        self._update_title()
        # If this image was already processed, restore its result immediately.
        self._render_from_cache()

    def _render_from_cache(self):
        p = self._current_view_path
        if not p:
            return
        data = self._result_cache.get(p)
        if not data:
            return
        if "masks" in data:
            self._render_current_masks()
        else:
            boxes = data.get('boxes', [])
            self._render_boxes(boxes)
            self._fill_table(boxes)

    def _render_current_masks(self):
        p = self._current_view_path
        if not p:
            return
        data = self._result_cache.get(p)
        if not data or "masks" not in data:
            return
        if self._pm_item is None:
            return
        pm = self._pm_item.pixmap()
        w = pm.width()
        h = pm.height()
        masks = data.get("masks")
        boxes = data.get("boxes") or []
        names = data.get("names") or []
        thr = (self.sld_score.value() / 100.0) if hasattr(self, "sld_score") else 0.0
        class_items = self._compute_class_masks(masks, boxes, names, w, h, thr)
        vis = data.get("class_visibility")
        if not isinstance(vis, list) or len(vis) != len(class_items):
            vis = [True] * len(class_items)
            data["class_visibility"] = vis
        self._render_class_masks(class_items, vis)
        self._fill_table_masks(class_items, vis)
        # fps label is only meaningful for video; clear otherwise
        if not self._video_source:
            try:
                self.lbl_fps.setText("")
            except Exception:
                pass

    def _pick_weights(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select weights", "", "Weights (*.pt *.onnx *.engine *.torchscript *.tflite *.xml *.bin);;All (*.*)")
        if not p:
            return
        self.ed_weights.setText(p)

    # navigation for folder mode (basic)
    def _prev_image(self):
        if not self._folder_images:
            return
        self._folder_idx = (self._folder_idx - 1) % len(self._folder_images)
        p = self._folder_images[self._folder_idx]
        self.ed_path.setText(p)
        self.list_images.blockSignals(True)
        self.list_images.setCurrentRow(self._folder_idx)
        self.list_images.blockSignals(False)
        self.set_preview_from_path(p)

    def _next_image(self):
        if not self._folder_images:
            return
        self._folder_idx = (self._folder_idx + 1) % len(self._folder_images)
        p = self._folder_images[self._folder_idx]
        self.ed_path.setText(p)
        self.list_images.blockSignals(True)
        self.list_images.setCurrentRow(self._folder_idx)
        self.list_images.blockSignals(False)
        self.set_preview_from_path(p)

    def _arrow_nav(self, delta: int):
        if self.cmb_mode.currentText() != "Folder" or not self._folder_images:
            return
        if delta < 0:
            self._prev_image()
        else:
            self._next_image()

    def _on_list_clicked(self, item: QtWidgets.QListWidgetItem):
        p = item.data(QtCore.Qt.UserRole)
        if not isinstance(p, str):
            return
        try:
            idx = self._folder_images.index(p)
        except ValueError:
            return
        self._folder_idx = idx
        self.ed_path.setText(p)
        self.set_preview_from_path(p)

    def _load_folder(self, d: str):
        import pathlib
        exts = {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}
        imgs = [str(p.resolve()) for p in sorted(pathlib.Path(d).glob("**/*")) if p.suffix.lower() in exts]
        self._folder_images = imgs
        self._folder_idx = 0 if imgs else -1
        self._result_cache.clear()
        self.list_images.clear()
        for p in imgs:
            it = QtWidgets.QListWidgetItem(pathlib.Path(p).name)
            it.setData(QtCore.Qt.UserRole, p)
            self.list_images.addItem(it)
        if imgs:
            self.list_images.setCurrentRow(0)
            self.set_preview_from_path(imgs[0])
        self._update_title()

    def run_inference(self):
        # build params for InferWorker
        model = self.ed_weights.text().strip()
        model_path_normalized = model
        try:
            from pathlib import Path as _Path
            mp = _Path(model)
            if mp.suffix.lower() == ".xml" and mp.exists():
                # Ultralytics expects the OpenVINO project directory, not the XML file itself
                model_path_normalized = str(mp.parent.resolve())
            elif mp.suffix.lower() == ".bin" and mp.exists():
                model_path_normalized = str(mp.parent.resolve())
            elif mp.is_dir():
                model_path_normalized = str(mp.resolve())
        except Exception:
            pass
        model = model_path_normalized
        conf = max(0, min(1.0, self.sld_score.value() / 100.0))
        imgsz = None  # use original image size
        device = "0"
        mode = self.cmb_mode.currentText()
        task = self.cmb_type.currentText().strip()

        if not model:
            QtWidgets.QMessageBox.warning(self, "weights", "Select weights first.")
            return

        if not task:
            QtWidgets.QMessageBox.warning(self, "Task Required", "Please select a <b>task</b> before running inference.")
            return

        self._active_task = task
        task_lower = task.lower()
        self._notified_no_detections = False

        if mode == "Current Image":
            # In this app, "Current Image" means the one currently displayed on
            # the Infer tab canvas, not the Label tab. Require that it exists.
            if not self._mode_user_selected:
                QtWidgets.QMessageBox.warning(self, "Source", "Select mode and source first.")
                return
            run_path = self._current_view_path
            if not run_path:
                QtWidgets.QMessageBox.warning(self, "No image", "Load an image into the Infer tab first.")
                return
            # keep current canvas preview; just run using this path
            params = dict(model=model, imgsz=imgsz, conf=conf, device=device, use_folder=False, folder="", task=task_lower)
            self._last_run_path = run_path
            self._start_worker(params, run_path)
            return

        if mode == "Single Image":
            path = self.ed_path.text().strip()
            if not path:
                QtWidgets.QMessageBox.warning(self, "path", "Choose an image file.")
                return
            self.set_preview_from_path(path)
            params = dict(model=model, imgsz=imgsz, conf=conf, device=device, use_folder=False, folder="", task=task_lower)
            self._last_run_path = path
            self._start_worker(params, path)
            return

        if mode == "Video":
            path = self.ed_path.text().strip()
            if not path:
                QtWidgets.QMessageBox.warning(self, "path", "Choose a video file.")
                return
            try:
                from pathlib import Path as _P
                if not _P(path).exists():
                    QtWidgets.QMessageBox.warning(self, "path", "Video file not found.")
                    return
            except Exception:
                pass
            self._video_source = path
            self._current_view_path = path
            params = dict(model=model, imgsz=imgsz, conf=conf, device=device, use_folder=False, folder="", task=task_lower, use_video=True, video_path=path)
            self._last_run_path = path
            self._start_worker(params, path)
            return

        if mode == "Folder":
            d = (self._folder_root or self.ed_path.text().strip())
            if not d:
                QtWidgets.QMessageBox.warning(self, "Folder", "Load a folder first.")
                return
            # normalize UI state
            if not self._folder_root:
                self._folder_root = d
                try:
                    self.ed_path.setText(d)
                except Exception:
                    pass
            # ensure folder list is loaded
            if not self._folder_images:
                self._load_folder(d)
            if not self._folder_images:
                QtWidgets.QMessageBox.information(self, "Folder", "No images in folder.")
                return
            # clear previous single-image results; a new folder run should refresh results
            try:
                self._result_cache.clear()
            except Exception:
                pass
            # ensure we have a preview target (prefer current selection if valid)
            cur_row = self.list_images.currentRow() if self.list_images.count() > 0 else -1
            if 0 <= cur_row < len(self._folder_images):
                target = self._folder_images[cur_row]
                self._folder_idx = cur_row
            else:
                target = self._folder_images[0]
                self._folder_idx = 0
                self.list_images.setCurrentRow(0)
            if self._current_view_path != target:
                self.set_preview_from_path(target)
            params = dict(model=model, imgsz=imgsz, conf=conf, device=device, use_folder=True, folder=d, task=task_lower)
            # stream inference will feed via sig_each; pass None for single image
            self._start_worker(params, None)
            return

        QtWidgets.QMessageBox.information(self, "Mode", "Unsupported mode.")

    def _start_worker(self, params: dict, img_path: str | None):
        # If model path changed (e.g., new .onnx), clear cache to avoid stale overlays
        try:
            model_path = params.get("model", "")
            if model_path and model_path != self._last_model_path:
                self._result_cache.clear()
                self._last_model_path = model_path
        except Exception:
            pass
        # Stop previous worker if still around
        if self.worker is not None:
            try:
                if hasattr(self.worker, "stop"):
                    self.worker.stop()
            except Exception:
                pass
            try:
                # give previous worker enough time to finish to avoid premature deletion
                self.worker.wait(2000)
            except Exception:
                pass
            self.worker = None
        self._clear_overlays()
        self.btn_save_overlay.setEnabled(False)
        self.worker = InferWorker(params, img_path)
        try:
            self.worker.setParent(self)  # keep thread alive until we clean it up
        except Exception:
            pass
        self.worker.sig_log.connect(self._on_worker_log)
        self.worker.sig_done_single.connect(self._on_single)
        self.worker.sig_each.connect(lambda p, r: self._on_folder_each(p, r))
        try:
            self.worker.sig_video_frame.connect(self._on_video_frame)
        except Exception:
            pass
        self.worker.sig_done_folder.connect(lambda c: self._on_worker_finished())
        self.worker.finished.connect(self._on_worker_finished)
        self._infer_start_time = time.time()
        self.worker.start()

    def _on_single(self, result: dict):
        # Save and render the single-image result
        p = self._last_run_path or self._current_view_path
        if not p:
            return
        if "masks" in result:
            masks = result.get("masks")
            names = result.get("names") or []
            boxes = result.get("boxes") or []
            self._result_cache[p] = {"masks": masks, "names": names, "boxes": boxes, "class_visibility": None}
            if p == self._current_view_path:
                self._render_current_masks()
            self._maybe_notify_no_masks(masks)
        else:
            boxes = result.get("boxes") or []
            names = result.get("names") or []
            self._result_cache[p] = {'boxes': boxes, 'names': names}
            if p == self._current_view_path:
                self._render_boxes(boxes)
                self._fill_table(boxes)
            self._maybe_notify_no_detections(boxes)

    def _on_folder_each(self, path: str, result: dict):
        # cache; render if this is the image currently shown
        if not path:
            return
        # Normalize to absolute resolved paths to avoid mismatch (e.g., ONNX providers)
        try:
            p_norm = str(QtCore.QFileInfo(path).absoluteFilePath())
        except Exception:
            try:
                from pathlib import Path as _P
                p_norm = str(_P(path).resolve())
            except Exception:
                p_norm = path
        if "masks" in result:
            masks = result.get("masks")
            names = result.get("names") or []
            boxes = result.get("boxes") or []
            self._result_cache[p_norm] = {"masks": masks, "names": names, "boxes": boxes, "class_visibility": None}
        else:
            boxes = result.get("boxes") or []
            names = result.get("names") or []
            self._result_cache[p_norm] = {'boxes': boxes, 'names': names}
        cur = self._current_view_path
        if cur:
            same = False
            try:
                # robust same-file check when possible
                if os.path.exists(p_norm) and os.path.exists(cur):
                    same = os.path.samefile(p_norm, cur)
            except Exception:
                pass
            if not same:
                try:
                    from pathlib import Path as _P
                    same = _P(p_norm).resolve() == _P(cur).resolve()
                except Exception:
                    same = (p_norm == cur) or (os.path.basename(p_norm) == os.path.basename(cur))
            if same:
                if "masks" in result:
                    self._render_current_masks()
                    self._maybe_notify_no_masks(result.get("masks"))
                else:
                    boxes = result.get("boxes") or []
                    self._render_boxes(boxes)
                    self._fill_table(boxes)
                    self._maybe_notify_no_detections(boxes)

    def _on_video_frame(self, frame, result: dict):
        # frame: BGR numpy array
        path = self._video_source or self._last_run_path or "__video__"
        self._current_view_path = path
        self._set_preview_from_frame(frame)
        self._update_title()
        # fps measurement
        try:
            import time as _time
            now = _time.perf_counter()
            if self._video_last_ts is not None:
                dt = max(1e-6, now - self._video_last_ts)
                fps = 1.0 / dt
                self.lbl_fps.setText(f"{fps:.1f} FPS")
            self._video_last_ts = now
        except Exception:
            pass
        data = self._result_cache.get(path, {})
        # Preserve visibility if possible
        vis = data.get("class_visibility") if isinstance(data, dict) else None
        data = {
            "masks": result.get("masks"),
            "names": result.get("names") or [],
            "boxes": result.get("boxes") or [],
            "class_visibility": vis
        }
        self._result_cache[path] = data
        if data.get("masks") is not None:
            self._render_current_masks()
            self._maybe_notify_no_masks(data.get("masks"))
        else:
            boxes = data.get("boxes") or []
            self._render_boxes(boxes)
            self._fill_table(boxes)
            self._maybe_notify_no_detections(boxes)

    def _on_worker_log(self, message: str):
        try:
            print(message)
        except Exception:
            pass

    def _render_boxes(self, boxes: list):
        if self._pm_item is None:
            return
        self._clear_overlays()
        scn = self.canvas.scene()
        fill_alpha = int(self.sld_alpha.value() / 100.0 * 160)
        # confidence threshold
        thr = (self.sld_score.value() / 100.0) if hasattr(self, 'sld_score') else 0.0
        self._bbox_items = []
        self._bbox_base_pen = []
        for i, b in enumerate(boxes):
            try:
                if float(b.get('conf', 0.0)) < thr:
                    continue
            except Exception:
                pass
            x1, y1, x2, y2 = b.get("xyxy", (0, 0, 0, 0))
            r = QtCore.QRectF(QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2)).normalized()

            name = b.get("name", str(b.get("cls", "")))
            class_color = self._color_for_class(str(name))
            border_col = QtGui.QColor(class_color)
            pen = QtGui.QPen(border_col); pen.setWidthF(2.0); pen.setStyle(QtCore.Qt.DashLine)
            fill_color = QtGui.QColor(class_color)
            fill_color.setAlpha(max(45, fill_alpha))
            brush = QtGui.QBrush(fill_color)

            # bbox
            rect_item = scn.addRect(r, pen, brush)
            rect_item.setZValue(10)
            self._overlay_items.append(rect_item)
            self._bbox_items.append(rect_item)
            self._bbox_base_pen.append(QtGui.QPen(pen))

            # label text
            conf = float(b.get("conf", 0.0)) * 100.0
            label = f"{name} {conf:.1f}%"

            text_item = QtWidgets.QGraphicsTextItem(label)
            f = text_item.font(); f.setBold(True); text_item.setFont(f)
            text_item.setDefaultTextColor(QtCore.Qt.white)
            text_item.setZValue(12)
            text_item.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations, True)
            scn.addItem(text_item)

            # background pill behind text
            pad_x, pad_y = 3, 3
            br = text_item.boundingRect()
            bg_rect = QtCore.QRectF(0, 0, br.width() + pad_x * 2, br.height() + pad_y * 2)
            bg_color = QtGui.QColor(class_color); bg_color.setAlpha(220)
            bg_item = scn.addRect(bg_rect, QtGui.QPen(QtCore.Qt.NoPen), QtGui.QBrush(bg_color))
            bg_item.setZValue(11)
            bg_item.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations, True)

            # place centered above bbox; clamp within bbox width and move inside if needed
            bg_w, bg_h = bg_rect.width(), bg_rect.height()
            px = r.left() + (r.width() - bg_w) / 2.0
            py = r.top() - (bg_h)  # just above the box
            # if going above top, place inside near top
            if py < r.top():
                py = r.top()
            # clamp horizontally within bbox
            px = max(r.left(), min(px, r.right() - bg_w))
            # clamp vertically inside bbox if needed
            if py + bg_h > r.bottom():
                py = r.bottom() - bg_h

            bg_item.setPos(px, py)
            text_item.setPos(px + pad_x, py + pad_y)

            self._overlay_items.extend([bg_item, text_item])

        QtCore.QTimer.singleShot(0, lambda: self.canvas.fitInView(self._pm_item, QtCore.Qt.KeepAspectRatio))

    def _fill_table(self, boxes: list):
        dt: QtWidgets.QTableWidget = self.tbl_detections
        self._table_mode = "boxes"
        self._table_block = True
        dt.setRowCount(0)
        for i, b in enumerate(boxes):
            dt.insertRow(i)
            # Placeholder (non-interactive) for Show column
            placeholder = QtWidgets.QTableWidgetItem("")
            placeholder.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
            dt.setItem(i, 0, placeholder)
            name = b.get("name", str(b.get("cls", "")))
            conf = float(b.get("conf", 0.0))
            x1,y1,x2,y2 = b.get("xyxy", (0,0,0,0))
            cols = [str(i), name, f"{conf:.2f}", f"{int(x1)},{int(y1)}", f"{int(x2)},{int(y2)}"]
            for j, val in enumerate(cols):
                dt.setItem(i, j + 1, QtWidgets.QTableWidgetItem(val))
        self._table_block = False
        if dt.rowCount() > 0:
            dt.setCurrentCell(0, 0)
            self._highlight_row(0)
        else:
            try:
                self.tbl_detections.clearSelection()
            except Exception:
                pass
        self.btn_save_overlay.setEnabled(True)

    def _fill_table_masks(self, class_items: list[dict], visibility: list[bool] | None = None):
        dt: QtWidgets.QTableWidget = self.tbl_detections
        self._table_mode = "masks"
        self._table_block = True
        dt.setRowCount(0)
        vis_list = visibility if isinstance(visibility, (list, tuple)) else None
        if vis_list is None or len(vis_list) != len(class_items):
            vis_list = [True] * len(class_items)
        for i, item in enumerate(class_items):
            dt.insertRow(i)
            # visible toggle
            chk = QtWidgets.QTableWidgetItem()
            chk.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsSelectable)
            chk.setCheckState(QtCore.Qt.Checked if vis_list[i] else QtCore.Qt.Unchecked)
            dt.setItem(i, 0, chk)
            cid = item.get("cid", i)
            cls_name = item.get("name", str(cid))
            cols = [str(cid), cls_name, "-", "-", "-"]
            for j, val in enumerate(cols):
                dt.setItem(i, j + 1, QtWidgets.QTableWidgetItem(val))
        self._table_block = False
        if dt.rowCount() > 0:
            dt.setCurrentCell(0, 1)
            self._highlight_row(0)
        else:
            try:
                self.tbl_detections.clearSelection()
            except Exception:
                pass
        self.btn_save_overlay.setEnabled(True)

    def _maybe_notify_no_detections(self, boxes: list | None):
        if not self._active_task or self._active_task.lower() != "object detection":
            return
        if boxes:
            return
        if self._notified_no_detections:
            return
        self._notified_no_detections = True
        QtWidgets.QMessageBox.information(self, "Inference", "No detections were found.")

    def _maybe_notify_no_masks(self, masks: np.ndarray | None):
        if not self._active_task or not self._active_task.lower().startswith("instance"):
            return
        empty = True
        try:
            if isinstance(masks, np.ndarray) and masks.size > 0 and masks.max() > 0:
                empty = False
        except Exception:
            empty = True
        if not empty:
            return
        if self._notified_no_detections:
            return
        self._notified_no_detections = True
        QtWidgets.QMessageBox.information(self, "Inference", "No masks were found.")

    def _on_task_changed(self, text: str):
        self._active_task = text.strip() or ""
        self._update_export_buttons()

    def _update_export_buttons(self):
        if not hasattr(self, "btn_export_masks"):
            return
        task = (self._active_task or "").lower()
        seg_mode = task.startswith("instance")
        self.btn_export_masks.setEnabled(seg_mode)

    def _export_masks(self):
        if not self._current_view_path:
            QtWidgets.QMessageBox.information(self, "Export masks", "No image selected.")
            return
        data = self._result_cache.get(self._current_view_path) if hasattr(self, "_result_cache") else None
        if not data or "masks" not in data:
            QtWidgets.QMessageBox.information(self, "Export masks", "No masks available for this image.")
            return
        masks = data.get("masks")
        if masks is None:
            QtWidgets.QMessageBox.information(self, "Export masks", "No masks available for this image.")
            return
        # Determine output path
        base_dir = os.path.dirname(self._current_view_path)
        base_name = os.path.splitext(os.path.basename(self._current_view_path))[0]
        suggested = os.path.join(base_dir, f"{base_name}_mask.png")
        save_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save mask PNG", suggested, "PNG Images (*.png)")
        if not save_path:
            return
        # Load original size
        try:
            import cv2
            orig = cv2.imread(self._current_view_path, cv2.IMREAD_GRAYSCALE)
            H, W = orig.shape[:2]
        except Exception:
            pm = self._pm_item.pixmap() if self._pm_item else None
            if pm is None or pm.isNull():
                QtWidgets.QMessageBox.warning(self, "Export masks", "Cannot determine image size.")
                return
            W = pm.width(); H = pm.height()
        try:
            masks_np = np.asarray(masks)
        except Exception:
            QtWidgets.QMessageBox.warning(self, "Export masks", "Mask data is invalid.")
            return
        if masks_np.ndim < 3 or masks_np.shape[0] == 0:
            QtWidgets.QMessageBox.information(self, "Export masks", "No masks available for this image.")
            return
        combined = np.zeros((H, W), dtype=np.uint8)
        for idx in range(masks_np.shape[0]):
            try:
                m_small = masks_np[idx]
                m_bin = (m_small > 0.5).astype(np.uint8)
                m_up = cv2.resize(m_bin, (W, H), interpolation=cv2.INTER_NEAREST)
                combined = np.clip(combined + (m_up * 255), 0, 255).astype(np.uint8)
            except Exception:
                continue
        try:
            import cv2
            ok = cv2.imwrite(save_path, combined)
        except Exception:
            ok = False
        if ok:
            QtWidgets.QMessageBox.information(self, "Export masks", f"Saved masks to:\n{save_path}")
        else:
            QtWidgets.QMessageBox.warning(self, "Export masks", "Failed to save masks.")

    def _on_table_select(self):
        row = self.tbl_detections.currentRow()
        if row < 0:
            return
        self._highlight_row(row)

    def _on_table_item_changed(self, item: QtWidgets.QTableWidgetItem):
        if self._table_block:
            return
        if self._table_mode != "masks":
            return
        if item.column() != 0:
            return
        row = item.row()
        visible = item.checkState() == QtCore.Qt.Checked
        cur = self._current_view_path
        if not cur:
            return
        data = self._result_cache.get(cur, {})
        masks = data.get("masks")
        if masks is None or not isinstance(masks, np.ndarray):
            return
        # recompute class list for alignment
        pm = self._pm_item.pixmap() if self._pm_item else None
        if pm is None or pm.isNull():
            return
        w, h = pm.width(), pm.height()
        thr = (self.sld_score.value() / 100.0) if hasattr(self, "sld_score") else 0.0
        class_items = self._compute_class_masks(masks, data.get("boxes") or [], data.get("names") or [], w, h, thr)
        if row >= len(class_items):
            return
        vis_list = data.get("class_visibility")
        if not isinstance(vis_list, list) or len(vis_list) != len(class_items):
            vis_list = [True] * len(class_items)
        vis_list[row] = visible
        data["class_visibility"] = vis_list
        # Re-render masks with updated visibility
        self._render_class_masks(class_items, vis_list)

    def _highlight_row(self, row: int):
        if self._table_mode == "masks":
            # Skip highlight in video mode to reduce overhead/visual noise
            if self._video_source:
                return
            # outline selected class mask
            p = self._current_view_path
            if not p:
                return
            data = self._result_cache.get(p, {})
            masks = data.get("masks")
            if masks is None or self._pm_item is None:
                return
            pm = self._pm_item.pixmap()
            w, h = pm.width(), pm.height()
            thr = (self.sld_score.value() / 100.0) if hasattr(self, "sld_score") else 0.0
            class_items = self._compute_class_masks(masks, data.get("boxes") or [], data.get("names") or [], w, h, thr)
            if 0 <= row < len(class_items):
                self._render_mask_outline(class_items[row].get("mask"))
            return

        if not self._bbox_items or row < 0 or row >= len(self._bbox_items):
            return
        hi_pen = QtGui.QPen(QtGui.QColor(235, 64, 52)); hi_pen.setWidthF(2.6); hi_pen.setStyle(QtCore.Qt.SolidLine)
        for i, it in enumerate(list(self._bbox_items)):
            if not isinstance(it, QtWidgets.QGraphicsRectItem):
                continue
            try:
                if i == row:
                    it.setPen(hi_pen)
                else:
                    if 0 <= i < len(self._bbox_base_pen):
                        it.setPen(QtGui.QPen(self._bbox_base_pen[i]))
                    else:
                        default_pen = QtGui.QPen(QtGui.QColor(37, 99, 235))
                        default_pen.setWidthF(2.0); default_pen.setStyle(QtCore.Qt.DashLine)
                        it.setPen(default_pen)
            except RuntimeError:
                # item might have been deleted with scene switch
                pass

    def _on_worker_finished(self):
        try:
            self.btn_save_overlay.setEnabled(True)
        except Exception:
            pass
        # Make sure the worker really stopped before dropping our reference
        w = self.worker
        if w is not None:
            try:
                if w.isRunning():
                    w.wait(5000)
            except Exception:
                pass
            try:
                w.deleteLater()
            except Exception:
                pass
        try:
            if hasattr(self, "_infer_start_time"):
                elapsed = time.time() - self._infer_start_time
                print(f"[DEBUG] inference elapsed: {elapsed:.3f} s")
        except Exception:
            pass
        self.worker = None

    def _cleanup_worker(self):
        w = self.worker
        if w is not None:
            try:
                if hasattr(w, "stop"):
                    w.stop()
            except Exception:
                pass
            try:
                # Try to stop nicely; extend wait to reduce "thread still running" aborts
                w.wait(5000)
            except Exception:
                pass

    def eventFilter(self, obj, ev):
        return super().eventFilter(obj, ev)

    def _update_title(self):
        name = "-"
        if self._current_view_path:
            from pathlib import Path
            try:
                name = Path(self._current_view_path).name
            except Exception:
                name = self._current_view_path
            if self._folder_images:
                try:
                    idx = self._folder_images.index(self._current_view_path)
                    self.lbl_page.setText(f"{idx+1} / {len(self._folder_images)}")
                except ValueError:
                    self.lbl_page.setText("0 / 0")
            else:
                self.lbl_page.setText("0 / 0")
        else:
            self.lbl_page.setText("0 / 0")
        self.lbl_img_filename.setText(name)

    # ----- export helpers -----
    def _on_display_changed(self, *_):
        # update numeric editors text
        try:
            self.ed_score_val.setText(f"{self.sld_score.value()/100.0:.2f}")
        except Exception:
            pass
        try:
            self.ed_alpha_val.setText(f"{self.sld_alpha.value():.0f}")
        except Exception:
            pass
        # re-render current image from cache
        self._render_from_cache()

    def _on_score_edited(self):
        text = (self.ed_score_val.text() or "0").replace('%','')
        try:
            v = float(text)
        except Exception:
            v = self.sld_score.value()/100.0
        v = max(0.0, min(1.0, v))
        self.sld_score.setValue(int(round(v*100)))
        self._render_from_cache()

    def _on_alpha_edited(self):
        text = (self.ed_alpha_val.text() or "0").replace('%','')
        try:
            v = int(text)
        except Exception:
            v = self.sld_alpha.value()
        v = max(0, min(100, v))
        self.sld_alpha.setValue(v)
        self._render_from_cache()

    def _export_txt(self):
        # Export YOLO txt for each cached image result
        if not self._result_cache:
            QtWidgets.QMessageBox.information(self, "Export", "No inference results to export.")
            return
        exportable = [(path, data) for path, data in self._result_cache.items() if (data.get('boxes') or [])]
        if not exportable:
            QtWidgets.QMessageBox.information(self, "Export", "No detections available to export.")
            return
        from pathlib import Path
        written = 0
        for path, data in exportable:
            try:
                boxes = data.get('boxes', []) or []
                if path is None:
                    continue
                p = Path(path)
                # load image size
                img = QtGui.QImage(path)
                if img.isNull():
                    pm = QtGui.QPixmap(path)
                    W = pm.width(); H = pm.height()
                else:
                    W = img.width(); H = img.height()
                if W <= 0 or H <= 0:
                    continue
                lines = []
                names = data.get('names') or []
                for b in boxes:
                    x1,y1,x2,y2 = b.get('xyxy', (0,0,0,0))
                    # normalize
                    cx = ((x1 + x2) / 2.0) / float(W)
                    cy = ((y1 + y2) / 2.0) / float(H)
                    ww = abs(x2 - x1) / float(W)
                    hh = abs(y2 - y1) / float(H)
                    # class id
                    cid = b.get('cls')
                    if cid is None:
                        name = b.get('name')
                        try:
                            cid = names.index(name) if name in names else 0
                        except Exception:
                            cid = 0
                    try:
                        cid = int(cid)
                    except Exception:
                        cid = 0
                    lines.append(f"{cid} {cx:.6f} {cy:.6f} {ww:.6f} {hh:.6f}")
                out_txt = p.with_suffix('.txt')
                out_txt.write_text("\n".join(lines), encoding='utf-8')
                written += 1
            except Exception:
                pass
        QtWidgets.QMessageBox.information(self, "Export TXT", f"Exported TXT for {written} images.")

    def _export_json(self):
        # Aggregate all cached results into a single JSON file
        if not self._result_cache:
            QtWidgets.QMessageBox.information(self, "Export", "No inference results to export.")
            return
        exportable = [(path, data) for path, data in self._result_cache.items() if (data.get('boxes') or [])]
        if not exportable:
            QtWidgets.QMessageBox.information(self, "Export", "No detections available to export.")
            return
        from pathlib import Path
        import json
        # choose default directory
        default_dir = None
        if self._folder_root:
            default_dir = self._folder_root
        elif self._current_view_path:
            default_dir = str(Path(self._current_view_path).parent)
        else:
            default_dir = str(Path.cwd())
        save_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save inference_results.json",
            str(Path(default_dir) / "inference_results.json"),
            "JSON (*.json)"
        )
        if not save_path:
            return
        result = {}
        for path, data in exportable:
            try:
                p = Path(path)
                fname = p.name
                boxes = data.get('boxes', []) or []
                names = data.get('names', []) or []
                # image size metadata
                img = QtGui.QImage(path)
                if img.isNull():
                    pm = QtGui.QPixmap(path)
                    W = pm.width(); H = pm.height()
                else:
                    W = img.width(); H = img.height()
                dets = []
                for b in boxes:
                    x1,y1,x2,y2 = b.get('xyxy', (0,0,0,0))
                    cid = b.get('cls')
                    name = b.get('name')
                    conf = float(b.get('conf', 0.0)) if b.get('conf') is not None else 0.0
                    if cid is None:
                        try:
                            cid = names.index(name) if name in names else 0
                        except Exception:
                            cid = 0
                    try:
                        cid = int(cid)
                    except Exception:
                        cid = 0
                    dets.append({
                        'class_id': cid,
                        'class_name': name if name is not None else str(cid),
                        'confidence': conf,
                        'bbox': {'x1': float(x1), 'y1': float(y1), 'x2': float(x2), 'y2': float(y2)}
                    })
                result[fname] = {
                    'image_size': [int(W), int(H)],
                    'detections': dets,
                }
            except Exception:
                pass
        try:
            Path(save_path).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
            QtWidgets.QMessageBox.information(self, "Export JSON", f"Saved: {save_path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export JSON", f"Failed: {e}")
