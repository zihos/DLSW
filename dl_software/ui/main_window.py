"""Main window composed of styled tabs."""

from __future__ import annotations

import os
import subprocess

from PySide6 import QtCore, QtGui, QtWidgets

from .styles import APP_QSS
from .tabs import InferTab, LabelTab, TrainTab
from ..label_tool import Project, ProjectDialog


class DLMainWindow(QtWidgets.QMainWindow):
    """Tabbed main window used by the DL software."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vision Label · Train · Infer")
        self.resize(1400, 900)

        mb = self.menuBar()
        menu_project = mb.addMenu("Project")
        menu_settings = mb.addMenu("Settings")
        menu_help = mb.addMenu("Help")
        act_open_project = menu_project.addAction("Open Project…")
        act_open_project.triggered.connect(self._open_project_dialog)
        menu_settings.addAction("Preferences…")
        act_about = menu_help.addAction("Wiki")
        act_about.triggered.connect(self._open_about)

        tabs = QtWidgets.QTabWidget()
        tabs.setDocumentMode(True)
        tabs.setMovable(True)
        tabs.setElideMode(QtCore.Qt.ElideRight)
        self.tabs = tabs

        self.label_tab = LabelTab()
        self.label_tool = self.label_tab.get_controller()
        self.label_tool.set_status_bar_proxy(self.statusBar())
        self.train_tab = TrainTab()
        self.infer_tab = InferTab()

        tabs.addTab(self.label_tab, "Label")
        tabs.addTab(self.train_tab, "Train")
        tabs.addTab(self.infer_tab, "Infer")

        self.start_page = self._build_start_page()
        self.pages = QtWidgets.QStackedWidget()
        self.pages.addWidget(self.start_page)
        self.pages.addWidget(self.tabs)
        # 시작 시에는 프로젝트 선택/생성 페이지로 진입해 흐름을 명확히 분리한다.
        self.pages.setCurrentWidget(self.start_page)
        self.setCentralWidget(self.pages)

        self._wire_tabs()
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.statusBar().showMessage("Ready")
        self._refresh_project_list()

    def _open_about(self):
        """Open the project wiki in the default browser."""
        url = "https://github.com/zihos/DLSW/wiki"

        # Launch xdg-open with a cleaned env and silenced stderr to avoid noisy GTK warnings from snap Firefox.
        env = os.environ.copy()
        env.pop("GTK_MODULES", None)
        env.pop("GTK_PATH", None)
        env.pop("GTK2_RC_FILES", None)
        try:
            subprocess.Popen(
                ["xdg-open", url],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
                start_new_session=True,
            )
        except Exception:
            # Fallback to Qt handler if xdg-open is unavailable.
            QtGui.QDesktopServices.openUrl(QtCore.QUrl(url))

    def _build_start_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(20)

        page.setStyleSheet(
            """
            QWidget#startPage {
                background: #0f172a;
                color: #e5e7eb;
            }
            QLabel#title {
                font-size: 26px;
                font-weight: 700;
                color: #e5e7eb;
            }
            QLabel#subtitle {
                color: #cbd5e1;
                font-size: 13px;
            }
            QLabel[class~="section"] {
                color: #cbd5e1;
                font-weight: 600;
                letter-spacing: 0.4px;
            }
            QPushButton[class~="start"] {
                background: #2563eb;
                color: white;
                font-weight: 600;
                border: none;
                padding: 10px 16px;
                border-radius: 6px;
                min-width: 160px;
            }
            QPushButton[class~="start"]:hover { background: #1d4ed8; }
            QPushButton[class~="start"]:pressed { background: #1e40af; }
            QPushButton[class~="start"][class~="secondary"] {
                background: transparent;
                color: #cbd5e1;
                border: 1px solid #475569;
            }
            QPushButton[class~="start"][class~="secondary"]:hover { background: rgba(255,255,255,0.04); }
            QPushButton[class~="link"] {
                border: none;
                background: transparent;
                color: #93c5fd;
                text-align: left;
                padding: 6px 4px;
                font-size: 13px;
            }
            QPushButton[class~="link"]:hover {
                color: #bfdbfe;
                text-decoration: underline;
                background: rgba(255,255,255,0.05);
            }
            """
        )
        page.setObjectName("startPage")

        title = QtWidgets.QLabel("Vision Label")
        title.setObjectName("title")
        layout.addWidget(title)

        subtitle = QtWidgets.QLabel("프로젝트를 선택하거나 새로 만들어서 시작하세요.")
        subtitle.setWordWrap(True)
        subtitle.setObjectName("subtitle")
        layout.addWidget(subtitle)

        # Start actions
        lbl_start = QtWidgets.QLabel("START")
        lbl_start.setProperty("class", "section")
        layout.addWidget(lbl_start)

        start_row = QtWidgets.QHBoxLayout()
        start_row.setSpacing(12)
        self.btn_open_project = QtWidgets.QPushButton("Open Project…")
        self.btn_open_project.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn_open_project.setProperty("class", "start")
        self.btn_open_project.setObjectName("btnOpenProject")
        self.btn_open_project.setStyleSheet("")  # ensure style sheet applies

        self.btn_new_project = QtWidgets.QPushButton("Create Project…")
        self.btn_new_project.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn_new_project.setProperty("class", "start secondary")
        self.btn_new_project.setStyleSheet("")

        start_row.addWidget(self.btn_open_project)
        start_row.addWidget(self.btn_new_project)
        start_row.addStretch(1)
        layout.addLayout(start_row)

        # Recent projects
        lbl_recent = QtWidgets.QLabel("RECENT")
        lbl_recent.setProperty("class", "section")
        layout.addWidget(lbl_recent)

        self.recent_container = QtWidgets.QWidget()
        recent_layout = QtWidgets.QVBoxLayout(self.recent_container)
        recent_layout.setContentsMargins(0, 0, 0, 0)
        recent_layout.setSpacing(4)
        self.recent_layout = recent_layout
        layout.addWidget(self.recent_container)

        self.btn_open_project.clicked.connect(lambda: self._open_project_dialog(allow_create=False))
        self.btn_new_project.clicked.connect(self._create_project_prompt)

        layout.addStretch(1)
        return page

    def _refresh_project_list(self):
        if not hasattr(self, "recent_layout"):
            return
        # clear previous buttons/labels
        while self.recent_layout.count():
            it = self.recent_layout.takeAt(0)
            w = it.widget()
            if w:
                w.deleteLater()

        projects = self.label_tool.project_manager.list_projects()[:5]
        if not projects:
            empty = QtWidgets.QLabel("최근 프로젝트가 없습니다.")
            empty.setStyleSheet("color:#94a3b8;")
            self.recent_layout.addWidget(empty)
            return

        for project in projects:
            btn = QtWidgets.QPushButton(f"{project.name} — {project.meta.created_at.split('T')[0]}")
            btn.setFlat(True)
            btn.setCursor(QtCore.Qt.PointingHandCursor)
            btn.setProperty("class", "link")
            btn.setStyleSheet("")
            btn.clicked.connect(lambda _, p=project: self._on_project_selected(p))
            self.recent_layout.addWidget(btn)
        self.recent_layout.addStretch(1)

    def _open_project_dialog(self, allow_create: bool = True):
        dlg = ProjectDialog(self.label_tool.project_manager, self, allow_create=allow_create)
        if dlg.exec() != QtWidgets.QDialog.Accepted or dlg.selected_project is None:
            return
        self._on_project_selected(dlg.selected_project)

    def _create_project_prompt(self):
        name, ok = QtWidgets.QInputDialog.getText(self, "Create Project", "Project name:")
        if not ok:
            return
        name = name.strip()
        if not name:
            QtWidgets.QMessageBox.warning(self, "Invalid name", "프로젝트 이름을 입력하세요.")
            return
        try:
            project = self.label_tool.project_manager.create_project(name)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Create failed", str(e))
            return
        self._on_project_selected(project)

    def _on_project_selected(self, project: Project):
        self.label_tool._load_project(project)
        self.statusBar().showMessage(f"Project: {project.name}")
        self.pages.setCurrentWidget(self.tabs)
        self.tabs.setCurrentWidget(self.label_tab)
        self._refresh_project_list()

    def _wire_tabs(self):
        """Connect tab controls to the shared controller."""
        # Export Dataset lives in Label tab now; no Train tab button wiring

        # Label tab workflow: Train button opens Train tab with embedded params
        self.label_tool.btn_train.clicked.connect(lambda: self.tabs.setCurrentWidget(self.train_tab))

        self.infer_tab.btn_save_overlay.setText("Run Inference")
        self.infer_tab.set_label_tool(self.label_tool)
        self.infer_tab.btn_save_overlay.clicked.connect(self.infer_tab.run_inference)
        if hasattr(self.infer_tab, 'btn_export_json'):
            self.infer_tab.btn_export_json.setDisabled(False)

    def _on_tab_changed(self, idx: int):
        w = self.tabs.widget(idx)
        if w is self.infer_tab:
            mode = self.infer_tab.cmb_mode.currentText() if hasattr(self.infer_tab, 'cmb_mode') else ""
            # Do not auto-mirror Label tab image. Only restore explicit Infer context.
            try:
                if mode == "Folder" and getattr(self.infer_tab, "_folder_images", None):
                    imgs = self.infer_tab._folder_images
                    if len(imgs) > 0:
                        idx_cur = getattr(self.infer_tab, "_folder_idx", 0)
                        if not isinstance(idx_cur, int) or idx_cur < 0:
                            idx_cur = 0
                        idx_cur = max(0, min(idx_cur, len(imgs)-1))
                        self.infer_tab.set_preview_from_path(imgs[idx_cur])
                        return
                if mode == "Single Image":
                    p_single = self.infer_tab.ed_path.text().strip()
                    if p_single:
                        self.infer_tab.set_preview_from_path(p_single)
                        return
                # For Current Image / Video / Webcam or empty state: leave canvas as-is (placeholder)
            except Exception:
                pass


def create_app() -> tuple[QtWidgets.QApplication, DLMainWindow]:
    """Utility to build the QApplication + main window ready to show."""
    app = QtWidgets.QApplication([])
    app.setStyleSheet(APP_QSS)
    win = DLMainWindow()
    return app, win
