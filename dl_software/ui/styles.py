"""Application-wide style definitions."""

APP_QSS = """
* { font-family: 'Noto Sans','Inter','Arial'; font-size: 14px; }
QMainWindow { background: #F5F6F8; }

/* ======= VSCode-like Menu Bar ======= */
QMenuBar { background:#FFFFFF; border-bottom:1px solid #C9D0D9; }
QMenuBar::item { padding:6px 10px; margin:0 2px; color:#111827; }
QMenuBar::item:selected { background:#EEF2FF; color:#111827; }
QMenu { background:#FFFFFF; border:1px solid #CBD3DE; }
QMenu::item { padding:6px 16px; color:#111827; }
QMenu::item:selected { background:#EEF2FF; }

/* ======= Chrome-like Tabs (tight, no gaps) ======= */
QTabWidget::pane {
  margin-top: 0;
  border: 1px solid #C9D0D9;
  border-radius: 8px;
  background: #FFFFFF;
}
QTabBar {
  qproperty-drawBase: 0;
  background: #EEF1F6;
  border: 0;
  padding-top: 0;
  padding-left: 4px;
}
QTabBar::tab {
  padding: 8px 16px;
  margin: 0;
  border: 1px solid #C9D0D9;
  border-bottom: none;
  border-top-left-radius: 12px;
  border-top-right-radius: 12px;
  border-bottom-left-radius: 0;
  border-bottom-right-radius: 0;
  background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #F5F6F8, stop:1 #ECEFF3);
  color: #343B46;
  font-weight: 700;
  font-size: 13.5px;
}
QTabBar::tab + QTabBar::tab { margin-left: -1px; }
QTabBar::tab:hover { background: #F9FAFB; }
QTabBar::tab:selected {
  background: #FFFFFF;
  color: #0F172A;
  border-color: #AEB7C3;
  border-bottom: none;
}

/* ======= Big Section Header (text only: bigger + bolder) ======= */
QLabel[role="header-title"] {
  font-size: 20px;
  font-weight: 900;
  color: #0B1220;
}

/* ======= Group Panels & Titles (text only: bigger + bolder, no bg) ======= */
QGroupBox {
  background:#FFFFFF;
  border:1px solid #D7DCE3;
  border-radius:8px;
  margin-top:22px;
  padding-top:14px;
}
QGroupBox::title {
  subcontrol-origin: margin;
  subcontrol-position: top left;
  left: 1px;
  padding: 0 8px;

  color:#0B1220;
  font-weight: 900;
  font-size: 20px;
  letter-spacing: 0.2px;
}

/* Lists / Tables / Text */
QListWidget, QTreeView, QTableView, QTextEdit, QPlainTextEdit {
  background:#FFFFFF; border:1px solid #D7DCE3; border-radius:6px; outline:0;
  selection-background-color: rgba(37,99,235,0.12);
  selection-color:#111827;
}
QHeaderView::section {
  background:#F4F6FA;
  border:none;
  border-bottom:1px solid #D7DCE3;
  padding:6px 8px;
  color:#374151;
  font-weight: 800;
}

/* Inputs */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
  background:#FFFFFF; border:1px solid #D7DCE3; border-radius:6px;
  padding:6px 8px; color:#111827;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
  border:2px solid #2563EB;
  padding:5px 7px;
}

/* Slider */
QSlider::groove:horizontal { height:6px; background:#E5E7EB; border-radius:3px; margin:8px 10px; }
QSlider::handle:horizontal {
  width:18px; height:18px; margin:-7px 0;
  background:#2563EB; border-radius:9px; border:1px solid rgba(0,0,0,0.06);
}

/* Buttons */
QPushButton { background:#2563EB; color:#FFFFFF; border:none; border-radius:8px; padding:8px 12px; }
QPushButton:hover { background:#1E55C9; }
QPushButton:disabled { background:#AEB6C2; color:#FFFFFF; }

/* Subtle hint labels */
QLabel[hint='subtle'] { color:#4B5563; }

/* Floating dark palette */
#floatingPalette { background:rgba(20,25,35,220); border-radius:10px; }
#floatingPalette QToolButton { color:#E7E9EE; font-size:14px; border:none; padding:8px 6px; border-radius:6px; }
#floatingPalette QToolButton:hover { background:rgba(255,255,255,0.08); }
#floatingPalette QToolButton:checked { background:rgba(37,99,235,0.22); color:#AFC5FF; border:1px solid rgba(37,99,235,0.55); }
#floatingPalette QLabel, #floatingPalette QFrame { color:rgba(255,255,255,0.65); }
"""
