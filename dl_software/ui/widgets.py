"""Reusable UI widgets used across tabs."""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt


class Header(QtWidgets.QWidget):
    """Section header with optional subtitle."""

    def __init__(self, title: str, subtitle: str = ""):
        super().__init__()
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 8, 8)

        ttl = QtWidgets.QLabel(title)
        ttl.setProperty("role", "header-title")
        ttl.setWordWrap(True)

        sub = QtWidgets.QLabel(subtitle)
        sub.setWordWrap(True)
        sub.setProperty("hint", "subtle")

        lay.addWidget(ttl)
        if subtitle:
            lay.addWidget(sub)


class VSep(QtWidgets.QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameShape(QtWidgets.QFrame.VLine)
        self.setFrameShadow(QtWidgets.QFrame.Sunken)


class HSep(QtWidgets.QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameShape(QtWidgets.QFrame.HLine)
        self.setFrameShadow(QtWidgets.QFrame.Sunken)


class TitledGroup(QtWidgets.QGroupBox):
    def __init__(self, title: str):
        super().__init__(title)
        font = self.font()
        font.setPointSize(18)
        font.setBold(True)
        self.setFont(font)


def configure_combo(combo: QtWidgets.QComboBox, *, show_all: bool = False) -> QtWidgets.QComboBox:
    """Ensure consistent sizing and scrollbar styling for combo boxes."""
    combo.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
    view = QtWidgets.QListView()
    view.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    view.setStyleSheet("QScrollBar:vertical { width: 14px; }")
    combo.setView(view)
    if show_all:
        count = combo.count()
        if count:
            combo.setMaxVisibleItems(count)
            row_height = view.sizeHintForRow(0)
            if row_height > 0:
                view.setMinimumHeight(row_height * count + view.frameWidth() * 2)
    return combo


def wrap_expanding(widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
    """Wrap a widget in an expanding container so it fills form columns."""
    container = QtWidgets.QWidget()
    container.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
    layout = QtWidgets.QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    layout.addWidget(widget)
    if not isinstance(widget, QtWidgets.QCheckBox):
        widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, widget.sizePolicy().verticalPolicy())
    else:
        layout.addStretch(1)
    return container


class CanvasWithFloatingTools(QtWidgets.QWidget):
    """Graphics view placeholder with floating palette used in the Skeleton."""

    def __init__(self):
        super().__init__()
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        self.view = QtWidgets.QGraphicsView()
        self.view.setScene(QtWidgets.QGraphicsScene())
        self.view.setBackgroundBrush(QtGui.QColor("#FFFFFF"))
        rect_pen = QtGui.QPen(QtGui.QColor("#C6CDD6"), 1, Qt.DashLine)
        self.view.scene().addRect(0, 0, 960, 600, rect_pen)
        lay.addWidget(self.view, 1)

        self.palette = QtWidgets.QFrame(self.view.viewport())
        self.palette.setObjectName("floatingPalette")
        self.palette.setAttribute(Qt.WA_StyledBackground, True)
        pl = QtWidgets.QVBoxLayout(self.palette)
        pl.setContentsMargins(8, 8, 8, 8)
        pl.setSpacing(4)

        def tool(text: str) -> QtWidgets.QToolButton:
            b = QtWidgets.QToolButton()
            b.setText(text)
            b.setCheckable(True)
            b.setToolButtonStyle(Qt.ToolButtonTextOnly)
            b.setCursor(Qt.PointingHandCursor)
            b.setFixedWidth(40)
            return b

        for t in ("✋", "▭", "🔺", "🧹"):
            pl.addWidget(tool(t))
        pl.addWidget(self._hsep())
        for t in ("↶", "↷"):
            pl.addWidget(tool(t))
        pl.addWidget(self._hsep())
        for t in ("◀", "▶"):
            pl.addWidget(tool(t))
        pl.addWidget(self._hsep())
        pl.addWidget(tool("💾"))
        pl.addStretch(1)

        shadow = QtWidgets.QGraphicsDropShadowEffect(self.palette)
        shadow.setBlurRadius(22)
        shadow.setOffset(0, 4)
        shadow.setColor(QtGui.QColor(0, 0, 0, 160))
        self.palette.setGraphicsEffect(shadow)
        self.view.viewport().installEventFilter(self)

    def _hsep(self) -> QtWidgets.QFrame:
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setStyleSheet("color: rgba(255,255,255,90);")
        return line

    def resizeEvent(self, e: QtGui.QResizeEvent):
        super().resizeEvent(e)
        self._position_palette()

    def showEvent(self, e: QtGui.QShowEvent):
        super().showEvent(e)
        QtCore.QTimer.singleShot(0, self._position_palette)

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if obj is self.view.viewport() and event.type() == QtCore.QEvent.Resize:
            self._position_palette()
        return super().eventFilter(obj, event)

    def _position_palette(self):
        vp = self.view.viewport()
        if vp.width() <= 0 or vp.height() <= 0:
            return
        self.palette.adjustSize()
        w = self.palette.sizeHint().width()
        h = self.palette.sizeHint().height()
        self.palette.move(vp.width() - w - 12, max(0, (vp.height() - h) // 2))
