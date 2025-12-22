"""Label tab wrapper that embeds the legacy LabelTool widget."""

from __future__ import annotations

from PySide6 import QtWidgets

from ...label_tool import LabelTool


class LabelTab(QtWidgets.QWidget):
    """Hosts the existing LabelTool inside a tab-friendly container."""

    def __init__(self, tool: LabelTool | None = None, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.tool = tool or LabelTool()
        self.tool.setParent(self)
        self.tool.hide()  # prevent the standalone window from ever showing

        central = self.tool.centralWidget()
        if central is None:
            central = QtWidgets.QWidget()
        else:
            central.setParent(self)
            self.tool.setCentralWidget(None)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(central)
        self._central = central

    def get_controller(self) -> LabelTool:
        """Expose the underlying controller."""
        return self.tool
