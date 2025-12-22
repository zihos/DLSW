"""Application bootstrap."""

from __future__ import annotations

import sys

from PySide6 import QtWidgets

from .ui.main_window import create_app


def run_app() -> int:
    """Run the DL software application."""
    app, win = create_app()
    win.show()
    return app.exec()


def main():
    sys.exit(run_app())


if __name__ == "__main__":
    main()
