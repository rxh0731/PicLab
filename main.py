"""图片实验室独立版应用入口。"""

from __future__ import annotations

import sys

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QMainWindow

import config
from data.log_manager import LogManager
from ui.pages.image_lab_page import ImageLabPage
from ui.theme import apply_theme


class ImageLabWindow(QMainWindow):
    """图片实验室主窗口。"""

    def __init__(self) -> None:
        super().__init__()
        self._page_confirmed_close = False
        self.setWindowTitle("图片实验室")
        self.setMinimumSize(1100, 720)
        self.resize(1440, 900)
        icon = QIcon(config.WINDOW_ICON_FILE)
        if not icon.isNull():
            self.setWindowIcon(icon)
        self.page = ImageLabPage(self)
        self.page.home_requested.connect(self._close_from_page)
        self.setCentralWidget(self.page)

    def _close_from_page(self) -> None:
        """页面已经完成未保存内容确认，直接关闭窗口。"""

        self._page_confirmed_close = True
        self.close()

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if not self._page_confirmed_close and not self.page.confirm_close():
            event.ignore()
            return
        self.page.shutdown()
        super().closeEvent(event)


def create_application(arguments: list[str] | None = None) -> QApplication:
    """创建图片实验室 Qt 应用。"""

    app = QApplication(arguments if arguments is not None else sys.argv)
    app.setApplicationDisplayName("图片实验室")
    app.setApplicationName("图片实验室")
    icon = QIcon(config.WINDOW_ICON_FILE)
    if not icon.isNull():
        app.setWindowIcon(icon)
    apply_theme(app)
    return app


def main() -> int:
    """启动独立图片实验室。"""

    log_manager = LogManager()
    log_manager.open()
    log_manager.write("正在启动图片实验室独立版")
    app = create_application()
    window = ImageLabWindow()
    window.showMaximized()
    exit_code = app.exec()
    log_manager.write(f"图片实验室退出，退出码：{exit_code}")
    log_manager.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
