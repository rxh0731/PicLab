"""图片实验室独立版运行配置。"""

from __future__ import annotations

import os
import sys

if getattr(sys, "frozen", False):
    SCRIPT_DIR = os.path.dirname(os.path.abspath(sys.executable))
else:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

RESOURCE_DIR = SCRIPT_DIR
LOG_FILE = os.path.join(SCRIPT_DIR, "piclab.log")
LOG_MAX_BYTES = 2 * 1024 * 1024
WINDOW_ICON_FILE = os.path.join(RESOURCE_DIR, "assets", "piclab_icon.png")

