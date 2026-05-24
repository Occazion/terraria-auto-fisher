#!/usr/bin/env python3
"""AutoFisher entry point.

The app is split across several modules — see CLAUDE.md for the layout.
This file only configures logging, clears temp patterns from previous runs,
and spins up the Qt main loop.
"""

import logging
import sys

from PyQt6.QtWidgets import QApplication

from paths import cleanup_temp_patterns
from ui import AppUi


__version__ = '2'
__author__ = 'Yehor Bondarchuk'


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    cleanup_temp_patterns()
    app = QApplication(sys.argv)
    window = AppUi(version=__version__)
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
