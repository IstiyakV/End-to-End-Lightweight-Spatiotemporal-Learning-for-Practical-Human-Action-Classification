"""
HAR Control Center — Launch GUI
Usage: python gui.py
"""

import sys
import multiprocessing
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gui.app import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
