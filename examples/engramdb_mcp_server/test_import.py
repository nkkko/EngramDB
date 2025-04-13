#!/usr/bin/env python3

import sys
print(f"Python version: {sys.version}")
print(f"Python path: {sys.path}")

try:
    import engramdb_py
    print(f"Successfully imported engramdb_py: {engramdb_py}")
except ImportError as e:
    print(f"Error importing engramdb_py: {e}")

try:
    import engramdb
    print(f"Successfully imported engramdb: {engramdb}")
except ImportError as e:
    print(f"Error importing engramdb: {e}")