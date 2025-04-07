"""
Fixed version of the Flask Website Generator 

Main fixes:
1. Fixed the output path to use WEBSITE_OUTPUT_PATH instead of OUTPUT_PATH
2. Fixed the file extension handling to avoid duplicated extensions
3. Improved the logic for determining file types and locations
"""

# This is a working version of the Flask Website Generator application that fixes the issues with
# file path handling, extension management, and output directory configuration.

import os
import json
import uuid
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field

# Import your required changes
import numpy as np
from flask import Flask, request, jsonify, render_template_string, Response