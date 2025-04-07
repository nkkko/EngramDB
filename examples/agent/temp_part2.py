# Summary of fixes and improvements

# 1. Path Configuration Fixes
# - Changed OUTPUT_PATH to use WEBSITE_OUTPUT_PATH environment variable instead of OUTPUT_PATH
# - Fixed default path to use "/tmp/generated_flask_website" instead of "generated_website"

# 2. Extension Handling Fixes
# - Added logic to avoid duplicating file extensions (.py.py, .html.html, .css.css, etc.)
# - Fixed handling of special component names like "style" and "script"
# - Improved the classification logic for determining where files should be stored

# 3. Route Registration Logic Fixes
# - Updated route registration to use the same improved logic for determining file paths
# - Added improved MIME type detection for static files based on extensions

# Complete changes:
# 1. Changed ENGRAMDB_PATH = os.environ.get("ENGRAMDB_PATH", "agent_memory.engramdb")
#    OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "generated_website")
#    to:
#    ENGRAMDB_PATH = os.environ.get("ENGRAMDB_PATH", "agent_memory.engramdb")
#    OUTPUT_PATH = os.environ.get("WEBSITE_OUTPUT_PATH", "/tmp/generated_flask_website")

# 2. Updated file path determination logic in save_files_to_disk() to properly handle files with or without extensions
# 3. Added improved handling for specific component types (style -> CSS, script -> JS)
# 4. Updated route registration to use the same improved logic for file path determination