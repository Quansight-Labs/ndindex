# Make sure the doctests in the docs can import the module
import sys
sys.path.insert(0, '.')

collect_ignore = ["setup.py", "docs/conf.py"]
