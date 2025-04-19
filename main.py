"""
Main entry point for the Flask UI application.
This file redirects to the actual app entry point.
"""

import os
import sys

# Add the current directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the Flask app
from run import app

if __name__ == '__main__':
    app.run(debug=True)
