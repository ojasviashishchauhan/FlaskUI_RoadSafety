from flask import Flask
from flask_login import LoginManager
from flask_mongoengine import MongoEngine
from flask_bootstrap import Bootstrap
import os
from dotenv import load_dotenv
import datetime
import re # Import re for regex

# --- Custom Jinja Filter Definition ---
# Define the function *before* it's used
def do_regex_search(value, pattern):
    """Apply regex search in Jinja template."""
    if value is None: value = '' # Handle None values
    return re.search(pattern, str(value)) # Ensure value is string
# -------------------------------------

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
Bootstrap(app)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-for-testing')
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 2048 * 1024 * 1024  # 2 GB max upload
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'heic', 'heif'}  # Add supported image formats

# MongoDB configuration using Flask-MongoEngine
app.config['MONGODB_SETTINGS'] = {
    'db': 'flask_ui_db',      # Database name
    'host': 'localhost',     # Hostname
    'port': 27017            # Port
}

# Initialize MongoEngine
db = MongoEngine()
db.init_app(app)

# Initialize login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Context Processor to inject current year
@app.context_processor
def inject_now():
    return {'now': datetime.datetime.now()}

# Register the custom filter *after* defining the function
app.jinja_env.filters['regex_search'] = do_regex_search

# Import routes AFTER initializing extensions
from app import routes 
from app import models # Import models to ensure they are registered with MongoEngine 