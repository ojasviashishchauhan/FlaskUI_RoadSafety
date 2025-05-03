try:
    from flask import Flask
    from flask_login import LoginManager
    from flask_mongoengine import MongoEngine
    from flask_bootstrap import Bootstrap
    print("All imports are working correctly!")
except ImportError as e:
    print(f"Import error: {e}") 