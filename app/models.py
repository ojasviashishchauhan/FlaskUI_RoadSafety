from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from app import db, login_manager
from bson.objectid import ObjectId

class User(UserMixin):
    def __init__(self, username, email, password=None, _id=None, password_hash=None, is_admin=False):
        self.username = username
        self.email = email
        if password:
            self.password_hash = generate_password_hash(password)
        else:
            self.password_hash = password_hash  # Use existing hash if provided
        self._id = _id
        self.is_admin = is_admin # Add admin flag

    def check_password(self, password):
        if not self.password_hash:
            return False  # Cannot check password if hash is missing
        return check_password_hash(self.password_hash, password)

    def save(self):
        if not self._id:
            user_data = {
                'username': self.username,
                'email': self.email,
                'password_hash': self.password_hash,
                'is_admin': self.is_admin # Save admin flag
            }
            result = db.users.insert_one(user_data)
            self._id = result.inserted_id
        return self._id

    @staticmethod
    def get_by_username(username):
        user_data = db.users.find_one({'username': username})
        if user_data:
            return User(
                username=user_data['username'],
                email=user_data['email'],
                _id=user_data['_id'],
                password_hash=user_data.get('password_hash'),
                is_admin=user_data.get('is_admin', False) # Load admin flag
            )
        return None

    @staticmethod
    def get_by_email(email):
        user_data = db.users.find_one({'email': email})
        if user_data:
            return User(
                username=user_data['username'],
                email=user_data['email'],
                _id=user_data['_id'],
                password_hash=user_data.get('password_hash'),
                is_admin=user_data.get('is_admin', False) # Load admin flag
            )
        return None

    @staticmethod
    def get_by_id(user_id):
        try: # Add try-except for invalid ObjectId
            user_data = db.users.find_one({'_id': ObjectId(user_id)})
        except Exception: # Handle potential ObjectId conversion error
            return None
        if user_data:
            return User(
                username=user_data['username'],
                email=user_data['email'],
                _id=user_data['_id'],
                password_hash=user_data.get('password_hash'),
                is_admin=user_data.get('is_admin', False) # Load admin flag
            )
        return None

    def get_id(self):
        return str(self._id)

class Image:
    def __init__(self, filename, user_id, metadata=None, upload_time=None, _id=None):
        self.filename = filename
        self.user_id = user_id
        self.metadata = metadata or {}
        self.upload_time = upload_time
        self._id = _id

    def save(self):
        image_data = {
            'filename': self.filename,
            'user_id': ObjectId(self.user_id),
            'metadata': self.metadata,
            'upload_time': self.upload_time
        }
        result = db.images.insert_one(image_data)
        self._id = result.inserted_id
        return self._id

    @staticmethod
    def get_by_user_id(user_id):
        # Query for images where user_id matches either the string OR the ObjectId
        try:
            user_id_obj = ObjectId(user_id)
            query = {
                '$or': [
                    {'user_id': user_id},       # Match string ID (for older data)
                    {'user_id': user_id_obj}  # Match ObjectId (for newer data)
                ]
            }
            return list(db.images.find(query).sort('upload_time', -1))
        except Exception as e:
            # Fallback or handle error if ObjectId conversion fails (shouldn't for valid user_id)
            print(f"[ERROR] Error in get_by_user_id query for user {user_id}: {e}")
            # Attempt query with just string ID as fallback?
            try:
                 return list(db.images.find({'user_id': user_id}).sort('upload_time', -1))
            except Exception:
                 return [] # Return empty list if all fails

    @staticmethod
    def get_by_id(image_id):
        try: # Add try-except for invalid ObjectId
            return db.images.find_one({'_id': ObjectId(image_id)})
        except Exception:
            return None

    @staticmethod
    def delete_by_id(image_id):
        """Deletes an image record from the database by its ID."""
        try:
            result = db.images.delete_one({'_id': ObjectId(image_id)})
            return result.deleted_count > 0
        except Exception:
            return False

@login_manager.user_loader
def load_user(user_id):
    return User.get_by_id(user_id) 