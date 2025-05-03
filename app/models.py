from datetime import datetime
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from app import login_manager
from bson import ObjectId
from mongoengine import Document, StringField, DateTimeField, DictField, FloatField, ListField, ObjectIdField, BooleanField

class User(Document, UserMixin):
    username = StringField(required=True, unique=True)
    email = StringField(required=True, unique=True)
    password_hash = StringField(required=True)
    role = StringField(default='user', choices=['user', 'admin'])
    created_at = DateTimeField(default=datetime.utcnow)

    meta = {
        'collection': 'users',
        'indexes': [
            'username',
            'email'
        ],
        'strict': False
    }

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def get_id(self):
        return str(self.id)

    @property
    def is_admin(self):
        return self.role == 'admin'

class Image(Document):
    filename = StringField()
    original_filename = StringField()
    file_path = StringField()
    user_id = ObjectIdField(required=True)
    upload_time = DateTimeField(default=datetime.utcnow)
    media_type = StringField(default='image', choices=['image', 'video'])
    completion_time = DateTimeField()
    processing_status = StringField(default='pending')
    error_message = StringField()
    image_type = StringField()
    metadata = DictField(default={})
    location = DictField(default={})
    prediction_results = DictField(default={})
    confidence_score = FloatField()
    processing_time = FloatField()
    annotated_image_path = StringField()

    meta = {
        'collection': 'images',
        'indexes': [
            'user_id',
            'upload_time',
            'processing_status',
            'filename',
            'media_type'
        ],
        'strict': False
    }

    def to_dict(self):
        """Convert document to dictionary suitable for templates"""
        data = self.to_mongo().to_dict()
        data['id'] = str(data.pop('_id'))
        
        if isinstance(data.get('upload_time'), datetime):
            data['upload_time'] = data['upload_time'].isoformat()
        if isinstance(data.get('completion_time'), datetime):
            data['completion_time'] = data['completion_time'].isoformat()
            
        if 'user_id' in data and isinstance(data['user_id'], ObjectId):
             data['user_id'] = str(data['user_id'])

        return data

@login_manager.user_loader
def load_user(user_id):
    try:
        return User.objects(id=user_id).first()
    except Exception as e:
        print(f"Error loading user {user_id}: {e}")
        return None 