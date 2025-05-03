from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, EmailField, SelectField
from wtforms.validators import DataRequired, Email, EqualTo, Length, ValidationError
from wtforms import MultipleFileField
from wtforms.validators import InputRequired
from flask_wtf.file import FileAllowed
from app.models import User
from app.utils import allowed_file
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage

# Define allowed video extensions
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv'} # Add more as needed
# Define allowed image extensions
ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'heic', 'heif'}

def allowed_video_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS

def allowed_image_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=3, max=25)])
    email = EmailField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6, max=40)])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Register')

    def validate_username(self, username):
        user = User.objects(username=username.data).first() # Use MongoEngine query
        if user:
            raise ValidationError('Username already exists. Please choose a different one.')

    def validate_email(self, email):
        user = User.objects(email=email.data).first() # Use MongoEngine query
        if user:
            raise ValidationError('Email already registered. Please use a different one.')

class ImageUploadForm(FlaskForm):
    images = MultipleFileField('Select Images (up to 100)', validators=[DataRequired()])
    image_type = SelectField('Damage Type', choices=[
        ('Potholes', 'Potholes'),
        ('Longitudinal', 'Longitudinal Cracks'),
        ('Transverse', 'Transverse Cracks'),
        ('Alligator', 'Alligator Cracks'),
        ('Edge', 'Edge Cracks'),
        ('Reflection', 'Reflection Cracks'),
        ('All', 'All Damage Types')
    ], validators=[DataRequired()], default='Potholes')
    submit = SubmitField('Upload Images')

    def validate_images(self, field):
        if not field.data:
            raise ValidationError('Please select at least one file.')
            
        # Check if too many files are being uploaded
        if len(field.data) > 100:
            raise ValidationError('You can upload a maximum of 100 images at once.')
            
        invalid_files = []
        for upload in field.data:
            if not upload.filename:
                continue
            filename = secure_filename(upload.filename)
            if not allowed_image_file(filename):
                invalid_files.append(filename)
                
        if invalid_files:
            if len(invalid_files) > 5:
                # Show only the first few invalid files to avoid overly long error messages
                file_list = ", ".join(invalid_files[:5]) + f" and {len(invalid_files) - 5} more"
            else:
                file_list = ", ".join(invalid_files)
                
            raise ValidationError(f'Invalid image file type(s): {file_list}. Allowed types: {", ".join(ALLOWED_IMAGE_EXTENSIONS)}')

# New Video Upload Form
class VideoUploadForm(FlaskForm):
    videos = MultipleFileField('Select Videos (up to 100)', validators=[DataRequired()])
    # Updated damage type selection field with expanded options
    image_type = SelectField('Damage Type to Focus On', choices=[
        ('Potholes', 'Potholes'),
        ('Longitudinal', 'Longitudinal Cracks'),
        ('Transverse', 'Transverse Cracks'),
        ('Alligator', 'Alligator Cracks'),
        ('Edge', 'Edge Cracks'),
        ('Reflection', 'Reflection Cracks'),
        ('All', 'All Damage Types')
    ], validators=[DataRequired()], default='All')
    submit = SubmitField('Upload Videos')

    def validate_videos(self, field):
        print(f"--- Starting validate_videos --- Data type: {type(field.data)}")
        if not field.data:
            raise ValidationError('Please select at least one video file.')
            
        # Check if too many files are being uploaded
        if len(field.data) > 100:
            raise ValidationError('You can upload a maximum of 100 videos at once.')
            
        invalid_files = []
        for i, upload in enumerate(field.data):
            print(f"Validating video item {i}: {repr(upload)}, Type: {type(upload)}")
            if isinstance(upload, FileStorage) and upload.filename:
                filename = secure_filename(upload.filename)
                if not allowed_video_file(filename):
                    invalid_files.append(filename)
            elif isinstance(upload, str) and not upload:
                print(f"Warning: Skipping item {i} because it is an empty string.")
                continue
            elif not getattr(upload, 'filename', None):
                # Skip items with no filename
                continue
            else:
                print(f"ERROR: Unexpected item type {type(upload)} found in videos field: {repr(upload)}")
                raise ValidationError('Unexpected data received during video validation.')
                
        if invalid_files:
            if len(invalid_files) > 5:
                # Show only the first few invalid files to avoid overly long error messages
                file_list = ", ".join(invalid_files[:5]) + f" and {len(invalid_files) - 5} more"
            else:
                file_list = ", ".join(invalid_files)
                
            raise ValidationError(f'Invalid video file type(s): {file_list}. Allowed types: {", ".join(ALLOWED_VIDEO_EXTENSIONS)}') 