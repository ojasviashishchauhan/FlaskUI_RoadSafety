from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, EmailField, SelectField
from wtforms.validators import DataRequired, Email, EqualTo, Length, ValidationError
from wtforms import MultipleFileField
from wtforms.validators import InputRequired
from flask_wtf.file import FileAllowed
from app.models import User
from app.utils import allowed_file
from werkzeug.utils import secure_filename

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
        user = User.get_by_username(username.data)
        if user:
            raise ValidationError('Username already exists. Please choose a different one.')

    def validate_email(self, email):
        user = User.get_by_email(email.data)
        if user:
            raise ValidationError('Email already registered. Please use a different one.')

class ImageUploadForm(FlaskForm):
    images = MultipleFileField('Select Images', validators=[DataRequired()])
    image_type = SelectField('Image Type', choices=[
        ('Potholes', 'Potholes'),
        ('Longitudinal', 'Longitudinal Cracks'),
        ('Transverse', 'Transverse Cracks'),
        ('Alligator', 'Alligator Cracks')
    ], validators=[DataRequired()])
    submit = SubmitField('Upload Images')

    def validate_images(self, field):
        if not field.data:
            raise ValidationError('Please select at least one file.')
        for upload in field.data:
            filename = secure_filename(upload.filename)
            if not allowed_file(filename):
                raise ValidationError('Invalid file type.') 