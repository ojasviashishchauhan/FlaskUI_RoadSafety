# Road AI Safety

A Flask web application that allows users to upload images, extract metadata including EXIF data and GPS coordinates, and visualize the image locations on a map.

## Features

- User authentication (signup, login, logout)
- Image upload and storage
- Automatic extraction of image metadata (size, format, date taken, GPS coordinates)
- Interactive map view of image locations
- CSV export of metadata
- Mobile-responsive UI
- get mongodb database by -> mongorestore--uri="mongodb://localhost:27017/"--db=flask_ui_db./dump/flask_ui_db

## Technology Stack

- **Backend**: Flask, Python
- **Database**: MongoDB
- **Frontend**: Bootstrap 5, Leaflet.js for maps
- **Authentication**: Flask-Login
- **Forms**: Flask-WTF
- **Image Processing**: Pillow
- **Data Export**: Pandas

## Installation

1. Clone the repository:

   ```
   git clone <repository-url>
   cd flaskUI
   ```
2. Create and activate a virtual environment:

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:

   ```
   pip install -r requirements.txt
   ```
4. Set up environment variables:
   Create a `.env` file in the project root with the following:

   ```
   SECRET_KEY=your_secret_key
   MONGO_URI=mongodb://localhost:27017/
   ```
5. Make sure MongoDB is running locally or update the MONGO_URI in `.env`

## Running the Application

```
python run.py
```

The application will be available at http://127.0.0.1:5000/

## Project Structure

```
flaskUI/
├── app/
│   ├── __init__.py         # Flask app initialization
│   ├── routes.py           # Route definitions
│   ├── models.py           # Database models
│   ├── forms.py            # Form definitions
│   ├── utils.py            # Utility functions
│   ├── static/             # Static files (CSS, JS)
│   │   └── css/
│   │       └── style.css   # Custom CSS
│   └── templates/          # HTML templates
│       ├── base.html       # Base template
│       ├── index.html      # Homepage
│       ├── login.html      # Login page
│       ├── register.html   # Registration page
│       ├── dashboard.html  # User dashboard
│       ├── upload.html     # Upload form
│       ├── image_details.html # Image details
│       └── map.html        # Map view
├── uploads/                # Uploaded images storage
├── run.py                  # Application entry point
├── main.py                 # Alternative entry point
├── requirements.txt        # Dependencies
└── README.md               # Project documentation
```

## Future Enhancements

- Multiple image uploads
- Image search functionality
- Filter images by date/location
- Image editing capabilities
- Admin panel for user management
- Social sharing options
