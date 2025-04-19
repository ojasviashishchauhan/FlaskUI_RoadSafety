# simple_heic_test.py
from PIL import Image
from pillow_heif import register_heif_opener
import sys

register_heif_opener()

heic_path = '/Users/ojasvi/PycharmProjects/flaskUI/IMG_3381.HEIC' # Use the actual path
jpeg_path = '/Users/ojasvi/PycharmProjects/flaskUI/jpg.jpg'

try:
    print(f"Attempting to open: {heic_path}")
    img = Image.open(heic_path)
    print(f"Successfully opened: {img.format} {img.size} {img.mode}")
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img.save(jpeg_path, "JPEG")
    print(f"Successfully converted and saved to: {jpeg_path}")
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
