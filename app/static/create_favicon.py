from PIL import Image, ImageDraw, ImageFont
import os

# Create a new image with a white background
size = (32, 32)
img = Image.new('RGB', size, 'white')
draw = ImageDraw.Draw(img)

# Draw a simple road icon
draw.rectangle([8, 14, 24, 18], fill='black')  # Road
draw.polygon([12, 8, 20, 8, 24, 14, 8, 14], fill='red')  # Warning triangle

# Save as ICO
img.save('favicon.ico', format='ICO', sizes=[(32, 32)]) 