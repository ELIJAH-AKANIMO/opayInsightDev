"""Generate a small placeholder demo GIF illustrating the UI steps.
Usage: python scripts/make_demo_gif.py
This will create webapp/demo.gif
"""
from PIL import Image, ImageDraw, ImageFont
import os

os.makedirs('webapp', exist_ok=True)
frames = []
width, height = 640, 360
bg = (30, 41, 59)
font_color = (255, 255, 255)
try:
    font = ImageFont.truetype('arial.ttf', 24)
except Exception:
    from PIL import ImageFont
    font = ImageFont.load_default()

texts = [
    'Fraud Detection UI - Demo',
    '1) Load sample data',
    '2) Click Score (top alerts shown)',
    '3) Inspect SHAP and mark reviewed'
]

for t in texts:
    im = Image.new('RGB', (width, height), color=bg)
    d = ImageDraw.Draw(im)
    w, h = d.textsize(t, font=font)
    d.text(((width - w) / 2, (height - h) / 2), t, font=font, fill=font_color)
    frames.append(im)

out_path = 'webapp/demo.gif'
frames[0].save(out_path, format='GIF', append_images=frames[1:], save_all=True, duration=900, loop=0)
print('Created', out_path)