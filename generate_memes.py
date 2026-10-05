import os
import cv2
import numpy as np

MEME_DIR = os.path.join(os.path.dirname(__file__), "memes")
os.makedirs(MEME_DIR, exist_ok=True)

MEME_DEFS = {
    "cat_meme.jpg": ("Cat Meme", (200, 200), (255, 200, 100)),
    "dog_meme.jpg": ("Dog Meme", (200, 200), (180, 220, 255)),
    "person_meme.jpg": ("Person Meme", (200, 200), (100, 255, 180)),
    "tv_meme.jpg": ("TV Meme", (200, 200), (200, 100, 255)),
    "phone_meme.jpg": ("Phone Meme", (200, 200), (100, 200, 255)),
    "pizza_meme.jpg": ("Pizza Meme", (200, 200), (100, 255, 255)),
    "banana_meme.jpg": ("Banana Meme", (200, 200), (255, 255, 100)),
    "car_meme.jpg": ("Car Meme", (200, 200), (200, 200, 255)),
    "bird_meme.jpg": ("Bird Meme", (200, 200), (255, 180, 180)),
    "horse_meme.jpg": ("Horse Meme", (200, 200), (180, 255, 180)),
}

for filename, (text, size, color) in MEME_DEFS.items():
    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    img[:] = color
    font_scale = 1.2
    thickness = 3
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
    x = (size[0] - tw) // 2
    y = (size[1] + th) // 2
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_DUPLEX,
                font_scale, (0, 0, 0), thickness)
    cv2.putText(img, "PLACEHOLDER", (10, size[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 50, 50), 1)
    path = os.path.join(MEME_DIR, filename)
    cv2.imwrite(path, img)
    print(f"Created {path}")

print("\nDone! Replace these placeholder images with real memes in the memes/ folder.")
