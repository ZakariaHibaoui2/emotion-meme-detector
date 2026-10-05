import os
import cv2
import random


MEME_DIR = os.path.join(os.path.dirname(__file__), "memes")

CLASS_MEME_MAP = {
    "cat": ["cat_meme.jpg"],
    "dog": ["dog_meme.jpg"],
    "person": ["person_meme.jpg"],
    "tv": ["tv_meme.jpg"],
    "cell phone": ["phone_meme.jpg"],
    "pizza": ["pizza_meme.jpg"],
    "banana": ["banana_meme.jpg"],
    "car": ["car_meme.jpg"],
    "bird": ["bird_meme.jpg"],
    "horse": ["horse_meme.jpg"],
}

EMOTION_MEME_MAP = {
    "Happy": ["happy_cat_meme.jpg"],
    "Sad": ["sad_cat_meme.jpg"],
    "Angry": ["angry_cat_meme.jpg"],
    "Surprised": ["surprised_cat_meme.jpg"],
    "Fear": ["fear_cat_meme.jpg"],
    "Disgust": ["disgust_cat_meme.jpg"],
    "Neutral": ["neutral_cat_meme.jpg"],
}


class MemeOverlay:
    def __init__(self, size: tuple = (200, 200)):
        self.size = size
        self._cache: dict[str, list] = {}
        self._emotion_cache: dict[str, list] = {}

    def _load_meme(self, filename: str):
        path = os.path.join(MEME_DIR, filename)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        return cv2.resize(img, self.size)

    def get_meme_for_class(self, class_name: str):
        candidates = CLASS_MEME_MAP.get(class_name)
        if not candidates:
            return None
        key = f"{class_name}_{self.size[0]}x{self.size[1]}"
        if key not in self._cache:
            loaded = [self._load_meme(m) for m in candidates]
            self._cache[key] = [m for m in loaded if m is not None]
        if not self._cache[key]:
            return None
        return random.choice(self._cache[key])

    def get_meme_for_emotion(self, emotion: str):
        candidates = EMOTION_MEME_MAP.get(emotion)
        if not candidates:
            return None
        key = f"emotion_{emotion}_{self.size[0]}x{self.size[1]}"
        if key not in self._emotion_cache:
            loaded = [self._load_meme(m) for m in candidates]
            self._emotion_cache[key] = [m for m in loaded if m is not None]
        if not self._emotion_cache[key]:
            return None
        return random.choice(self._emotion_cache[key])

    def overlay(self, frame, meme_img):
        h, w = frame.shape[:2]
        mh, mw = meme_img.shape[:2]
        x1, y1 = w - mw - 10, 10
        x2, y2 = x1 + mw, y1 + mh

        if meme_img.shape[2] == 4:
            alpha = meme_img[:, :, 3] / 255.0
            for c in range(3):
                frame[y1:y2, x1:x2, c] = (
                    alpha * meme_img[:, :, c]
                    + (1 - alpha) * frame[y1:y2, x1:x2, c]
                )
        else:
            frame[y1:y2, x1:x2] = meme_img

        return frame

    @staticmethod
    def detected_classes(result):
        classes = set()
        if result.boxes is not None:
            for cls_id in result.boxes.cls:
                class_name = result.names.get(int(cls_id), "")
                classes.add(class_name)
        return classes
