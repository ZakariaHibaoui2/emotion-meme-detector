import argparse
import cv2
from detector import Detector
from emotion_detector import EmotionDetector
from meme_overlay import MemeOverlay, CLASS_MEME_MAP, EMOTION_MEME_MAP


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time YOLO object detection")
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO model name or path")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"],
                        help="Inference device")
    parser.add_argument("--source", default="0",
                        help="Video source: 0 for webcam, or path to video/image")
    parser.add_argument("--memes", action="store_true", default=True,
                        help="Enable meme overlays (default: on)")
    parser.add_argument("--emotion-memes", action="store_true", default=True,
                        help="Show emotion-based cat memes (default: on)")
    return parser.parse_args()


def main():
    args = parse_args()

    detector = Detector(model_name=args.model, device=args.device)
    emotion_detector = EmotionDetector() if args.emotion_memes else None
    meme_overlay = MemeOverlay() if args.memes else None

    print(f"Object memes: {', '.join(CLASS_MEME_MAP.keys())}")
    print(f"Emotion memes: {', '.join(EMOTION_MEME_MAP.keys())}")

    if args.source == "0":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(args.source)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    window_name = "Object Detection [Q to quit]"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = detector.detect(frame, conf=args.conf)
        annotated = detector.draw(frame, result)

        if meme_overlay:
            detected = MemeOverlay.detected_classes(result)
            for cls in detected:
                meme_img = meme_overlay.get_meme_for_class(cls)
                if meme_img is not None:
                    annotated = meme_overlay.overlay(annotated, meme_img)

        if emotion_detector:
            emotion, confidence = emotion_detector.detect_emotion(annotated)
            if emotion:
                meme_img = meme_overlay.get_meme_for_emotion(emotion)
                if meme_img is not None:
                    annotated = meme_overlay.overlay(annotated, meme_img)
                cv2.putText(annotated, f"{emotion} {confidence:.0%}", (10, 30),
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow(window_name, annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
