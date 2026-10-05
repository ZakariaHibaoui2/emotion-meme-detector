# Emotion Meme Detector — Real-Time YOLO + Facial Emotion Recognition

> A webcam app that detects objects with **YOLO11** and reads facial expressions with a **CNN emotion classifier**, then pops up a matching meme in real time. 😹

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO11-111F68)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

---

## How it works

```
Webcam frame ──► YOLO11 object detection ──► boxes + labels (person, cat, dog, phone, pizza…)
            └──► Face crop ──► Mini-Xception CNN (PyTorch) ──► 7 emotions
                                   Angry · Disgust · Fear · Happy · Sad · Surprised · Neutral
            ──► MemeOverlay: picks the meme mapped to the detected class or emotion and blends it into the frame
```

- **Detector** (`detector.py`): Ultralytics YOLO wrapper (CPU / CUDA / Apple MPS)
- **EmotionDetector** (`emotion_detector.py`): Xception-style CNN with depthwise-separable convolutions and residual blocks, loading pretrained weights from `models/emotion_model.t7`
- **MemeOverlay** (`meme_overlay.py`): class→meme and emotion→meme maps, alpha-blended overlay
- **generate_memes.py**: regenerates the placeholder meme images with OpenCV

## Run

```bash
git clone https://github.com/ZakariaHibaoui2/emotion-meme-detector.git
cd emotion-meme-detector
pip install -r requirements.txt

python main.py                      # webcam, CPU
python main.py --device cuda        # NVIDIA GPU
python main.py --source video.mp4 --conf 0.4
```

YOLO weights (`yolo11n.pt`) download automatically on first run. Press **q** to quit.

| Flag | Default | Description |
|---|---|---|
| `--model` | `yolo11n.pt` | YOLO model name or path |
| `--conf` | `0.5` | Detection confidence threshold |
| `--device` | `cpu` | `cpu`, `cuda` or `mps` |
| `--source` | `0` | Webcam index, or a video/image path |

## Author

**Zakaria Hibaoui** — [GitHub](https://github.com/ZakariaHibaoui2) · [LinkedIn](https://www.linkedin.com/in/zakaria-hibaoui-08b235337/)

## License

[MIT](LICENSE)
