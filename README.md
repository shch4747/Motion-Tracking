# Markerless Body Motion Tracking — Webcam Game Controller

A real-time full-body pose estimation system that maps physical gestures to keyboard inputs, enabling control of games using body movements. No specialist hardware required — runs on a standard webcam at ~30 FPS.

## How It Works

MediaPipe's pose estimation model detects 33 skeletal landmarks per frame. The system tracks the relative positions of key joints (wrists, shoulders, hips, ankles) across frames and maps specific pose transitions to keyboard events via `pynput`.

### Detected gestures and mapped keys

| Gesture | Detection Logic | Key |
|---|---|---|
| Left hand raised | Left wrist y < left shoulder y | `A` |
| Right hand raised | Right wrist y < right shoulder y | `D` |
| Both hands raised | Both wrists above both shoulders | `Space` |
| Duck | Hip y drops > 5% below calibrated baseline | `S` |
| Jump | Ankle y decreases by > 1% frame-over-frame | `W` |
| Reset | Wrists horizontally separated by > 5% of frame width | Recalibrates baseline |

Y-coordinates in normalized MediaPipe space decrease upward, so "above" means a smaller y value.

### Calibration

On the first frame, the system records the standing hip position as a baseline. Duck detection is relative to this baseline, making it robust to different camera heights and body sizes without manual configuration.

## Setup

```bash
pip install opencv-python mediapipe pynput
```

```bash
python main.py
```

Stand in frame, wait for the first frame to calibrate, then use gestures to control the game. Press `q` to quit.

## Tunable Parameters

At the top of `main.py`:

```python
min_detection_confidence = 0.5   # MediaPipe landmark detection threshold
min_tracking_confidence  = 0.5   # MediaPipe tracking threshold across frames
duck_threshold           = 0.05  # Hip drop fraction to trigger duck
jump_threshold           = 0.01  # Ankle rise fraction per frame to trigger jump
```

Increase thresholds to reduce false positives; decrease for more sensitive detection. Values may need adjustment based on camera distance and lighting.

## Tested With

- Subway Surfers (browser)
- Chrome Dino game

The gesture-to-key mapping is game-agnostic — any game controllable via W/A/S/D/Space works without modification.

## Tech

Python, MediaPipe, OpenCV, pynput
