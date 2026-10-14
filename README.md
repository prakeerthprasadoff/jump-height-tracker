# Jump Height Tracker

A webcam game that measures how high you jump. Stand in front of your camera, jump, and it shows your height, air time, and all-time best.

Built with OpenCV and MediaPipe Pose.

## How it works

The jump height is calculated from **flight time**: `h = g·t² / 8`. This is the same method used by the validated *My Jump* app. Because it only needs the time between takeoff and landing, the result doesn't depend on how far you stand from the camera, so there's nothing to calibrate.

1. MediaPipe tracks your feet and hips in each frame.
2. When you stand still for about a second, the game marks the ground line under your feet.
3. Takeoff and landing are the moments your lowest foot point crosses a small height above that line. These times are interpolated between frames for better precision.
4. Flight time gives the jump height. A correction adds back the small threshold height.

## Setup

Requires Python 3.10 to 3.12.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python jump_game.py
```

The pose model (`models/pose_landmarker_lite.task`, about 6 MB) is included in the repo.

On macOS, the first run asks for camera permission for your terminal or editor.

## Playing

1. Step back until your whole body, feet included, is in the frame.
2. Stand still until **JUMP!** appears.
3. Jump. Your result shows when you land. Your best score is saved to `highscore.json`.

| Key | Action |
| --- | --- |
| `C` | Switch to the next camera (for example, built-in camera ↔ iPhone Continuity Camera) |
| `R` | Reset best score |
| `S` | Show/hide the skeleton |
| `M` | Mirror the view on/off |
| `Q` / `Esc` | Quit |

### Options

```bash
.venv/bin/python jump_game.py --camera 0                  # choose a camera by index
.venv/bin/python jump_game.py --video clip.mov --fps 240  # analyze a recorded (slo-mo) clip
```

## Accuracy

Accuracy is limited mostly by the frame rate:

| Camera | Simulated error |
| --- | --- |
| 30 fps webcam | up to about ±4 cm |
| 60 fps | under 1 cm |
| 240 fps slo-mo clip | under 1 cm |

For the most accurate results, record a slo-mo clip on your phone and use `--video`.

Tucking your legs before landing makes your air time longer and inflates the score. Land with straight legs for a fair number.
