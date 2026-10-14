#!/usr/bin/env python3
"""Vertical Jump: a webcam jump-height game built on OpenCV + MediaPipe Pose.

Jump height comes from flight time (h = g * t^2 / 8), the same method used by
camera-based tools like the My Jump app, so no distance calibration is needed.
Takeoff and landing are found where the lowest foot point crosses a small
threshold above the standing ground line, interpolated between frames.

Run:   .venv/bin/python jump_game.py                 # webcam
       .venv/bin/python jump_game.py --video clip.mov  # recorded clip (slo-mo = better accuracy)
Keys:  Q/Esc quit   R reset best   S toggle skeleton   M toggle mirror   C switch camera
"""
import argparse
import json
import math
import random
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

HERE = Path(__file__).resolve().parent
MODEL = HERE / "models" / "pose_landmarker_lite.task"
SCORE_FILE = HERE / "highscore.json"
WINDOW = "Vertical Jump"
G = 9.81

# MediaPipe pose landmark indices
L_HIP, R_HIP = 23, 24
L_ANKLE, R_ANKLE = 27, 28
FOOT_POINTS = (29, 30, 31, 32)  # heels + toes
REQUIRED = (L_HIP, R_HIP, L_ANKLE, R_ANKLE, *FOOT_POINTS)
SKELETON = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
            (23, 25), (25, 27), (24, 26), (26, 28), (27, 29), (29, 31), (27, 31),
            (28, 30), (30, 32), (28, 32)]

# BGR colors
WHITE = (255, 255, 255)
GREY = (170, 170, 170)
GOLD = (40, 200, 255)
GREEN = (90, 230, 120)
CYAN = (255, 210, 60)
RED = (80, 80, 255)
FONT = cv2.FONT_HERSHEY_DUPLEX

RATINGS = [(70, "LEGENDARY"), (60, "ELITE"), (50, "EXPLOSIVE"), (40, "ATHLETIC"),
           (30, "SOLID"), (20, "NICE"), (0, "WARM UP")]
RESULT_SECS = 4.0


class JumpDetector:
    """State machine over (ground_y, hip_y, leg_len) pixel observations.

    SEARCHING -> CALIBRATING (stand still) -> READY -> AIRBORNE -> back to CALIBRATING.
    Heights are expressed in leg lengths so thresholds work at any camera distance.
    """
    STILL_SECS = 0.8   # how long you must stand still to (re)set the ground line
    STILL_STD = 0.012  # allowed jitter (std dev) while "still", in leg lengths
    TAKEOFF = 0.08     # feet this far above ground => definitely airborne
    CONTACT = 0.025    # flight is timed between crossings of this level
    LEG_M = 0.87       # typical hip-to-ankle length; converts CONTACT to meters
    MIN_FLIGHT, MAX_FLIGHT = 0.12, 1.2
    LOST_GRACE = 0.4   # tolerate brief tracking dropouts mid-air

    def __init__(self):
        self.reset()

    def reset(self):
        self.state = "SEARCHING"
        self.hist = deque()
        self.ground = self.hip = self.leg = None
        self.progress = 0.0
        self.last_seen = -1e9

    def rise(self, sample):
        return (self.ground - sample[1]) / self.leg

    def update(self, t, obs):
        """obs = (ground_y, hip_y, leg_len) in pixels or None. Returns a jump dict on landing."""
        if obs is None:
            if self.state == "AIRBORNE" and t - self.last_seen < self.LOST_GRACE:
                return None
            if self.state != "SEARCHING":
                self.reset()
            return None

        self.last_seen = t
        self.hist.append((t, *obs))
        while t - self.hist[0][0] > 2.5:
            self.hist.popleft()

        if self.state == "AIRBORNE":
            return self._airborne(t)

        still = self._still_window()
        if self.state in ("SEARCHING", "CALIBRATING"):
            self.state = "CALIBRATING"
            if still:
                self._set_ground(still)
                self.state = "READY"
            return None

        # READY
        if still:
            self._set_ground(still)
        r = self.rise(self.hist[-1])
        if r > self.TAKEOFF:
            self.takeoff_t = self._takeoff_time()
            self.peak_rise = r
            self.peak_hip = 0.0
            self.state = "AIRBORNE"
        elif r < -self.TAKEOFF:  # walked toward the camera; ground line is stale
            self.state = "CALIBRATING"
        return None

    def _still_window(self):
        """Trailing samples whose feet/hip jitter stays small, if they span STILL_SECS."""
        t_now, _, _, leg = self.hist[-1]
        tol2 = (self.STILL_STD * leg) ** 2
        n = sg = sg2 = sh = sh2 = 0.0
        start = len(self.hist) - 1
        for i in range(len(self.hist) - 1, -1, -1):
            _, g, h, _ = self.hist[i]
            n += 1
            sg, sg2, sh, sh2 = sg + g, sg2 + g * g, sh + h, sh2 + h * h
            if sg2 / n - (sg / n) ** 2 > tol2 or sh2 / n - (sh / n) ** 2 > tol2:
                break
            start = i
        span = t_now - self.hist[start][0]
        self.progress = min(1.0, span / self.STILL_SECS)
        return list(self.hist)[start:] if span >= self.STILL_SECS else None

    def _set_ground(self, window):
        arr = np.array(window)
        self.ground = float(np.median(arr[:, 1]))
        self.hip = float(np.median(arr[:, 2]))
        self.leg = float(np.median(arr[:, 3]))

    def _crossing(self, a, b):
        """Interpolated time where rise passes CONTACT between samples a and b."""
        ra, rb = self.rise(a), self.rise(b)
        if ra == rb:
            return b[0]
        f = (self.CONTACT - ra) / (rb - ra)
        return a[0] + min(max(f, 0.0), 1.0) * (b[0] - a[0])

    def _takeoff_time(self):
        h = self.hist
        for i in range(len(h) - 2, -1, -1):
            if self.rise(h[i]) <= self.CONTACT:
                return self._crossing(h[i], h[i + 1])
        return h[0][0]

    def _airborne(self, t):
        cur = self.hist[-1]
        r = self.rise(cur)
        self.peak_rise = max(self.peak_rise, r)
        self.peak_hip = max(self.peak_hip, (self.hip - cur[2]) / self.leg)

        if t - self.takeoff_t > self.MAX_FLIGHT:  # not a jump (walked away, etc.)
            self.state = "CALIBRATING"
            return None
        if r > self.CONTACT:
            return None

        land_t = self._crossing(self.hist[-2], cur) if len(self.hist) > 1 else t
        flight = land_t - self.takeoff_t
        self.state = "CALIBRATING"
        if not (self.MIN_FLIGHT <= flight <= self.MAX_FLIGHT) or self.peak_hip < self.TAKEOFF / 2:
            return None
        # Time spent above level c on a parabola of apex h gives exactly h - c, so add c back.
        cm = (G * flight ** 2 / 8 + self.CONTACT * self.LEG_M) * 100
        return {"cm": cm, "flight": flight}


def observe(lms, w, h):
    """Pull (ground_y, hip_y, leg_len) in pixels from pose landmarks, if the full body is visible."""
    if lms is None:
        return None
    for i in REQUIRED:
        lm = lms[i]
        if (lm.visibility or 0) < 0.5 or not (0.0 <= lm.y <= 1.0):
            return None
    ground = max(lms[i].y for i in FOOT_POINTS) * h
    hip_x = (lms[L_HIP].x + lms[R_HIP].x) / 2 * w
    hip_y = (lms[L_HIP].y + lms[R_HIP].y) / 2 * h
    ank_x = (lms[L_ANKLE].x + lms[R_ANKLE].x) / 2 * w
    ank_y = (lms[L_ANKLE].y + lms[R_ANKLE].y) / 2 * h
    leg = math.hypot(hip_x - ank_x, hip_y - ank_y)
    return (ground, hip_y, leg) if leg > 40 else None


# ---------- drawing helpers ----------

def text(img, s, org, scale, color, thick=2, anchor="left"):
    (w, h), _ = cv2.getTextSize(s, FONT, scale, thick)
    x, y = org
    if anchor == "center":
        x -= w // 2
    elif anchor == "right":
        x -= w
    off = max(2, int(round(scale * 2)))
    cv2.putText(img, s, (x + off, y + off), FONT, scale, (0, 0, 0), thick, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), FONT, scale, color, thick, cv2.LINE_AA)
    return w, h


def panel(img, x1, y1, x2, y2, alpha=0.6, color=(18, 18, 18)):
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(img.shape[1], int(x2)), min(img.shape[0], int(y2))
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    roi[:] = (roi * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)


def rating(cm):
    return next(label for floor, label in RATINGS if cm >= floor)


class Game:
    def __init__(self):
        self.det = JumpDetector()
        self.best = self._load_best()
        self.jumps = []
        self.result = None       # (jump dict, shown_at, is_new_best)
        self.particles = []
        self.show_skeleton = True
        self.fps = deque(maxlen=30)
        self.last_t = None

    # ----- persistence -----
    def _load_best(self):
        try:
            return float(json.loads(SCORE_FILE.read_text())["best_cm"])
        except (OSError, ValueError, KeyError):
            return 0.0

    def _save_best(self):
        SCORE_FILE.write_text(json.dumps({"best_cm": round(self.best, 1)}))

    def reset_best(self):
        self.best = 0.0
        self.jumps.clear()
        self._save_best()

    # ----- per frame -----
    def step(self, t, frame, lms):
        H, W = frame.shape[:2]
        dt = 0.0 if self.last_t is None else t - self.last_t
        if dt > 0:
            self.fps.append(1.0 / dt)
        self.last_t = t

        jump = self.det.update(t, observe(lms, W, H))
        if jump:
            new_best = jump["cm"] > self.best
            if new_best:
                self.best = jump["cm"]
                self._save_best()
                self._burst(W, H)
            self.jumps.append(jump)
            self.result = (jump, t, new_best)
        if self.result and t - self.result[1] > RESULT_SECS:
            self.result = None

        self.draw(frame, t, dt, lms)

    def _burst(self, W, H):
        colors = [GOLD, GREEN, CYAN, RED, WHITE]
        for _ in range(160):
            a = random.uniform(0, 2 * math.pi)
            v = random.uniform(200, 750)
            self.particles.append([W / 2, H * 0.42, math.cos(a) * v, math.sin(a) * v - 300,
                                   random.choice(colors), random.uniform(1.2, 2.2)])

    # ----- rendering -----
    def draw(self, img, t, dt, lms):
        H, W = img.shape[:2]
        s = H / 720
        det = self.det

        if self.show_skeleton and lms is not None:
            pts = [(int(l.x * W), int(l.y * H)) for l in lms]
            col = GREEN if det.state == "AIRBORNE" else CYAN
            for a, b in SKELETON:
                cv2.line(img, pts[a], pts[b], col, max(2, int(3 * s)), cv2.LINE_AA)
            for i in range(11, 33):
                cv2.circle(img, pts[i], max(3, int(4 * s)), WHITE, -1, cv2.LINE_AA)
        if det.ground is not None and det.state in ("READY", "AIRBORNE"):
            gy = int(det.ground)
            cv2.line(img, (0, gy), (W, gy), GREEN, max(1, int(2 * s)), cv2.LINE_AA)

        self._draw_top(img, s)
        self._draw_meter(img, s, t)
        self._draw_particles(img, dt)
        if self.result:
            self._draw_result(img, s, t)
        else:
            self._draw_status(img, s, t)
        self._draw_bottom(img, s)

    def _draw_top(self, img, s):
        W = img.shape[1]
        panel(img, 0, 0, W, 64 * s, 0.65)
        text(img, "VERTICAL JUMP", (int(20 * s), int(44 * s)), 1.1 * s, WHITE, 2)
        text(img, f"JUMPS {len(self.jumps)}", (W // 2, int(44 * s)), 0.9 * s, GREY, 2, "center")
        text(img, f"BEST {self.best:.1f} cm", (W - int(20 * s), int(44 * s)), 1.1 * s, GOLD, 2, "right")

    def _draw_meter(self, img, s, t):
        H, W = img.shape[:2]
        top, bot = int(110 * s), H - int(110 * s)
        x1, x2 = W - int(70 * s), W - int(40 * s)
        max_cm = max(80, math.ceil(self.best * 1.15 / 10) * 10)
        to_y = lambda cm: int(bot - (bot - top) * min(cm, max_cm) / max_cm)

        panel(img, x1 - 50 * s, top - 30 * s, x2 + 25 * s, bot + 20 * s, 0.5)
        cv2.rectangle(img, (x1, top), (x2, bot), GREY, 1, cv2.LINE_AA)
        for cm in range(0, max_cm + 1, 10):
            y = to_y(cm)
            cv2.line(img, (x1 - int(6 * s), y), (x1, y), GREY, 1, cv2.LINE_AA)
            text(img, str(cm), (x1 - int(10 * s), y + int(5 * s)), 0.4 * s, GREY, 1, "right")

        if self.result:
            jump, shown, new_best = self.result
            val = jump["cm"] * min(1.0, (t - shown) / 0.6)
            fill = GOLD if new_best else GREEN
        elif self.jumps:
            val, fill = self.jumps[-1]["cm"], GREEN
        else:
            val, fill = 0.0, GREEN
        if val > 0:
            cv2.rectangle(img, (x1 + 2, to_y(val)), (x2 - 2, bot - 1), fill, -1)
        if self.best > 0:
            y = to_y(self.best)
            cv2.line(img, (x1 - int(4 * s), y), (x2 + int(4 * s), y), GOLD, max(2, int(3 * s)), cv2.LINE_AA)
            text(img, "BEST", ((x1 + x2) // 2, y - int(8 * s)), 0.4 * s, GOLD, 1, "center")
        text(img, "cm", ((x1 + x2) // 2, top - int(10 * s)), 0.5 * s, GREY, 1, "center")

    def _draw_status(self, img, s, t):
        H, W = img.shape[:2]
        det = self.det
        cx, cy = W // 2, int(H * 0.2)
        if det.state == "SEARCHING":
            text(img, "STEP BACK", (cx, cy), 1.6 * s, WHITE, 3, "center")
            text(img, "whole body, feet included, in frame", (cx, cy + int(45 * s)), 0.8 * s, GREY, 2, "center")
        elif det.state == "CALIBRATING":
            text(img, "STAND STILL", (cx, cy), 1.6 * s, WHITE, 3, "center")
            bw, bh = int(320 * s), int(14 * s)
            by = cy + int(30 * s)
            cv2.rectangle(img, (cx - bw // 2, by), (cx + bw // 2, by + bh), GREY, 1, cv2.LINE_AA)
            cv2.rectangle(img, (cx - bw // 2, by), (cx - bw // 2 + int(bw * det.progress), by + bh), CYAN, -1)
        elif det.state == "READY":
            pulse = 1.0 + 0.08 * math.sin(t * 8)
            text(img, "JUMP!", (cx, cy + int(10 * s)), 2.6 * s * pulse, GREEN, 5, "center")
        elif det.state == "AIRBORNE":
            text(img, "AIRBORNE", (cx, cy), 1.8 * s, GOLD, 4, "center")

    def _draw_result(self, img, s, t):
        H, W = img.shape[:2]
        jump, shown, new_best = self.result
        age = t - shown
        cm = jump["cm"] * min(1.0, age / 0.6)  # count-up animation
        cx, cy = W // 2, int(H * 0.42)

        panel(img, cx - 330 * s, cy - 170 * s, cx + 330 * s, cy + 150 * s, 0.7)
        if new_best and int(age * 4) % 2 == 0:
            text(img, "NEW HIGH SCORE!", (cx, cy - int(115 * s)), 1.3 * s, GOLD, 3, "center")
        elif not new_best:
            text(img, rating(jump["cm"]), (cx, cy - int(115 * s)), 1.2 * s, CYAN, 3, "center")
        text(img, f"{cm:.1f} cm", (cx, cy + int(20 * s)), 3.2 * s, GOLD if new_best else WHITE, 6, "center")
        text(img, f"{cm / 2.54:.1f} in   |   air time {jump['flight']:.2f} s",
             (cx, cy + int(75 * s)), 0.85 * s, GREY, 2, "center")
        if new_best:
            text(img, rating(jump["cm"]), (cx, cy + int(122 * s)), 0.9 * s, CYAN, 2, "center")
        else:
            gap = self.best - jump["cm"]
            text(img, f"{gap:.1f} cm off your best", (cx, cy + int(122 * s)), 0.8 * s, GREY, 2, "center")

    def _draw_particles(self, img, dt):
        H, W = img.shape[:2]
        alive = []
        for p in self.particles:
            p[0] += p[2] * dt
            p[1] += p[3] * dt
            p[3] += 900 * dt
            p[5] -= dt
            if p[5] > 0 and 0 <= p[0] < W and p[1] < H:
                cv2.circle(img, (int(p[0]), int(p[1])), 5, p[4], -1, cv2.LINE_AA)
                alive.append(p)
        self.particles = alive

    def _draw_bottom(self, img, s):
        H, W = img.shape[:2]
        panel(img, 0, H - 50 * s, W, H, 0.65)
        recent = "  ".join(f"{j['cm']:.1f}" for j in self.jumps[-6:][::-1])
        text(img, f"RECENT  {recent}" if recent else "RECENT  -", (int(20 * s), H - int(17 * s)),
             0.65 * s, WHITE, 1)
        fps = np.mean(self.fps) if self.fps else 0
        text(img, f"{fps:.0f} fps   R reset   S skeleton   M mirror   C camera   Q quit",
             (W - int(20 * s), H - int(17 * s)), 0.55 * s, GREY, 1, "right")


def open_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 60)
    return cap


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=1,
                    help="webcam index (on this Mac: 1 = MacBook camera, 0 = iPhone Continuity Camera)")
    ap.add_argument("--video", help="analyze a recorded clip instead of the webcam")
    ap.add_argument("--fps", type=float, help="override clip frame rate (e.g. 240 for slo-mo)")
    args = ap.parse_args()

    if not MODEL.exists():
        raise SystemExit(f"Missing pose model at {MODEL}. Download pose_landmarker_lite.task into models/.")

    live = args.video is None
    cam = args.camera
    if live:
        cap = open_camera(cam)
        if not cap.isOpened() and cam != 0:  # e.g. iPhone not connected, so the Mac camera is index 0
            cam = 0
            cap = open_camera(cam)
    else:
        cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit("Could not open camera/video. On macOS, allow camera access for your terminal/VS Code "
                         "in System Settings > Privacy & Security > Camera.")
    clip_fps = args.fps or cap.get(cv2.CAP_PROP_FPS) or 30.0

    landmarker = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ))

    game = Game()
    mirror = live
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    t0, frame_idx, last_ms = time.perf_counter(), 0, -1

    while True:
        ok, frame = cap.read()
        if not ok:
            if live:
                continue
            break
        t = time.perf_counter() - t0 if live else frame_idx / clip_fps
        frame_idx += 1
        if mirror:
            frame = cv2.flip(frame, 1)

        ts_ms = max(int(t * 1000), last_ms + 1)
        last_ms = ts_ms
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms)
        lms = result.pose_landmarks[0] if result.pose_landmarks else None

        game.step(t, frame, lms)
        cv2.imshow(WINDOW, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("r"):
            game.reset_best()
        elif key == ord("s"):
            game.show_skeleton = not game.show_skeleton
        elif key == ord("c") and live:
            cap.release()
            cam = cam + 1
            cap = open_camera(cam)
            if not cap.isOpened():
                cam = 0
                cap = open_camera(cam)
            game.det.reset()
        elif key == ord("m"):
            mirror = not mirror

    if not live:
        print("Jumps:", ", ".join(f"{j['cm']:.1f} cm ({j['flight']:.3f} s)" for j in game.jumps) or "none")
        cv2.waitKey(0)
    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
