import cv2
import numpy as np
import mediapipe as mp
import time
from collections import deque

# -----------------------------
# Camera performance settings
# -----------------------------
CAM_INDEX = 0
CAM_W, CAM_H = 1280, 720     # try 1280x720 for sharpness
FPS_TARGET = 30

# -----------------------------
# Drawing settings
# -----------------------------
BRUSH_MIN, BRUSH_MAX = 3, 35
brush_size = 10
draw_color = (0, 0, 255)     # default: red (BGR)
mode_text = "IDLE"

# Smoothing factor for tip motion (0..1) higher = smoother but more lag
SMOOTH_ALPHA = 0.35

# Pinch threshold (normalized distance in pixels later)
PINCH_ON = 40   # smaller = pinch detected (depends on resolution)
PINCH_OFF = 55  # hysteresis

# -----------------------------
# UI toolbar layout
# -----------------------------
TOOLBAR_H = 90
BTN_W = 180
BTN_GAP = 18
LEFT_PAD = 20

# Buttons: (label, color or None, type)
# type: "color", "eraser", "clear", "undo", "save"
buttons = [
    ("Red",   (0, 0, 255), "color"),
    ("Green", (0, 255, 0), "color"),
    ("Blue",  (255, 0, 0), "color"),
    ("Yellow",(0, 255, 255), "color"),
    ("Eraser",(0, 0, 0), "eraser"),
    ("Clean", None, "clear"),   # clean canvas
    ("Undo",  None, "undo"),
    ("Save",  None, "save"),
]

# Undo history
UNDO_MAX = 12
history = deque(maxlen=UNDO_MAX)

# -----------------------------
# MediaPipe hands
# -----------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# -----------------------------
# Helpers
# -----------------------------
def fingers_up(lm, w, h):
    """
    Returns booleans for [thumb, index, middle, ring, pinky] up/down.
    Basic orientation assumption: camera mirrored, palm facing camera.
    """
    # landmark indexes
    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]

    out = [False]*5

    # Thumb: compare x (works reasonably for mirrored front camera)
    out[0] = lm[tips[0]].x > lm[pips[0]].x  # may flip depending on hand; fine for our modes

    # Other fingers: tip.y < pip.y means up
    for i in range(1, 5):
        out[i] = lm[tips[i]].y < lm[pips[i]].y

    return out

def draw_toolbar(img):
    """Draw top toolbar and return list of button rectangles (x1,y1,x2,y2,type,label,color)."""
    h, w, _ = img.shape
    cv2.rectangle(img, (0, 0), (w, TOOLBAR_H), (35, 35, 35), -1)

    rects = []
    x = LEFT_PAD
    y1, y2 = 12, TOOLBAR_H - 12

    for label, color, btype in buttons:
        x1, x2 = x, x + BTN_W
        # button background
        if btype == "color":
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
            text_color = (0, 0, 0) if label in ("Yellow",) else (255, 255, 255)
        elif btype == "eraser":
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)
            text_color = (255, 255, 255)
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), (70, 70, 70), -1)
            text_color = (255, 255, 255)

        cv2.rectangle(img, (x1, y1), (x2, y2), (200, 200, 200), 2)
        cv2.putText(img, label, (x1+18, y2-22), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

        rects.append((x1, y1, x2, y2, btype, label, color))
        x = x2 + BTN_GAP

    # brush size indicator
    cv2.putText(img, f"Brush: {brush_size}", (w - 220, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    return rects

def inside_rect(x, y, rect):
    x1, y1, x2, y2, *_ = rect
    return x1 <= x <= x2 and y1 <= y <= y2

def save_canvas(canvas):
    ts = time.strftime("%Y%m%d_%H%M%S")
    name = f"air_canvas_{ts}.png"
    cv2.imwrite(name, canvas)
    return name

# -----------------------------
# Main
# -----------------------------
cap = cv2.VideoCapture(CAM_INDEX)

# Try to force better FPS / sharpness
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)

# On some webcams this helps latency
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

# Drawing canvas
ret, frame = cap.read()
if not ret:
    raise RuntimeError("Camera read failed.")
frame = cv2.flip(frame, 1)
h, w, _ = frame.shape
canvas = np.zeros((h, w, 3), dtype=np.uint8)

# Previous points for drawing
prev_x, prev_y = None, None
smooth_x, smooth_y = None, None
pinching = False

# For FPS display
prev_time = time.time()
fps = 0

print("Air Canvas Pro running.")
print("Controls:")
print("- Selection mode: Index + Middle up (2 fingers) -> choose tool from toolbar")
print("- Drawing mode: Index up only -> draw")
print("- Pinch (thumb+index) while drawing -> change brush size")
print("Press 'c' to clear, 'u' undo, 's' save, 'q' quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror
    frame = cv2.flip(frame, 1)
    frame_h, frame_w, _ = frame.shape

    # MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    # Toolbar
    overlay = frame.copy()
    rects = draw_toolbar(overlay)

    # Default mode
    mode_text = "IDLE"
    finger_tip = None

    if res.multi_hand_landmarks:
        hand = res.multi_hand_landmarks[0]
        lm = hand.landmark

        # draw landmarks lightly (optional)
        # mp_draw.draw_landmarks(overlay, hand, mp_hands.HAND_CONNECTIONS)

        up = fingers_up(lm, frame_w, frame_h)
        thumb_up, index_up, middle_up, ring_up, pinky_up = up

        # Index tip coords
        ix = int(lm[8].x * frame_w)
        iy = int(lm[8].y * frame_h)
        finger_tip = (ix, iy)

        # Smoothing
        if smooth_x is None:
            smooth_x, smooth_y = ix, iy
        else:
            smooth_x = int(SMOOTH_ALPHA * ix + (1 - SMOOTH_ALPHA) * smooth_x)
            smooth_y = int(SMOOTH_ALPHA * iy + (1 - SMOOTH_ALPHA) * smooth_y)

        sx, sy = smooth_x, smooth_y

        # Pinch detection (thumb tip + index tip distance)
        tx = int(lm[4].x * frame_w)
        ty = int(lm[4].y * frame_h)
        pinch_dist = int(((sx - tx)**2 + (sy - ty)**2) ** 0.5)

        # ----- Selection Mode (Index + Middle up) -----
        if index_up and middle_up and not ring_up and not pinky_up:
            mode_text = "SELECTION"
            prev_x, prev_y = None, None

            # highlight selection cursor
            cv2.circle(overlay, (sx, sy), 10, (255, 255, 255), -1)

            # If fingertip is inside toolbar, select tool
            if sy < TOOLBAR_H:
                for r in rects:
                    if inside_rect(sx, sy, r):
                        _, _, _, _, btype, label, col = r
                        if btype == "color":
                            draw_color = col
                        elif btype == "eraser":
                            draw_color = (0, 0, 0)
                        elif btype == "clear":
                            history.append(canvas.copy())
                            canvas[:] = 0
                        elif btype == "undo":
                            if len(history) > 0:
                                canvas = history.pop()
                        elif btype == "save":
                            name = save_canvas(canvas)
                            print(f"Saved: {name}")
                        time.sleep(0.18)  # debounce clicks
                        break

        # ----- Drawing Mode (Index only) -----
        elif index_up and not middle_up and not ring_up and not pinky_up:
            mode_text = "DRAW"

            # Brush size adjustment via pinch while drawing
            if not pinching and pinch_dist < PINCH_ON:
                pinching = True
            elif pinching and pinch_dist > PINCH_OFF:
                pinching = False

            if pinching:
                # Map pinch distance to brush size (smaller dist = bigger brush)
                # clamp and invert
                val = max(10, min(120, pinch_dist))
                # invert mapping
                mapped = int(np.interp(val, [10, 120], [BRUSH_MAX, BRUSH_MIN]))
                brush_size = int(np.clip(mapped, BRUSH_MIN, BRUSH_MAX))
                cv2.putText(overlay, "PINCH: Brush Size", (20, frame_h - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

            # draw pointer
            cv2.circle(overlay, (sx, sy), brush_size, draw_color, -1)

            # Save state occasionally before drawing strokes (for undo)
            if prev_x is None:
                history.append(canvas.copy())

            if prev_x is not None:
                cv2.line(canvas, (prev_x, prev_y), (sx, sy), draw_color, brush_size)

            prev_x, prev_y = sx, sy

        else:
            mode_text = "IDLE"
            prev_x, prev_y = None, None
            pinching = False

    else:
        prev_x, prev_y = None, None
        pinching = False
        smooth_x, smooth_y = None, None

    # Combine canvas with frame (clean blending)
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, inv = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)
    inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)

    base = cv2.bitwise_and(overlay, inv)
    out = cv2.bitwise_or(base, canvas)

    # Mode/FPS overlay
    now = time.time()
    fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, (now - prev_time)))
    prev_time = now

    cv2.putText(out, f"MODE: {mode_text}", (20, TOOLBAR_H + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
    cv2.putText(out, f"FPS: {int(fps)}", (20, TOOLBAR_H + 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

    cv2.imshow("Air Canvas Pro", out)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break
    elif key == ord('c'):
        history.append(canvas.copy())
        canvas[:] = 0
    elif key == ord('u'):
        if len(history) > 0:
            canvas = history.pop()
    elif key == ord('s'):
        name = save_canvas(canvas)
        print(f"Saved: {name}")

cap.release()
cv2.destroyAllWindows()
hands.close()
