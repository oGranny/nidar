from picamera2 import Picamera2
from libcamera import controls
import cv2
import numpy as np
import time

# ===================== CONFIG =====================
BOARD_SIZE = (9, 6)          # Inner corners (columns, rows)
SQUARE_SIZE_MM = 25.4        # Physical chessboard square size
MIN_CAPTURES = 25
RESOLUTION = (1280, 720)     # Use SAME resolution as flight
# =================================================

# Prepare object points (0,0,0), (1,0,0) ...
objp = np.zeros((BOARD_SIZE[0] * BOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:BOARD_SIZE[0], 0:BOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE_MM

objpoints = []
imgpoints = []

criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)

# ===================== CAMERA SETUP =====================
picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"size": RESOLUTION, "format": "RGB888"}
)
picam2.configure(config)

# Manual focus + locked AF (important)
picam2.set_controls({
    "AfMode": controls.AfModeEnum.Manual,
    "LensPosition": 0.0,   # Infinity focus
    "AeEnable": True
})

picam2.start()
time.sleep(1)
# ======================================================

print("\nControls:")
print("  c → capture frame")
print("  q → calibrate & quit\n")

img_shape = None

while True:
    frame = picam2.capture_array()
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    if img_shape is None:
        img_shape = gray.shape[::-1]

    found, corners = cv2.findChessboardCorners(gray, BOARD_SIZE, None)

    display = frame.copy()

    if found:
        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria
        )
        cv2.drawChessboardCorners(display, BOARD_SIZE, corners2, found)

    # Overlay info
    cv2.putText(
        display,
        f"Captures: {len(objpoints)} / {MIN_CAPTURES}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        display,
        "Press 'c' to capture | 'q' to finish",
        (20, display.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    cv2.imshow("Live Camera Calibration", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("c") and found:
        objpoints.append(objp)
        imgpoints.append(corners2)
        print(f"Captured frame {len(objpoints)}")

    elif key == ord("q"):
        break

# ===================== CALIBRATION =====================
picam2.stop()
cv2.destroyAllWindows()

if len(objpoints) < MIN_CAPTURES:
    raise RuntimeError("Not enough captures for calibration")

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, img_shape, None, None
)

if not ret:
    raise RuntimeError("Calibration failed")

fx = float(mtx[0, 0])
fy = float(mtx[1, 1])
cx = float(mtx[0, 2])
cy = float(mtx[1, 2])

print("\n================= CALIBRATION RESULT =================")
print(f"fx = {fx:.2f}")
print(f"fy = {fy:.2f}")
print(f"cx = {cx:.2f}")
print(f"cy = {cy:.2f}")
print("=====================================================")

# Optional: save for reuse
np.savez(
    "camera_intrinsics.npz",
    fx=fx, fy=fy, cx=cx, cy=cy, dist=dist
)
print("\nSaved to camera_intrinsics.npz")
