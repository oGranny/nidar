"""Pixel-to-GPS coordinate transforms for high-altitude drone imagery."""

from __future__ import annotations

import math
from typing import Sequence, Tuple, Mapping

import numpy as np


def _rotation_matrix_enu(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Body → ENU rotation using aerospace convention.
    Roll  about X (forward)
    Pitch about Y (right)
    Yaw   about Z (down → mapped to ENU)
    """

    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    # ZYX rotation (yaw → pitch → roll)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp,     cp * sr,                cp * cr],
        ]
    )


def pixel_to_ground_gps(
    pixel: Sequence[float],
    config: Mapping[str, float],
    drone_lat: float,
    drone_lon: float,
    altitude_m: float,
    *,
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
) -> Tuple[float, float] | None:
    """
    Project a pixel to the ground plane and return (lat, lon).
    Returns None if the ray does not intersect the ground.
    """

    fx = float(config["DEFAULT_FX"])
    fy = float(config["DEFAULT_FY"])
    cx = float(config["DEFAULT_CX"])
    cy = float(config["DEFAULT_CY"])

    if len(pixel) != 2:
        raise ValueError("pixel must be (u, v)")
    if altitude_m <= 0:
        raise ValueError("altitude_m must be positive (AGL)")

    u, v = map(float, pixel)

    # --- Camera ray (OpenCV convention) ---
    x = (u - cx) / fx
    y = (v - cy) / fy
    ray_cam = np.array([x, y, 1.0])

    # --- Camera → body frame correction ---
    # Camera:  x right, y down, z forward
    # Body:    x forward, y right, z down
    cam_to_body = np.array(
        [
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
        ]
    )

    ray_body = cam_to_body @ ray_cam

    # --- Body → ENU rotation ---
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)

    R = _rotation_matrix_enu(roll, pitch, yaw)
    ray_enu = R @ ray_body

    # ENU: Z is UP → ground is at z = -altitude
    if ray_enu[2] >= 0:
        return None  # looking above horizon

    t = altitude_m / (-ray_enu[2])

    east_m = t * ray_enu[0]
    north_m = t * ray_enu[1]

    # --- Convert meters → lat/lon ---
    meters_per_deg_lat = 111_111.0
    meters_per_deg_lon = meters_per_deg_lat * math.cos(math.radians(drone_lat))

    lat = drone_lat + north_m / meters_per_deg_lat
    lon = drone_lon + east_m / meters_per_deg_lon

    return lat, lon


__all__ = ["pixel_to_ground_gps"]


def calibrate_camera_from_images(
	image_paths: Iterable[str], board_size: Tuple[int, int] = (9, 6), square_size_mm: float = 25.4
) -> Tuple[float, float, float, float]:
	"""Estimate camera intrinsics (fx, fy, cx, cy) from chessboard images.

	This uses OpenCV's findChessboardCorners + calibrateCamera. Provide a set of
	images of a flat chessboard taken with the Pi Camera covering the frame. The
	board_size is the inner-corners count (columns, rows). square_size_mm is the
	physical square size and is only needed to get a properly scaled reprojection
	(pixels focal length is independent of units but OpenCV requires real-world
	coordinates).

	Returns (fx, fy, cx, cy) in pixels.

	Example:
		fx, fy, cx, cy = calibrate_camera_from_images(glob.glob('calib/*.jpg'))

	Requires: opencv-python (cv2).
	"""
	try:
		import cv2
	except Exception as e:
		raise RuntimeError("OpenCV (cv2) is required for camera calibration") from e

	# prepare object points like (0,0,0), (1,0,0), ... scaled by square size
	objp = np.zeros((board_size[0] * board_size[1], 3), dtype=np.float32)
	objp[:, :2] = (
		np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2).astype(np.float32) * square_size_mm
	)

	objpoints = []  # 3d points in real world space
	imgpoints = []  # 2d points in image plane.

	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

	img_shape = None
	for p in image_paths:
		img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
		if img is None:
			continue
		if img_shape is None:
			img_shape = img.shape[::-1]
		ret, corners = cv2.findChessboardCorners(img, board_size, None)
		if not ret:
			continue
		corners2 = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
		objpoints.append(objp)
		imgpoints.append(corners2)

	if not objpoints:
		raise RuntimeError("No valid chessboard detections found in provided images")

	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)
	if not ret:
		raise RuntimeError("calibrateCamera failed")

	fx = float(mtx[0, 0])
	fy = float(mtx[1, 1])
	cx = float(mtx[0, 2])
	cy = float(mtx[1, 2])
	return fx, fy, cx, cy
