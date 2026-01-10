"""Pixel-to-GPS coordinate transforms for high-altitude drone imagery."""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np
from typing import Iterable

# Default pinhole intrinsics (pixels) matching Raspberry Pi Camera Module 3.
# Resolution assumed: 4608x2592, focal length: ~4.28 mm, pixel pitch: 1.4 um.

def _rotation_matrix(roll_rad: float, pitch_rad: float, yaw_rad: float, heading_rad: float) -> np.ndarray:
	"""Return the ZYX rotation that maps camera rays into world ENU coordinates."""

	def rx(angle: float) -> np.ndarray:
		return np.array(
			[[1.0, 0.0, 0.0], [0.0, math.cos(angle), -math.sin(angle)], [0.0, math.sin(angle), math.cos(angle)]]
		)

	def ry(angle: float) -> np.ndarray:
		return np.array(
			[[math.cos(angle), 0.0, math.sin(angle)], [0.0, 1.0, 0.0], [-math.sin(angle), 0.0, math.cos(angle)]]
		)

	def rz(angle: float) -> np.ndarray:
		return np.array(
			[[math.cos(angle), -math.sin(angle), 0.0], [math.sin(angle), math.cos(angle), 0.0], [0.0, 0.0, 1.0]]
		)

	return rz(heading_rad) @ rz(yaw_rad) @ ry(pitch_rad) @ rx(roll_rad)


def pixel_to_ground_gps(
	pixel: Sequence[float],
	config,
	drone_lat: float,
	drone_lon: float,
	altitude_m: float,
	heading_deg: float = 0.0,
	*,
	roll_deg: float,
	pitch_deg: float,
	yaw_deg: float,
) -> Tuple[float, float] | None:
	"""Project a pixel location down to the ground plane and return (lat, lon).

	Returns ``None`` if the view vector never hits the ground plane (e.g., when the
	ray points upward because of a bad pitch estimate).
	"""
	fx = config.get("DEFAULT_FX"),
	fy = config.get("DEFAULT_FY"),
	cx = config.get("DEFAULT_CX"),
	cy = config.get("DEFAULT_CY"),

	if len(pixel) != 2:
		raise ValueError("pixel must contain (x, y)")
	if altitude_m <= 0:
		raise ValueError("altitude_m must be positive")

	u, v = float(pixel[0]), float(pixel[1])
	x_cam = (u - cx) / fx
	y_cam = (v - cy) / fy
	ray_cam = np.array([x_cam, y_cam, 1.0])

	roll_rad = math.radians(roll_deg)
	pitch_rad = math.radians(pitch_deg)
	yaw_rad = math.radians(yaw_deg)
	heading_rad = math.radians(heading_deg)

	rotation = _rotation_matrix(roll_rad, pitch_rad, yaw_rad, heading_rad)
	ray_world = rotation @ ray_cam

	if ray_world[2] >= 0:
		return None

	t = altitude_m / (-ray_world[2])
	east_m = t * ray_world[0]
	north_m = t * ray_world[1]

	meters_per_deg_lat = 111_111.0
	meters_per_deg_lon = meters_per_deg_lat * math.cos(math.radians(drone_lat))

	lat = drone_lat + (north_m / meters_per_deg_lat)
	lon = drone_lon + (east_m / meters_per_deg_lon)
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
