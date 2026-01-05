"""Pixel-to-GPS coordinate transforms for high-altitude drone imagery."""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np

# Camera orientation constants (degrees). Update if the payload mounting changes.
CAMERA_ROLL_DEG = 0.0
CAMERA_PITCH_DEG = 0.0
CAMERA_YAW_DEG = 0.0

# Default pinhole intrinsics (pixels) matching Raspberry Pi Camera Module 3.
# Resolution assumed: 4608x2592, focal length: ~4.28 mm, pixel pitch: 1.4 um.
DEFAULT_FX = 3057.0
DEFAULT_FY = 3057.0
DEFAULT_CX = 2304.0
DEFAULT_CY = 1296.0


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
	drone_lat: float,
	drone_lon: float,
	altitude_m: float,
	heading_deg: float = 0.0,
	*,
	fx: float = DEFAULT_FX,
	fy: float = DEFAULT_FY,
	cx: float = DEFAULT_CX,
	cy: float = DEFAULT_CY,
	roll_deg: float = CAMERA_ROLL_DEG,
	pitch_deg: float = CAMERA_PITCH_DEG,
	yaw_deg: float = CAMERA_YAW_DEG,
) -> Tuple[float, float] | None:
	"""Project a pixel location down to the ground plane and return (lat, lon).

	Returns ``None`` if the view vector never hits the ground plane (e.g., when the
	ray points upward because of a bad pitch estimate).
	"""

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
