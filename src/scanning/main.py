"""Run the Hailo detection pipeline and geotag person detections via MAVLink telemetry."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # type: ignore

import hailo
from hailo_apps.hailo_app_python.apps.detection_simple.detection_pipeline_simple import (
	GStreamerDetectionApp,
)
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
from pymavlink import mavutil

from transform import pixel_to_ground_gps
from scanning.geotag import save_coordinate_with_threshold


MAVLINK_CONNECTION = os.environ.get("GARUDA_MAVLINK_CONN", "udp:127.0.0.1:14550")
GEOTAG_DISTANCE_THRESHOLD_M = float(os.environ.get("GARUDA_GEOTAG_THRESHOLD_M", "10"))


@dataclass
class TelemetrySnapshot:
	latitude: float
	longitude: float
	relative_alt_m: float
	roll_deg: float
	pitch_deg: float
	yaw_deg: float
	heading_deg: float


class MavlinkTelemetry:
	"""Thin wrapper around pymavlink for the fields needed to geotag detections."""

	def __init__(self, connection_string: str = MAVLINK_CONNECTION) -> None:
		self._conn = mavutil.mavlink_connection(connection_string, autoreconnect=True)
		try:
			self._conn.wait_heartbeat(timeout=10)
		except Exception:
			print("Warning: Did not receive MAVLink heartbeat; telemetry will stay empty until connection is restored.")

		self._lat: Optional[float] = None
		self._lon: Optional[float] = None
		self._rel_alt_m: Optional[float] = None
		self._roll_deg: Optional[float] = None
		self._pitch_deg: Optional[float] = None
		self._yaw_deg: Optional[float] = None
		self._heading_deg: Optional[float] = None

	def poll(self) -> None:
		msg = self._conn.recv_match(blocking=False)
		while msg is not None:
			msg_type = msg.get_type()
			if msg_type == "GLOBAL_POSITION_INT":
				self._lat = msg.lat / 1e7
				self._lon = msg.lon / 1e7
				rel_alt_mm = getattr(msg, "relative_alt", None)
				alt_mm = getattr(msg, "alt", None)
				self._rel_alt_m = (rel_alt_mm if rel_alt_mm is not None else alt_mm) / 1000.0 if (rel_alt_mm or alt_mm) else None
				if getattr(msg, "hdg", None) is not None and msg.hdg != 65535:
					self._heading_deg = msg.hdg / 100.0
			elif msg_type == "ATTITUDE":
				self._roll_deg = (msg.roll or 0.0) * 57.2957795
				self._pitch_deg = (msg.pitch or 0.0) * 57.2957795
				self._yaw_deg = (msg.yaw or 0.0) * 57.2957795
			elif msg_type == "VFR_HUD":
				self._heading_deg = msg.heading

			msg = self._conn.recv_match(blocking=False)

	def snapshot(self) -> Optional[TelemetrySnapshot]:
		if None in (self._lat, self._lon, self._rel_alt_m, self._roll_deg, self._pitch_deg, self._yaw_deg):
			return None
		heading = self._heading_deg if self._heading_deg is not None else self._yaw_deg
		if heading is None:
			return None
		return TelemetrySnapshot(
			latitude=self._lat,
			longitude=self._lon,
			relative_alt_m=self._rel_alt_m,
			roll_deg=self._roll_deg,
			pitch_deg=self._pitch_deg,
			yaw_deg=self._yaw_deg,
			heading_deg=heading,
		)


class user_app_callback_class(app_callback_class):
	def __init__(self, telemetry: MavlinkTelemetry) -> None:
		super().__init__()
		self.total_people = 0
		self.total_frames = 0
		self.telemetry = telemetry


def get_caps_from_pad(pad) -> Tuple[Optional[str], Optional[int], Optional[int]]:
	"""Read format, width, height from the pad's current caps."""

	if pad is None:
		return None, None, None
	caps = pad.get_current_caps() or pad.get_pad_template_caps()
	if not caps or caps.get_size() == 0:
		return None, None, None
	try:
		structure = caps.get_structure(0)
		fmt = structure.get_value("format") if structure.has_field("format") else None
		width = structure.get_value("width")
		height = structure.get_value("height")
		return fmt, int(width), int(height)
	except Exception:
		return None, None, None


def _geotag_person_detection(
	bbox_pixels: Sequence[int], user_data: user_app_callback_class, width: Optional[int], height: Optional[int]
) -> Optional[Tuple[float, float]]:
	telemetry = user_data.telemetry
	telemetry.poll()
	snapshot = telemetry.snapshot()
	if snapshot is None or width is None or height is None:
		return None

	xmin_px, ymin_px, xmax_px, ymax_px = bbox_pixels
	center_pixel = ((xmin_px + xmax_px) / 2.0, (ymin_px + ymax_px) / 2.0)
	gps = pixel_to_ground_gps(
		pixel=center_pixel,
		drone_lat=snapshot.latitude,
		drone_lon=snapshot.longitude,
		altitude_m=snapshot.relative_alt_m,
		heading_deg=snapshot.heading_deg,
		roll_deg=snapshot.roll_deg,
		pitch_deg=snapshot.pitch_deg,
		yaw_deg=snapshot.yaw_deg,
	)
	if gps is None:
		return None

	save_coordinate_with_threshold(gps, threshold_meters=GEOTAG_DISTANCE_THRESHOLD_M)
	return gps


def app_callback(pad, info, user_data: user_app_callback_class):
	user_data.increment()
	buffer = info.get_buffer()
	if buffer is None:
		return Gst.PadProbeReturn.OK

	fmt, width, height = get_caps_from_pad(pad)
	roi = hailo.get_roi_from_buffer(buffer)
	detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

	people_count = 0
	string_to_print = f"Frame count: {user_data.get_count()}\n"

	for det in detections:
		label = det.get_label()
		conf = det.get_confidence()

		try:
			bbox = det.get_bbox()
		except Exception:
			bbox = None

		xmin_n = ymin_n = xmax_n = ymax_n = None
		xmin_px = ymin_px = xmax_px = ymax_px = None

		if bbox is not None:
			xmin_n = bbox.xmin()
			ymin_n = bbox.ymin()
			xmax_n = bbox.xmax()
			ymax_n = bbox.ymax()

			if width is not None and height is not None:
				xmin_px = int(xmin_n * width)
				ymin_px = int(ymin_n * height)
				xmax_px = int(xmax_n * width)
				ymax_px = int(ymax_n * height)

		geotag = None
		if label == "person":
			people_count += 1
			user_data.total_people += 1
			if all(v is not None for v in (xmin_px, ymin_px, xmax_px, ymax_px)):
				geotag = _geotag_person_detection((xmin_px, ymin_px, xmax_px, ymax_px), user_data, width, height)

		det_line = f"Detection: {label}  Confidence: {conf:.2f}"
		if xmin_n is not None:
			det_line += f"  BBox(norm): [{xmin_n:.3f}, {ymin_n:.3f}, {xmax_n:.3f}, {ymax_n:.3f}]"
		if xmin_px is not None:
			det_line += f"  BBox(px): [{xmin_px}, {ymin_px}, {xmax_px}, {ymax_px}]"
		if geotag is not None:
			det_line += f"  Geotag(lat,lon): [{geotag[0]:.6f}, {geotag[1]:.6f}]"
		string_to_print += det_line + "\n"

	user_data.total_frames += 1
	running_average = user_data.total_people / user_data.total_frames if user_data.total_frames else 0.0

	header = (
		f"People detected in frame: {people_count}\n"
		f"Running average people per frame: {running_average:.2f}\n"
	)
	full_output = f"Frame count: {user_data.get_count()}\n" + header + string_to_print
	print(full_output)

	return Gst.PadProbeReturn.OK


def main() -> None:
	telemetry = MavlinkTelemetry(MAVLINK_CONNECTION)
	user_data = user_app_callback_class(telemetry)
	app = GStreamerDetectionApp(app_callback, user_data)
	app.run()


if __name__ == "__main__":
	main()
