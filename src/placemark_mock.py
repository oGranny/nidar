#!/usr/bin/env python3
"""Entrypoint for the Hailo-based person scanner that emits unique geo placemarks."""
from __future__ import annotations

import argparse
import csv
import math
import os
import threading
import time
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # type: ignore
import hailo  # type: ignore
from hailo_apps.hailo_app_python.apps.detection_simple.detection_pipeline_simple import (
    GStreamerDetectionApp,
)
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class

from geo_transform import person_gps_from_bbox

Gst.init(None)


# ---------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------
def get_caps_from_pad(pad):
    """Return (format, width, height) for a pad if available."""
    if pad is None:
        return None, None, None
    caps = pad.get_current_caps() or pad.get_pad_template_caps()
    if not caps or caps.get_size() == 0:
        return None, None, None
    try:
        structure = caps.get_structure(0)
        fmt = structure.get_value("format") if structure.has_field("format") else None
        width = int(structure.get_value("width"))
        height = int(structure.get_value("height"))
        return fmt, width, height
    except Exception:
        return None, None, None


# ---------------------------------------------------------------
# MAVLink geo provider
# ---------------------------------------------------------------
try:
    from pymavlink import mavutil
except Exception as exc:  # pragma: no cover - import guard
    raise RuntimeError("pymavlink is required. Install with: pip install pymavlink") from exc


class MAVLinkGeoProvider:
    """Background MAVLink listener that keeps the latest telemetry sample."""

    def __init__(
        self,
        connection_str: str = "udp:127.0.0.1:14550",
        serial_baud: int = 115200,
        wait_heartbeat: bool = True,
    ) -> None:
        self._lock = threading.Lock()
        self.lat: Optional[float] = None
        self.lon: Optional[float] = None
        self.alt: Optional[float] = None
        self.ts: Optional[float] = None
        self.groundspeed: Optional[float] = None
        self.heading: Optional[float] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.connection_str = connection_str
        self.serial_baud = serial_baud
        self.wait_heartbeat = wait_heartbeat
        self.start()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._running = True
        self._thread = threading.Thread(target=self._mav_thread, daemon=True)
        self._thread.start()

    def _mav_thread(self) -> None:
        conn = None
        try:
            if self.connection_str.startswith("/") or self.connection_str.lower().startswith("com"):
                conn = mavutil.mavlink_connection(
                    self.connection_str, baud=self.serial_baud, autoreconnect=True
                )
            else:
                conn = mavutil.mavlink_connection(self.connection_str, autoreconnect=True)

            if self.wait_heartbeat:
                try:
                    conn.wait_heartbeat(timeout=5)
                except Exception:
                    pass

            while self._running:
                msg = conn.recv_match(blocking=True, timeout=1)
                if msg is None:
                    continue

                msg_type = msg.get_type()
                now = time.time()

                if msg_type == "GLOBAL_POSITION_INT":
                    lat = getattr(msg, "lat", None)
                    lon = getattr(msg, "lon", None)
                    alt_mm = getattr(msg, "alt", None)
                    if lat is not None and lon is not None:
                        with self._lock:
                            self.lat = float(lat) / 1e7
                            self.lon = float(lon) / 1e7
                            self.alt = float(alt_mm) / 1000.0 if alt_mm is not None else self.alt
                            self.ts = now

                elif msg_type == "GPS_RAW_INT":
                    lat = getattr(msg, "lat", None)
                    lon = getattr(msg, "lon", None)
                    alt_mm = getattr(msg, "alt", None)
                    fix_type = getattr(msg, "fix_type", None)
                    if lat is not None and lon is not None and (fix_type is None or fix_type >= 2):
                        with self._lock:
                            self.lat = float(lat) / 1e7
                            self.lon = float(lon) / 1e7
                            self.alt = float(alt_mm) / 1000.0 if alt_mm is not None else self.alt
                            self.ts = now

                elif msg_type == "VFR_HUD":
                    gs = getattr(msg, "groundspeed", None)
                    heading = getattr(msg, "heading", None)
                    with self._lock:
                        if gs is not None:
                            self.groundspeed = float(gs)
                        if heading is not None and heading >= 0:
                            self.heading = float(heading)
                        self.ts = now

                elif msg_type == "ATTITUDE":
                    with self._lock:
                        self.ts = now

        except Exception as exc:
            print(f"[MAVLinkGeoProvider] Exception in MAV thread: {exc}")
        finally:
            try:
                if conn:
                    conn.close()
            except Exception:
                pass

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

    def get_latest(self) -> Tuple[Optional[float], ...]:
        with self._lock:
            return (self.lat, self.lon, self.alt, self.ts, self.groundspeed, self.heading)


# ---------------------------------------------------------------
# Legacy CSV logger (per detection)
# ---------------------------------------------------------------
class LegacyGeotagCsvLogger:
    def __init__(self, csv_path: str = "detections_geotags.csv") -> None:
        self.csv_path = csv_path
        self._ensure_header()

    def _ensure_header(self) -> None:
        if os.path.exists(self.csv_path):
            return
        with open(self.csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "record_time_utc",
                    "frame_count",
                    "detection_label",
                    "confidence",
                    "bbox_x1",
                    "bbox_y1",
                    "bbox_x2",
                    "bbox_y2",
                    "people_in_frame",
                    "telemetry_ts",
                    "lat",
                    "lon",
                    "alt_m",
                    "groundspeed_m_s",
                    "heading_deg",
                ]
            )

    def log_detection(
        self,
        frame_count: int,
        det_label: str,
        confidence: float,
        bbox: Optional[Tuple[float, ...]],
        people_in_frame: int,
        telemetry: Optional[Tuple[Optional[float], ...]],
    ) -> None:
        lat = lon = alt = ts = groundspeed = heading = None
        if telemetry:
            lat, lon, alt, ts, groundspeed, heading = telemetry
        rec_time = datetime.utcnow().isoformat() + "Z"
        bbox_x1 = bbox[0] if (bbox and len(bbox) > 0) else ""
        bbox_y1 = bbox[1] if (bbox and len(bbox) > 1) else ""
        bbox_x2 = bbox[2] if (bbox and len(bbox) > 2) else ""
        bbox_y2 = bbox[3] if (bbox and len(bbox) > 3) else ""
        with open(self.csv_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    rec_time,
                    frame_count,
                    det_label,
                    round(confidence, 4),
                    bbox_x1,
                    bbox_y1,
                    bbox_x2,
                    bbox_y2,
                    people_in_frame,
                    ts,
                    lat,
                    lon,
                    alt,
                    groundspeed,
                    heading,
                ]
            )


# ---------------------------------------------------------------
# Placemark CSV logger (unique locations)
# ---------------------------------------------------------------
class PlacemarkCsvLogger:
    header = [
        "placemark_id",
        "first_seen_utc",
        "frame_count",
        "lat",
        "lon",
        "alt_m",
        "heading_deg",
        "groundspeed_m_s",
        "confidence",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "telemetry_ts",
        "people_in_frame",
    ]

    def __init__(self, csv_path: str, distance_threshold_m: float = 5.0) -> None:
        self.csv_path = csv_path
        self.distance_threshold_m = max(distance_threshold_m, 0.5)
        self._lock = threading.Lock()
        self._entries: List[Tuple[float, float]] = []
        self._next_id = 1
        self._prepare_existing_entries()

    def _prepare_existing_entries(self) -> None:
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow(self.header)
            return
        with open(self.csv_path, "r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    lat = float(row["lat"])
                    lon = float(row["lon"])
                    self._entries.append((lat, lon))
                    placemark_id = int(row["placemark_id"])
                    self._next_id = max(self._next_id, placemark_id + 1)
                except (ValueError, KeyError):
                    continue

    @staticmethod
    def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        rad = math.radians
        dlat = rad(lat2 - lat1)
        dlon = rad(lon2 - lon1)
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(rad(lat1)) * math.cos(rad(lat2)) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return 6371000.0 * c

    def register(  # noqa: PLR0913 - clarity preferred over packing args
        self,
        frame_count: int,
        bbox: Tuple[int, int, int, int],
        confidence: float,
        telemetry: Optional[Tuple[Optional[float], ...]],
        people_in_frame: int,
        location: Tuple[float, float],
    ) -> Optional[Dict[str, object]]:
        person_lat, person_lon = location
        with self._lock:
            for lat, lon in self._entries:
                if self._haversine_m(lat, lon, person_lat, person_lon) < self.distance_threshold_m:
                    return None

            alt = heading = groundspeed = ts = None
            if telemetry:
                lat, lon, alt, ts, groundspeed, heading = telemetry

            rec_time = datetime.utcnow().isoformat() + "Z"
            placemark = {
                "placemark_id": self._next_id,
                "first_seen_utc": rec_time,
                "frame_count": frame_count,
                "lat": round(person_lat, 7),
                "lon": round(person_lon, 7),
                "alt_m": alt,
                "heading_deg": heading,
                "groundspeed_m_s": groundspeed,
                "confidence": round(confidence, 4),
                "bbox_x1": bbox[0],
                "bbox_y1": bbox[1],
                "bbox_x2": bbox[2],
                "bbox_y2": bbox[3],
                "telemetry_ts": ts,
                "people_in_frame": people_in_frame,
            }

            with open(self.csv_path, "a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.header)
                writer.writerow(placemark)

            self._entries.append((person_lat, person_lon))
            self._next_id += 1
            return placemark


# ---------------------------------------------------------------
# User callback state
# ---------------------------------------------------------------
class user_app_callback_class(app_callback_class):
    def __init__(self) -> None:
        super().__init__()
        self.total_people = 0
        self.total_frames = 0


# ---------------------------------------------------------------
# Detection callback factory
# ---------------------------------------------------------------
def build_app_callback(
    mav_provider: MAVLinkGeoProvider,
    placemark_logger: PlacemarkCsvLogger,
    legacy_logger: Optional[LegacyGeotagCsvLogger],
    enable_legacy_log: bool,
    distance_threshold_m: float,
):
    def _callback(pad, info, user_data):  # type: ignore[override]
        user_data.increment()
        buffer = info.get_buffer()
        if buffer is None:
            return Gst.PadProbeReturn.OK

        _, width, height = get_caps_from_pad(pad)
        roi = hailo.get_roi_from_buffer(buffer)
        detections: Iterable = roi.get_objects_typed(hailo.HAILO_DETECTION)
        detection_infos: List[Dict[str, object]] = []

        for detection in detections:
            label = detection.get_label()
            confidence = detection.get_confidence()
            bbox_obj = None
            try:
                bbox_obj = detection.get_bbox()
            except Exception:
                bbox_obj = None

            bbox_pixels: Optional[Tuple[int, int, int, int]] = None
            bbox_norm: Optional[Tuple[float, float, float, float]] = None

            if bbox_obj:
                xmin_n = bbox_obj.xmin()
                ymin_n = bbox_obj.ymin()
                xmax_n = bbox_obj.xmax()
                ymax_n = bbox_obj.ymax()
                bbox_norm = (xmin_n, ymin_n, xmax_n, ymax_n)
                if width and height:
                    bbox_pixels = (
                        int(xmin_n * width),
                        int(ymin_n * height),
                        int(xmax_n * width),
                        int(ymax_n * height),
                    )

            detection_infos.append(
                {
                    "label": label,
                    "confidence": confidence,
                    "bbox_pixels": bbox_pixels,
                    "bbox_norm": bbox_norm,
                }
            )

        people_count = sum(1 for entry in detection_infos if entry["label"] == "person")
        user_data.total_people += people_count
        user_data.total_frames += 1
        latest_geo = mav_provider.get_latest()

        for entry in detection_infos:
            if entry["label"] != "person":
                continue

            confidence = entry["confidence"]
            bbox_pixels = entry["bbox_pixels"]
            bbox_norm = entry["bbox_norm"]
            bbox_for_log = bbox_pixels or bbox_norm

            placemark_message = None
            if (
                bbox_pixels
                and latest_geo
                and latest_geo[0] is not None
                and latest_geo[1] is not None
                and latest_geo[2] is not None
            ):
                lat, lon, alt, _ts, _gs, heading = latest_geo
                heading_val = heading if heading is not None else 0.0
                location = person_gps_from_bbox(
                    bbox_pixels[0],
                    bbox_pixels[1],
                    bbox_pixels[2],
                    bbox_pixels[3],
                    lat,
                    lon,
                    alt,
                    heading_val,
                )
                if location:
                    placemark = placemark_logger.register(
                        frame_count=user_data.get_count(),
                        bbox=bbox_pixels,
                        confidence=confidence,
                        telemetry=latest_geo,
                        people_in_frame=people_count,
                        location=location,
                    )
                    if placemark:
                        placemark_message = (
                            "New placemark #{id} at lat={lat:.6f}, lon={lon:.6f} (confidence={conf})"
                        ).format(
                            id=placemark["placemark_id"],
                            lat=placemark["lat"],
                            lon=placemark["lon"],
                            conf=placemark["confidence"],
                        )

            if enable_legacy_log and legacy_logger and bbox_for_log:
                legacy_logger.log_detection(
                    frame_count=user_data.get_count(),
                    det_label="person",
                    confidence=confidence,
                    bbox=bbox_for_log,
                    people_in_frame=people_count,
                    telemetry=latest_geo,
                )

            if placemark_message:
                print(placemark_message)

        summary = (
            f"Frame {user_data.get_count()} | people: {people_count} | "
            f"avg/frame: {user_data.total_people / max(user_data.total_frames, 1):.2f} | "
            f"placemark threshold: {distance_threshold_m} m"
        )
        print(summary)
        if latest_geo and latest_geo[0] is not None:
            lat, lon, alt, ts, gs, hdg = latest_geo
            ts_str = (
                datetime.utcfromtimestamp(float(ts)).isoformat() + "Z"
                if ts is not None
                else "N/A"
            )
            print(
                f"Telemetry lat={lat:.6f} lon={lon:.6f} alt={alt} gs={gs} hdg={hdg} ts={ts_str}"
            )
        else:
            print("Telemetry: No Fix / No Data")

        return Gst.PadProbeReturn.OK

    return _callback


# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start the placemark scanner that logs unique person detections."
    )
    parser.add_argument(
        "--mavlink",
        default="udp:127.0.0.1:14550",
        help="MAVLink endpoint (e.g. udp:127.0.0.1:14550 or /dev/ttyUSB0)",
    )
    parser.add_argument(
        "--serial-baud",
        type=int,
        default=57600,
        help="Baud rate for serial MAVLink endpoints (if applicable)",
    )
    parser.add_argument(
        "--placemark-csv",
        default="person_placemarks.csv",
        help="Destination CSV for unique placemark entries",
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=5.0,
        help="Meters between placemarks before a new entry is created",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Also append each person detection to the legacy detections_geotags.csv",
    )
    parser.add_argument(
        "--legacy-csv",
        default="detections_geotags.csv",
        help="Path to the legacy CSV file used when --log is provided",
    )
    return parser.parse_args()


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main() -> None:
    args = parse_args()
    mav_provider = MAVLinkGeoProvider(
        connection_str=args.mavlink, serial_baud=args.serial_baud, wait_heartbeat=True
    )
    placemark_logger = PlacemarkCsvLogger(
        csv_path=args.placemark_csv, distance_threshold_m=args.distance_threshold
    )
    legacy_logger = LegacyGeotagCsvLogger(csv_path=args.legacy_csv) if args.log else None
    user_data = user_app_callback_class()
    callback = build_app_callback(
        mav_provider=mav_provider,
        placemark_logger=placemark_logger,
        legacy_logger=legacy_logger,
        enable_legacy_log=args.log,
        distance_threshold_m=args.distance_threshold,
    )

    app = GStreamerDetectionApp(callback, user_data)
    print("Starting placemark scanner with MAVLink geotagging.")
    print(f"Connecting to MAVLink endpoint: {args.mavlink}")
    try:
        app.run()
    except KeyboardInterrupt:
        print("Stopping scanner...")
    finally:
        mav_provider.stop()


if __name__ == "__main__":
    main()
