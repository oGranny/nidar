#!/usr/bin/env python3
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import hailo
import csv
import threading
import time
from datetime import datetime
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.hailo_app_python.apps.detection_simple.detection_pipeline_simple import GStreamerDetectionApp

Gst.init(None)

# -------------------------
# Helper: Get Pad Caps (for pixel conversion)
# -------------------------
def get_caps_from_pad(pad):
    """
    Read format, width, height from the pad's current caps.
    Returns (format_str, width, height) or (None, None, None) if unavailable.
    """
    if pad is None:
        return None, None, None
    caps = pad.get_current_caps()
    if not caps:
        caps = pad.get_pad_template_caps()
    if not caps or caps.get_size() == 0:
        return None, None, None
    try:
        structure = caps.get_structure(0)
        fmt = None
        if structure.has_field("format"):
            fmt = structure.get_value("format")
        width = structure.get_value("width")
        height = structure.get_value("height")
        # ensure ints
        return fmt, int(width), int(height)
    except Exception:
        return None, None, None

# -------------------------
# MAVLink GeoProvider (pymavlink)
# -------------------------
try:
    from pymavlink import mavutil
except Exception as e:
    raise RuntimeError("pymavlink is required. Install with: pip install pymavlink") from e

class MAVLinkGeoProvider:
    """
    Background MAVLink listener that extracts location/telemetry and stores the latest values.
    """
    def __init__(self, connection_str='udp:127.0.0.1:14550', serial_baud=115200, wait_heartbeat=True):
        self._lock = threading.Lock()
        self.lat = None           # degrees (float)
        self.lon = None           # degrees (float)
        self.alt = None           # meters (float)
        self.ts = None            # unix epoch seconds
        self.groundspeed = None   # m/s
        self.heading = None       # degrees (0-360)
        self._running = False
        self._thread = None
        self.connection_str = connection_str
        self.serial_baud = serial_baud
        self.wait_heartbeat = wait_heartbeat
        self.start()

    def start(self):
        if self._thread is not None:
            return
        self._running = True
        self._thread = threading.Thread(target=self._mav_thread, daemon=True)
        self._thread.start()

    def _mav_thread(self):
        conn = None
        try:
            if self.connection_str.startswith('/') or self.connection_str.lower().startswith('com'):
                conn = mavutil.mavlink_connection(self.connection_str, baud=self.serial_baud, autoreconnect=True)
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

                if msg_type == 'GLOBAL_POSITION_INT':
                    lat = getattr(msg, 'lat', None)
                    lon = getattr(msg, 'lon', None)
                    alt_mm = getattr(msg, 'alt', None)
                    if lat is not None and lon is not None:
                        with self._lock:
                            self.lat = float(lat) / 1e7
                            self.lon = float(lon) / 1e7
                            if alt_mm is not None:
                                self.alt = float(alt_mm) / 1000.0
                            self.ts = now

                elif msg_type == 'GPS_RAW_INT':
                    lat = getattr(msg, 'lat', None)
                    lon = getattr(msg, 'lon', None)
                    alt_mm = getattr(msg, 'alt', None)
                    fix_type = getattr(msg, 'fix_type', None)
                    if lat is not None and lon is not None:
                        if fix_type is None or fix_type >= 2:
                            with self._lock:
                                self.lat = float(lat) / 1e7
                                self.lon = float(lon) / 1e7
                                if alt_mm is not None:
                                    self.alt = float(alt_mm) / 1000.0
                                self.ts = now

                elif msg_type == 'VFR_HUD':
                    gs = getattr(msg, 'groundspeed', None)
                    heading = getattr(msg, 'heading', None)
                    with self._lock:
                        if gs is not None:
                            self.groundspeed = float(gs)
                        if heading is not None and heading >= 0:
                            self.heading = float(heading)
                        self.ts = now

                elif msg_type == 'ATTITUDE':
                    with self._lock:
                        self.ts = now

        except Exception as e:
            print(f"[MAVLinkGeoProvider] Exception in MAV thread: {e}")
        finally:
            try:
                if conn:
                    conn.close()
            except Exception:
                pass

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

    def get_latest(self):
        with self._lock:
            return (self.lat, self.lon, self.alt, self.ts, self.groundspeed, self.heading)

# -------------------------
# CSV Logger for geotags
# -------------------------
class GeotagCsvLogger:
    def __init__(self, csv_path="detections_geotags.csv"):
        self.csv_path = csv_path
        try:
            with open(self.csv_path, "r", newline="") as f:
                pass
        except FileNotFoundError:
            with open(self.csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "record_time_utc", "frame_count", "detection_label", "confidence",
                    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
                    "people_in_frame", "telemetry_ts",
                    "lat", "lon", "alt_m", "groundspeed_m_s", "heading_deg"
                ])

    def log_detection(self, frame_count, det_label, confidence, bbox, people_in_frame, telemetry):
        lat = lon = alt = ts = groundspeed = heading = None
        if telemetry:
            lat, lon, alt, ts, groundspeed, heading = telemetry
        
        rec_time = datetime.utcnow().isoformat() + "Z"
        
        # Handle bbox being a tuple (x1, y1, x2, y2) or object with attributes
        bbox_x1 = bbox.x1 if hasattr(bbox, "x1") else (bbox[0] if bbox else "")
        bbox_y1 = bbox.y1 if hasattr(bbox, "y1") else (bbox[1] if bbox else "")
        bbox_x2 = bbox.x2 if hasattr(bbox, "x2") else (bbox[2] if bbox else "")
        bbox_y2 = bbox.y2 if hasattr(bbox, "y2") else (bbox[3] if bbox else "")
        
        with open(self.csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([rec_time, frame_count, det_label, round(confidence, 4), 
                        bbox_x1, bbox_y1, bbox_x2, bbox_y2, 
                        people_in_frame, ts, lat, lon, alt, groundspeed, heading])

# -------------------------
# User callback + app
# -------------------------
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.total_people = 0
        self.total_frames = 0

# Configure MAVLink
MAVLINK_CONNECTION = 'udp:127.0.0.1:14550'   # Adjust as needed
SERIAL_BAUD = 57600
mav_provider = MAVLinkGeoProvider(connection_str=MAVLINK_CONNECTION, serial_baud=SERIAL_BAUD)
csv_logger = GeotagCsvLogger(csv_path="detections_geotags.csv")

def app_callback(pad, info, user_data):
    user_data.increment()
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # 1. Get video dimensions for pixel conversion
    fmt, width, height = get_caps_from_pad(pad)

    # 2. Get Detections
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    # 3. Count people
    people_count = 0
    for det in detections:
        if det.get_label() == "person":
            people_count += 1

    user_data.total_people += people_count
    user_data.total_frames += 1

    running_average = user_data.total_people / user_data.total_frames if user_data.total_frames > 0 else 0.0

    # 4. Prepare Summary String
    string_to_print = (
        f"Frame count: {user_data.get_count()}\n"
        f"People detected: {people_count}\n"
        f"Avg people/frame: {round(running_average, 2)}\n"
    )

    # 5. Get Telemetry
    latest_geo = mav_provider.get_latest()

    # 6. Process Detections & Log
    for detection in detections:
        det_label = detection.get_label()
        det_conf = detection.get_confidence()
        
        # --- BBOX EXTRACTION FIX ---
        try:
            bbox_obj = detection.get_bbox()
        except Exception:
            bbox_obj = None

        xmin_n = ymin_n = xmax_n = ymax_n = None
        xmin_px = ymin_px = xmax_px = ymax_px = None
        bbox_for_log = None

        if bbox_obj:
            # Normalized coordinates (0.0 - 1.0)
            xmin_n = bbox_obj.xmin()
            ymin_n = bbox_obj.ymin()
            xmax_n = bbox_obj.xmax()
            ymax_n = bbox_obj.ymax()

            # Calculate Pixel coordinates
            if width is not None and height is not None:
                xmin_px = int(xmin_n * width)
                ymin_px = int(ymin_n * height)
                xmax_px = int(xmax_n * width)
                ymax_px = int(ymax_n * height)
                bbox_for_log = (xmin_px, ymin_px, xmax_px, ymax_px)
            else:
                # Fallback to normalized if caps failed
                bbox_for_log = (xmin_n, ymin_n, xmax_n, ymax_n)

        # Append to print string
        det_line = f"Detection: {det_label} Conf: {round(det_conf, 2)}"
        if bbox_for_log:
            det_line += f" BBox: {bbox_for_log}"
        string_to_print += det_line + "\n"

        # Log to CSV if person
        if det_label == "person":
            csv_logger.log_detection(
                frame_count=user_data.get_count(),
                det_label=det_label,
                confidence=det_conf,
                bbox=bbox_for_log,  # Pass the tuple (px or norm)
                people_in_frame=people_count,
                telemetry=latest_geo
            )

    # 7. Print Telemetry
    if latest_geo and latest_geo[0] is not None:
        lat, lon, alt, ts, gs, hdg = latest_geo
        ts_str = datetime.utcfromtimestamp(float(ts)).isoformat() + "Z" if ts else "N/A"
        string_to_print += f"Telemetry: lat={lat:.6f} lon={lon:.6f} alt={alt} gs={gs} hdg={hdg} ts={ts_str}\n"
    else:
        string_to_print += "Telemetry: No Fix / No Data\n"

    print(string_to_print)
    return Gst.PadProbeReturn.OK

# -------------------------
# Run app
# -------------------------
if __name__ == "__main__":
    try:
        user_data = user_app_callback_class()
        app = GStreamerDetectionApp(app_callback, user_data)
        print("Starting hailo detection app with MAVLink geotagging.")
        print(f"Connecting to MAVLink endpoint: {MAVLINK_CONNECTION}")
        app.run()
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        mav_provider.stop()