import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import hailo
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.hailo_app_python.apps.detection_simple.detection_pipeline_simple import GStreamerDetectionApp

class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.total_people = 0
        self.total_frames = 0

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

def app_callback(pad, info, user_data):
    # increment frame counter (keeps the same behavior you already used)
    user_data.increment()
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # read frame size from the pad caps so we can convert normalized bbox -> pixels
    fmt, width, height = get_caps_from_pad(pad)

    # Get ROI and detections
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    people_count = 0
    string_to_print = f"Frame count: {user_data.get_count()}\n"

    for det in detections:
        label = det.get_label()
        conf = det.get_confidence()

        # bbox is a HailoBBox object (normalized coordinates)
        try:
            bbox = det.get_bbox()
        except Exception:
            bbox = None

        # default normalized coords if bbox missing
        xmin_n = ymin_n = xmax_n = ymax_n = None
        xmin_px = ymin_px = xmax_px = ymax_px = None

        if bbox is not None:
            # HailoBBox provides methods xmin(), ymin(), xmax(), ymax()
            xmin_n = bbox.xmin()
            ymin_n = bbox.ymin()
            xmax_n = bbox.xmax()
            ymax_n = bbox.ymax()

            if width is not None and height is not None:
                xmin_px = int(xmin_n * width)
                ymin_px = int(ymin_n * height)
                xmax_px = int(xmax_n * width)
                ymax_px = int(ymax_n * height)

        # If this is a person, update counts
        if label == "person":
            people_count += 1
            user_data.total_people += 1

        # Build print line for this detection (include both normalized and pixel coords if available)
        det_line = f"Detection: {label}  Confidence: {round(conf,2)}"
        if xmin_n is not None:
            det_line += f"  BBox(norm): [{xmin_n:.3f}, {ymin_n:.3f}, {xmax_n:.3f}, {ymax_n:.3f}]"
        if xmin_px is not None:
            det_line += f"  BBox(px): [{xmin_px}, {ymin_px}, {xmax_px}, {ymax_px}]"
        string_to_print += det_line + "\n"

    user_data.total_frames += 1
    running_average = user_data.total_people / user_data.total_frames if user_data.total_frames > 0 else 0.0

    header = (
        f"People detected in frame: {people_count}\n"
        f"Running average people per frame: {round(running_average, 2)}\n"
    )
    # Prepend header before per-detection lines
    full_output = f"Frame count: {user_data.get_count()}\n" + header + string_to_print
    print(full_output)

    return Gst.PadProbeReturn.OK

if __name__ == "__main__":
    user_data = user_app_callback_class()
    app = GStreamerDetectionApp(app_callback, user_data)
    app.run()