import time
import math
from typing import List, Tuple, Dict, Optional
import sys
sys.path.append('src/scanning')
from transform import pixel_to_ground_gps
import numpy as np
import cv2
import onnxruntime as ort
from pymavlink import mavutil


# ==================== USER INPUT ====================
CONNECTION_STRING = "udp:127.0.0.1:14550"

# Primary route points to visit in order (lat, lon, altRelMeters)
WAYPOINTS: List[Tuple[float, float, float]] = [
    (22.255122, 84.907263, 15),
    (22.255063, 84.906966, 15),
    (22.254918, 84.907342, 15),
    (22.254766, 84.906967, 15),
    (22.254668, 84.907258, 15),
]

# Camera source options for Arducam:
#   "libcamera" - Use picamera2 with libcamera (recommended, no v4l2 needed)
#   0 or "/dev/video0" - Use V4L2 interface (requires legacy camera enabled in raspi-config)
#   Check available devices with: ls /dev/video*
CAMERA_SOURCE = "libcamera"  # Change to 0 if you enable legacy camera support

# YOLO model
MODEL_PATH = "models/yolo11s.onnx"
CONF_THRESH = 0.05
IOU_THRESH = 0.75
INPUT_SIZE = 640  # typical for YOLOv5/8/11 onnx

TAKEOFF_ALT = 15
REACH_RADIUS_M = 5.0
RTL_AT_END = True
DETECTION_ATTEMPTS = 5  # Number of YOLO runs at each waypoint

# Servo defaults (used by angle_to_pwm)
SERVO_MIN_ANGLE = 0.0
SERVO_MAX_ANGLE = 180.0
SERVO_MIN_PWM = 1000
SERVO_MAX_PWM = 2000

# Drop mechanism configuration
DROP_SERVOS = [6, 10, 7, 9, 8]  # Servo outputs for drop in order
DROP_ANGLE_OPEN = 90.0  # Angle to open/drop
DROP_ANGLE_CLOSE = 0.0  # Angle to close
DROP_DWELL_SEC = 1.0  # Time to hold open before closing

# Camera mount configuration
# Camera is fixed facing downwards on the drone
CAMERA_PITCH_OFFSET_DEG = -90.0  # -90 = straight down (fixed mount)
CAMERA_ROLL_OFFSET_DEG = 0.0
CAMERA_YAW_OFFSET_DEG = 0.0
# ===================================================
 

# ================= MAVLINK HELPERS ==================
def connect_vehicle():
    print(f"Connecting to {CONNECTION_STRING}...")
    master = mavutil.mavlink_connection(CONNECTION_STRING)
    master.wait_heartbeat()
    print(f"Connected. SYS={master.target_system} COMP={master.target_component}")
    return master


def gps_prearm_check(master, min_fix=3, min_sats=6):
    print("\n=== Pre-arm GPS check ===")
    while True:
        msg = master.recv_match(type="GPS_RAW_INT", timeout=5)
        if not msg:
            # print("Waiting GPS…") 
            continue
        fix = msg.fix_type
        sats = msg.satellites_visible
        print(f"GPS FIX={fix} SATS={sats}")
        if fix >= min_fix and sats >= min_sats:
            print("✔ GPS OK — PRE-ARM PASS")
            break
        time.sleep(1)


def set_mode(master, mode):
    mode_map = master.mode_mapping()
    if mode not in mode_map:
        raise Exception(f"Mode {mode} not supported")
    master.mav.set_mode_send(
        master.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_map[mode]
    )
    print(f"Set Mode → {mode}")
    time.sleep(1)


def arm(master):
    print("\n=== Arming ===")
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        1, 0, 0, 0, 0, 0, 0
    )
    while True:
        hb = master.recv_match(type="HEARTBEAT", timeout=5)
        if hb:
            armed = (hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) != 0
            print("ARMED:", armed)
            if armed:
                print("✔ Vehicle Armed")
                break
        time.sleep(0.5)


def takeoff(master, alt):
    print("\n=== Takeoff ===")
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0,
        0, 0, 0, 0, 0, 0, alt
    )
    print(f"Takeoff to {alt}m")
    time.sleep(4)


def guided_goto(master, lat, lon, alt):
    # Use SET_POSITION_TARGET_GLOBAL_INT for GUIDED navigation
    type_mask = (
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_VX_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_VY_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_VZ_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
    )
    
    # Use 0 for time_boot_ms; ArduPilot ignores it for position targets
    master.mav.set_position_target_global_int_send(
        0,  # time_boot_ms (not critical for this command)
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
        type_mask,
        int(lat * 1e7),
        int(lon * 1e7),
        float(alt),
        0, 0, 0,
        0, 0, 0,
        0, 0
    )
    print(f"→ GUIDED GOTO lat={lat:.7f} lon={lon:.7f} alt={alt:.1f}m")


def get_current_position(master) -> Optional[Tuple[float, float, float]]:
    msg = master.recv_match(type="GLOBAL_POSITION_INT", timeout=2)
    if not msg:
        return None
    lat = msg.lat / 1e7
    lon = msg.lon / 1e7
    alt_rel = msg.relative_alt / 1000.0  # mm → m
    return (lat, lon, alt_rel)


def get_heading(master) -> Optional[float]:
    """Get current heading in degrees (0-360, 0=North)."""
    msg = master.recv_match(type="VFR_HUD", timeout=2)
    if not msg:
        return None
    return float(msg.heading)


def get_attitude(master) -> Optional[Tuple[float, float, float]]:
    """Get drone attitude from flight controller IMU.
    Returns (roll, pitch, yaw) in degrees.
    Roll: positive = right wing down
    Pitch: positive = nose up
    Yaw: 0-360, 0=North
    """
    msg = master.recv_match(type="ATTITUDE", timeout=2)
    if not msg:
        return None
    roll_deg = math.degrees(msg.roll)
    pitch_deg = math.degrees(msg.pitch)
    yaw_deg = math.degrees(msg.yaw)
    # Normalize yaw to 0-360
    if yaw_deg < 0:
        yaw_deg += 360
    return (roll_deg, pitch_deg, yaw_deg)


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6378137.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c


# ================= SERVO HELPERS ====================

def angle_to_pwm(angle_deg: float,
                 *, min_angle: float = SERVO_MIN_ANGLE,
                 max_angle: float = SERVO_MAX_ANGLE,
                 min_pwm: int = SERVO_MIN_PWM,
                 max_pwm: int = SERVO_MAX_PWM) -> int:
    angle = max(min(angle_deg, max_angle), min_angle)
    span_a = max_angle - min_angle if max_angle != min_angle else 1.0
    span_p = max_pwm - min_pwm
    pwm = int(round(min_pwm + (angle - min_angle) * (span_p / span_a)))
    return max(min(pwm, max_pwm), min_pwm)

def set_servo_angle(master, servo_output: int, pwm: int):
    print(f"Sending SERVO{servo_output} -> PWM {pwm}")
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
        0,
        float(servo_output),  # param1: servo number
        float(pwm),           # param2: PWM value
        0, 0, 0, 0, 0
    )

def reset_servos(master):
    set_servo_angle(master, 6, 1000)
    set_servo_angle(master, 7, 1000)
    set_servo_angle(master, 8, 1000)
    set_servo_angle(master, 9, 2000)
    set_servo_angle(master, 10, 2000)

def drop_packet(master, servo_output):
    map = {
        6: 2400,
        7: 2400,
        8: 2400,
        9: 600,
        10: 600,
    }
    set_servo_angle(master, servo_output, map[servo_output])
    time.sleep(1)

def wait_until_reached(master, target_lat, target_lon, radius_m=REACH_RADIUS_M, timeout_s=120) -> bool:
    print(f"Waiting until within {radius_m}m of target…")
    t0 = time.time()
    # return True
    while time.time() - t0 < timeout_s:
        pos = get_current_position(master)
        if pos is None:
            continue
        lat, lon, _ = pos
        d = haversine_m(lat, lon, target_lat, target_lon)
        print(f"Distance to target: {d:.1f}m")
        if d <= radius_m:
            print("✔ Target reached")
            return True
        time.sleep(1)
    print("Timeout waiting to reach target")
    return False


# ================= YOLO HELPERS ======================
COCO_NAMES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
    "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"
]


def letterbox(img, new_shape=INPUT_SIZE, color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    pad_h, pad_w = new_shape - nh, new_shape - nw
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(resized, pad_h // 2, pad_h - pad_h // 2, pad_w // 2, pad_w - pad_w // 2,
                                cv2.BORDER_CONSTANT, value=color)
    return padded, r, (pad_w // 2, pad_h // 2)


def nms(boxes, scores, iou_threshold):
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        ious = iou(boxes[i], boxes[idxs[1:]])
        idxs = idxs[1:][ious < iou_threshold]
    return keep


def iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_box + area_boxes - inter
    return inter / (union + 1e-6)


class YoloDetector:
    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.out_names = [o.name for o in self.session.get_outputs()]

    def infer(self, frame: np.ndarray, display=False):
        img, r, (pad_w, pad_h) = letterbox(frame, INPUT_SIZE)
        img_input = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0  # BGR->RGB, HWC->CHW
        img_input = np.expand_dims(img_input, 0)

        outputs = self.session.run(self.out_names, {self.input_name: img_input})
        out = outputs[0]
        # Normalize output to YOLOv8 format: (N, 84) where [cx,cy,w,h, 80 class scores]
        # Common ONNX exports: (1,84,8400) or (1,8400,84)
        if out.ndim == 3:
            if out.shape[1] == 84:      # (1, 84, 8400)
                out = out[0].T          # -> (8400, 84)
            elif out.shape[2] == 84:    # (1, 8400, 84)
                out = out[0]            # -> (8400, 84)
            else:
                out = out.reshape(-1, out.shape[-1])

        # YOLOv8/v11: [x,y,w,h, class_scores...] (no objectness). Some exports emit logits; apply sigmoid fallback.
        boxes = []
        scores = []
        labels = []
        
        if out.size == 0:
            if display:
                cv2.imshow("YOLO Detection", frame)
                cv2.waitKey(1)
            return []

        # Ensure float32
        out = out.astype(np.float32, copy=False)

        # Split boxes and class scores
        bx = out[:, :4]
        cls = out[:, 4:]

        # Sigmoid fallback if values look like logits (outside [0,1])
        if cls.max() > 1.0 or cls.min() < 0.0:
            cls = 1.0 / (1.0 + np.exp(-cls))

        # For each prediction, take top class
        cls_ids = np.argmax(cls, axis=1)
        cls_confs = cls[np.arange(cls.shape[0]), cls_ids]

        # Filter by confidence
        keep_mask = cls_confs >= float(CONF_THRESH)
        bx = bx[keep_mask]
        cls_ids = cls_ids[keep_mask]
        cls_confs = cls_confs[keep_mask]

        # Convert cx,cy,w,h to x1,y1,x2,y2 and map back to original image coords
        if bx.size:
            cx = bx[:, 0]
            cy = bx[:, 1]
            w = bx[:, 2]
            h = bx[:, 3]
            x1 = cx - w / 2.0
            y1 = cy - h / 2.0
            x2 = cx + w / 2.0
            y2 = cy + h / 2.0
            x1 = (x1 - pad_w) / r
            y1 = (y1 - pad_h) / r
            x2 = (x2 - pad_w) / r
            y2 = (y2 - pad_h) / r
            boxes = np.stack([x1, y1, x2, y2], axis=1)
            scores = cls_confs.astype(np.float32)
            labels = [COCO_NAMES[i] if i < len(COCO_NAMES) else str(i) for i in cls_ids]
        
        if len(boxes) == 0:
            if display:
                cv2.imshow("YOLO Detection", frame)
                cv2.waitKey(1)
            return []

        keep = nms(np.asarray(boxes, dtype=np.float32), np.asarray(scores, dtype=np.float32), IOU_THRESH)
        results = []
        for i in keep:
            results.append({"label": labels[i], "conf": float(scores[i]), "xyxy": boxes[i].tolist()})
        
        # Draw on frame if display enabled
        if display:
            vis_frame = frame.copy()
            for res in results:
                x1, y1, x2, y2 = [int(v) for v in res["xyxy"]]
                label_text = f"{res['label']} {res['conf']:.2f}"
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis_frame, label_text, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow("YOLO Detection", vis_frame)
            cv2.waitKey(1)
        
        return results


def choose_target_label(detections, allowed: Optional[List[str]] = None) -> Optional[str]:
    if not detections:
        return None
    if allowed:
        detections = [d for d in detections if d["label"] in allowed]
        if not detections:
            return None
    # pick the highest confidence
    detections.sort(key=lambda d: d["conf"], reverse=True)
    return detections[0]["label"]


# =================== MAIN FLOW =======================
def main():
    master = connect_vehicle()
    gps_prearm_check(master)
    set_mode(master, "LOITER")
    arm(master)
    time.sleep(2)

    set_mode(master, "GUIDED")
    takeoff(master, TAKEOFF_ALT)

    # Prepare camera + detector
    reset_servos(master)    

    if CAMERA_SOURCE == "libcamera":
        try:
            from picamera2 import Picamera2
            picam2 = Picamera2()
            picam2.start()
            cap = None
            print("✓ Arducam initialized")
        except ImportError as e:
            print(f"ERROR: picamera2 failed: {e}")
            print("Make sure to run: sudo apt-get install python3-picamera2")
            return
    else:
        cap = cv2.VideoCapture(CAMERA_SOURCE)
        if not cap.isOpened():
            print("ERROR: Cannot open camera source:", CAMERA_SOURCE)
            return
        picam2 = None
    
    detector = YoloDetector(MODEL_PATH)

    # Track which drop servo to use next
    drop_index = 0

    try:
        for idx, (lat, lon, alt) in enumerate(WAYPOINTS):
            print(f"\n=== Leg {idx+1}/{len(WAYPOINTS)} → ({lat:.7f},{lon:.7f},{alt:.1f}m) ===")
            guided_goto(master, lat, lon, alt)
            reached = wait_until_reached(master, lat, lon, REACH_RADIUS_M, timeout_s=180)

            # Run YOLO multiple times at this waypoint
            print(f"Running YOLO detection at point (up to {DETECTION_ATTEMPTS} attempts)…")
            person_detected = False
            
            for attempt in range(1, DETECTION_ATTEMPTS + 1):
                print(f"  Attempt {attempt}/{DETECTION_ATTEMPTS}")
                
                # Capture frame
                if picam2 is not None:
                    frame = picam2.capture_array()
                    # Convert RGBA to BGR for OpenCV/YOLO
                    if frame.shape[2] == 4:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                else:
                    ret, frame = cap.read()
                    if not ret:
                        print("  WARN: No frame from camera; skipping this attempt")
                        time.sleep(0.5)
                        continue
                
                # Run detection
                dets = detector.infer(frame, display=True)
                print(f"  Detections: {dets}")
                
                # Check if person detected
                person_dets = [d for d in dets if d["label"] == "person"]
                if person_dets:
                    # Pick highest confidence person
                    person_dets.sort(key=lambda d: d["conf"], reverse=True)
                    person = person_dets[0]
                    
                    # Calculate center pixel of bounding box
                    x1, y1, x2, y2 = person["xyxy"]
                    pixel_x = (x1 + x2) / 2
                    pixel_y = (y1 + y2) / 2
                    
                    # Get drone position and IMU attitude
                    drone_pos = get_current_position(master)
                    heading = get_heading(master)
                    attitude = get_attitude(master)
                    
                    if drone_pos and heading is not None and attitude is not None:
                        drone_lat, drone_lon, drone_alt = drone_pos
                        drone_roll, drone_pitch, drone_yaw = attitude
                        
                        # Apply camera mount offsets to drone attitude
                        camera_pitch = drone_pitch + CAMERA_PITCH_OFFSET_DEG
                        camera_roll = drone_roll + CAMERA_ROLL_OFFSET_DEG
                        camera_yaw = drone_yaw + CAMERA_YAW_OFFSET_DEG
                        
                        # Debug: print input values
                        print(f"    DEBUG: drone_pos={drone_lat:.7f}, {drone_lon:.7f}, alt={drone_alt:.1f}m")
                        print(f"    DEBUG: drone attitude: roll={drone_roll:.1f}°, pitch={drone_pitch:.1f}°, yaw={drone_yaw:.1f}°")
                        print(f"    DEBUG: camera orientation: pitch={camera_pitch:.1f}°, pixel=({pixel_x:.0f}, {pixel_y:.0f})")
                        
                        # Convert pixel to GPS coordinates using actual drone attitude
                        target_gps = pixel_to_ground_gps(
                            pixel=[pixel_x, pixel_y],
                            drone_lat=drone_lat,
                            drone_lon=drone_lon,
                            altitude_m=drone_alt,
                            heading_deg=heading,
                            pitch_deg=camera_pitch,
                            roll_deg=camera_roll,
                            yaw_deg=camera_yaw
                        )
                        
                        if target_gps:
                            target_lat, target_lon = target_gps
                            print(f"  ✓ PERSON DETECTED at pixel ({pixel_x:.0f}, {pixel_y:.0f})")
                            print(f"  → GPS: {target_lat:.7f}, {target_lon:.7f}")
                            guided_goto(master, target_lat, target_lon, drone_alt)
                            wait_until_reached(master, target_lat, target_lon, REACH_RADIUS_M, timeout_s=180)
                            
                            # Drop packet at detected location
                            if drop_index < len(DROP_SERVOS):
                                drop_packet(master, DROP_SERVOS[drop_index])
                                time.sleep(1)
                                drop_index += 1
                            else:
                                print("  WARN: No more drop servos available")
                            
                            person_detected = True
                            break  # Skip remaining attempts
                        else:
                            print("  WARN: Could not calculate ground coordinates (ray doesn't hit ground)")
                    else:
                        print("  WARN: Could not get drone position/heading")
                
                # Small delay between attempts
                if attempt < DETECTION_ATTEMPTS:
                    time.sleep(0.3)
            
            if not person_detected:
                if drop_index < len(DROP_SERVOS):
                    drop_packet(master, DROP_SERVOS[drop_index])
                    drop_index += 1
                else:
                    print("  WARN: No more drop servos available")

                print("  No person detected in any attempt; continuing to next waypoint.")

        if RTL_AT_END:
            print("\n=== RTL ===")
            master.mav.command_long_send(
                master.target_system,
                master.target_component,
                mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH,
                0, 0, 0, 0, 0, 0, 0, 0
            )
            print("Commanded Return-to-Launch")

        print("\nMission complete.")
    finally:
        time.sleep(2)
        reset_servos(master)
        if picam2 is not None:
            picam2.stop()
        elif cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()