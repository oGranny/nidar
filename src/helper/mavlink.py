from typing import Optional, Tuple
from pymavlink import mavutil
import time
import math

def connect_vehicle(config):
    print(f"Connecting to {config.get("MAVPROXY_URL")}...")
    master = mavutil.mavlink_connection(config.get("MAVPROXY_URL"))
    print(f"Connecting to {config.get("MAVPROXY_URL")}...")
    master.wait_heartbeat()
    print(f"Connected. SYS={master.target_system} COMP={master.target_component}")
    return master


def gps_prearm_check(master, config, min_fix=3, min_sats=6):
    print("\n=== Pre-arm GPS check ===")
    while True:
        msg = master.recv_match(type="GPS_RAW_INT", timeout=5)
        if not msg:
            print("Waiting GPS…") 
            continue
        fix = msg.fix_type
        sats = msg.satellites_visible
        print(f"GPS FIX={fix} SATS={sats}")
        if fix >= min_fix and sats >= min_sats:
            print("✔ GPS OK — PRE-ARM PASS")
            break
        time.sleep(1)


def set_mode(master, config, mode):
    mode_map = master.mode_mapping()
    if mode not in mode_map:
        raise Exception(f"Mode {mode} not supported")
    master.mav.set_mode_send(
        master.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_map[mode]
    )
    print(f"Set Mode → {mode}")
    time.sleep(config.get("DLY_PREARM_CHK", 1))


def arm(master, config):
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
        time.sleep(config.get("DLY_ARM", 1)) 


def takeoff(master, config, alt):
    print("\n=== Takeoff ===")
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0,
        0, 0, 0, 0, 0, 0, alt
    )
    print(f"Takeoff to {alt}m")
    time.sleep(config.get("DLY_TAKEOFF", 4))


def guided_goto(master, config, lat, lon, alt):
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


def get_current_position(master, config) -> Optional[Tuple[float, float, float]]:
    msg = master.recv_match(type="GLOBAL_POSITION_INT", timeout=2)
    if not msg:
        return None
    lat = msg.lat / 1e7
    lon = msg.lon / 1e7
    alt_rel = msg.relative_alt / 1000.0  # mm → m
    return (lat, lon, alt_rel)


def get_heading(master, config) -> Optional[float]:
    """Get current heading in degrees (0-360, 0=North)."""
    msg = master.recv_match(type="VFR_HUD", timeout=2)
    if not msg:
        return None
    return float(msg.heading)


def get_attitude(master, config) -> Optional[Tuple[float, float, float]]:
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


def wait_until_reached(master, config, target_lat, target_lon) -> bool:
    radius_m = config.get("REACH_RADIUS_GUIDED", 5)
    timeout_s = config.get("TIMEOUT_REACH_TARGET_GUIDED", 1800)
    print(f"Waiting until within {radius_m}m of target…")
    t0 = time.time()
    return True
    while time.time() - t0 < timeout_s:
        pos = get_current_position(master)
        if pos is None:
            continue
        lat, lon, _ = pos
        d = haversine_m(lat, lon, target_lat, target_lon)
        print(f"Distance to target: {d:.1f}m")
        if d <= radius_m:
            print(f"✔ Target reached: {pos}")
            return True
        time.sleep(config.get("DLY_POS_CHK_GUIDED", 2))
    print("Timeout waiting to reach target")
    return False

def rtl(master):
    print("\n=== RTL ===")
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH,
        0, 0, 0, 0, 0, 0, 0, 0
    )
    print("Commanded Return-to-Launch")

