"""
Tracklet Database Module

Stores tracklet GPS coordinates in SQLite database using drone telemetry
and camera projection to convert pixel coordinates to ground GPS.
"""

import math
import os
import sqlite3
import sys
import json
from threading import Lock
from typing import Mapping, Optional, Tuple
from pymavlink import mavutil
# Fix imports: Add src root to sys.path to allow sibling package imports
current_file_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate: src/hailo/object_detection -> src/hailo -> src
src_root = os.path.abspath(os.path.join(current_file_dir, "../../"))

# Insert at beginning of path to ensure priority
if src_root not in sys.path:
    sys.path.insert(0, src_root)

try:
    # Primary import attempt assuming 'scanning' is a package in 'src'
    from scanning.transform import gps_distance_m, pixel_to_ground_gps
except ImportError:
    # Fallback: Add 'scanning' dir directly to path (legacy support)
    scanning_dir = os.path.join(src_root, "scanning")
    if scanning_dir not in sys.path:
        sys.path.append(scanning_dir)
    from transform import pixel_to_ground_gps, gps_distance_m

# Thread-safe database lock
_db_lock = Lock()

# Default database path
DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "tracklets.db")

url = "udp:127.0.0.1:14550"
print(f"Connecting to {"udp:127.0.0.1:14550"}...")
master = mavutil.mavlink_connection(url)
print(f"Connecting to {"udp:127.0.0.1:14550"}...")
master.wait_heartbeat()
print(f"Connected. SYS={master.target_system} COMP={master.target_component}")

def get_attitude(master, config=None) -> Optional[Tuple[float, float, float]]:
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

def get_current_position(master, config=None) -> Optional[Tuple[float, float, float]]:
    msg = master.recv_match(type="GLOBAL_POSITION_INT", timeout=2)
    if not msg:
        return None
    lat = msg.lat / 1e7
    lon = msg.lon / 1e7
    alt_rel = msg.relative_alt / 1000.0  # mm → m
    return (lat, lon, alt_rel)


def get_connection(db_path=None):
    """
    Get a SQLite database connection.
    
    Args:
        db_path (str, optional): Path to the database file.
        
    Returns:
        sqlite3.Connection: Database connection object.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    return sqlite3.connect(db_path)


def init_database(db_path=None):
    """
    Initialize the SQLite database and create the tracklets table if it doesn't exist.
    
    Args:
        db_path (str, optional): Path to the database file.
    """
    with _db_lock:
        conn = get_connection(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tracklets (
                track_id INTEGER PRIMARY KEY,
                class_id INTEGER NOT NULL,
                class_name TEXT,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                confidence REAL,
                drone_lat REAL,
                drone_lon REAL,
                drone_alt REAL,
                first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()


def get_drone_telemetry() -> Optional[dict]:
    """
    Get current drone GPS position and attitude from mavlink.
    
    Args:
        master: pymavlink mavutil connection object.
        
    Returns:
        dict with lat, lon, alt, roll, pitch, yaw or None if unavailable.
    """
    # math is already imported globally
    
    if master is None:
        return None
    
    telemetry = {}
    a = get_current_position(master)
    b = get_attitude(master)
    print(a, b)
    
    # Get GPS position
    if a:
        telemetry['lat'] = a[0]
        telemetry['lon'] = a[1]
        telemetry['alt'] = a[2]
    else:
        return None
    
    # Get attitude
    if b:
        telemetry['roll'] = b[0]
        telemetry['pitch'] = b[1]
        telemetry['yaw'] = b[2]
    else:
        # Default to level flight if attitude unavailable
        telemetry['roll'] = 0.0
        telemetry['pitch'] = 0.0
        telemetry['yaw'] = 0.0
    
    return telemetry


DIST_THRESHOLD_M = 4.0

def broadcast_database(conn):
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM tracklets")
        rows = cursor.fetchall()
        
        # Get column names
        columns = [description[0] for description in cursor.description]
        data = []
        for row in rows:
            data.append(dict(zip(columns, row)))
            
        json_data = json.dumps(data)
        
        # Send via MAVLink STATUSTEXT in chunks
        chunk_size = 50
        if master:
            for i in range(0, len(json_data), chunk_size):
                chunk = json_data[i:i+chunk_size]
                master.mav.statustext_send(mavutil.mavlink.MAV_SEVERITY_INFO, chunk.encode('utf-8'))
                
    except Exception as e:
        print(f"Error broadcasting database: {e}")

def upsert_tracklet(
    track_id: int,
    class_id: int,
    xmin: int,
    ymin: int,
    xmax: int,
    ymax: int,
    config: Mapping[str, float],
    class_name: str = None,
    confidence: float = None,
    db_path: str = None,
) -> Optional[Tuple[float, float]]:
    """
    Insert or update a tracklet record with GPS coordinates.
    Uses live drone telemetry from mavlink master connection.
    
    Args:
        track_id: Unique tracker ID.
        class_id: Class index of the detected object.
        xmin, ymin, xmax, ymax: Bounding box coordinates in pixels.
        config: Camera intrinsics config.
        class_name: Human-readable class name.
        confidence: Detection confidence score.
        db_path: Path to the database file.
        
    Returns:
        Tuple of (latitude, longitude) if successful, None otherwise.
    """
    # Get live telemetry from mavlink
    telemetry = get_drone_telemetry()
    if telemetry is None:
        print(f"Warning: No telemetry available for track_id {track_id}")
        return None

    drone_lat = telemetry["lat"]
    drone_lon = telemetry["lon"]
    altitude_m = telemetry["alt"]
    roll_deg = telemetry["roll"]
    pitch_deg = telemetry["pitch"]
    yaw_deg = telemetry["yaw"]

    gps_coords = pixel_to_ground_gps(
        pixel=((xmin+xmax)/2.0, (ymin+ymax)/2.0),
        config=config,
        drone_lat=drone_lat,
        drone_lon=drone_lon,
        altitude_m=altitude_m,
        roll_deg=roll_deg,
        pitch_deg=pitch_deg,
        yaw_deg=yaw_deg,
    )

    if gps_coords is None:
        print(f"Warning: Could not project track_id {track_id} to ground GPS")
        return None

    lat, lon = gps_coords

    # transform imports handled at top of file

    with _db_lock:
        conn = get_connection(db_path)
        cur = conn.cursor()

        # If track_id already exists, allow normal update
        cur.execute("SELECT 1 FROM tracklets WHERE track_id = ? LIMIT 1", (track_id,))
        exists_same_id = cur.fetchone() is not None

        if not exists_same_id:
            # New track_id -> compare to all existing tracklets
            cur.execute("SELECT track_id, latitude, longitude FROM tracklets")
            for existing_id, existing_lat, existing_lon in cur.fetchall():
                dist_m = gps_distance_m(lat, lon, float(existing_lat), float(existing_lon))
                if dist_m < DIST_THRESHOLD_M:
                    print(
                        f"Track {track_id} NOT inserted: already exists near track {existing_id} "
                        f"(dist={dist_m:.2f}m < {DIST_THRESHOLD_M}m)"
                    )
                    conn.close()
                    return None

        cur.execute(
            '''
            INSERT INTO tracklets (
                track_id, class_id, class_name, latitude, longitude,
                confidence, drone_lat, drone_lon, drone_alt, last_updated
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(track_id) DO UPDATE SET
                class_id = excluded.class_id,
                class_name = excluded.class_name,
                latitude = excluded.latitude,
                longitude = excluded.longitude,
                confidence = excluded.confidence,
                drone_lat = excluded.drone_lat,
                drone_lon = excluded.drone_lon,
                drone_alt = excluded.drone_alt,
                last_updated = CURRENT_TIMESTAMP
            ''',
            (track_id, class_id, class_name, lat, lon, confidence, drone_lat, drone_lon, altitude_m),
        )

        conn.commit()
        
        broadcast_database(conn)

        conn.close()

    return lat, lon


def get_tracklet(track_id, db_path=None):
    """
    Retrieve a tracklet record by track_id.
    
    Args:
        track_id (int): Unique tracker ID.
        db_path (str, optional): Path to the database file.
        
    Returns:
        dict or None: Tracklet data as a dictionary, or None if not found.
    """
    with _db_lock:
        conn = get_connection(db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM tracklets WHERE track_id = ?', (track_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'track_id': row[0],
                'class_id': row[1],
                'class_name': row[2],
                'latitude': row[3],
                'longitude': row[4],
                'confidence': row[5],
                'drone_lat': row[6],
                'drone_lon': row[7],
                'drone_alt': row[8],
                'first_seen': row[9],
                'last_updated': row[10]
            }
        return None


def get_all_tracklets(db_path=None):
    """
    Retrieve all tracklet records from the database.
    
    Args:
        db_path (str, optional): Path to the database file.
        
    Returns:
        list: List of tracklet dictionaries.
    """
    with _db_lock:
        conn = get_connection(db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM tracklets')
        rows = cursor.fetchall()
        conn.close()
        
        tracklets = []
        for row in rows:
            tracklets.append({
                'track_id': row[0],
                'class_id': row[1],
                'class_name': row[2],
                'latitude': row[3],
                'longitude': row[4],
                'confidence': row[5],
                'drone_lat': row[6],
                'drone_lon': row[7],
                'drone_alt': row[8],
                'first_seen': row[9],
                'last_updated': row[10]
            })
        return tracklets


def delete_tracklet(track_id, db_path=None):
    """
    Delete a tracklet record by track_id.
    
    Args:
        track_id (int): Unique tracker ID.
        db_path (str, optional): Path to the database file.
    """
    with _db_lock:
        conn = get_connection(db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM tracklets WHERE track_id = ?', (track_id,))
        conn.commit()
        conn.close()


def clear_all_tracklets(db_path=None):
    """
    Delete all tracklet records from the database.
    
    Args:
        db_path (str, optional): Path to the database file.
    """
    with _db_lock:
        conn = get_connection(db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM tracklets')
        conn.commit()
        conn.close()
