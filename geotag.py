#!/usr/bin/env python3
"""
Person Detection, Tracking & Geotagging Pipeline

Uses Hailo object detection with ByteTrack tracking.
Geotags detected persons and saves unique locations to CSV.
Only finalizes GPS when a track ID leaves the frame.

Usage:
    python geotag.py -n models/yolo11s7.hef -i camera --track
    python geotag.py -n /path/to/model.hef -i video.mp4 --track

Requires:
    - Hailo inference engine
    - pymavlink (optional, for real drone telemetry)
"""

import argparse
import csv
import json
import math
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add paths for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src', 'hailo')))

from src.scanning.transform import pixel_to_ground_gps

# ==================== CONFIGURATION ====================
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')
CSV_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'geotag_results.csv')
FINAL_GEOTAGS_PATH = os.path.join(os.path.dirname(__file__), 'final_geotags.csv')

# GPS distance threshold in meters - positions within this distance are considered the same
GPS_THRESHOLD_METERS = 5.0

# Load config
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)


@dataclass
class TrackGeotagData:
    """Stores geotagging data for a single tracked person."""
    track_id: int
    class_name: str
    geotags: List[Tuple[float, float]] = field(default_factory=list)
    centroids: List[Tuple[int, int]] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    
    def add_observation(self, centroid: Tuple[int, int], geotag: Optional[Tuple[float, float]] = None):
        """Add a new observation for this track."""
        self.centroids.append(centroid)
        self.timestamps.append(time.time())
        self.last_seen = time.time()
        if geotag:
            self.geotags.append(geotag)
    
    def get_final_gps(self) -> Optional[Tuple[float, float]]:
        """Get the final averaged GPS position."""
        if not self.geotags:
            return None
        avg_lat = sum(g[0] for g in self.geotags) / len(self.geotags)
        avg_lon = sum(g[1] for g in self.geotags) / len(self.geotags)
        return (avg_lat, avg_lon)
    
    def get_duration(self) -> float:
        """Get tracking duration in seconds."""
        return self.last_seen - self.first_seen


class DroneTelemetry:
    """Read-only MAVLink telemetry reader with simulated fallback."""
    
    def __init__(self, connection_string: str = None, simulated: bool = False):
        self.connection_string = connection_string
        self.simulated = simulated
        self.master = None
        self.connected = False
        
        # Cached telemetry (default simulated values)
        self.lat = 12.9716  # Default: Bangalore
        self.lon = 77.5946
        self.alt = 50.0  # 50m altitude
        self.heading = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        
        self._running = False
        self._thread = None
    
    def connect(self) -> bool:
        """Connect to drone (read-only)."""
        if self.simulated:
            print("✓ Using simulated telemetry")
            self.connected = True
            return True
        
        try:
            from pymavlink import mavutil
            print(f"Connecting to drone at {self.connection_string}...")
            self.master = mavutil.mavlink_connection(self.connection_string)
            self.master.wait_heartbeat(timeout=10)
            print(f"✓ Connected! SYS={self.master.target_system} COMP={self.master.target_component}")
            self.connected = True
            
            # Start background telemetry thread
            self._running = True
            self._thread = threading.Thread(target=self._telemetry_loop, daemon=True)
            self._thread.start()
            return True
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            print("  Falling back to simulated telemetry")
            self.simulated = True
            self.connected = True
            return True
    
    def _telemetry_loop(self):
        """Background thread to read telemetry."""
        while self._running:
            try:
                msg = self.master.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=0.1)
                if msg:
                    self.lat = msg.lat / 1e7
                    self.lon = msg.lon / 1e7
                    self.alt = msg.relative_alt / 1000.0
                
                msg = self.master.recv_match(type='VFR_HUD', blocking=True, timeout=0.1)
                if msg:
                    self.heading = float(msg.heading)
                
                msg = self.master.recv_match(type='ATTITUDE', blocking=True, timeout=0.1)
                if msg:
                    self.roll = math.degrees(msg.roll)
                    self.pitch = math.degrees(msg.pitch)
                    self.yaw = math.degrees(msg.yaw)
                    if self.yaw < 0:
                        self.yaw += 360
            except Exception:
                pass
    
    def get_telemetry(self) -> Tuple[float, float, float, float, float, float, float]:
        """Returns (lat, lon, alt, heading, roll, pitch, yaw)"""
        return (self.lat, self.lon, self.alt, self.heading, self.roll, self.pitch, self.yaw)
    
    def close(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)
        if self.master:
            self.master.close()


class GeotagManager:
    """
    Manages person geotagging, deduplication, and CSV storage.
    
    Logic:
    - Track each person by their track_id
    - When a track is lost (person leaves frame), compute final GPS
    - Check if this GPS is within threshold of any existing known geotag
    - If within threshold: don't add to final (duplicate)
    - If outside threshold: add as new unique geotag
    """
    
    def __init__(self, csv_path: str, final_csv_path: str, threshold_meters: float = 5.0):
        self.csv_path = csv_path
        self.final_csv_path = final_csv_path
        self.threshold_meters = threshold_meters
        
        # Active tracks being monitored
        self.active_tracks: Dict[int, TrackGeotagData] = {}
        
        # Known unique geotags (deduplicated)
        self.known_geotags: List[Dict] = []
        
        # All track history (for detailed CSV)
        self.all_tracks: List[Dict] = []
        
        # Load existing known geotags if file exists
        self._load_known_geotags()
        
        # Initialize CSV files
        self._init_csv_files()
    
    def _load_known_geotags(self):
        """Load existing known geotags from final CSV."""
        if os.path.exists(self.final_csv_path):
            try:
                with open(self.final_csv_path, 'r', newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        self.known_geotags.append({
                            'id': int(row['id']),
                            'lat': float(row['latitude']),
                            'lon': float(row['longitude']),
                            'class': row['class'],
                            'timestamp': row['timestamp'],
                            'track_ids': row.get('track_ids', str(row['id']))
                        })
                print(f"✓ Loaded {len(self.known_geotags)} existing geotags from {self.final_csv_path}")
            except Exception as e:
                print(f"⚠ Error loading existing geotags: {e}")
    
    def _init_csv_files(self):
        """Initialize CSV files with headers."""
        # Detailed track log
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'track_id', 'class', 'final_lat', 'final_lon',
                    'num_observations', 'duration_sec', 'timestamp',
                    'is_duplicate', 'matched_geotag_id'
                ])
        
        # Final unique geotags (recreate to ensure consistency)
        self._save_final_geotags()
    
    def _save_final_geotags(self):
        """Save known geotags to final CSV."""
        with open(self.final_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'latitude', 'longitude', 'class', 'timestamp', 'track_ids'])
            for geotag in self.known_geotags:
                writer.writerow([
                    geotag['id'],
                    f"{geotag['lat']:.7f}",
                    f"{geotag['lon']:.7f}",
                    geotag['class'],
                    geotag['timestamp'],
                    geotag['track_ids']
                ])
    
    @staticmethod
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two GPS coordinates in meters."""
        R = 6371000  # Earth radius in meters
        
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        
        a = math.sin(delta_phi / 2) ** 2 + \
            math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c
    
    def find_matching_geotag(self, lat: float, lon: float) -> Optional[Dict]:
        """Find if a GPS position matches any known geotag within threshold."""
        for geotag in self.known_geotags:
            distance = self.haversine_distance(lat, lon, geotag['lat'], geotag['lon'])
            if distance <= self.threshold_meters:
                return geotag
        return None
    
    def update_track(self, track_id: int, class_name: str, centroid: Tuple[int, int],
                     geotag: Optional[Tuple[float, float]] = None):
        """Update tracking data for a person."""
        if track_id not in self.active_tracks:
            self.active_tracks[track_id] = TrackGeotagData(
                track_id=track_id,
                class_name=class_name
            )
        
        self.active_tracks[track_id].add_observation(centroid, geotag)
    
    def finalize_track(self, track_id: int) -> Optional[Dict]:
        """
        Finalize a track when the person leaves the frame.
        
        Returns:
            Dict with track info and whether it was a duplicate
        """
        if track_id not in self.active_tracks:
            return None
        
        track_data = self.active_tracks.pop(track_id)
        final_gps = track_data.get_final_gps()
        
        result = {
            'track_id': track_id,
            'class': track_data.class_name,
            'final_gps': final_gps,
            'num_observations': len(track_data.centroids),
            'num_geotags': len(track_data.geotags),
            'duration': track_data.get_duration(),
            'is_duplicate': False,
            'matched_geotag_id': None
        }
        
        if final_gps:
            lat, lon = final_gps
            
            # Check if this matches any known geotag
            matching = self.find_matching_geotag(lat, lon)
            
            if matching:
                # Duplicate found - update existing geotag's track_ids
                result['is_duplicate'] = True
                result['matched_geotag_id'] = matching['id']
                
                # Update track_ids list
                track_ids = matching['track_ids'].split(',')
                if str(track_id) not in track_ids:
                    track_ids.append(str(track_id))
                    matching['track_ids'] = ','.join(track_ids)
                
                print(f"\n{'='*60}")
                print(f"🔄 DUPLICATE DETECTED: Track {track_id} matches Geotag #{matching['id']}")
                print(f"   Distance: {self.haversine_distance(lat, lon, matching['lat'], matching['lon']):.2f}m")
                print(f"   GPS: ({lat:.7f}, {lon:.7f})")
                print(f"{'='*60}\n")
            else:
                # New unique geotag
                new_id = len(self.known_geotags) + 1
                new_geotag = {
                    'id': new_id,
                    'lat': lat,
                    'lon': lon,
                    'class': track_data.class_name,
                    'timestamp': datetime.now().isoformat(),
                    'track_ids': str(track_id)
                }
                self.known_geotags.append(new_geotag)
                result['matched_geotag_id'] = new_id
                
                print(f"\n{'='*60}")
                print(f"✅ NEW UNIQUE GEOTAG: #{new_id}")
                print(f"   Track ID: {track_id} ({track_data.class_name})")
                print(f"   📍 GPS: ({lat:.7f}, {lon:.7f})")
                print(f"   Duration: {track_data.get_duration():.1f}s")
                print(f"   Observations: {len(track_data.geotags)}")
                print(f"{'='*60}\n")
                
                # Save updated final geotags
                self._save_final_geotags()
        else:
            print(f"\n⚠ Track {track_id} lost without GPS data")
        
        # Log to detailed CSV
        self._log_track_to_csv(result)
        
        return result
    
    def _log_track_to_csv(self, result: Dict):
        """Log track finalization to detailed CSV."""
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            gps = result['final_gps']
            writer.writerow([
                result['track_id'],
                result['class'],
                f"{gps[0]:.7f}" if gps else '',
                f"{gps[1]:.7f}" if gps else '',
                result['num_observations'],
                f"{result['duration']:.2f}",
                datetime.now().isoformat(),
                result['is_duplicate'],
                result['matched_geotag_id']
            ])
    
    def check_lost_tracks(self, current_track_ids: set) -> List[Dict]:
        """Check for tracks that are no longer active and finalize them."""
        lost_ids = set(self.active_tracks.keys()) - current_track_ids
        results = []
        
        for lost_id in lost_ids:
            result = self.finalize_track(lost_id)
            if result:
                results.append(result)
        
        return results
    
    def get_summary(self) -> Dict:
        """Get summary of geotag statistics."""
        return {
            'active_tracks': len(self.active_tracks),
            'unique_geotags': len(self.known_geotags),
            'threshold_meters': self.threshold_meters
        }


# Global instances
drone_telemetry: Optional[DroneTelemetry] = None
geotag_manager: Optional[GeotagManager] = None


def compute_geotag(centroid: Tuple[int, int], image_shape: tuple) -> Optional[Tuple[float, float]]:
    """
    Compute GPS coordinates for a detection centroid.
    
    Args:
        centroid: (x, y) pixel coordinates
        image_shape: (height, width, channels) of the frame
        
    Returns:
        (latitude, longitude) or None
    """
    global drone_telemetry
    
    if drone_telemetry is None or not drone_telemetry.connected:
        return None
    
    lat, lon, alt, heading, roll, pitch, yaw = drone_telemetry.get_telemetry()
    
    if alt <= 0:
        return None
    
    # Apply camera mount offsets
    camera_pitch = pitch + config.get("CAMERA_PITCH_OFFSET_DEG", 90)
    camera_roll = roll + config.get("CAMERA_ROLL_OFFSET_DEG", 0)
    camera_yaw = yaw + config.get("CAMERA_YAW_OFFSET_DEG", 0)
    
    try:
        result = pixel_to_ground_gps(
            pixel=centroid,
            config=config,
            drone_lat=lat,
            drone_lon=lon,
            altitude_m=alt,
            roll_deg=camera_roll,
            pitch_deg=camera_pitch,
            yaw_deg=camera_yaw
        )
        return result
    except Exception as e:
        return None


def geotag_callback(track_id: int, class_name: str, centroid: Tuple[int, int], 
                    bbox: List[int], image_shape: tuple):
    """
    Callback for each tracked detection.
    Called from the object detection post-processor.
    """
    global geotag_manager
    
    if geotag_manager is None:
        return
    
    # Compute geotag for this detection
    geotag = compute_geotag(centroid, image_shape)
    
    # Update track
    geotag_manager.update_track(track_id, class_name, centroid, geotag)


def on_tracks_update(current_track_ids: set):
    """
    Called after each frame to check for lost tracks.
    """
    global geotag_manager
    
    if geotag_manager is None:
        return []
    
    return geotag_manager.check_lost_tracks(current_track_ids)


def init_geotag_system(mavproxy_url: str = None, simulated: bool = False,
                       csv_path: str = None, final_csv_path: str = None,
                       threshold_meters: float = GPS_THRESHOLD_METERS):
    """Initialize the geotagging system."""
    global drone_telemetry, geotag_manager
    
    # Initialize telemetry
    drone_telemetry = DroneTelemetry(
        connection_string=mavproxy_url or config.get("MAVPROXY_URL"),
        simulated=simulated
    )
    drone_telemetry.connect()
    
    # Initialize geotag manager
    geotag_manager = GeotagManager(
        csv_path=csv_path or CSV_OUTPUT_PATH,
        final_csv_path=final_csv_path or FINAL_GEOTAGS_PATH,
        threshold_meters=threshold_meters
    )
    
    print(f"\n{'='*60}")
    print("GEOTAG SYSTEM INITIALIZED")
    print(f"  GPS Threshold: {threshold_meters}m")
    print(f"  Detailed CSV: {csv_path or CSV_OUTPUT_PATH}")
    print(f"  Final Geotags: {final_csv_path or FINAL_GEOTAGS_PATH}")
    print(f"  Known Geotags: {len(geotag_manager.known_geotags)}")
    print(f"{'='*60}\n")


def shutdown_geotag_system():
    """Shutdown the geotagging system and finalize all active tracks."""
    global drone_telemetry, geotag_manager
    
    if geotag_manager:
        # Finalize all remaining active tracks
        remaining_ids = list(geotag_manager.active_tracks.keys())
        for track_id in remaining_ids:
            geotag_manager.finalize_track(track_id)
        
        print(f"\n{'='*60}")
        print("GEOTAG SYSTEM SHUTDOWN")
        print(f"  Final Unique Geotags: {len(geotag_manager.known_geotags)}")
        print(f"{'='*60}\n")
    
    if drone_telemetry:
        drone_telemetry.close()


# ==================== MODIFIED POST-PROCESSOR ====================
# This section patches the object_detection post-processor to use our geotag system

def patch_post_processor():
    """
    Patch the object_detection post-processor to integrate geotagging.
    This modifies draw_detections to call our geotag callbacks.
    """
    try:
        from src.hailo.object_detection import object_detection_post_process as pp
        
        # Store original function
        original_draw_detections = pp.draw_detections
        
        def patched_draw_detections(detections: dict, img_out, labels, tracker=None, draw_trail=False):
            """Patched version that integrates geotagging."""
            global geotag_manager
            
            # Call original function
            result = original_draw_detections(detections, img_out, labels, tracker, draw_trail)
            
            # Track current IDs and trigger geotag updates
            if tracker and geotag_manager:
                boxes = detections["detection_boxes"]
                classes = detections["detection_classes"]
                num_detections = detections["num_detections"]
                
                # Get current track IDs from tracker
                dets_for_tracker = []
                for idx in range(num_detections):
                    box = boxes[idx]
                    score = detections["detection_scores"][idx]
                    dets_for_tracker.append([*box, score])
                
                if dets_for_tracker:
                    online_targets = tracker.update(np.array(dets_for_tracker), return_only=True)
                    current_ids = set()
                    
                    for track in online_targets:
                        track_id = track.track_id
                        current_ids.add(track_id)
                        
                        x1, y1, x2, y2 = track.tlbr
                        centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                        
                        # Find best matching class
                        best_idx = pp.find_best_matching_detection_index(track.tlbr, boxes)
                        class_name = labels[classes[best_idx]] if best_idx is not None else "unknown"
                        
                        # Call geotag callback
                        geotag_callback(
                            track_id=track_id,
                            class_name=class_name,
                            centroid=centroid,
                            bbox=[int(x1), int(y1), int(x2), int(y2)],
                            image_shape=img_out.shape
                        )
                    
                    # Check for lost tracks
                    on_tracks_update(current_ids)
            
            return result
        
        # Replace function
        pp.draw_detections = patched_draw_detections
        print("✓ Post-processor patched for geotagging")
        
    except Exception as e:
        print(f"⚠ Could not patch post-processor: {e}")


# ==================== STANDALONE INTEGRATION ====================

def integrate_with_object_detection():
    """
    Integrate geotagging with the object_detection pipeline.
    Call this before running object_detection.run_inference_pipeline()
    """
    init_geotag_system()
    
    # Set up callback in post-processor
    try:
        from src.hailo.object_detection import object_detection_post_process as pp
        
        # Set telemetry getter
        pp.set_drone_telemetry_getter(lambda: drone_telemetry.get_telemetry() if drone_telemetry else None)
        
        # Set track lost callback
        def on_track_lost(track_id: int, track_data: dict):
            if geotag_manager:
                geotag_manager.finalize_track(track_id)
        
        pp.set_track_lost_callback(on_track_lost)
        
        print("✓ Geotagging integrated with object_detection")
        
    except Exception as e:
        print(f"⚠ Integration error: {e}")


# ==================== MAIN ====================

def main():
    """
    Main function - can be run standalone or imported.
    
    Standalone usage:
        python geotag.py -n model.hef -i camera --track --simulated
    """
    parser = argparse.ArgumentParser(
        description="Person detection with geotagging and deduplication",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("-n", "--net", type=str, required=True, help="HEF model path")
    parser.add_argument("-i", "--input", type=str, default="camera", help="Input source")
    parser.add_argument("--track", action="store_true", help="Enable tracking (required)")
    parser.add_argument("--simulated", action="store_true", help="Use simulated telemetry")
    parser.add_argument("--threshold", type=float, default=GPS_THRESHOLD_METERS,
                        help=f"GPS deduplication threshold in meters (default: {GPS_THRESHOLD_METERS})")
    parser.add_argument("--csv", type=str, default=CSV_OUTPUT_PATH, help="Output CSV path")
    parser.add_argument("--final-csv", type=str, default=FINAL_GEOTAGS_PATH, help="Final geotags CSV path")
    parser.add_argument("-l", "--labels", type=str, 
                        default="src/hailo/common/coco.txt", help="Labels file")
    parser.add_argument("-s", "--save_stream_output", action="store_true")
    parser.add_argument("-o", "--output-dir", type=str, default="output")
    parser.add_argument("--show-fps", action="store_true")
    parser.add_argument("--draw-trail", action="store_true")
    
    args = parser.parse_args()
    
    if not args.track:
        print("⚠ Warning: --track flag is required for geotagging. Enabling automatically.")
        args.track = True
    
    # Initialize geotag system
    init_geotag_system(
        simulated=args.simulated,
        csv_path=args.csv,
        final_csv_path=args.final_csv,
        threshold_meters=args.threshold
    )
    
    try:
        # Import and run object detection pipeline
        from src.hailo.object_detection.object_detection import run_inference_pipeline
        
        # Integrate our callbacks
        integrate_with_object_detection()
        
        # Run the pipeline
        run_inference_pipeline(
            net=args.net,
            input=args.input,
            batch_size=1,
            labels=args.labels,
            output_dir=args.output_dir,
            save_stream_output=args.save_stream_output,
            enable_tracking=args.track,
            show_fps=args.show_fps,
            draw_trail=args.draw_trail
        )
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        shutdown_geotag_system()


if __name__ == "__main__":
    main()
