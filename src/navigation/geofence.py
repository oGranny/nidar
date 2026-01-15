import xml.etree.ElementTree as ET
from pymavlink import mavutil

def kml_to_boundary_coordinates(kml_file_path):
    """
    Parses a KML file and extracts the coordinates of the boundary.
    Assumes the KML contains a Polygon or LineString representing the boundary.

    Args:
        kml_file_path (str): Path to the KML file.

    Returns:
        list: A list of tuples (latitude, longitude) representing the boundary coordinates.
    """
    tree = ET.parse(kml_file_path)
    root = tree.getroot()

    # Namespace for KML usually needs to be handled
    namespace = {'kml': 'http://www.opengis.net/kml/2.2'}
    
    # Find the coordinates element. This is a simple implementation that looks for
    # the first <coordinates> tag within a Polygon or LineString.
    # It might need adjustment based on the specific structure of the KML files used.
    coordinates_element = root.find('.//kml:coordinates', namespace)
    
    if coordinates_element is None:
        # Fallback: try searching without namespace or check for different structure
        coordinates_element = root.find('.//coordinates')

    if coordinates_element is None or not coordinates_element.text:
        return []

    # Extract text and split by whitespace
    coord_text = coordinates_element.text.strip()
    coords_list = coord_text.split()

    boundary_coordinates = []
    for coord in coords_list:
        try:
            # KML coordinates are usually longitude,latitude,altitude
            parts = coord.split(',')
            if len(parts) >= 2:
                lon = float(parts[0])
                lat = float(parts[1])
                boundary_coordinates.append((lat, lon))
        except ValueError:
            continue

    return boundary_coordinates


def upload_geofence(master, coordinates):
    """
    Uploads a geofence to the flight controller using MAVLink.

    Args:
        master: The MAVLink connection object.
        coordinates (list): List of (latitude, longitude) tuples.
    """
    # Fence type: inclusion polygon (param value 1 for FENCE_TYPE usually refers to inclusion)
    # This example assumes usage of the FENCE_POINT protocol or relevant MAV_CMD_DO_FENCE_ENABLE if needed,
    # but typically it involves sending FENCE_POINT messages.
    
    # Check if we have coordinates
    if not coordinates:
        print("No coordinates to upload.")
        return

    # Limit to 250 points max
    if len(coordinates) > 250:
        coordinates = coordinates[:250]

    count = len(coordinates)
    
    # Send FENCE_TOTAL first
    # target_system, target_component, count
    # Note: ArduPilot specific mainly. modern protocol uses MAV_CMD_DO_FENCE_ENABLE or mission items
    # but let's stick to the common simple FENCE_POINT upload.
    
    # Actually, for newer ArduPilot firmware (> 4.0), geofences are often uploaded as MISSION_ITEMs 
    # with the sequence MAV_CMD_NAV_FENCE_... or using the specific FENCE_POINT message sequence.
    # The simplest legacy method is FENCE_POINT.
    
    # 1. Disable fence
    # master.mav.command_long_send(
    #     master.target_system, master.target_component,
    #     mavutil.mavlink.MAV_CMD_DO_FENCE_ENABLE, 0,
    #     0, 0, 0, 0, 0, 0, 0)

    # 2. Send total point count (deprecated in some protocols but useful for simple setup)
    # Some GCS implementations send a param set for FENCE_TOTAL.
    
    print(f"Uploading {count} fence points...")

    for i, (lat, lon) in enumerate(coordinates):
        # Index starts at 0 for return point? It depends on flight stack. 
        # Usually 0 is return point, 1..N correspond to polygon vertices.
        # But often we just upload vertices. 
        
        # message: FENCE_POINT
        # target_system, target_component, idx, count, lat, lng
        master.mav.fence_point_send(
            master.target_system, 
            master.target_component, 
            i, 
            count, 
            lat, 
            lon
        )
        print(f"Sent point {i}: {lat}, {lon}")

    print("Geofence upload complete.")


if __name__ == "__main__":
    # Example usage
    kml_path = "dts.kml"
    boundary_coords = kml_to_boundary_coordinates(kml_path)
    
    # Create MAVLink connection (adjust connection string as needed)
    connection_string = 'udp:127.0.0.1:14550'
    print(f"Connecting to {connection_string}...")
    master = mavutil.mavlink_connection(connection_string)
    master.wait_heartbeat()
    print("Heartbeat received!")

    upload_geofence(master, boundary_coords)

    # for lat, lon in boundary_coords:
    #     print(f"({lat}, {lon}),")