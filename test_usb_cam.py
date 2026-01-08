#!/usr/bin/env python3
"""
Simple USB camera test script
Tests cv2.VideoCapture with different device indices
"""

import cv2
import time

def test_camera(device_id):
    """Test a camera device"""
    print(f"\n=== Testing camera device {device_id} ===")
    
    cap = cv2.VideoCapture(device_id)
    
    if not cap.isOpened():
        print(f"✗ Cannot open device {device_id}")
        return False
    
    # Get camera properties
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"✓ Device {device_id} opened successfully")
    print(f"  Resolution: {int(width)}x{int(height)}")
    print(f"  FPS: {fps}")
    
    # Try to capture a few frames
    print("  Capturing frames (press 'q' to exit, 's' to save frame)...")
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print(f"  ✗ Failed to capture frame {frame_count + 1}")
            break
        
        frame_count += 1
        
        # Display the frame
        cv2.imshow(f'USB Camera Test - Device {device_id}', frame)
        
        # Add frame info overlay
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Device: {device_id}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to quit, 's' to save", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow(f'USB Camera Test - Device {device_id}', frame)
        
        # Wait for key press
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print(f"  Exiting... Captured {frame_count} frames")
            break
        elif key == ord('s'):
            filename = f"test_frame_{device_id}_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"  Saved frame to {filename}")
        
        # Auto-exit after 100 frames for quick test
        if frame_count >= 100:
            print(f"  Auto-exiting after {frame_count} frames")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"✓ Device {device_id} test complete. Captured {frame_count} frames.")
    return True


def main():
    print("USB Camera Test Script")
    print("=" * 50)
    
    # Test common device indices
    devices_to_test = [0, 1, 2]
    
    print("\nTesting USB camera devices...")
    print("Available devices:")
    
    working_devices = []
    
    for device_id in devices_to_test:
        cap = cv2.VideoCapture(device_id)
        if cap.isOpened():
            working_devices.append(device_id)
            print(f"  Device {device_id}: Available")
            cap.release()
        else:
            print(f"  Device {device_id}: Not available")
    
    if not working_devices:
        print("\n✗ No USB cameras found!")
        print("\nTroubleshooting:")
        print("  1. Check USB connection")
        print("  2. Run: ls /dev/video*")
        print("  3. Check camera permissions")
        return
    
    print(f"\n✓ Found {len(working_devices)} camera(s): {working_devices}")
    
    # Test each working device
    for device_id in working_devices:
        response = input(f"\nTest device {device_id}? (y/n): ").lower()
        if response == 'y':
            test_camera(device_id)
    
    print("\n=== Test Complete ===")


if __name__ == "__main__":
    main()
