#!/usr/bin/env python3
"""
Arducam test script using picamera2 and libcamera
Tests Arducam camera functionality on Raspberry Pi
"""

import time
import numpy as np
import cv2

def test_picamera2():
    """Test Arducam using picamera2 library"""
    print("\n=== Testing Arducam with picamera2 ===")
    
    try:
        from picamera2 import Picamera2
        print("✓ picamera2 library imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import picamera2: {e}")
        print("\nInstall with:")
        print("  sudo apt-get update")
        print("  sudo apt-get install -y python3-picamera2")
        return False
    
    try:
        # Initialize camera
        print("\nInitializing Arducam...")
        picam2 = Picamera2()
        
        # Get camera info
        print("\nCamera Information:")
        camera_config = picam2.create_preview_configuration()
        print(f"  Configuration: {camera_config}")
        
        # Start camera
        print("\nStarting camera...")
        picam2.start()
        print("✓ Camera started successfully")
        
        # Let camera warm up
        time.sleep(2)
        
        print("\nCapturing frames (press Ctrl+C to stop)...")
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                # Capture frame
                frame = picam2.capture_array()
                frame_count += 1
                
                # Convert RGBA to BGR if needed
                if len(frame.shape) == 3 and frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                elif len(frame.shape) == 3 and frame.shape[2] == 3:
                    # Already BGR or RGB, ensure BGR
                    pass
                
                # Calculate FPS
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                
                # Add overlay info
                h, w = frame.shape[:2]
                cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Resolution: {w}x{h}", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, "Press 'q' to quit, 's' to save", (10, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display frame
                cv2.imshow('Arducam Test', frame)
                
                # Handle key press
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print(f"\nExiting... Captured {frame_count} frames")
                    break
                elif key == ord('s'):
                    filename = f"arducam_test_{int(time.time())}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Saved frame to {filename}")
                
                # Print progress every 30 frames
                if frame_count % 30 == 0:
                    print(f"  Captured {frame_count} frames @ {fps:.1f} FPS")
                
                # Auto-exit after 300 frames (for automated testing)
                if frame_count >= 300:
                    print(f"\nAuto-exiting after {frame_count} frames")
                    break
        
        except KeyboardInterrupt:
            print(f"\n\nInterrupted by user. Captured {frame_count} frames")
        
        finally:
            # Cleanup
            picam2.stop()
            cv2.destroyAllWindows()
            
            elapsed = time.time() - start_time
            avg_fps = frame_count / elapsed if elapsed > 0 else 0
            print(f"\n✓ Test complete")
            print(f"  Total frames: {frame_count}")
            print(f"  Duration: {elapsed:.1f}s")
            print(f"  Average FPS: {avg_fps:.1f}")
        
        return True
    
    except Exception as e:
        print(f"\n✗ Error testing camera: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_libcamera_detection():
    """Check if libcamera can detect the camera"""
    print("\n=== Checking libcamera detection ===")
    
    import subprocess
    
    try:
        # Run libcamera-hello to detect camera
        print("Running: libcamera-hello --list-cameras")
        result = subprocess.run(
            ['libcamera-hello', '--list-cameras'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        print("\nOutput:")
        print(result.stdout)
        
        if result.returncode == 0:
            print("✓ libcamera detected camera(s)")
            return True
        else:
            print("✗ libcamera did not detect any cameras")
            print("\nError output:")
            print(result.stderr)
            return False
    
    except FileNotFoundError:
        print("✗ libcamera-hello not found")
        print("Install with: sudo apt-get install libcamera-apps")
        return False
    except subprocess.TimeoutExpired:
        print("✗ Command timed out")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    print("=" * 60)
    print("Arducam Test Script")
    print("=" * 60)
    
    # Check libcamera detection
    libcam_ok = test_libcamera_detection()
    
    if not libcam_ok:
        print("\n⚠ Warning: libcamera may not detect the camera")
        print("  But picamera2 might still work...")
    
    # Test with picamera2
    input("\nPress Enter to start picamera2 test (or Ctrl+C to exit)...")
    
    success = test_picamera2()
    
    if success:
        print("\n" + "=" * 60)
        print("✓ Arducam test PASSED")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("✗ Arducam test FAILED")
        print("=" * 60)
        print("\nTroubleshooting:")
        print("  1. Check camera connection to CSI port")
        print("  2. Enable camera in raspi-config:")
        print("     sudo raspi-config")
        print("     Interface Options → Camera → Enable")
        print("  3. Reboot: sudo reboot")
        print("  4. Check for updates:")
        print("     sudo apt-get update")
        print("     sudo apt-get install -y python3-picamera2")
        print("  5. Test with: libcamera-hello")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting...")
