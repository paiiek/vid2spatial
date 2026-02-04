#!/usr/bin/env python3
"""
Test OSC Spatial Sender.

This script:
1. Generates a test trajectory (circular motion)
2. Sends it via OSC
3. Optionally receives and prints the messages (for verification)
"""

import sys
sys.path.insert(0, "/home/seung/mmhoa/vid2spatial")

import time
import threading
import numpy as np
from vid2spatial_pkg.osc_sender import OSCSpatialSender


def generate_test_trajectory(duration_sec=5.0, fps=30.0):
    """Generate circular motion trajectory for testing."""
    num_frames = int(duration_sec * fps)
    trajectory = []

    for i in range(num_frames):
        t = i / fps
        # Circular motion: az goes 0 -> 360 deg, el oscillates
        az = np.radians(360 * t / duration_sec)  # Full rotation
        el = np.radians(15 * np.sin(2 * np.pi * t / 2))  # Oscillate ±15 deg
        dist = 2.0 + 0.5 * np.sin(2 * np.pi * t / 3)  # Distance varies 1.5-2.5m

        # Compute XYZ
        x = dist * np.sin(az) * np.cos(el)
        y = dist * np.sin(el)
        z = dist * np.cos(az) * np.cos(el)

        trajectory.append({
            "frame": i,
            "az": float(az),
            "el": float(el),
            "dist_m": float(dist),
            "x": float(x),
            "y": float(y),
            "z": float(z),
        })

    return trajectory


def start_osc_receiver(port=9000):
    """Start OSC receiver in background thread for verification."""
    try:
        from pythonosc import dispatcher, osc_server

        disp = dispatcher.Dispatcher()

        def print_handler(address, *args):
            print(f"  <- {address}: {args}")

        disp.map("/vid2spatial/*", print_handler)

        server = osc_server.ThreadingOSCUDPServer(("127.0.0.1", port), disp)
        print(f"[Receiver] Listening on port {port}")

        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.daemon = True
        server_thread.start()

        return server
    except Exception as e:
        print(f"[Receiver] Failed to start: {e}")
        return None


def test_single_frame():
    """Test sending single frame."""
    print("\n" + "=" * 50)
    print("TEST 1: Single Frame Send")
    print("=" * 50)

    sender = OSCSpatialSender(
        host="127.0.0.1",
        port=9001,  # Different port to avoid conflict
        distance_mode="normalized",
    )

    if sender.connect():
        sender.send_frame(
            az_deg=45.0,
            el_deg=10.0,
            dist_m=2.5,
            velocity_deg_s=30.0,
            timecode_s=1.5,
            frame_idx=45,
        )
        print("Sent: az=45°, el=10°, dist=2.5m, vel=30°/s, tc=1.5s")
        sender.disconnect()
    else:
        print("Connection failed")


def test_trajectory_stream():
    """Test streaming trajectory."""
    print("\n" + "=" * 50)
    print("TEST 2: Trajectory Stream (3 seconds)")
    print("=" * 50)

    # Generate test trajectory
    trajectory = generate_test_trajectory(duration_sec=3.0, fps=30.0)
    print(f"Generated {len(trajectory)} frames")

    sender = OSCSpatialSender(
        host="127.0.0.1",
        port=9001,
        distance_mode="normalized",
    )

    if sender.connect():
        frame_count = [0]

        def on_frame(idx, frame):
            frame_count[0] += 1
            if idx % 30 == 0:  # Print every second
                az_deg = np.degrees(frame['az'])
                el_deg = np.degrees(frame['el'])
                print(f"  Frame {idx}: az={az_deg:.1f}°, el={el_deg:.1f}°, dist={frame['dist_m']:.2f}m")

        start = time.time()
        sender.stream_trajectory(
            trajectory,
            fps=30.0,
            realtime=True,
            loop=False,
            on_frame=on_frame,
        )
        elapsed = time.time() - start

        print(f"\nStreamed {frame_count[0]} frames in {elapsed:.2f}s")
        sender.disconnect()
    else:
        print("Connection failed")


def test_with_real_trajectory():
    """Test with real tracking result."""
    print("\n" + "=" * 50)
    print("TEST 3: Real Trajectory from Tracking")
    print("=" * 50)

    import json
    from pathlib import Path

    # Try to load existing trajectory
    result_path = Path("/home/seung/mmhoa/vid2spatial/eval/comprehensive_results/robustness_layer_results.json")

    if not result_path.exists():
        print("No existing trajectory found, using synthetic")
        trajectory = generate_test_trajectory(duration_sec=2.0, fps=30.0)
    else:
        # Generate new trajectory from video
        try:
            from vid2spatial_pkg.hybrid_tracker import HybridTracker
            from vid2spatial_pkg.trajectory_stabilizer import rts_smooth_trajectory

            video_path = "/home/seung/mmhoa/vid2spatial/test_videos/marker_hd.mp4"

            print("Tracking video...")
            tracker = HybridTracker(device="cuda", box_threshold=0.15)
            result = tracker.track(
                video_path=video_path,
                text_prompt="colored marker",
                tracking_method="adaptive_k",
                estimate_depth=True,
            )

            # Get trajectory and apply RTS
            traj_3d = result.get_trajectory_3d(smooth=False)
            trajectory = rts_smooth_trajectory(traj_3d["frames"])
            print(f"Got {len(trajectory)} frames from tracking")

        except Exception as e:
            print(f"Tracking failed: {e}, using synthetic")
            trajectory = generate_test_trajectory(duration_sec=2.0, fps=30.0)

    # Stream
    sender = OSCSpatialSender(
        host="127.0.0.1",
        port=9001,
        distance_mode="normalized",
    )

    if sender.connect():
        print(f"Streaming {len(trajectory)} frames...")

        sender.stream_trajectory(
            trajectory[:90],  # First 3 seconds
            fps=30.0,
            realtime=True,
            loop=False,
        )

        sender.disconnect()


def main():
    print("=" * 50)
    print("OSC Spatial Sender Test")
    print("=" * 50)
    print("\nOSC Messages will be sent to 127.0.0.1:9001")
    print("To receive, run an OSC listener on that port")
    print("(e.g., in Max/MSP, SuperCollider, or another OSC client)")

    # Run tests
    test_single_frame()

    print("\nWaiting 1 second...")
    time.sleep(1)

    test_trajectory_stream()

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print("""
OSC Address Structure:
  /vid2spatial/azimuth    {float: -180 to 180 deg}
  /vid2spatial/elevation  {float: -90 to 90 deg}
  /vid2spatial/distance   {float: 0-1 normalized}
  /vid2spatial/velocity   {float: deg/s}
  /vid2spatial/timecode   {float: seconds}
  /vid2spatial/frame      {int: frame index}
  /vid2spatial/spatial    {az, el, dist, vel, tc} (bundled)
  /vid2spatial/xyz        {x, y, z} (cartesian)
""")


if __name__ == "__main__":
    main()
