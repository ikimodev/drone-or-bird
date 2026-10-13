import os
import sys
import argparse
import time
import cv2
from PIL import Image
import numpy as np

from defense_core import AirDefenseEnsemble
from hud_renderer import render_tactical_hud

def is_image_file(path):
    ext = os.path.splitext(str(path))[1].lower()
    return ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.avif']

def load_image(path):
    try:
        pil_img = Image.open(path).convert('RGB')
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception:
        return cv2.imread(path)

def main():
    parser = argparse.ArgumentParser(description="Military-Grade Air Defense C2 Drone vs Bird Detection System")
    parser.add_argument("--source", type=str, default="user_image.jpg", help="Path to video/image file, or '0' for webcam")
    parser.add_argument("--save", type=str, default=None, help="Optional output path to save processed result (image or video)")
    parser.add_argument("--headless", action="store_true", help="Run without GUI window and output directly to file")
    args = parser.parse_args()

    print("==================================================")
    print("Initializing Military-Grade Air Defense Ensemble...")
    print("==================================================")
    defense_engine = AirDefenseEnsemble()

    source = args.source

    # Case 1: Still image source
    if not source.isdigit() and is_image_file(source):
        print(f"Processing Aerial Image: {source}...")
        if not os.path.exists(source):
            print(f"Error: File '{source}' not found.")
            return

        frame = load_image(source)
        if frame is None:
            print(f"Error: Could not decode image '{source}'.")
            return

        t0 = time.time()
        targets = defense_engine.detect_image(frame)
        hud_frame = render_tactical_hud(frame, targets, show_trails=False)
        dt = (time.time() - t0) * 1000

        drone_count = sum(1 for t in targets if t.cls == 'drone')
        bird_count = sum(1 for t in targets if t.cls == 'bird')

        print(f"Detection completed in {dt:.1f}ms:")
        print(f"  Confirmed Drones (UAS): {drone_count}")
        print(f"  Confirmed Avian Birds:  {bird_count}")
        for t in targets:
            print(f"   * [{t.cls.upper()}] {t.sub_type} ({t.conf*100:.1f}%) | Threat: {t.threat_level}")

        save_path = args.save or f"c2_hud_{os.path.basename(source)}"
        cv2.imwrite(save_path, hud_frame)
        print(f"Output saved to: {save_path}")

        if not args.headless:
            window_name = f"Air Defense C2 - {os.path.basename(source)} (Press any key to close)"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, hud_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    # Case 2: Video or Webcam stream
    if source.isdigit():
        source = int(source)
        print(f"Connecting to live camera feed (Device #{source})...")
    else:
        print(f"Opening surveillance video stream: {source}...")
        if not os.path.exists(str(source)):
            print(f"Error: Could not find video file '{source}'.")
            return

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open source: {source}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    out_writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(args.save, fourcc, fps, (width, height))
        print(f"Recording tactical C2 stream to: {args.save}")

    window_name = "Air Defense C2 Surveillance (Press 'q' to exit)"
    if not args.headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, width, height)

    frame_count = 0
    t_start = time.time()
    defense_engine.reset_tracker()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream/Video playback completed.")
            break

        frame_count += 1
        targets = defense_engine.process_stream_frame(frame)
        hud_frame = render_tactical_hud(frame, targets, show_trails=True)

        if out_writer:
            out_writer.write(hud_frame)

        if not args.headless:
            cv2.imshow(window_name, hud_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("Operator requested termination.")
                break

    cap.release()
    if out_writer:
        out_writer.release()
    if not args.headless:
        cv2.destroyAllWindows()

    elapsed = max(0.001, time.time() - t_start)
    print(f"Surveillance session finished: {frame_count} frames processed at {frame_count/elapsed:.1f} FPS.")

if __name__ == '__main__':
    main()
