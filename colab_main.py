import cv2
from ultralytics import YOLO

def main():
    # Load the YOLOv8 model
    print("Loading YOLOv8 model...")
    model = YOLO('yolov8n.pt')

    video_path = 'video.mp4' 
    output_path = 'output_video.mp4'

    print(f"Opening video file: {video_path}...")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}.")
    else:
        # Get video properties for saving
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Setup VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print("Processing video... This might take a moment.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run inference
            results = model(frame, verbose=False)

            # Custom logic for "drone", "bird", or "Unknown"
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id].lower()

                if class_name == 'bird':
                    label = 'bird'
                    color = (0, 255, 0)  # Green
                elif class_name in ['drone', 'airplane', 'kite']:
                    label = 'drone'
                    color = (0, 0, 255)  # Red
                else:
                    label = 'Unknown'
                    color = (128, 128, 128)  # Gray

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                (text_width, text_height), _ = cv2.getTextSize(label, font, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), font, 0.6, (255, 255, 255), 2)

            # Write the processed frame to the output video
            out.write(frame)

        # Release resources
        cap.release()
        out.release()
        print(f"Done! Video saved as {output_path}")

if __name__ == '__main__':
    main()
