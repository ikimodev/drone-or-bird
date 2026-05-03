import cv2
from ultralytics import YOLO

def main():
    # Load the YOLOv8 model
    print("Loading YOLOv8 model...")
    model = YOLO('yolov8n.pt')

    # Open the video file
    video_path = 'video.mp4'  # ضع اسم أو مسار الفيديو هنا
    print(f"Opening video file: {video_path}...")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}.")
        return

    print("Video opened. Press 'q' to exit.")

    while True:
        # Read frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Run inference on the frame
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
                # Mapped airplane/kite to drone since base COCO lacks 'drone' class
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

        # Display the frame
        cv2.imshow("YOLOv8 Video Test", frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
