import cv2
import os
from ultralytics import YOLO

video_path = os.path.join(os.path.dirname(__file__), "data/crowd_video_2.mp4")

# Open video file
video = cv2.VideoCapture(video_path)

# Load YOLO model
od = YOLO("yolov8m.pt")

while True:
    ret, frame = video.read()
    if not ret:
        print("Video finished or cannot read frame")
        break

    results = od(frame)
    annotated_frame = results[0].plot()

    # class 0 in YOLO is person
    people_count = len([box for box in results[0].boxes if int(box.cls) == 0])

    # Resizing frame to 25% of original size
    resized_frame = cv2.resize(annotated_frame, None, fx=0.25, fy=0.25)
    # Add people counter text on frame
    cv2.putText(resized_frame, f"People Count: {people_count}", (10, 30),
                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Crowd Video (press q to quit)", resized_frame)

    # Check if window is closed or 'q' is pressed  (to exit i used gpt to figure out how to detect window close)
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty("Crowd Video (press q to quit)", cv2.WND_PROP_VISIBLE) < 1:
        break

# End of video processing and close resources
video.release()
cv2.destroyAllWindows()
