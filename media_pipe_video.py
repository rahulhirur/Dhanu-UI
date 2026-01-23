import cv2
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import sys


# 1. Define a callback function to handle results asynchronously
def print_result(result: vision.GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    # This function runs whenever a gesture is recognized
    if result.gestures:
        for gesture in result.gestures:
            print(f"Detected: {gesture[0].category_name} (Score: {gesture[0].score:.2f})")

# 2. Configuration options
model_path = 'gesture_recognizer.task' # Ensure you have downloaded this model

base_options = python.BaseOptions(model_asset_path=model_path)

options = vision.GestureRecognizerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM, # Critical for Webcam
    result_callback=print_result
)

# 3. Initialize the recognizer
with vision.GestureRecognizer.create_from_options(options) as recognizer:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("USER GENERATED ERROR")
        print("Error: Could not open camera at index 0.")
        print("Troubleshooting: Check permissions, cable, or try index 1.")
        sys.exit()
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 4. Preprocessing: Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # 5. Calculate monotonic timestamp in milliseconds
        timestamp_ms = int((time.time() - start_time) * 1000)

        # 6. Async Detection (Does not block the loop)
        recognizer.recognize_async(mp_image, timestamp_ms)

        # Display the feed
        cv2.imshow('Gesture Recognizer', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
