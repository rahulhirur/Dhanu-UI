
from media_pipe_video import GestureRecognitionApp

if __name__ == "__main__":
    app = GestureRecognitionApp(model_path='gesture_recognizer.task', camera_index=0)
    app.run()
    
    # Print summary of detected gestures
    gestures = app.get_detected_gestures()
    if gestures:
        print(f"\nTotal gestures detected: {len(gestures)}")
        for g in gestures[-5:]:  # Show last 5
            print(f"  - {g['gesture']} ({g['confidence']:.2f}) at {g['timestamp']}ms")