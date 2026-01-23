import cv2
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import sys


class GestureRecognitionApp:
    """Structured gesture recognition application using MediaPipe"""
    
    def __init__(self, model_path='gesture_recognizer.task', camera_index=0):
        """
        Initialize the gesture recognition app.
        
        Args:
            model_path (str): Path to the gesture recognizer model
            camera_index (int): Camera device index (default: 0)
        """
        self.model_path = model_path
        self.camera_index = camera_index
        self.recognizer = None
        self.cap = None
        self.start_time = None
        self.detected_gestures = []
    
    def setup_callback(self):
        """Create a callback function for gesture detection results"""
        def handle_gesture_result(result: vision.GestureRecognizerResult, 
                                 output_image: mp.Image, 
                                 timestamp_ms: int):
            """Callback function that runs when a gesture is recognized"""
            if result.gestures:
                for gesture in result.gestures:
                    gesture_name = gesture[0].category_name
                    confidence = gesture[0].score
                    self.detected_gestures.append({
                        'gesture': gesture_name,
                        'confidence': confidence,
                        'timestamp': timestamp_ms
                    })
                    print(f"Detected: {gesture_name} (Score: {confidence:.2f})")
        
        return handle_gesture_result
    
    def initialize_recognizer(self):
        """Initialize the MediaPipe gesture recognizer with options"""
        try:
            base_options = python.BaseOptions(model_asset_path=self.model_path)
            
            options = vision.GestureRecognizerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.LIVE_STREAM,
                result_callback=self.setup_callback()
            )
            
            self.recognizer = vision.GestureRecognizer.create_from_options(options)
            print(f"✓ Gesture recognizer initialized with model: {self.model_path}")
            return True
        except Exception as e:
            print(f"✗ Error initializing recognizer: {e}")
            return False
    
    def initialize_camera(self):
        """Initialize and configure the camera"""
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            print("USER GENERATED ERROR")
            print(f"Error: Could not open camera at index {self.camera_index}.")
            print("Troubleshooting: Check permissions, cable, or try a different index.")
            return False
        
        print(f"✓ Camera initialized (index: {self.camera_index})")
        return True
    
    def preprocess_frame(self, frame):
        """
        Convert and prepare frame for MediaPipe processing.
        
        Args:
            frame: OpenCV frame (BGR format)
            
        Returns:
            mp.Image: MediaPipe image object in RGB format
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    def get_timestamp(self):
        """Calculate timestamp in milliseconds since start"""
        return int((time.time() - self.start_time) * 1000)
    
    def process_frame(self, frame):
        """
        Process a single frame for gesture recognition.
        
        Args:
            frame: OpenCV frame to process
        """
        mp_image = self.preprocess_frame(frame)
        timestamp_ms = self.get_timestamp()
        self.recognizer.recognize_async(mp_image, timestamp_ms)
    
    def display_frame(self, frame, window_name='Gesture Recognizer'):
        """
        Display the frame and handle user input.
        
        Args:
            frame: Frame to display
            window_name (str): Name of the display window
            
        Returns:
            bool: True to continue, False to exit
        """
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        return key != ord('q')
    
    def run(self):
        """Main loop for gesture recognition"""
        if not self.initialize_recognizer():
            return False
        
        if not self.initialize_camera():
            return False
        
        self.start_time = time.time()
        
        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Process frame for gesture recognition
                self.process_frame(frame)
                
                # Display the feed
                if not self.display_frame(frame):
                    break
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        if self.recognizer:
            self.recognizer.close()
        print("\n✓ Resources cleaned up")
    
    def get_detected_gestures(self):
        """Get all detected gestures"""
        return self.detected_gestures
    
    def clear_gestures(self):
        """Clear the gesture history"""
        self.detected_gestures.clear()


# Example usage
if __name__ == "__main__":
    app = GestureRecognitionApp(model_path='gesture_recognizer.task', camera_index=0)
    app.run()
    
    # Print summary of detected gestures
    gestures = app.get_detected_gestures()
    if gestures:
        print(f"\nTotal gestures detected: {len(gestures)}")
        for g in gestures[-5:]:  # Show last 5
            print(f"  - {g['gesture']} ({g['confidence']:.2f}) at {g['timestamp']}ms")
