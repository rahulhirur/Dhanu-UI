import streamlit as st
import cv2
import threading
import queue
import sys
import os
from datetime import datetime
import json

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from MediaPipe_Model.media_pipe_video import GestureRecognitionApp
from S2LLM_Cerb.voice_to_text import listen_and_transcribe
from S2LLM_Cerb.llm import query_cerebras

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="DhanUI - Robot Control Interface",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .status-box { 
        padding: 15px; 
        border-radius: 10px; 
        margin: 10px 0;
        font-weight: bold;
    }
    .status-success {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .status-error {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    .status-info {
        background-color: #d1ecf1;
        color: #0c5460;
        border: 1px solid #bee5eb;
    }
    .status-warning {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeeba;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'camera_enabled' not in st.session_state:
    st.session_state.camera_enabled = False
    
if 'microphone_enabled' not in st.session_state:
    st.session_state.microphone_enabled = False
    
if 'camera_frame' not in st.session_state:
    st.session_state.camera_frame = None
    
if 'gesture_text' not in st.session_state:
    st.session_state.gesture_text = ""
    
if 'speech_text' not in st.session_state:
    st.session_state.speech_text = ""
    
if 'combined_text' not in st.session_state:
    st.session_state.combined_text = ""
    
if 'llm_output' not in st.session_state:
    st.session_state.llm_output = ""
    
if 'status_messages' not in st.session_state:
    st.session_state.status_messages = []
    
if 'gesture_recognizer' not in st.session_state:
    st.session_state.gesture_recognizer = None
    
if 'camera_thread' not in st.session_state:
    st.session_state.camera_thread = None

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def add_status_message(message: str, status_type: str = "info"):
    """Add a status message with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.status_messages.append({
        'message': message,
        'type': status_type,
        'time': timestamp
    })
    # Keep only last 20 messages
    if len(st.session_state.status_messages) > 20:
        st.session_state.status_messages = st.session_state.status_messages[-20:]

def display_status_messages():
    """Display all status messages"""
    st.subheader("📋 Status Messages")
    
    if not st.session_state.status_messages:
        st.info("No messages yet...")
    else:
        for msg in st.session_state.status_messages:
            css_class = f"status-box status-{msg['type']}"
            st.markdown(
                f"<div class='{css_class}'>[{msg['time']}] {msg['message']}</div>",
                unsafe_allow_html=True
            )

def initialize_gesture_recognizer():
    """Initialize the gesture recognition model"""
    try:
        add_status_message("Initializing gesture recognizer...", "info")
        model_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 
            'MediaPipe_Model', 
            'gesture_recognizer.task'
        )
        
        if not os.path.exists(model_path):
            add_status_message(f"Model not found at {model_path}", "error")
            return None
        
        recognizer = GestureRecognitionApp(
            model_path=model_path,
            camera_index=0
        )
        
        if recognizer.initialize_recognizer() and recognizer.initialize_camera():
            recognizer.start_time = __import__('time').time()
            add_status_message("✓ Gesture recognizer initialized successfully", "success")
            return recognizer
        else:
            add_status_message("✗ Failed to initialize gesture recognizer", "error")
            return None
            
    except Exception as e:
        add_status_message(f"✗ Error initializing gesture recognizer: {str(e)}", "error")
        return None

def capture_camera_frame():
    """Capture a single frame from camera"""
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            add_status_message("✗ Cannot open camera", "error")
            return None
        
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            return frame
        else:
            add_status_message("✗ Failed to capture frame", "error")
            return None
            
    except Exception as e:
        add_status_message(f"✗ Camera error: {str(e)}", "error")
        return None

def process_speech_to_text():
    """Process speech from microphone"""
    try:
        add_status_message("🎙️ Listening for speech...", "info")
        text = listen_and_transcribe()
        
        if text:
            st.session_state.speech_text = text
            add_status_message(f"✓ Speech recognized: {text}", "success")
            return text
        else:
            add_status_message("✗ Could not recognize speech", "warning")
            return ""
            
    except Exception as e:
        add_status_message(f"✗ Speech recognition error: {str(e)}", "error")
        return ""

def process_gesture_recognition(frame):
    """Process gesture from frame"""
    try:
        if frame is None:
            add_status_message("✗ No frame to process", "error")
            return ""
        
        if st.session_state.gesture_recognizer is None:
            add_status_message("✗ Gesture recognizer not initialized", "error")
            return ""
        
        add_status_message("👆 Processing gesture...", "info")
        
        # Process frame
        recognizer = st.session_state.gesture_recognizer
        recognizer.process_frame(frame)
        
        # Get latest detected gestures
        gestures = recognizer.get_detected_gestures()
        
        if gestures:
            latest_gesture = gestures[-1]
            gesture_name = latest_gesture['gesture']
            confidence = latest_gesture['confidence']
            
            st.session_state.gesture_text = gesture_name
            add_status_message(
                f"✓ Gesture detected: {gesture_name} (Confidence: {confidence:.2f})",
                "success"
            )
            recognizer.clear_gestures()
            return gesture_name
        else:
            add_status_message("⚠️ No gesture detected in frame", "warning")
            return ""
            
    except Exception as e:
        add_status_message(f"✗ Gesture recognition error: {str(e)}", "error")
        return ""

def gesture_to_action(gesture_name: str):
    """
    Map gesture name to action text
    
    Args:
        gesture_name: The detected gesture name
        
    Returns:
        tuple: (action_text, is_valid_gesture)
    """
    gesture_mapping = {
        'thumbs_up': 'Bring the cathether',
        'open_palm': 'Go back to neutral position',
        'closed_fist': 'Stop'
    }
    
    if gesture_name.lower() in gesture_mapping:
        action_text = gesture_mapping[gesture_name.lower()]
        return action_text, True
    else:
        return "", False

def process_gesture_action_to_llm(gesture_name: str):
    """
    Process gesture name by converting to action text and feeding to LLM
    
    Args:
        gesture_name: The detected gesture name
        
    Returns:
        str: LLM response with instructions
    """
    try:
        if not gesture_name.strip():
            add_status_message("⚠️ No gesture to process", "warning")
            return ""
        
        # Convert gesture to action text
        action_text, is_valid = gesture_to_action(gesture_name)
        
        if not is_valid:
            add_status_message(
                f"⚠️ Gesture '{gesture_name}' not recognized. Valid gestures: thumbs_up, open_palm, closed_fist",
                "warning"
            )
            return ""
        
        add_status_message(f"👆 Gesture '{gesture_name}' → Action: '{action_text}'", "info")
        
        # Create a prompt for LLM to generate instructions
        llm_prompt = f"Generate detailed step-by-step instructions for the following medical robot action: {action_text}"
        
        add_status_message(f"🤖 Processing action through LLM: '{action_text}'", "info")
        
        # Query LLM with the action text
        llm_response = query_cerebras(llm_prompt)
        
        if llm_response:
            st.session_state.llm_output = llm_response
            add_status_message("✓ LLM generated instructions successfully", "success")
            return llm_response
        else:
            add_status_message("✗ LLM returned empty response", "error")
            return ""
            
    except Exception as e:
        add_status_message(f"✗ Error processing gesture action: {str(e)}", "error")
        return ""

def process_llm_orchestrator(text_input: str):
    """Process text through LLM orchestrator"""
    try:
        if not text_input.strip():
            add_status_message("⚠️ No text to process through LLM", "warning")
            return ""
        
        add_status_message(f"🤖 LLM processing: '{text_input}'", "info")
        
        llm_response = query_cerebras(text_input)
        
        if llm_response:
            st.session_state.llm_output = llm_response
            add_status_message("✓ LLM processed successfully", "success")
            return llm_response
        else:
            add_status_message("✗ LLM returned empty response", "error")
            return ""
            
    except Exception as e:
        add_status_message(f"✗ LLM processing error: {str(e)}", "error")
        return ""

# ============================================================================
# MAIN UI LAYOUT
# ============================================================================

st.title("🤖 DhanUI - Robot Control Interface")
st.markdown("**Integrated Multi-Modal Input System**")

# Sidebar for controls
with st.sidebar:
    st.header("⚙️ Input Controls")
    
    # Camera Control
    st.subheader("📹 Camera Feed")
    camera_col1, camera_col2 = st.columns([3, 1])
    with camera_col1:
        st.session_state.camera_enabled = st.checkbox(
            "Enable Camera", 
            value=st.session_state.camera_enabled,
            key="camera_toggle"
        )
    
    if st.session_state.camera_enabled:
        st.info("📹 Camera feed enabled")
        if st.button("📸 Capture Frame for Gesture Recognition"):
            frame = capture_camera_frame()
            if frame is not None:
                st.session_state.camera_frame = frame
                add_status_message("✓ Frame captured", "success")
    else:
        st.warning("📹 Camera feed disabled")
    
    st.divider()
    
    # Microphone Control
    st.subheader("🎙️ Microphone Input")
    st.session_state.microphone_enabled = st.checkbox(
        "Enable Microphone",
        value=st.session_state.microphone_enabled,
        key="mic_toggle"
    )
    
    if st.session_state.microphone_enabled:
        st.info("🎙️ Microphone enabled")
        if st.button("🎤 Record & Transcribe Speech"):
            process_speech_to_text()
    else:
        st.warning("🎙️ Microphone disabled")
    
    st.divider()
    
    # Initialize Gesture Recognizer
    if st.button("🔧 Initialize Gesture Recognizer"):
        st.session_state.gesture_recognizer = initialize_gesture_recognizer()
    
    st.divider()
    
    # Process Button
    if st.button("⚡ Process Through LLM Orchestrator", use_container_width=True):
        combined = f"{st.session_state.gesture_text} {st.session_state.speech_text}".strip()
        if combined:
            st.session_state.combined_text = combined
            process_llm_orchestrator(combined)
        else:
            add_status_message("⚠️ No input data (gesture or speech) to process", "warning")

# Main content area
main_col1, main_col2 = st.columns(2)

# Left column: Input Display and Camera Frame
with main_col1:
    st.header("📥 Input Data")
    
    # Camera Frame Display
    if st.session_state.camera_enabled:
        st.subheader("📹 Camera Preview")
        if st.session_state.camera_frame is not None:
            frame_rgb = cv2.cvtColor(st.session_state.camera_frame, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, use_column_width=True)
        else:
            st.info("Click 'Capture Frame' to see preview")
    
    # Gesture Text Display
    st.subheader("👆 Gesture Recognition Result")
    if st.session_state.gesture_text:
        st.success(f"**Detected Gesture:** {st.session_state.gesture_text}")
    else:
        st.info("No gesture detected yet")
    
    # Speech Text Display
    st.subheader("🗣️ Speech-to-Text Result")
    if st.session_state.speech_text:
        st.success(f"**Transcribed Speech:** {st.session_state.speech_text}")
    else:
        st.info("No speech transcribed yet")
    
    # Combined Text Display
    st.subheader("🔗 Combined Input")
    if st.session_state.combined_text:
        st.info(f"**Input to LLM:** {st.session_state.combined_text}")
    else:
        st.info("Combined input will appear here")

# Right column: Status and LLM Output
with main_col2:
    st.header("📊 Processing Status & Results")
    
    # Status Messages
    display_status_messages()
    
    st.divider()
    
    # LLM Output
    st.subheader("🤖 LLM Orchestrator Output")
    if st.session_state.llm_output:
        try:
            # Try to parse as JSON and display nicely
            output_json = json.loads(st.session_state.llm_output)
            st.json(output_json)
            
            # Display summary
            st.subheader("📋 Task Summary")
            if 'intent' in output_json:
                st.write(f"**Intent:** {output_json['intent']}")
            if 'task_type' in output_json:
                st.write(f"**Task Type:** {output_json['task_type']}")
            if 'steps' in output_json:
                st.write("**Steps to Execute:**")
                for i, step in enumerate(output_json['steps'], 1):
                    st.write(f"{i}. {step.get('action', 'N/A')}")
        except json.JSONDecodeError:
            # Display as raw text if not valid JSON
            st.text_area("LLM Output:", value=st.session_state.llm_output, height=300, disabled=True)
    else:
        st.info("LLM output will appear here after processing")

# Footer
st.divider()
st.markdown("""
**System Status:** Ready
- 📹 Camera: """ + ("✓ Enabled" if st.session_state.camera_enabled else "✗ Disabled") + """
- 🎙️ Microphone: """ + ("✓ Enabled" if st.session_state.microphone_enabled else "✗ Disabled") + """
- 👆 Gesture Recognizer: """ + ("✓ Initialized" if st.session_state.gesture_recognizer else "✗ Not Initialized") + """
""")