import cv2
import streamlit as st

st.title("Webcam Live Feed")
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

while run:
    ret, frame = camera.read()
    
    # Check if the frame was successfully captured
    if not ret or frame is None:
        st.error("Can't receive frame (stream end?). Exiting ...")
        break

    # Now it's safe to convert color
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)
else:
    st.write('Stopped')

# Good practice: release the camera when the loop finishes
camera.release()
