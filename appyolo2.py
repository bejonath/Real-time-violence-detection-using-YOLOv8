import streamlit as st
import cv2
import numpy as np
import os
import tempfile
from ultralytics import YOLO
import cvzone
import math
import base64

model_yolo = YOLO(r"C:\Users\07jon\runs\detect\train16\weights\best.pt")

def custom_popup(message, type="info"):
    if type == "success":
        color = "#28a745"  # Green
    elif type == "warning":
        color = "#dc3545"  # Red
    else:
        color = "#17a2b8"  # Blue
    
    st.markdown(
        f"""
        <style>
        .popup {{
            position: fixed;
            z-index: 9999;
            left: 50%;
            top: 50%;
            transform: translate(-50%, -50%);
            width: 300px;
            background-color: {color};
            color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }}
        .popup-content {{
            text-align: center;
        }}
        </style>
        <div class="popup">
            <div class="popup-content">
                <h3>Conclusion</h3>
                <p>{message}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    classNames_1 = ['Non Violence', 'Violence']
    
    if not cap.isOpened():
        st.error(f"Error: Could not open video file {video_path}")
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create placeholders for displaying video frames
    video_frame = st.empty()

    violence_count = 0
    non_violence_count = 0
    frame_count = 0 
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (width, height))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = model_yolo(rgb_frame)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                if box.conf[0] > 0.40:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1
                    cvzone.cornerRect(frame, (x1, y1, w, h))
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])
                    #class_label = f'{classNames_1[cls]}'
                    #cvzone.putTextRect(frame, class_label, (max(0, x1), max(35, y1)))
        
                    if cls == 0:  # Non Violence
                        non_violence_count += 1
                    else:  # Violence
                        violence_count += 1

        video_frame.image(frame, channels='BGR')
        frame_count += 1

    cap.release()
    
    # Provide final conclusion in a colored pop-up
    if violence_count > non_violence_count:
        custom_popup("Violence detected", "warning")  # Red popup
    elif non_violence_count > violence_count:
        custom_popup("Violence not detected", "safe")  # Green popup
    else:
        custom_popup("The video contains an equal amount of violent and non-violent content", "info")  # Blue popup

    return None

    #return temp_output.name

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

def main():
    st.set_page_config(page_title="Smart Survelliance System", page_icon="🎥")

        # Add background image
    add_bg_from_local('D:/Projects/SIH2/bg.jpg')  # Replace with the path to your image
    
    # Custom CSS to style the app
    st.markdown("""
    <style>
    .reportview-container {
        background: rgba(0,0,0,0);
    }
    .main {
        
        padding: 20px;
        border-radius: 10px;
    }
    h1, h2, h3 {
        color: #1E1E1E;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        
        border-radius: 5px 5px 0px 0px;
        gap: 1px;
        padding-top: 6px;
        padding-bottom: 6px;
        padding-right: 10px;
        padding-left: 10px
    }
    .stTabs [aria-selected="true"] {
        background-color: #000000;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2 = st.tabs(["Detection", "About"])
    
    with tab1:
        st.title("Violence Detection")
        
        uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mpeg"])
        
        if uploaded_file:
            st.write("File uploaded successfully. Processing...")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name
            
            st.write(f"Temporary file created: {temp_file_path}")
            
            process_video(temp_file_path)
            
            # Clean up temporary file
            os.unlink(temp_file_path)
    
    with tab2:
        st.title("About Us")
        st.write("""

        This application uses state-of-the-art computer vision techniques to detect and classify actions in video footage. 
        Our system is built on the YOLO (You Only Look Once) object detection framework, which has been trained to 
        recognize various actions, with a focus on distinguishing between violent and non-violent behaviors.

        ### Applications:
        This technology can be applied in various fields, including:
        - Security and surveillance
        - Behavioral studies
        - Sports analysis
        - Public safety monitoring
        
        ### About Team:
        1. Emile Jonath
        2. Harmesh
        3. Jaswanth
        4. Pavan Kumar
        5. Sri Nandini
        6. Dhanush


        For more information or inquiries, please contact us at info@videoactiondetection.com
        """)

if __name__ == "__main__":
    main()