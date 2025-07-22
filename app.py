import streamlit as st
import cv2
import numpy as np
import tempfile
from lane_detection import canny, ROI, average_slope, display_line
from streamlit_lottie import st_lottie
import json

st.set_page_config(page_title="Lane Detection App", layout="wide")

# Load Lottie animation
@st.cache_data
def load_lottie(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

lottie_car = load_lottie("car.json")  # ensure 'car.json' Lottie file is in the same directory

st.title("🚗 Road Lane Detection App")

# Tabs Layout
tabs = st.tabs(["🏁 Detection", "ℹ️ About", "📞 Contact"])

with tabs[0]:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("Upload Image or Video")
        file = st.file_uploader("Choose an image or a video file", type=["jpg", "png", "mp4"])

        def process_image(image):
            lane_image = np.copy(image)
            edges = canny(lane_image)
            cropped = ROI(edges)
            lines = cv2.HoughLinesP(cropped, 2, np.pi / 180, 100, np.array([]), 40, 5)
            avg_lines = average_slope(lane_image, lines)
            line_img = display_line(lane_image, avg_lines)
            final = cv2.addWeighted(lane_image, 0.8, line_img, 1, 1)
            return final

        if file:
            file_type = file.type
            if file_type.startswith("image"):
                file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, 1)
                output = process_image(image)
                st.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), caption="Detected Lanes", use_container_width=True)

            elif file_type == "video/mp4":
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(file.read())
                cap = cv2.VideoCapture(tfile.name)

                stframe = st.empty()
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    output = process_image(frame)
                    output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
                    stframe.image(output_rgb, channels="RGB", use_container_width=True)
                cap.release()

    with col2:
        st_lottie(lottie_car, speed=1, loop=True, height=400, key="car")

with tabs[1]:
    st.header("About")
    st.markdown("""
    This app uses basic computer vision techniques such as:
    - Canny edge detection
    - Region of Interest masking
    - Hough Line Transform
    - Lane averaging and overlay

    Built with ❤️ using OpenCV and Streamlit.
                - Rishi Karmakar
    """)

with tabs[2]:
    st.header("Contact")
    st.markdown("""
    For feedback or collaborations, reach out at:
    - 📧 **rkarmakar2k03@gmail.com**
    - 💼 [LinkedIn](https://www.linkedin.com/in/rishi-karmakar-136a21246/)
    """)