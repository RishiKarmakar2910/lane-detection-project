# lane-detection-project
## 🚗 Road Lane Detection

A computer vision project to detect road lane lines from images and videos using OpenCV. Built with Python and Streamlit, this project helps simulate the first steps in autonomous vehicle navigation.

## 📁 Project Structure

- `lane_detection.py` – Core logic using Canny + ROI masking
- `app.py` – Streamlit app for image/video upload and processing
- `test_image.jpg` – Sample input image
- `test.mp4` – Sample input video
- `car.json` – Lottie animation asset (optional UI enhancement)

## 🛠️ Tech Stack

- Python
- OpenCV
- Streamlit
- NumPy
- Lottie (for animations)

## 🚀 How to Run

### 🔧 Run using Python
```bash
python lane_detection.py
```
### 🔧 Run using Streamlit
```bash
streamlit run app.py
