import cv2
import numpy as np

def canny(image):
    """Applies grayscale, Gaussian blur, and Canny edge detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def ROI(image):
    """Applies a triangular mask to keep region of interest."""
    height = image.shape[0]
    triangle = np.array([[ (200, height), (1100, height), (550, 250) ]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def display_line(image, lines):
    """Draws lines on a blank image."""
    line_image = np.zeros_like(image)
    if lines is not None and len(lines) > 0:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image

def coordinates(image, line_parameters):
    """Converts slope and intercept to x1, y1, x2, y2 coordinates."""
    slope, intercept = line_parameters
    if slope == 0: slope = 0.01
    y1 = image.shape[0]
    y2 = int(y1 * (3 / 5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def average_slope(image, lines):
    """Averages left and right lines' slopes and intercepts."""
    left_fit = []
    right_fit = []
    if lines is None:
        return []

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope, intercept = parameters
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    line_result = []

    if left_fit:
        left_fit_avg = np.average(left_fit, axis=0)
        left_line = coordinates(image, left_fit_avg)
        line_result.append(left_line)

    if right_fit:
        right_fit_avg = np.average(right_fit, axis=0)
        right_line = coordinates(image, right_fit_avg)
        line_result.append(right_line)

    return np.array(line_result)

def process_video(video_path):
    """Reads video and applies lane detection on each frame."""
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        canny1 = canny(frame)
        cropped_image = ROI(canny1)
        lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), 40, 5)
        average_lines = average_slope(frame, lines)
        line_image = display_line(frame, average_lines)
        final_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
        cv2.imshow("Result", final_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def process_image(image):
    """Processes a single image and returns the lane-detected image."""
    canny_img = canny(image)
    roi = ROI(canny_img)
    lines = cv2.HoughLinesP(roi, 2, np.pi/180, 100, np.array([]), 40, 5)
    avg_lines = average_slope(image, lines)
    line_img = display_line(image, avg_lines)
    result = cv2.addWeighted(image, 0.8, line_img, 1, 1)
    return result


if __name__ == "__main__":
    video_file = "test.mp4"  # Change this to any video file path
    process_video(video_file)
