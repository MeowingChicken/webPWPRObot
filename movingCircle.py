# Moving Circle Detection
# v1.0.1
# 1. Import necessary libraries: OpenCV (cv2) for image processing, NumPy (np) for numerical operations, and sys for system-specific functions.
# 2. Define a function `detect_can_bottom_video` that takes the path to a video file as input.
# 3. Inside the function, attempt to open the video file. If it fails, raise an error.
# 4. Initialize a frame counter.
# 5. Start a continuous loop to read frames from the video.
# 6. If a frame cannot be read (end of video), break the loop.
# 7. Process only every fifth frame to improve performance.
# 8. Convert the frame to grayscale and apply a Gaussian blur to reduce noise.
# 9. Use the OpenCV `HoughCircles` function to detect circles within the processed image.
# 10. If any circles are found, sort them to identify the largest one, which is assumed to be the can bottom.
# 11. Draw the largest detected circle and its center point on the original color frame.
# 12. Print the center coordinates (cx, cy) and radius of the detected circle to the console.
# 13. Display the resulting frame with the drawn circle in a window.
# 14. Wait for a short period and check if the 'q' key is pressed; if so, break the loop.
# 15. After the loop, release the video capture object and close all OpenCV windows.
# 16. Define a `main` function to call the `detect_can_bottom_video` function with a specific video file name (`movie.mp4`).
# 17. The script execution starts here, calling the `main` function if the script is run directly.

import cv2
import numpy as np
import sys

def detect_can_bottom_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video stream or file: {video_path}")

    frameNumber = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frameNumber % 5 == 0:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Blur to reduce noise
            gray = cv2.GaussianBlur(gray, (9, 9), 2)
            output = frame.copy()
            img_height = gray.shape[0]

            # Detect circles
            circles = cv2.HoughCircles(
                gray,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=img_height // 2,
                param1=120,
                param2=80,
                minRadius=90,
                maxRadius=105
            )

            if circles is not None:
                circles = np.uint16(np.around(circles[0]))
                # Pick the largest circle (assumed can bottom)
                cx, cy, radius = max(circles, key=lambda c: c[2])

                # Draw detected circle
                cv2.circle(output, (cx, cy), radius, (0, 255, 67), 4)
                cv2.circle(output, (cx, cy), 4, (0, 255, 67), -1)

                print(cx, cy, radius)

            cv2.imshow("Rolling Soda Can Detection (HoughCircles)", output)

        frameNumber += 1

        if cv2.waitKey(24) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    detect_can_bottom_video("movie.mp4")
    return 0

if __name__ == "__main__":
    sys.exit(main())
