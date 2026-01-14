import cv2
import numpy as np
import sys

def detect_can_bottom_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise IOError(f"Cannot open video stream or file: {video_path}")
    frameNumber=0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frameNumber%5 == 0:
            
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
                print(cx,cy,radius)
            cv2.imshow("Rolling Soda Can Detection (HoughCircles)", output)
        frameNumber+=1
        if cv2.waitKey(24) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    detect_can_bottom_video("movie.mp4")
    return 0

if __name__ == "__main__":
    sys.exit(main())
