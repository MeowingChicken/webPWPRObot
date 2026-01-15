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

# Define the function to perform can bottom detection in a video 
def detect_can_bottom_video(video_path): 
    # Open the video file
    cap = cv2.VideoCapture(video_path) 

    # Check if the video opened successfully
    if not cap.isOpened(): 
        # Raise an error if the file cannot be opened
        raise IOError(f"Cannot open video stream or file: {video_path}") 

    frameNumber = 0
    # Loop through each frame in the video
    while True: 
        # Read the next frame
        ret, frame = cap.read() 

        # If no frame is returned (end of video), break the loop
        if not ret: 
            break

        # Process only every 5th frame for efficiency
        if frameNumber % 5 == 0:
            # Convert the frame to grayscale for circle detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

            # Apply Gaussian blur to reduce noise and help with circle detection
            gray = cv2.GaussianBlur(gray, (9, 9), 2) 

            # Create a copy of the original frame to draw the detection results on
            output = frame.copy() 

            # Get the height of the image, used for setting the minimum distance between circles
            img_height = gray.shape[0] 

            # Detect circles using the Hough Circle Transform
            circles = cv2.HoughCircles(
                gray, # Input image (grayscale)
                cv2.HOUGH_GRADIENT, # Detection method
                dp=1.2, # Inverse ratio of the accumulator resolution to the image resolution
                minDist=img_height // 2, # Minimum distance between the centers of detected circles
                param1=120, # Gradient value threshold for Canny edge detector (higher threshold)
                param2=80, # Accumulator threshold for the circle centers (lower threshold)
                minRadius=90, # Minimum circle radius to detect
                maxRadius=105 # Maximum circle radius to detect
            ) 

            # If circles are found
            if circles is not None:
                # Convert the (x, y, radius) coordinates to integers
                circles = np.uint16(np.around(circles[0])) 

                # Pick the largest circle found (assumed to be the can bottom)
                cx, cy, radius = max(circles, key=lambda c: c[2]) 

                # Draw the outer circle on the output frame
                cv2.circle(output, (cx, cy), radius, (0, 255, 67), 4) # Green color, thickness 4

                # Draw the center of the circle on the output frame
                cv2.circle(output, (cx, cy), 4, (0, 255, 67), -1) # Green color, filled

                # Print the center coordinates and radius to the console
                print(cx, cy, radius) 

            # Display the resulting frame with detections
            cv2.imshow("Rolling Soda Can Detection (HoughCircles)", output) 

        frameNumber += 1
        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(24) & 0xFF == ord('q'): 
            break

    # Release the video capture object and close all OpenCV windows
    cap.release() 
    cv2.destroyAllWindows() 

# Main function to run the script
def main(): 
    # Call the detection function with a specific video file path
    detect_can_bottom_video("movie.mp4") 
    return 0

# Entry point of the script
if __name__ == "__main__": 
    # Exit the system with the return code from main()
    sys.exit(main())


