"""
1. Load the input image from file
2. Convert the image to grayscale
3. Apply Gaussian blur to reduce noise
4. Determine the image height and compute minimum
   and maximum allowable circle radii
5. Use the Hough Circle Transform to detect circles
   in the image
6. If no circles are detected, stop execution
7. Round and convert detected circle parameters
   to integer values
8. Select the circle with the largest radius,
   assuming it corresponds to the outer edge
   of the soda can
9. Draw the detected circle and its center
   on a copy of the original image
10. Display the resulting image in a window
"""

import cv2
import numpy as np
import sys

def detect_can_bottom(image_path):
    # Load the image in color format
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError("Image could not be loaded.")

    # Convert the image to grayscale for processing
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and smooth edges
    gray_img = cv2.GaussianBlur(gray_img, (9, 9), 2)

    # Get the image height to scale circle size limits
    img_height = gray_img.shape[0]

    # Define minimum and maximum radius for circle detection
    r_min = int(img_height * 0.25)
    r_max = int(img_height * 0.60)

    # Detect circles using the Hough Circle Transform
    detected = cv2.HoughCircles(
        gray_img,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=img_height,
        param1=120,
        param2=40,
        minRadius=r_min,
        maxRadius=r_max
    )

    # Stop execution if no circles are found
    if detected is None:
        raise RuntimeError("No circles detected.")

    # Convert detected circle parameters to integer values
    detected = np.int32(np.around(detected[0]))

    # Select the circle with the largest radius (outer edge)
    largest_circle = sorted(detected, key=lambda c: c[2], reverse=True)[0]
    cx, cy, radius = largest_circle

    # Create a copy of the original image for drawing
    output = image.copy()

    # Draw the detected outer circle on the image
    cv2.circle(output, (cx, cy), radius, (0, 255, 67), 7)

    # Draw the center point of the circle
    cv2.circle(output, (cx, cy), 4, (0, 255, 67), 5)

    # Display the final result
    cv2.imshow("Detected Soda Can Bottom", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    # Specify the image file to process
    image_file = "soda_can_top.jpeg"

    # Run the soda can bottom detection
    detect_can_bottom(image_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
