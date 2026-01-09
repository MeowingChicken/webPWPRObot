# --- PSEUDOCODE ---
# Circle Detection
# v1.0.0
# 1. Import necessary libraries: OpenCV and NumPy.
# 2. Define the path to the image file.
# 3. Load the image from the specified path.
# 4. Check if the image was loaded successfully; exit if not found.
# 5. Resize the image to a consistent maximum dimension (800 pixels) to standardize Hough Circle parameters across different input images.
# 6. Create a copy of the original image to draw the results on.
# 7. Pre-process the image for circle detection:
# a. Convert the image to grayscale.
# b. Apply a heavy median blur to reduce noise, suppress small details (like a pull-tab), and smooth the metallic surface.
# 8. Use the Hough Circle Transform (HoughCircles) to detect potential circles:
# a. Use the HOUGH_GRADIENT method.
# b. Define strict parameters for 'minDist', 'minRadius', and 'maxRadius' to specifically target only the large outer edge of the can.
# c. Use Canny edge threshold (param1) and accumulator threshold (param2) suitable for the pre-processed image.
# 9. Check if any circles were detected:
# a. If circles are found:
# i. Convert the circle parameters (x, y, radius) to integers.
# ii. Sort the detected circles by radius in descending order.
# iii. Select the largest detected circle (the first entry after sorting).
# iv. Draw the selected circle's circumference in green and its center in red on the output image.
# v. Print the detected center coordinates and radius.
# b. If no circles are found:
# i. Print an error message suggesting a parameter adjustment.
# 10. Display the image with the drawn circle until a key is pressed.
# 11. Close all OpenCV windows.

import cv2
import numpy as np

# --- COMMENTED CODE ---
# Define the path to the input image
path = '/Users/pl1021319/Documents/circle/soda_can.jpeg'

# Load the image from the specified file path
img = cv2.imread(path)

# Check if the image variable is empty (meaning the file was not found)
if img is None:
    print(f"Error: Image not found at {path}")
else:
    # 1. Resize for consistent parameter behavior across different image resolutions
    height, width = img.shape[:2]
    max_dim = 800
    # Calculate the scaling factor to ensure the largest dimension fits within max_dim
    scale = max_dim / max(height, width)
    # Resize the image using the calculated scale
    img = cv2.resize(img, (int(width * scale), int(height * scale)))

    # Create a copy of the resized image to draw detection results on
    output = img.copy()

    # 2. Advanced Pre-processing for Metal Surfaces
    # Convert the image from BGR (OpenCV default) to grayscale for edge detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply a heavy median blur (11x11 kernel) to suppress high-frequency noise,
    # small details (like pull-tabs or text), and smooth the reflective metallic surface.
    blurred = cv2.medianBlur(gray, 11)

    # 3. Detect ONLY the big circumference using the Hough Gradient method
    circles = cv2.HoughCircles(
        blurred,              # Input image (grayscale and blurred)
        cv2.HOUGH_GRADIENT,   # Detection method
        dp=1.2,               # Inverse ratio of the accumulator resolution to the image resolution (1 is same res, 2 is half)
        minDist=int(max_dim / 2),  # Minimum distance between the centers of detected circles (prevents multiple detections of the same can)
        param1=50,            # Canny edge detection upper threshold (lower threshold is half of this)
        param2=40,            # Accumulator threshold for the circle centers (lower values detect more circles, higher values detect fewer but 'better' ones)
        minRadius=int(max_dim * 0.25),  # Minimum radius in pixels (must be at least 25% of image size)
        maxRadius=int(max_dim * 0.5)    # Maximum radius in pixels (no larger than 50% of image size)
    )

    # 4. Process the detection results
    if circles is not None:
        # Convert the floating point circle parameters (x, y, radius) to unsigned 16-bit integers
        circles = np.uint16(np.around(circles))

        # Sort detected circles by radius in descending order to prioritize the largest one
        circles_sorted = sorted(circles[0, :], key=lambda x: x[2], reverse=True)

        # Unpack the parameters of the largest circle
        x, y, r = circles_sorted[0]

        # Draw the big circumference on the output image (Green color, thickness 4)
        cv2.circle(output, (x, y), r, (0, 255, 0), 4)

        # Draw the center point on the output image (Red color, filled circle)
        cv2.circle(output, (x, y), 5, (0, 0, 255), -1) # Corrected syntax for thickness argument

        # Print the results to the console
        print(f"Detected: Center({x}, {y}), Radius {r}")
    else:
        # Message if no circles met the strict criteria
        print("No circle found. Try lowering param2 to 30.")

    # 5. Display the results
    cv2.imshow("Corrected Detection", output)
    # Wait indefinitely until a key is pressed
    cv2.waitKey(0)
    # Close all open OpenCV windows
    cv2.destroyAllWindows()
