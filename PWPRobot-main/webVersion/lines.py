import cv2
import numpy as np

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    Only keeps the region of the image defined by the polygon
    formed by "vertices". The rest of the image is black.
    """
    mask = np.zeros_like(img)
    # Filling the polygon with white
    cv2.fillPoly(mask, vertices, 255)
    # Returning the image only where mask pixels are non-zero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def make_points(image, line):
    """
    Extends the line segment to full lane line length based on slope/intercept.
    """
    if line is None:
        return None
    slope, intercept = line
    # Bottom of the image
    y1 = int(image.shape[0])
    # A point slightly below the middle for a 'horizon'
    y2 = int(y1 * (3/5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [[x1, y1, x2, y2]]

def average_slope_intercept(image, lines):
    """
    Averages the left and right lane lines to single solid lines.
    """
    left_fit = []
    right_fit = []
    if lines is None:
        return None, None
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        # Slope logic assumes image origin (0,0) is top-left
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0) if left_fit else None
    right_fit_average = np.average(right_fit, axis=0) if right_fit else None
    left_line = make_points(image, left_fit_average)
    right_line = make_points(image, right_fit_average)
    return left_line, right_line

def draw_lines(img, lines, color=(0, 255, 0), thickness=2):
    """
    Draws lines on the image.
    """
    if lines is None:
        return
    for line in lines:
        if line is not None:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def calculate_center_line_points(left_line, right_line):
    """
    Calculates the midpoint between two lane lines.
    """
    if left_line is None or right_line is None:
        return None

    lx1, ly1, lx2, ly2 = left_line[0]
    rx1, ry1, rx2, ry2 = right_line[0]

    mx1 = int((lx1 + rx1) / 2)
    my1 = ly1
    mx2 = int((lx2 + rx2) / 2)
    my2 = ly2

    return [[mx1, my1, mx2, my2]]

# Initialize video capture (use 0 for primary camera)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera.")
    # Create a black image if camera fails for demonstration purposes
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # In a real scenario, you'd exit or use a static image/video file
    # For now, we'll break after the first loop iteration
    cap = None

while cap:
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot receive frame.")
        break
    
    # --- IMAGE PROCESSING ---
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_frame, 50, 150)

    # Define region of interest vertices (a trapezoid covering the road)
    height, width = frame.shape[:2]
    # Vertices (bottom left, top left, top right, bottom right)
    vertices = np.array([[(width*0.1, height), (width*0.4, height*0.6), (width*0.6, height*0.6), (width*0.9, height)]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)

    # Use Hough lines
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 50, minLineLength=10, maxLineGap=50)

    # Calculate averaged lane lines
    left_line, right_line = average_slope_intercept(frame, lines)

    # Create an overlay image for drawing lines
    line_image = np.zeros_like(frame)
    draw_lines(line_image, [left_line, right_line], color=(0, 255, 0), thickness=5)

    # Calculate and draw center line
    center_line = calculate_center_line_points(left_line, right_line)
    draw_lines(line_image, [center_line], color=(0, 0, 255), thickness=3)

    # Combine original image with the line image overlay
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

    # --- DISPLAY ---
    cv2.imshow('Lane and Center Line Detection', combo_image)
    cv2.imshow('Masked Edges (Processing View)', masked_edges)

    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if cap:
    cap.release()
cv2.destroyAllWindows()
