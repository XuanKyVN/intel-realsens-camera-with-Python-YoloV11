 import numpy as np
 import pyrealsense2 as rs
 import cv2
 from realsense_depth import *

 # Initialize DepthCamera
 dc = DepthCamera()

try:
    ret, depth_image, color_image = dc.get_frame()
    if not ret:
        print("Failed to get frames.")
    else:
        # Process the color image to find corners
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        edged = cv2.Canny(gray, 50, 100)
        edged = cv2.dilate(edged, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1)

        cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        corner_positions = []  # List to hold the corners' pixel positions

        for c in cnts:
            if cv2.contourArea(c) < 10000:  # Adjust threshold as needed
                continue

            box = cv2.minAreaRect(c)
            box = cv2.boxPoints(box)
            box = order_points(box)
            for (x, y) in box:
                corner_positions.append((int(x), int(y)))  # Save corner pixel positions
                cv2.circle(color_image, (int(x), int(y)), 5, (0, 0, 255), -1)  # Mark corners with red dots

        cv2.imshow("depth", depth_image)
        # Print the list of corner positions
        print("Corner positions (pixels):", corner_positions)

        # Use the corner pixel positions to find depth
        corner_depths = []
        for (x, y) in corner_positions:
            depth = depth_image[y, x]
            corner_depths.append(depth)

        # Print the depth of each corner
        print("Corner depths:", corner_depths)

        # Show the image with detected box
        cv2.imshow('Detect Box', color_image)
        cv2.waitKey(0)


finally:
    dc.release()
    cv2.destroyAllWindows()