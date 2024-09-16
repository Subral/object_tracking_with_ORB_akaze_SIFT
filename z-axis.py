import cv2
import numpy as np
from collections import deque

# Define the camera intrinsic parameters (fx, fy, cx, cy)
# These values should come from camera calibration
K = np.array([[640, 0, 320],
              [0, 640, 240],
              [0, 0, 1]], dtype=np.float32)  # Example intrinsic matrix for a 640x480 camera

# Function to apply exponential moving average filter
def exponential_moving_average(current, previous, alpha=0.1):
    current = np.array(current, dtype=np.float32)
    previous = np.array(previous, dtype=np.float32)
    return alpha * current + (1 - alpha) * previous

# Function to extract center coordinates from bounding boxes
def get_box_center(box):
    box = np.int32(box)
    x_coords = [pt[0][0] for pt in box]
    y_coords = [pt[0][1] for pt in box]
    center_x = (max(x_coords) + min(x_coords)) / 2
    center_y = (max(y_coords) + min(y_coords)) / 2
    return [center_x, center_y]

# Pose estimation using solvePnP
def estimate_pose(src_pts, dst_pts, K):
    # Make sure we have at least 4 points for solvePnP
    if len(dst_pts) < 4 or len(src_pts) < 4:
        print("Not enough points to calculate pose.")
        return None

    # Assuming the object is planar, we define the real-world 3D coordinates (Z = 0).
    # You need to provide the real-world coordinates of the object corners.
    object_points = np.array([[0, 0, 0], 
                              [0, 1, 0], 
                              [1, 1, 0], 
                              [1, 0, 0]], dtype=np.float32)  # Real-world points (Z=0)
    
    # Destination points in the 2D image.
    image_points = dst_pts.reshape(-1, 2)
    
    # Check if we have exactly 4 points for both object and image
    if object_points.shape[0] < 4 or image_points.shape[0] < 4:
        print("Error: Not enough 3D-2D point correspondences.")
        return None
    
    # Use only the first 4 points if there are more
    object_points = object_points[:4]
    image_points = image_points[:4]

    # SolvePnP to get rotation and translation vectors
    success, rvec, tvec = cv2.solvePnP(object_points, image_points, K, None)
    
    if success:
        # tvec contains the translation vector, and tvec[2] is the Z-axis (depth).
        z_depth = tvec[2][0]  # Extract Z-component for depth
        return z_depth
    else:
        return None

# Function to track the object
def track_object():
    cap = cv2.VideoCapture(0)
    image_to_track = cv2.imread('python/pepsi_logo.jpg', 0)

    if image_to_track is None:
        print("Error: Unable to load image.")
        return

    # AKAZE detector and feature matcher setup
    akaze = cv2.AKAZE_create(threshold=0.001)
    kp1, des1 = akaze.detectAndCompute(image_to_track, None)

    past_boxes = deque(maxlen=5)  # Use deque for efficient popping from the left
    past_centers = deque(maxlen=5)  # Store past centers for filtering

    # Matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    # Kalman filter setup
    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    
    # Tune process noise covariance (Q) and measurement noise covariance (R)
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.01
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1

    last_center = [0.0, 0.0]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp2, des2 = akaze.detectAndCompute(gray_frame, None)

        if des2 is not None:
            matches = bf.knnMatch(des1, des2, k=2)

            good_matches = []
            for m_n in matches:
                if len(m_n) == 2:
                    m, n = m_n
                    if m.distance < 0.6 * n.distance:
                        good_matches.append(m)

            if len(good_matches) > 60:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)

                if M is not None and np.linalg.cond(M) < 1e10:
                    h, w = image_to_track.shape
                    pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
                    dst = cv2.perspectiveTransform(pts, M)

                    # Add current bounding box and center to history
                    past_boxes.append(dst)
                    current_center = get_box_center(dst)
                    past_centers.append(current_center)

                    # Apply exponential moving average to stabilize the center coordinates
                    smoothed_center = exponential_moving_average(current_center, last_center, alpha=0.2)
                    last_center = smoothed_center

                    # Kalman correction and prediction
                    kalman.correct(np.array([[np.float32(smoothed_center[0])], [np.float32(smoothed_center[1])]]))
                    prediction = kalman.predict()
                    predicted_center = np.int32(prediction[:2].flatten())

                    # Estimate the Z-axis (depth) using solvePnP
                    z_depth = estimate_pose(src_pts, dst_pts, K)
                    if z_depth is not None:
                        print(f"Estimated Depth (Z-axis): {z_depth:.2f}")

                    # Draw bounding box and predicted center
                    frame = cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
                    frame = cv2.circle(frame, tuple(predicted_center), 5, (0, 0, 255), -1)

                    # Print tracking data (for debugging or further use)
                    print(f"Smoothed X: {smoothed_center[0]:.2f}, Y: {smoothed_center[1]:.2f}")

        # Display the frame
        cv2.imshow('Object Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

# Start object tracking
track_object()
