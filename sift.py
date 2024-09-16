# import cv2
# import numpy as np
# from collections import deque

# # Define the camera intrinsic parameters (fx, fy, cx, cy)
# K = np.array([[640, 0, 320],
#               [0, 640, 240],
#               [0, 0, 1]], dtype=np.float32)  # Example intrinsic matrix for a 640x480 camera

# # Function to apply exponential moving average filter
# def exponential_moving_average(current, previous, alpha=0.1):
#     current = np.array(current, dtype=np.float32)
#     previous = np.array(previous, dtype=np.float32)
#     return alpha * current + (1 - alpha) * previous

# # Function to extract center coordinates from bounding boxes
# def get_box_center(box):
#     box = np.int32(box)
#     x_coords = [pt[0][0] for pt in box]
#     y_coords = [pt[0][1] for pt in box]
#     center_x = (max(x_coords) + min(x_coords)) / 2
#     center_y = (max(y_coords) + min(y_coords)) / 2
#     return [center_x, center_y]

# # Estimate pose using solvePnP for Z-depth (3D translation)
# def estimate_pose(src_pts, dst_pts, K):
#     if len(src_pts) < 4 or len(dst_pts) < 4:
#         return None

#     object_points = np.array([[0, 0, 0], 
#                               [0, 1, 0], 
#                               [1, 1, 0], 
#                               [1, 0, 0]], dtype=np.float32)

#     image_points = np.float32(dst_pts.reshape(-1, 2))

#     if object_points.shape[0] != image_points.shape[0] or image_points.shape[0] < 4:
#         return None

#     success, rvec, tvec = cv2.solvePnP(object_points, image_points, K, None)
    
#     if success:
#         z_depth = tvec[2][0]  # Z-component of translation vector
#         return z_depth
#     else:
#         return None

# # Function to track the object
# def track_object():
#     cap = cv2.VideoCapture(0)
#     image_to_track = cv2.imread('python/pepsi_logo.jpg', 0)

#     if image_to_track is None:
#         print("Error: Unable to load image.")
#         return

#     # SIFT detector and feature matcher setup
#     sift = cv2.SIFT_create()
#     kp1, des1 = sift.detectAndCompute(image_to_track, None)

#     past_boxes = deque(maxlen=5)
#     past_centers = deque(maxlen=5)

#     bf = cv2.BFMatcher()

#     kalman = cv2.KalmanFilter(4, 2)
#     kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
#     kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
#     kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.01
#     kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1

#     last_center = [0.0, 0.0]
#     consistent_detection_count = 0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         kp2, des2 = sift.detectAndCompute(gray_frame, None)

#         if des2 is not None:
#             matches = bf.knnMatch(des1, des2, k=2)

#             good_matches = []
#             for m_n in matches:
#                 if len(m_n) == 2:
#                     m, n = m_n
#                     if m.distance < 0.7 * n.distance:  # Stricter ratio test
#                         good_matches.append(m)

#             if len(good_matches) > 10:  # Ensure enough good matches
#                 src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
#                 dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

#                 M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

#                 if M is not None and np.linalg.cond(M) < 1e10:
#                     h, w = image_to_track.shape
#                     pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
#                     dst = cv2.perspectiveTransform(pts, M)

#                     current_center = get_box_center(dst)
#                     past_centers.append(current_center)

#                     smoothed_center = exponential_moving_average(current_center, last_center, alpha=0.2)
#                     last_center = smoothed_center

#                     kalman.correct(np.array([[np.float32(smoothed_center[0])], [np.float32(smoothed_center[1])]]))
#                     prediction = kalman.predict()
#                     predicted_center = np.int32(prediction[:2].flatten())

#                     consistent_detection_count += 1
#                     if consistent_detection_count >= 3:  # Require consistent detection for 3 frames
#                         print(f"Coordinates: X={predicted_center[0]}, Y={predicted_center[1]}")
#                         consistent_detection_count = 0  # Reset after reporting
                        
#                     frame = cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
#                     frame = cv2.circle(frame, tuple(predicted_center), 5, (0, 0, 255), -1)
#                 else:
#                     consistent_detection_count = 0  # Reset counter if homography fails
#             else:
#                 consistent_detection_count = 0  # Reset counter if not enough good matches

#         # Display the frame (optional for debugging)
#         cv2.imshow('Object Tracking', frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
            
#     cap.release()
#     cv2.destroyAllWindows()

# # Start object tracking
# track_object()



import cv2
import numpy as np
from collections import deque

K = np.array([[640, 0, 320],
              [0, 640, 240],
              [0, 0, 1]], dtype=np.float32)  

def exponential_moving_average(current, previous, alpha=0.1):
    current = np.array(current, dtype=np.float32)
    previous = np.array(previous, dtype=np.float32)
    return alpha * current + (1 - alpha) * previous

def get_box_center(box):
    box = np.int32(box)
    x_coords = [pt[0][0] for pt in box]
    y_coords = [pt[0][1] for pt in box]
    center_x = (max(x_coords) + min(x_coords)) / 2
    center_y = (max(y_coords) + min(y_coords)) / 2
    return [center_x, center_y]

def estimate_pose(src_pts, dst_pts, K):
    if len(dst_pts) < 4 or len(src_pts) < 4:
        return None

    object_points = np.array([[0, 0, 0], 
                              [0, 1, 0], 
                              [1, 1, 0], 
                              [1, 0, 0]], dtype=np.float32) 

    image_points = dst_pts.reshape(-1, 2)

    if object_points.shape[0] < 4 or image_points.shape[0] < 4:
        return None

    object_points = object_points[:4]
    image_points = image_points[:4]

    success, rvec, tvec = cv2.solvePnP(object_points, image_points, K, None)
    
    if success:
        z_depth = tvec[2][0] 
        return z_depth
    else:
        return None

def track_object():
    cap = cv2.VideoCapture(0)
    image_to_track = cv2.imread('python/pepsi_logo.jpg', 0)

    if image_to_track is None:
        print("Error: Unable to load image.")
        return

    akaze = cv2.AKAZE_create(threshold=0.001)
    kp1, des1 = akaze.detectAndCompute(image_to_track, None)

    past_boxes = deque(maxlen=5)
    past_centers = deque(maxlen=5)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
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

            if len(good_matches) > 10:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)

                if M is not None and np.linalg.cond(M) < 1e10:
                    h, w = image_to_track.shape
                    pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
                    dst = cv2.perspectiveTransform(pts, M)

                    current_center = get_box_center(dst)
                    past_centers.append(current_center)

                    smoothed_center = exponential_moving_average(current_center, last_center, alpha=0.2)
                    last_center = smoothed_center

                    kalman.correct(np.array([[np.float32(smoothed_center[0])], [np.float32(smoothed_center[1])]]))
                    prediction = kalman.predict()
                    predicted_center = np.int32(prediction[:2].flatten())

                    z_depth = estimate_pose(src_pts, dst_pts, K)
                    if z_depth is not None:
                        print(f"Coordinates: X={predicted_center[0]:.2f}, Y={predicted_center[1]:.2f}, Z={z_depth:.2f}")

                    frame = cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
                    frame = cv2.circle(frame, tuple(predicted_center), 5, (0, 0, 255), -1)

        cv2.imshow('Object Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
track_object()
