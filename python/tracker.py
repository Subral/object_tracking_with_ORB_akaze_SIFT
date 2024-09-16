import cv2
import numpy as np
import asyncio
import websockets
import threading
import json
import base64

# Global variables to store object position, depth, and video feed
avg_center = None
z_depth = None
frame_base64 = None
lock = threading.Lock()

# WebSocket server to send coordinates and video feed
async def send_coordinates(websocket, path):
    global avg_center, z_depth, frame_base64
    try:
        while True:
            with lock:
                if avg_center is not None and z_depth is not None and frame_base64 is not None:
                    coordinates = {
                        "x": float(avg_center[0]),
                        "y": float(avg_center[1]),
                        "z": 1.0,
                        "frame": frame_base64
                    }
                    await websocket.send(json.dumps(coordinates))
            await asyncio.sleep(0.1)
    except websockets.ConnectionClosed:
        print("WebSocket connection closed")
    except Exception as e:
        print(f"Error in WebSocket: {e}")

def start_websocket_server():
    asyncio.set_event_loop(asyncio.new_event_loop())
    start_server = websockets.serve(send_coordinates, "localhost", 8765)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()

def track_object():
    global avg_center, z_depth, frame_base64

    cap = cv2.VideoCapture(0)
    image_to_track = cv2.imread('python/pepsi_logo.jpg', 0)

    if image_to_track is None:
        print("Error: Unable to load image.")
        return

    akaze = cv2.AKAZE_create(threshold=0.001)
    kp1, des1 = akaze.detectAndCompute(image_to_track, None)

    past_boxes = []
    N = 5

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.05

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
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)

            if len(good_matches) > 40:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 1.5)

                if M is not None:
                    h, w = image_to_track.shape
                    pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
                    dst = cv2.perspectiveTransform(pts, M)

                    past_boxes.append(dst)
                    if len(past_boxes) > N:
                        past_boxes.pop(0)

                    avg_box = np.mean(past_boxes, axis=0)

                    area_original = h * w
                    area_tracked = cv2.contourArea(np.int32(dst))
                    if area_tracked > 0:
                        z_depth = area_original / area_tracked
                    else:
                        z_depth = 0

                    avg_center = np.mean(avg_box, axis=0).flatten()
                    kalman.correct(np.array([[np.float32(avg_center[0])], [np.float32(avg_center[1])]]))
                    prediction = kalman.predict()
                    predicted_center = np.int32(prediction[:2].flatten())

                    frame = cv2.polylines(frame, [np.int32(avg_box)], True, (0, 255, 0), 3, cv2.LINE_AA)
                    frame = cv2.circle(frame, tuple(predicted_center), 5, (0, 0, 255), -1)

                    # Convert the frame to base64
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')

                    print(f"X: {avg_center[0]:.2f}, Y: {avg_center[1]:.2f}, Z: {z_depth:.2f}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Start the WebSocket server in a separate thread
websocket_thread = threading.Thread(target=start_websocket_server)
websocket_thread.start()

track_object()