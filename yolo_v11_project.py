import cv2
import numpy as np
import torch
import math
from ultralytics import YOLO
from collections import defaultdict
import time
from scipy.spatial import distance as dist
import csv
from datetime import datetime
import os


device = 'mps'

# if torch.backends.mps.is_available():
#     device = torch.device("mps")

model = YOLO("yolo11s.pt")
model.to(device)

# Load the video
video = cv2.VideoCapture('save.mp4')
output_filename = 'output_yt.mp4'
width, height = 1280, 720
videoOut = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))

# track = True
track = False



yolo_classes = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

# yolo_classes = [
#     'person',        # pedestrians
#     'bicycle',       # cyclists
#     'car',          # personal vehicles
#     'motorcycle',   # two-wheelers
#     'bus',          # public transport
#     'truck',        # commercial vehicles
#     'traffic light' # traffic signals
# ]


def overlay_transparent(background, foreground, angle, x, y, objSize=50):
    original_frame = background.copy()
    foreground = cv2.resize(foreground, (objSize, objSize))


    # Get the shape of the foreground image
    rows, cols, channels = foreground.shape

    # Calculate the center of the foreground image
    center_x = int(cols / 2)
    center_y = int(rows / 2)

    # Rotate the foreground image
    M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1)
    foreground = cv2.warpAffine(foreground, M, (cols, rows))

    # Overlay the rotated foreground image onto the background image
    for row in range(rows):
        for col in range(cols):
            if x + row < background.shape[0] and y + col < background.shape[1]:
                alpha = foreground[row, col, 3] / 255.0
                background[x + row, y + col] = alpha * foreground[row, col, :3] + (1 - alpha) * background[x + row, y + col]

    # Blend the foreground and background ROI using cv2.addWeighted
    result = background

    return result


def simulate_object(background, object_class, x, y):
    # Load the object image based on the class
    object_img = cv2.imread(f'assets/{object_class}.png', cv2.IMREAD_UNCHANGED)
    if object_img is None:
        return background
    # Simulate the object by overlaying it onto the background image
    # object_img = cv2.resize(object_img, (100, 100))
    background[y:y+100, x:x+100] = overlay_transparent(background[y:y+100, x:x+100], object_img, 0, 0, 0)

    return background



def add_myCar_overlay(background):
    overlay_img = cv2.imread('assets/MyCar.png', cv2.IMREAD_UNCHANGED)
    # Get the shape of the overlay image
    rows, cols, _ = overlay_img.shape
    x = 550
    y = background.shape[0] - 200

    # Overlay the image onto the background
    overlay_img = overlay_transparent(background[y:y+rows, x:x+cols], overlay_img, 0, 0, 0, objSize=250)
    background[y:y+rows, x:x+cols] = overlay_img

    return background


object_history = defaultdict(list)  # Store position history for each object
object_speeds = {}  # Store calculated speeds
heat_map = None  # Global heat map
last_positions = {}  # Store last known positions
last_timestamps = {}  # Store timestamps for speed calculation
speed_records = {}  # Store speed records for each object: {id: {'speeds': [], 'logged': False}}
csv_filename = 'speed_violations.csv'

# Create CSV file with headers if it doesn't exist
if not os.path.exists(csv_filename):
    with open(csv_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Object ID', 'Average Speed (km/h)', 'Timestamp', 'Remarks'])

def calculate_speed(pos1, pos2, time_diff):
    """Calculate speed in pixels per second"""
    distance = np.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
    speed = distance / time_diff if time_diff > 0 else 0
    # Convert pixel speed to km/h and divide by 2
    speed_kmh = (speed * 3.6 * 0.1) / 2  # Adjusted speed calculation
    return speed_kmh

def update_heat_map(position, heat_map):
    """Update heat map with new position"""
    global width, height
    if heat_map is None:
        heat_map = np.zeros((height, width), dtype=np.float32)
    
    # Increase heat intensity and radius
    cv2.circle(heat_map, position, 20, 0.3, -1)
    
    # Apply stronger Gaussian blur
    heat_map = cv2.GaussianBlur(heat_map, (15, 15), 0)
    
    # Adjust decay rate (0.98 for slower decay, 0.8 for faster decay)
    heat_map *= 0.90
    
    # Add threshold to remove very weak heat signatures
    heat_map[heat_map < 0.1] = 0
    
    return heat_map

def update_speed_record(obj_id, speed):
    """Update speed record and log violations to CSV using average speed"""
    global speed_records
    
    # Initialize record if not exists
    if obj_id not in speed_records:
        speed_records[obj_id] = {'speeds': [], 'logged': False}
    
    # Add new speed to list
    speed_records[obj_id]['speeds'].append(speed)
    
    # Calculate average speed
    avg_speed = sum(speed_records[obj_id]['speeds']) / len(speed_records[obj_id]['speeds'])
    # Check for violation and log if not already logged
    if (avg_speed) > 80 and not speed_records[obj_id]['logged']:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(csv_filename, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                obj_id,
                f"{avg_speed}",  # Multiply by 2 for original speed scale
                timestamp,
                "Speed Limit Violated (>80 km/h)"
            ])
        speed_records[obj_id]['logged'] = True
    
    return avg_speed

next_object_id = 0
tracked_objects = {}  # format: {id: {'coords': (x,y), 'class': class_name, 'last_seen': timestamp}}
MAX_DISAPPEARED = 30  # Increased frames before removing tracked object
DISTANCE_THRESHOLD = 70  # Increased distance threshold for matching



def filter_detections(detections, min_size=20):
    """Filter out potentially false detections"""
    filtered_detections = []
    for detection in detections:
        box = detection.boxes[0]
        xmin, ymin, xmax, ymax = box.xyxy[0].tolist()
        width = xmax - xmin
        height = ymax - ymin
        
        # Filter by size
        if width < min_size or height < min_size:
            continue
            
        # Filter by position (remove detections at image edges)
        if xmin < 10 or ymin < 10 or xmax > frame.shape[1]-10 or ymax > frame.shape[0]-10:
            continue
            
        filtered_detections.append(detection)
    
    return filtered_detections

# Remove these lines as they're using undefined variables
# detections = results[0]
# filtered_detections = filter_detections(detections)

def assign_object_id(current_objects, tracked_objects):
    """Match current detections with tracked objects and assign IDs"""
    global next_object_id
    
    # Add Kalman filtering for prediction
    if not hasattr(assign_object_id, 'kalman_filters'):
        assign_object_id.kalman_filters = {}
    
    current_time = time.time()
    matched_objects = []
    
    # Initialize Kalman filter for new objects
    for obj_id in tracked_objects:
        if obj_id not in assign_object_id.kalman_filters:
            kf = cv2.KalmanFilter(4, 2)
            kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                           [0, 1, 0, 0]], np.float32)
            kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                          [0, 1, 0, 1],
                                          [0, 0, 1, 0],
                                          [0, 0, 0, 1]], np.float32)
            assign_object_id.kalman_filters[obj_id] = kf
    
    # If there are no current objects, return empty list and clear tracked objects
    if len(current_objects) == 0:
        tracked_objects.clear()
        return matched_objects

    # If there are no tracked objects yet, create new IDs for all current objects
    if len(tracked_objects) == 0:
        for obj in current_objects:
            tracked_objects[next_object_id] = {
                'coords': obj[0],
                'class': obj[1],
                'last_seen': current_time
            }
            matched_objects.append([obj[0], obj[1], next_object_id])
            next_object_id += 1
        return matched_objects

    # Get centroids of current and tracked objects
    current_centroids = np.array([obj[0] for obj in current_objects])
    tracked_centroids = np.array([obj['coords'] for obj in tracked_objects.values()])
    
    # Calculate distance matrix between current and tracked objects
    D = dist.cdist(current_centroids, tracked_centroids)
    
    # Find the closest matches
    rows = D.min(axis=1).argsort()
    cols = D.argmin(axis=1)[rows]
    
    used_rows = set()
    used_cols = set()
    
    # Match objects based on minimum distance
    tracked_ids = list(tracked_objects.keys())
    for (row, col) in zip(rows, cols):
        if row in used_rows or col in used_cols:
            continue
            
        if D[row, col] > DISTANCE_THRESHOLD:
            continue
            
        object_id = tracked_ids[col]
        tracked_objects[object_id] = {
            'coords': current_objects[row][0],
            'class': current_objects[row][1],
            'last_seen': current_time
        }
        matched_objects.append([
            current_objects[row][0],
            current_objects[row][1],
            object_id
        ])
        used_rows.add(row)
        used_cols.add(col)
    
    # Add new objects that weren't matched
    for row in range(len(current_objects)):
        if row in used_rows:
            continue
        tracked_objects[next_object_id] = {
            'coords': current_objects[row][0],
            'class': current_objects[row][1],
            'last_seen': current_time
        }
        matched_objects.append([
            current_objects[row][0],
            current_objects[row][1],
            next_object_id
        ])
        next_object_id += 1
    
    # Remove old objects
    current_tracked = tracked_objects.copy()
    for obj_id in current_tracked:
        if current_time - tracked_objects[obj_id]['last_seen'] > MAX_DISAPPEARED / 30.0:  # assuming 30 fps
            tracked_objects.pop(obj_id)
    
    return matched_objects

def plot_object_bev(transformed_image_with_centroids, src_points, dst_points, objs_):
    global heat_map, object_history, object_speeds, last_positions, last_timestamps
    
    # Create a raw transformed image
    raw_transformed = np.zeros_like(transformed_image_with_centroids)
    
    # Get the transformation matrix
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # Apply the perspective transform to the original frame
    raw_transformed = cv2.warpPerspective(frame, M, (width, height))
    
    # Start with a black background for object visualization
    transformed_image_with_centroids = np.zeros_like(transformed_image_with_centroids)
    
    persObjs = []
    current_time = time.time()

    # Transform all objects first
    transformed_objects = []
    for obj_ in objs_:
        if obj_:
            centroid_coords = np.array([[obj_[0]]], dtype=np.float32)
            transformed_coords = cv2.perspectiveTransform(centroid_coords, M)
            transformed_coords_ = tuple(transformed_coords[0][0].astype(int))
            transformed_objects.append([transformed_coords_, obj_[1]])
    
    # Assign IDs to transformed objects
    tracked_transformed_objects = assign_object_id(transformed_objects, tracked_objects)

    # Draw and track objects
    for obj_ in tracked_transformed_objects:
        transformed_coords_, class_name, obj_id = obj_
        x, y = transformed_coords_
        
        if 0 <= x < transformed_image_with_centroids.shape[1] and 0 <= y < transformed_image_with_centroids.shape[0]:
            # Update object history using object ID
            if obj_id not in object_history:
                object_history[obj_id] = []
            object_history[obj_id].append(transformed_coords_)
            
            # Calculate speed only for vehicles
            vehicle_classes = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']
            if class_name in vehicle_classes:
                # Speed calculation code here
                if obj_id in last_positions and obj_id in last_timestamps:
                    time_diff = current_time - last_timestamps[obj_id]
                    current_speed = calculate_speed(last_positions[obj_id], transformed_coords_, time_diff)
                    object_speeds[obj_id] = current_speed
                    avg_speed = update_speed_record(obj_id, current_speed)
                    
                # Update last position and timestamp for vehicles
                last_positions[obj_id] = transformed_coords_
                last_timestamps[obj_id] = current_time
            
            # Draw object and trail for all objects
            cv2.circle(transformed_image_with_centroids, transformed_coords_, radius=10, 
                      color=(0, 0, 255), thickness=-1)
            
            # Draw label
            label = f"ID:{obj_id} {class_name}"
            if obj_id in object_speeds:  # Only show speed for vehicles
                label += f" {object_speeds[obj_id]:.1f}km/h"
            
            cv2.putText(transformed_image_with_centroids, label,
                      (x + 15, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                      0.5, (0, 0, 255), 2)
            
            # Draw trail for all objects
            if len(object_history[obj_id]) > 1:
                points = np.array(object_history[obj_id], dtype=np.int32)
                cv2.polylines(transformed_image_with_centroids, [points], False, 
                            (0, 0, 255), 2)
            
            persObjs.append([transformed_coords_, class_name])
            
            # Update heat map
            heat_map = update_heat_map(transformed_coords_, heat_map)

    # Overlay heat map with black background
    if heat_map is not None:
        # Normalize heat map for better visibility
        heat_map_norm = cv2.normalize(heat_map, None, 0, 255, cv2.NORM_MINMAX)
        heat_map_colored = cv2.applyColorMap(heat_map_norm.astype(np.uint8), 
                                           cv2.COLORMAP_JET)
        
        # Create mask for non-zero values
        mask = (heat_map > 0.1).astype(np.uint8)
        mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=1)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # Overlay with adjusted alpha
        overlay = cv2.addWeighted(transformed_image_with_centroids, 0.7,
                                heat_map_colored * mask, 0.5, 0)
        transformed_image_with_centroids = overlay

    # Limit history length to prevent memory issues
    for obj_id in speed_records:
        if len(speed_records[obj_id]['speeds']) > 30:  # Keep last 30 speed readings
            speed_records[obj_id]['speeds'] = speed_records[obj_id]['speeds'][-30:]

    # Return both the processed image and the raw transformed image
    return transformed_image_with_centroids, persObjs, raw_transformed

frame_count = 0
centroid_prev_frame = []
tracking_objects = {}
tracking_id = 0

# Add these global variables at the top with other globals
selected_points = []
selecting_points = True

# Add this new function for mouse callback
def mouse_callback(event, x, y, flags, param):
    global selected_points, selecting_points
    if event == cv2.EVENT_LBUTTONDOWN and len(selected_points) < 4:
        selected_points.append([x, y])
        # Draw circle at selected point
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Select Points", frame)
        
        if len(selected_points) == 4:
            selecting_points = False
            cv2.destroyWindow("Select Points")

# Modify the main loop to include point selection at start
# Replace the coordinate definition section with:
success, frame = video.read()
if not success:
    print("Failed to read first frame")
    exit()

frame = cv2.resize(frame, (width, height))

# Point selection phase
cv2.namedWindow("Select Points")
cv2.setMouseCallback("Select Points", mouse_callback)
print("Please select 4 points in this order:")
print("1. Bottom-left\n2. Top-left\n3. Top-right\n4. Bottom-right")

while selecting_points:
    cv2.imshow("Select Points", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit()

# Convert selected points to numpy array
src_points = np.float32(selected_points)
dst_points = np.float32([
    [200, 720],    # Bottom-left
    [200, 0],      # Top-left
    [1080, 0],     # Top-right
    [1080, 720]    # Bottom-right
])

# Reset video capture to start
video.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Continue with the main processing loop
while True:
    # Read the next frame
    success, frame = video.read()
    if not success:
        print("Failed to read video frame - video may have ended")
        break
        
    frame = cv2.resize(frame, (width, height))
    frame_count += 1

    # Perform object detection on the frame
    results = model(frame, 
        imgsz=640,  # Increased image size
        conf=0.25,  # Increased confidence threshold
        iou=0.45    # Adjusted IOU threshold
    )
    detections = results[0]

    # Filter out potentially false detections
    filtered_detections = filter_detections(detections)

    # Create a black image with the same size as the video frames
    image_ = np.zeros((height, width, 3), dtype=np.uint8)
    simulated_image = image_.copy()
    transformed_image_with_centroids = image_.copy()
    transformed_image_to_sim = image_.copy()
    simObjs = image_.copy()

    objs = []
    centroid_curr_frame = []


    # After detection loop, debug print
    print(f"Number of detected objects: {len(objs)}")

    # Before transformation
    if len(objs) > 0:
        transformed_image_with_centroids, persObjs_, raw_transformed = plot_object_bev(
            transformed_image_with_centroids, src_points, dst_points, objs)
        
        # Debug print
        print(f"Number of transformed objects: {len(persObjs_)}")

    #####################
    ##  OBJ DETECTION  ##
    #####################
    for detection in filtered_detections:    
        # Get the bounding box coordinates
        box = detection.boxes[0]  # Get the first (and only) box
        xmin, ymin, xmax, ymax = box.xyxy[0].tolist()  # Convert to regular coordinates
        confidence = box.conf[0].item()  # Get confidence score
        class_id = box.cls[0].item()  # Get class ID
        
        centroid_x = int((xmin + xmax) // 2)
        centroid_y = int((ymin + ymax) // 2)

        # Check if class is in our list and confidence is good
        if confidence >= 0.15:
            # Draw bounding box on the frame
            color = (0, 0, 255)
            object_label = f"{yolo_classes[int(class_id)]}: {confidence:.2f}"
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
            cv2.putText(frame, object_label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)
            
            # Always add to centroid_curr_frame and objs
            centroid_curr_frame.append([(centroid_x, centroid_y), yolo_classes[int(class_id)]])
            objs.append([(centroid_x, centroid_y), yolo_classes[int(class_id)]])
            
            # Debug print
            print(f"Added object: {yolo_classes[int(class_id)]} at ({centroid_x}, {centroid_y})")

    # Debug print after detection loop
    print(f"Total objects added: {len(objs)}")

    #####################
    ## OBJECT TRACKING ##
    #####################
    if track:
        if frame_count <= 2:
            for pt1, class_id in centroid_curr_frame:
                for pt2, class_id in centroid_prev_frame:
                    dist = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
                    if dist < 50:
                        tracking_objects[tracking_id] = pt1, class_id
                        tracking_id += 1
        else:
            tracking_objects_copy = tracking_objects.copy()
            for obj_id, pt2 in tracking_objects_copy.items():
                objects_exists = False
                for pt1, class_id in centroid_curr_frame:
                    dist = math.hypot(pt2[0][0] - pt1[0], pt2[0][1] - pt1[1])
                    if dist < 20:
                        tracking_objects[obj_id] = pt1, class_id
                        objects_exists = True
                        continue
                if not objects_exists:
                    tracking_objects.pop(obj_id)


        for obj_id, pt1 in tracking_objects.items():
            cv2.circle(frame, pt1[0], 3, (0, 255, 255), -1)
            # cv2.putText(frame, str(obj_id)+' '+str(pt1[1]), pt1[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)
            if track:
                objs.append([pt1[0], pt1[1]])

        centroid_prev_frame = centroid_curr_frame.copy()


    #####################
    ##        BEV      ##
    #####################
    # Define the source points (region of interest) in the original image
    x1, y1 = 200, 720    # Bottom-left (moved inward)
    x2, y2 = 320, 480    # Top-left (adjusted for better perspective)
    x3, y3 = 960, 480    # Top-right (adjusted for better perspective)
    x4, y4 = 1080, 720   # Bottom-right (moved inward)
    src_points = np.float32([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])

    # Define the destination points (desired output perspective)
    u1, v1 = 200, 720    # Bottom-left
    u2, v2 = 200, 0      # Top-left
    u3, v3 = 1080, 0     # Top-right
    u4, v4 = 1080, 720   # Bottom-right
    dst_points = np.float32([[u1, v1], [u2, v2], [u3, v3], [u4, v4]])


    # perspectivs plot and objs
    transformed_image_with_centroids, persObjs_, raw_transformed = plot_object_bev(
        transformed_image_with_centroids, src_points, dst_points, objs)

    ### plot objs overlays
    for persObj_ in persObjs_:
        simObjs = simulate_object(transformed_image_to_sim, persObj_[1], persObj_[0][0], persObj_[0][1])
    # Add the car_img overlay to the simulated image
    # simulated_image = add_myCar_overlay(simObjs)
    simulated_image = simObjs  # Just use the simulated objects without the static car


    videoOut.write(simulated_image)
    # Display the simulated image and frame
    cv2.imshow("Video", frame)
    cv2.imshow("Simulated Objects", simulated_image)
    cv2.imshow('Transformed Frame', transformed_image_with_centroids)
    # cv2.imwrite('test.jpg', simulated_image)

    # Add this new window display
    cv2.imshow("Raw BEV Transform", raw_transformed)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
video.release()
videoOut.release()
cv2.destroyAllWindows()
