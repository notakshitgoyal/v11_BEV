import time
import torch
import psutil
import numpy as np
import cv2
from ultralytics import YOLO
from statistics import mean
import json
from datetime import datetime

def benchmark_model(model_path, video_path, num_frames=100, device='mps', conf_threshold=0.25):
    """
    Benchmark YOLO model performance.
    
    Args:
        model_path (str): Path to YOLO model
        video_path (str): Path to test video
        num_frames (int): Number of frames to test
        device (str): Device to run inference on ('cpu', 'cuda', 'mps')
        conf_threshold (float): Confidence threshold for detections
    """
    # Initialize results dictionary
    results = {
        'model_path': model_path,
        'device': device,
        'video_path': video_path,
        'num_frames': num_frames,
        'conf_threshold': conf_threshold,
        'metrics': {}
    }
    
    # Load model
    try:
        print(f"Loading model from {model_path}...")
        model = YOLO(model_path)
        model.to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load video
    try:
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise Exception("Could not open video file")
    except Exception as e:
        print(f"Error loading video: {e}")
        return

    # Initialize metrics
    inference_times = []
    preprocessing_times = []
    postprocessing_times = []
    fps_values = []
    detection_counts = []
    memory_usage = []
    confidence_scores = []
    class_distributions = {}

    print("\nStarting benchmark...")
    print(f"Processing {num_frames} frames...")

    for i in range(num_frames):
        if i % 10 == 0:
            print(f"Processing frame {i}/{num_frames}")

        # Read frame
        success, frame = video.read()
        if not success:
            break

        # Record memory usage
        memory_usage.append(psutil.Process().memory_info().rss / 1024 / 1024)  # MB

        # Measure preprocessing time
        preprocess_start = time.time()
        frame = cv2.resize(frame, (640, 640))
        preprocess_time = time.time() - preprocess_start
        preprocessing_times.append(preprocess_time)

        # Measure inference time
        start_time = time.time()
        results_frame = model(frame, verbose=False, conf=conf_threshold)
        inference_time = time.time() - start_time
        inference_times.append(inference_time)

        # Measure postprocessing time
        postprocess_start = time.time()
        detections = results_frame[0]
        
        # Collect detection statistics
        for det in detections.boxes.data:
            _, _, _, _, conf, cls = det.tolist()
            confidence_scores.append(conf)
            
            # Update class distribution
            cls_id = int(cls)
            if cls_id in class_distributions:
                class_distributions[cls_id] += 1
            else:
                class_distributions[cls_id] = 1

        detection_counts.append(len(detections))
        postprocess_time = time.time() - postprocess_start
        postprocessing_times.append(postprocess_time)

        # Calculate FPS
        total_time = preprocess_time + inference_time + postprocess_time
        fps = 1 / total_time if total_time > 0 else 0
        fps_values.append(fps)

    # Calculate performance metrics
    total_detections = sum(detection_counts)
    avg_confidence = mean(confidence_scores) if confidence_scores else 0

    metrics = {
        'performance': {
            'average_inference_time': f"{mean(inference_times):.4f}s",
            'average_preprocessing_time': f"{mean(preprocessing_times):.4f}s",
            'average_postprocessing_time': f"{mean(postprocessing_times):.4f}s",
            'average_fps': f"{mean(fps_values):.2f}",
            'min_fps': f"{min(fps_values):.2f}",
            'max_fps': f"{max(fps_values):.2f}",
            'average_memory_usage_mb': f"{mean(memory_usage):.2f}",
            'max_memory_usage_mb': f"{max(memory_usage):.2f}"
        },
        'detection_statistics': {
            'total_frames_processed': len(fps_values),
            'total_detections': total_detections,
            'average_detections_per_frame': f"{mean(detection_counts):.2f}",
            'average_confidence_score': f"{avg_confidence:.4f}",
            'class_distribution': class_distributions
        }
    }

    results['metrics'] = metrics
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"benchmark_results_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

    # Print results
    print("\nBenchmark Results:")
    print("-" * 50)
    print("\nPerformance Metrics:")
    for key, value in metrics['performance'].items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    print("\nDetection Statistics:")
    for key, value in metrics['detection_statistics'].items():
        if key != 'class_distribution':
            print(f"{key.replace('_', ' ').title()}: {value}")
    
    print("\nClass Distribution:")
    for class_id, count in metrics['detection_statistics']['class_distribution'].items():
        print(f"Class {class_id}: {count} detections ({count/total_detections*100:.2f}%)")

    print(f"\nDetailed results saved to: {output_file}")

    # Cleanup
    video.release()

if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "yolo11s.pt"    # Update with your model path
    VIDEO_PATH = "save.mp4"      # Update with your video path
    NUM_FRAMES = 100             # Number of frames to process
    DEVICE = 'mps'              # 'cpu', 'cuda', or 'mps'
    CONF_THRESHOLD = 0.25       # Confidence threshold for detections

    benchmark_model(
        model_path=MODEL_PATH,
        video_path=VIDEO_PATH,
        num_frames=NUM_FRAMES,
        device=DEVICE,
        conf_threshold=CONF_THRESHOLD
    )