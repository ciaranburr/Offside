#!/usr/bin/env python3
"""
Basic Soccer Detector
Step 1: Detect players and ball using YOLO
"""

import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from pathlib import Path

class BasicSoccerDetector:
    def __init__(self):
        """Initialize the detector with YOLO model"""
        print("🚀 Loading YOLO model...")
        self.model = YOLO('yolov8n.pt')  # Nano version for speed
        
        # COCO classes we care about
        self.target_classes = {
            0: 'person',      # Players and referees
            32: 'sports ball' # Soccer ball
        }
        print("✅ Model loaded successfully!")
    
    def detect(self, image_path):
        """
        Detect players and ball in an image
        
        Args:
            image_path: Path to image file
            
        Returns:
            list: Detected objects with bounding boxes
        """
        print(f"🔍 Analyzing: {Path(image_path).name}")
        
        # Run YOLO detection
        results = self.model(image_path, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # Only keep players and balls with good confidence
                    if class_id in self.target_classes and confidence > 0.4:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        
                        detection = {
                            'class': self.target_classes[class_id],
                            'confidence': confidence,
                            'bbox': [x1, y1, x2, y2],
                            'center': [(x1 + x2) / 2, (y1 + y2) / 2]
                        }
                        detections.append(detection)
        
        # Sort by confidence (best first)
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        print(f"📊 Found {len(detections)} objects")
        return detections
    
    def visualize(self, image_path, detections):
        """Draw bounding boxes on the image"""
        # Load image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Colors for different objects
        colors = {
            'person': (0, 255, 0),      # Green for people
            'sports ball': (255, 0, 0)  # Red for ball
        }
        
        # Draw each detection
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = map(int, det['bbox'])
            color = colors[det['class']]
            
            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Add label
            label = f"{det['class']}: {det['confidence']:.2f}"
            cv2.putText(image, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Number the detection
            cv2.putText(image, str(i+1), (x1+5, y1+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return image
    
    def analyze_image(self, image_path, show_result=True):
        """Complete analysis of a soccer image"""
        print(f"\n{'='*50}")
        print(f"ANALYZING: {Path(image_path).name}")
        print(f"{'='*50}")
        
        # Detect objects
        detections = self.detect(image_path)
        
        # Separate players and ball
        players = [d for d in detections if d['class'] == 'person']
        balls = [d for d in detections if d['class'] == 'sports ball']
        
        # Print summary
        print(f"\n📊 DETECTION SUMMARY:")
        print(f"   Players: {len(players)}")
        print(f"   Balls: {len(balls)}")
        print(f"   Total: {len(detections)}")
        
        # Show details
        if players:
            print(f"\n👥 PLAYERS DETECTED:")
            for i, player in enumerate(players, 1):
                x, y = player['center']
                conf = player['confidence']
                print(f"   Player {i}: ({x:.0f}, {y:.0f}) confidence: {conf:.2f}")
        
        if balls:
            print(f"\n⚽ BALLS DETECTED:")
            for i, ball in enumerate(balls, 1):
                x, y = ball['center']
                conf = ball['confidence']
                print(f"   Ball {i}: ({x:.0f}, {y:.0f}) confidence: {conf:.2f}")
        
        # Create visualization
        result_image = self.visualize(image_path, detections)
        
        # Show result
        if show_result:
            plt.figure(figsize=(12, 8))
            plt.imshow(result_image)
            plt.title(f"Detection Results: {len(players)} players, {len(balls)} balls")
            plt.axis('off')
            plt.show()
        
        return {
            'detections': detections,
            'players': players,
            'balls': balls,
            'image': result_image
        }

# Test function
def test_detector():
    """Test the detector on your dataset"""
    detector = BasicSoccerDetector()
    
    # Test on your images
    test_images = [
        "../data/offside",
        "../data/onside", 
        "../data/close"
    ]
    
    for folder in test_images:
        folder_path = Path(folder)
        if folder_path.exists():
            print(f"\n🔍 Testing images in: {folder}")
            
            # Get first image from folder
            images = list(folder_path.glob("*.jpg")) + list(folder_path.glob("*.png"))
            if images:
                result = detector.analyze_image(images[0])
                
                # Wait for user input before next image
                input("Press Enter to continue to next image...")
            else:
                print(f"No images found in {folder}")

if __name__ == "__main__":
    test_detector()