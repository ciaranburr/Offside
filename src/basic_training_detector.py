#!/usr/bin/env python3
"""
Visual Feedback Soccer Detector
1. Shows model predictions with bounding boxes
2. Click to correct/add missing detections
3. Train on your corrections
"""

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from ultralytics import YOLO

class VisualFeedbackDetector:
    def __init__(self):
        """Initialize both YOLO (for initial predictions) and custom model (for learning)"""
        print("🚀 Initializing Visual Feedback Detector...")
        
        # YOLO for initial predictions
        self.yolo_model = YOLO('yolov8n.pt')
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"💻 Using device: {self.device}")
        
        # Training data storage
        self.training_data = []
        self.load_training_data()
        
        # Current image data
        self.current_image = None
        self.current_image_path = None
        self.current_predictions = []
        self.user_corrections = []
        
        print("✅ Detector ready!")
    
    def analyze_image(self, image_path):
        """
        Step 1: Get YOLO predictions and show them visually
        """
        print(f"\n🔍 Analyzing: {Path(image_path).name}")
        
        # Store current image
        self.current_image_path = image_path
        self.current_image = cv2.imread(image_path)
        self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        
        # Get YOLO predictions
        results = self.yolo_model(image_path, verbose=False)
        self.current_predictions = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i, box in enumerate(boxes):
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # Only keep players and balls
                    if class_id in [0, 32] and confidence > 0.3:  # person, sports ball
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        
                        obj_type = 'player' if class_id == 0 else 'ball'
                        
                        prediction = {
                            'id': i,
                            'type': obj_type,
                            'bbox': [x1, y1, x2, y2],
                            'confidence': confidence,
                            'status': 'predicted'  # predicted, confirmed, rejected
                        }
                        self.current_predictions.append(prediction)
        
        print(f"📊 Found {len(self.current_predictions)} predictions")
        
        # Show predictions visually
        self.show_predictions()
        
        return self.current_predictions
    
    def show_predictions(self):
        """Show image with prediction bounding boxes"""
        if self.current_image is None:
            return
        
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        ax.imshow(self.current_image)
        ax.set_title(f"Model Predictions - Click to add missing objects", fontsize=14)
        
        # Draw prediction boxes
        for pred in self.current_predictions:
            x1, y1, x2, y2 = pred['bbox']
            w, h = x2 - x1, y2 - y1
            
            # Color based on type and status
            if pred['type'] == 'ball':
                color = 'red' if pred['status'] == 'predicted' else 'darkred'
                label = f"⚽ Ball: {pred['confidence']:.2f}"
            else:
                color = 'green' if pred['status'] == 'predicted' else 'darkgreen'
                label = f"👤 Player: {pred['confidence']:.2f}"
            
            # Draw rectangle
            rect = patches.Rectangle((x1, y1), w, h, linewidth=2, 
                                   edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            
            # Add label
            ax.text(x1, y1-10, label, fontsize=10, color=color, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            
            # Add ID number
            ax.text(x1+5, y1+20, str(pred['id']), fontsize=12, color='white',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=color))
        
        # Instructions
        instructions = """
        Instructions:
        - Close this window when done reviewing
        - Each prediction has an ID number
        - Remember which predictions are correct/incorrect
        """
        ax.text(0.02, 0.98, instructions, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
               facecolor='yellow', alpha=0.8))
        
        ax.axis('off')
        plt.tight_layout()
        plt.show()
    
    def get_feedback(self):
        """
        Step 2: Get user feedback on predictions
        """
        print("\n📝 FEEDBACK TIME!")
        print("Review each prediction and tell me if it's correct:")
        
        corrections = []
        
        for pred in self.current_predictions:
            print(f"\nPrediction ID {pred['id']}: {pred['type']} (confidence: {pred['confidence']:.2f})")
            
            while True:
                response = input("Is this correct? (y)es / (n)o / (s)kip: ").lower().strip()
                
                if response in ['y', 'yes']:
                    pred['status'] = 'confirmed'
                    corrections.append({
                        'prediction_id': pred['id'],
                        'action': 'confirm',
                        'type': pred['type'],
                        'bbox': pred['bbox']
                    })
                    print("✅ Confirmed!")
                    break
                
                elif response in ['n', 'no']:
                    pred['status'] = 'rejected'
                    corrections.append({
                        'prediction_id': pred['id'],
                        'action': 'reject'
                    })
                    print("❌ Rejected!")
                    break
                
                elif response in ['s', 'skip']:
                    print("⏭️  Skipped")
                    break
                
                else:
                    print("Please enter 'y', 'n', or 's'")
        
        # Ask about missing objects
        print("\n🔍 Are there any missing balls or players?")
        
        while True:
            missing = input("Add missing object? (b)all / (p)layer / (d)one: ").lower().strip()
            
            if missing in ['d', 'done']:
                break
            
            elif missing in ['b', 'ball']:
                print("Click on the ball location in the image (approximate center)")
                ball_coords = self.click_to_add('ball')
                if ball_coords:
                    corrections.append({
                        'action': 'add',
                        'type': 'ball',
                        'center': ball_coords
                    })
            
            elif missing in ['p', 'player']:
                print("Click on the player location in the image")
                player_coords = self.click_to_add('player')
                if player_coords:
                    corrections.append({
                        'action': 'add',
                        'type': 'player',
                        'center': player_coords
                    })
            
            else:
                print("Please enter 'b', 'p', or 'd'")
        
        return corrections
    
    def click_to_add(self, object_type):
        """
        Show image and let user click to add missing objects
        """
        print(f"🖱️  Click on the {object_type} location, then close the window")
        
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        ax.imshow(self.current_image)
        ax.set_title(f"Click on the {object_type} location", fontsize=14)
        
        clicks = []
        
        def on_click(event):
            if event.inaxes != ax:
                return
            
            x, y = event.xdata, event.ydata
            clicks.append((x, y))
            
            # Draw a marker where clicked
            color = 'red' if object_type == 'ball' else 'blue'
            ax.plot(x, y, 'o', color=color, markersize=10)
            ax.text(x+10, y-10, f"New {object_type}", color=color, fontsize=12,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            fig.canvas.draw()
        
        fig.canvas.mpl_connect('button_press_event', on_click)
        ax.axis('off')
        plt.show()
        
        if clicks:
            print(f"✅ Added {object_type} at {clicks[-1]}")
            return clicks[-1]
        else:
            print("❌ No clicks detected")
            return None
    
    def save_training_example(self, corrections):
        """
        Step 3: Save the corrected data for training
        """
        # Create training example
        training_example = {
            'image_path': self.current_image_path,
            'image_size': {
                'width': self.current_image.shape[1],
                'height': self.current_image.shape[0]
            },
            'original_predictions': self.current_predictions.copy(),
            'corrections': corrections,
            'timestamp': str(Path(self.current_image_path).stat().st_mtime)
        }
        
        # Build final ground truth
        ground_truth = []
        
        # Add confirmed predictions
        for correction in corrections:
            if correction['action'] == 'confirm':
                ground_truth.append({
                    'type': correction['type'],
                    'bbox': correction['bbox'],
                    'source': 'confirmed_prediction'
                })
            elif correction['action'] == 'add':
                # Convert click to bounding box (approximate)
                x, y = correction['center']
                size = 30 if correction['type'] == 'ball' else 80
                bbox = [x-size//2, y-size//2, x+size//2, y+size//2]
                
                ground_truth.append({
                    'type': correction['type'],
                    'bbox': bbox,
                    'source': 'user_added'
                })
        
        training_example['ground_truth'] = ground_truth
        
        # Save to training data
        self.training_data.append(training_example)
        self.save_training_data()
        
        print(f"💾 Saved training example with {len(ground_truth)} objects")
    
    def save_training_data(self):
        """Save training data to JSON"""
        with open('visual_training_data.json', 'w') as f:
            json.dump(self.training_data, f, indent=2)
        print(f"📁 Training data saved ({len(self.training_data)} examples)")
    
    def load_training_data(self):
        """Load existing training data"""
        try:
            with open('visual_training_data.json', 'r') as f:
                self.training_data = json.load(f)
            print(f"📁 Loaded {len(self.training_data)} training examples")
        except FileNotFoundError:
            self.training_data = []
            print("📁 No existing training data found")
    
    def show_training_stats(self):
        """Show statistics about training data"""
        if not self.training_data:
            print("❌ No training data available")
            return
        
        total_examples = len(self.training_data)
        total_objects = sum(len(ex['ground_truth']) for ex in self.training_data)
        
        ball_count = 0
        player_count = 0
        
        for example in self.training_data:
            for obj in example['ground_truth']:
                if obj['type'] == 'ball':
                    ball_count += 1
                else:
                    player_count += 1
        
        print(f"\n📊 TRAINING DATA STATISTICS:")
        print(f"   Total images: {total_examples}")
        print(f"   Total objects: {total_objects}")
        print(f"   Balls: {ball_count}")
        print(f"   Players: {player_count}")
        print(f"   Avg objects per image: {total_objects/total_examples:.1f}")
    
    def complete_feedback_loop(self, image_path):
        """
        Complete workflow: predict -> feedback -> save
        """
        print("🎯 Starting complete feedback loop...")
        
        # Step 1: Analyze and show predictions
        predictions = self.analyze_image(image_path)
        
        # Step 2: Get user feedback
        corrections = self.get_feedback()
        
        # Step 3: Save training data
        self.save_training_example(corrections)
        
        print("✅ Feedback loop completed!")
        return len(corrections)

def main():
    """Interactive training session"""
    print("🎮 Visual Feedback Soccer Detector")
    print("="*50)
    
    detector = VisualFeedbackDetector()
    
    while True:
        print("\nOptions:")
        print("1. Analyze single image (full feedback loop)")
        print("2. Quick prediction only")
        print("3. Show training data stats")
        print("4. Exit")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            image_path = input("Enter image path: ").strip()
            if Path(image_path).exists():
                corrections_count = detector.complete_feedback_loop(image_path)
                print(f"🎉 Added {corrections_count} corrections to training data!")
            else:
                print("❌ Image not found!")
        
        elif choice == '2':
            image_path = input("Enter image path: ").strip()
            if Path(image_path).exists():
                detector.analyze_image(image_path)
                print("👀 Review the predictions above")
            else:
                print("❌ Image not found!")
        
        elif choice == '3':
            detector.show_training_stats()
        
        elif choice == '4':
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice!")

if __name__ == "__main__":
    main()