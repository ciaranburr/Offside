# AI-Powered Soccer Offside Detection System

An intelligent computer vision system that uses machine learning to detect players, referees, and the ball in soccer images. This project serves as the foundation for automated offside detection in soccer matches.

## 🚀 Features

- **Player Detection**: Automatically identifies all players in soccer images
- **Ball Detection**: Locates soccer balls with high accuracy
- **Position Analysis**: Analyzes player positions and field distribution
- **Visual Output**: Generates annotated images with bounding boxes
- **JSON Export**: Saves detailed analysis results for further processing
- **Command Line Interface**: Easy-to-use CLI for batch processing

## 🛠️ Technology Stack

- **Python 3.9+**
- **PyTorch** - Deep learning framework
- **YOLOv8** - State-of-the-art object detection
- **OpenCV** - Computer vision operations
- **Matplotlib** - Visualization
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning utilities

## 📁 Project Structure

```
offside/
├── data/
│   ├── raw/           # Raw soccer images
│   └── processed/     # Processed data
├── src/
│   └── soccer_detector.py  # Main detection system
├── models/            # Trained model files
├── results/           # Output images and analysis
└── README.md
```

## 🔧 Setup Instructions

### For New Users (Fresh Setup)

1. **Clone the repository**
   ```bash
   git clone https://github.com/ciaranburr/Offside.git
   cd offside
   ```

2. **Install Miniconda** (if not already installed)
   ```bash
   brew install miniconda
   conda init zsh
   # Restart your terminal or run: source ~/.zshrc
   ```

3. **Create conda environment**
   ```bash
   conda create -n soccer-detection python=3.9
   conda activate soccer-detection
   ```

4. **Install dependencies**
   ```bash
   # Install scientific computing packages
   conda install -c conda-forge numpy matplotlib pillow scikit-learn
   
   # Install ML packages
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   pip install ultralytics opencv-python
   ```

5. **Verify installation**
   ```bash
   python -c "import torch; print('✅ PyTorch:', torch.__version__)"
   python -c "import cv2; print('✅ OpenCV works')"
   python -c "from ultralytics import YOLO; print('✅ YOLO works')"
   ```

### For Returning to the Project

1. **Navigate to project directory**
   ```bash
   cd path/to/offside
   ```

2. **Activate conda environment**
   ```bash
   conda activate soccer-detection
   ```

3. **You're ready to go!**

## 🎯 Quick Start

### Test with Sample Image

1. **Download a test image**
   ```bash
   curl -o data/raw/soccer_test.jpg "https://images.unsplash.com/photo-1574629810360-7efbbe195018?w=800"
   ```

2. **Run detection**
   ```bash
   cd src
   python soccer_detector.py ../data/raw/soccer_test.jpg --output ../results --show
   ```

### Command Line Usage

```bash
# Basic usage
python soccer_detector.py <image_path>

# With options
python soccer_detector.py image.jpg --output results/ --confidence 0.6 --show

# Batch processing example
for img in ../data/raw/*.jpg; do
    python soccer_detector.py "$img" --output ../results/
done
```

### Options

- `--output, -o`: Output directory for results
- `--confidence, -c`: Detection confidence threshold (default: 0.5)
- `--model, -m`: YOLO model variant (yolov8n.pt, yolov8s.pt, etc.)
- `--show, -s`: Display result image

## 📊 Output

The system generates:

1. **Annotated Images**: Visual results with bounding boxes
2. **JSON Analysis**: Detailed statistics and metadata
3. **Console Summary**: Quick overview of detections

### Example Output

```
🏈 Processing Soccer Image: match_photo.jpg
==================================================
🔍 Running detection on: match_photo.jpg
📊 Found 12 objects
📸 Result saved to: results/match_photo_detected.jpg
📊 Analysis saved to: results/match_photo_analysis.json

📊 DETECTION SUMMARY
Players detected: 11
Balls detected: 1
Average player confidence: 0.78
Field distribution - Left: 4, Center: 3, Right: 4
==================================================
```

## 🔮 Future Development

This project is designed to be extended with:

1. **Team Classification**: Distinguish between different teams using jersey colors
2. **Pose Estimation**: Integration with Meta's Sapiens for detailed player poses
3. **Offside Detection**: ML model to determine offside violations
4. **Real-time Processing**: Video stream analysis
5. **Web Dashboard**: Interactive web interface

## 🧪 Development Workflow

### Adding New Features

1. **Create feature branch**
   ```bash
   git checkout -b feature/team-classification
   ```

2. **Develop and test**
   ```bash
   python soccer_detector.py test_image.jpg --show
   ```

3. **Commit changes**
   ```bash
   git add .
   git commit -m "Add team classification feature"
   git push origin feature/team-classification
   ```

### Testing

```bash
# Test on multiple images
for img in data/raw/*.jpg; do
    echo "Testing: $img"
    python src/soccer_detector.py "$img" --confidence 0.6
done
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📚 Technical Details

### Model Information

- **Base Model**: YOLOv8 (You Only Look Once)
- **Classes Detected**: Person (players/referees), Sports Ball
- **Input Resolution**: 640x640 (automatically resized)
- **Framework**: PyTorch

### Performance Notes

- **CPU-only setup** for development (GPU optional)
- **Processing time**: ~1-3 seconds per image
- **Memory usage**: ~1-2GB RAM
- **Accuracy**: 70-90% depending on image quality

## ⚠️ Troubleshooting

### Common Issues

1. **Import hanging**: 
   ```bash
   # Try CPU-only PyTorch
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```

2. **Image not found**:
   ```bash
   # Check file path and permissions
   ls -la data/raw/
   ```

3. **Environment issues**:
   ```bash
   # Recreate environment
   conda deactivate
   conda env remove -n soccer-detection
   conda create -n soccer-detection python=3.9
   ```

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

## 👨‍💻 Author

**Ciaran Burr**
- Duke University Computer Science & Statistical Science
- Email: ciaranburr2005@gmail.com
- GitHub: [@ciaranburr](https://github.com/ciaranburr)

## 🙏 Acknowledgments

- Meta AI for Sapiens foundation models
- Ultralytics for YOLOv8
- OpenCV community
- PyTorch team

---

**Built with ❤️ for the future of soccer analytics**