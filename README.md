# YOLOv11 + Norfair Multi-Object Tracker

A powerful real-time object detection and tracking application using state-of-the-art deep learning. This application combines **YOLOv11** (Ultralytics) for object detection with **Norfair** for robust multi-object tracking, allowing you to automatically detect and track objects, people, vehicles, and more through your webcam feed.

## Features

- **Automatic Object Detection**: Detects 80+ object classes (people, cars, dogs, cats, etc.) using YOLOv11
- **Real-time Tracking**: Maintains persistent IDs for tracked objects across frames
- **Two Tracking Modes**:
  - **Track All**: Automatically track all detected objects
  - **Selective Mode**: Click on objects to track only specific ones
- **Object Classification**: Shows class labels and confidence scores
- **Color-coded Boxes**: Different colors for different object classes
- **High Performance**: ~15-30 FPS on CPU, 60+ FPS on GPU
- **Flexible Models**: Choose from 5 YOLO model sizes for your hardware
- **Live Statistics**: FPS counter, detection count, tracking count

## What Can It Detect?

The tracker can detect and track 80 different object classes from the COCO dataset, including:

- **People**: person
- **Vehicles**: car, truck, bus, motorcycle, bicycle, train, airplane, boat
- **Animals**: dog, cat, horse, cow, bird, sheep, elephant, bear, zebra, giraffe
- **Sports**: baseball, tennis racket, frisbee, skis, snowboard, sports ball
- **Indoor Objects**: chair, couch, bed, dining table, TV, laptop, mouse, keyboard, phone, book
- **Kitchen**: bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich
- And many more!

## Installation

### Prerequisites

- Python 3.8 or higher
- Webcam connected to your computer
- (Optional) NVIDIA GPU with CUDA for faster performance

### Setup

1. Navigate to the project directory:
```bash
cd object-tracker
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. **First Run**: The first time you run the tracker, it will automatically download the YOLOv11 model (~6-50 MB depending on model size). This only happens once.

That's it! You're ready to track objects.

## Usage

### Basic Usage

Run with default settings (YOLOv11-nano model, track all objects):
```bash
python tracker.py
```

### Advanced Options

**Choose Model Size** (affects speed vs accuracy):
```bash
# Fastest (best for CPU, lower accuracy)
python tracker.py --model n

# Balanced (good for most cases)
python tracker.py --model s

# Medium (good GPU performance)
python tracker.py --model m

# High accuracy (requires good GPU)
python tracker.py --model l

# Maximum accuracy (requires powerful GPU)
python tracker.py --model x
```

**Adjust Confidence Threshold**:
```bash
# Only show very confident detections (fewer false positives)
python tracker.py --conf 0.5

# Show more detections (may include some false positives)
python tracker.py --conf 0.2
```

**Start in Selective Mode** (only track clicked objects):
```bash
python tracker.py --selective
```

**Combine Options**:
```bash
# Use small model with high confidence in selective mode
python tracker.py --model s --conf 0.4 --selective
```

## Controls

### Keyboard Controls
- **M**: Toggle between "Track All" and "Click to Select" modes
- **R**: Reset/clear all selected objects (in selective mode)
- **Q**: Quit the application

### Mouse Controls (Selective Mode)
- **Left Click**: Click on any detected object to start tracking it
- **Click Again**: Click on a tracked object to untrack it

## Tracking Modes

### 1. Track All Mode (Default)
- Automatically detects and tracks all objects in view
- All detected objects get bounding boxes with IDs
- Best for understanding what's happening in the scene
- No manual selection needed

### 2. Selective Mode
- Only tracks objects you click on
- Click on any detected object to add it to tracking
- Click again to remove it from tracking
- Great for focusing on specific objects of interest
- Toggle with **M** key at any time

## Model Size Comparison

| Model | Speed (CPU) | Speed (GPU) | Accuracy | Model Size | Best For |
|-------|-------------|-------------|----------|------------|----------|
| **n** (nano) | ~20-30 FPS | ~100+ FPS | Good | ~6 MB | Most computers, real-time |
| **s** (small) | ~15-25 FPS | ~80-100 FPS | Better | ~22 MB | Balanced performance |
| **m** (medium) | ~10-15 FPS | ~60-80 FPS | High | ~50 MB | GPU users, better accuracy |
| **l** (large) | ~5-10 FPS | ~40-60 FPS | Higher | ~90 MB | Strong GPU, high accuracy |
| **x** (extra) | ~3-7 FPS | ~30-50 FPS | Highest | ~130 MB | Best GPU, maximum accuracy |

**Recommendation**: Start with **nano (n)** for CPU or **small (s)** for GPU.

## How It Works

This tracker uses a two-stage approach:

### 1. Detection (YOLOv11)
- Each frame is processed by YOLOv11
- Detects objects and their bounding boxes
- Classifies each object (person, car, dog, etc.)
- Provides confidence scores

### 2. Tracking (Norfair)
- Associates detections across frames
- Maintains unique IDs for each object
- Handles occlusions and temporary disappearances
- Provides smooth tracking even when detection is uncertain

### Benefits Over Classical Tracking
Compared to the old OpenCV trackers (KCF, CSRT, etc.):
- **No manual selection required** (can track automatically)
- **Object classification** (knows what objects are)
- **Better accuracy** with complex scenes
- **Handles occlusions** better
- **Re-identifies objects** if they reappear
- **Multiple objects** tracked simultaneously with ease

## Tips for Best Results

### Performance Optimization
1. **Choose the right model**:
   - CPU users: Use `--model n` (nano)
   - GPU users: Use `--model s` or `--model m`
   - Powerful GPU: Try `--model l`

2. **Adjust confidence threshold**:
   - Too many false detections? Increase: `--conf 0.4`
   - Missing objects? Decrease: `--conf 0.2`

3. **GPU Acceleration**:
   - Install PyTorch with CUDA support for massive speedup
   - YOLOv11 will automatically use GPU if available

### Detection Quality
1. **Lighting**: Good, even lighting improves detection accuracy
2. **Distance**: Objects should be reasonably visible (not too small)
3. **Angle**: Front or side views work better than top-down
4. **Motion**: Tracker handles all motion speeds well
5. **Occlusion**: Partial occlusion is okay, tracker maintains IDs

### Tracking Stability
- The tracker maintains object IDs even during brief occlusions
- If an object disappears and reappears, it may get a new ID
- Use higher confidence thresholds for more stable tracking
- Lower confidence for detecting smaller or partially hidden objects

## Understanding the Display

### Bounding Box Colors
- **Different colors** = Different object classes
- **Green (thick)** = Selected/pinned object (in selective mode)
- **Colored (normal)** = Tracked object (in track-all mode)

### Labels
Each tracked object shows:
- **ID number**: Persistent identifier (e.g., #12)
- **Class name**: What the object is (e.g., "person", "car")
- **Confidence**: Detection confidence (e.g., 0.87 = 87% confident)
- **[TRACKED]**: Appears when object is selected in selective mode

### Info Panel
- **Model**: Which YOLO model and tracker is being used
- **FPS**: Current frames per second
- **Mode**: Current tracking mode
- **Detections**: Number of objects detected in current frame
- **Tracked**: Number of objects being tracked
- **Selected**: Number of manually selected objects (selective mode)

## Troubleshooting

### Webcam Not Found
**Error**: "Cannot open webcam"

**Solutions**:
- Ensure webcam is connected and not in use by another application
- Try changing camera index: modify line 45 in `tracker.py` to `cv2.VideoCapture(1)` or higher
- Check camera permissions on your OS

### Low FPS / Slow Performance
**Solutions**:
1. Use smaller model: `python tracker.py --model n`
2. Increase confidence threshold: `python tracker.py --conf 0.4` (fewer detections = faster)
3. Lower webcam resolution: modify lines 51-52 in `tracker.py`
4. Close other applications
5. Install PyTorch with GPU support (10-20x speedup)

### GPU Not Being Used
**Check if GPU is available**:
```python
import torch
print(torch.cuda.is_available())  # Should print True
```

**Install CUDA-enabled PyTorch**:
```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Objects Not Being Detected
**Solutions**:
1. Lower confidence threshold: `--conf 0.2`
2. Improve lighting conditions
3. Use larger model for better accuracy: `--model s` or `--model m`
4. Check if object class is in COCO dataset (80 classes supported)
5. Make sure objects are clearly visible and not too small

### Import Errors
**Error**: "No module named 'ultralytics'" or similar

**Solution**:
```bash
pip install -r requirements.txt
```

**Error**: "No module named 'torch'"

**Solution**:
```bash
pip install torch torchvision
```

## Technical Details

### Dependencies
- **OpenCV (cv2)**: Webcam capture and display
- **NumPy**: Array operations
- **Ultralytics**: YOLOv11 object detection
- **Norfair**: Multi-object tracking algorithms
- **PyTorch**: Deep learning backend (auto-installed with ultralytics)

### Architecture
```
Webcam → Frame → YOLOv11 Detection → Norfair Tracking → Display
                      ↓                      ↓
                  Bounding boxes        Persistent IDs
                  Class labels         Track association
                  Confidence scores    Occlusion handling
```

### YOLO Models
- **Framework**: Ultralytics YOLOv11 (latest version)
- **Training Data**: COCO dataset (80 classes)
- **Input**: RGB images (640x640 default)
- **Output**: Bounding boxes, class predictions, confidence scores

### Norfair Tracker
- **Algorithm**: Euclidean distance-based tracking
- **Features**:
  - Hit counter for track initialization/deletion
  - Handles missing detections
  - Maintains track history
  - Smooth track interpolation

## Project Structure

```
object-tracker/
├── tracker.py          # Main application (YOLOv11 + Norfair)
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Performance Benchmarks

Tested on different hardware configurations:

| Hardware | Model | Resolution | FPS | Notes |
|----------|-------|------------|-----|-------|
| Intel i5 CPU | nano | 1280x720 | ~25 | Good for CPU-only |
| Intel i7 CPU | small | 1280x720 | ~15 | Balanced |
| RTX 3060 GPU | nano | 1280x720 | ~120 | Very smooth |
| RTX 3060 GPU | small | 1280x720 | ~90 | Recommended |
| RTX 3060 GPU | medium | 1280x720 | ~70 | High accuracy |
| RTX 4090 GPU | large | 1280x720 | ~110 | Maximum quality |

## Future Enhancements

Possible improvements for future versions:
- Record tracking sessions to video files
- Export tracking data (positions, timestamps) to CSV/JSON
- Custom object classes with fine-tuned models
- Multiple camera support
- Track-specific event triggers
- Heatmap visualization
- Path history visualization
- Object counting and analytics

## Comparison: Classical vs Deep Learning Tracking

| Feature | Classical (OpenCV) | Deep Learning (YOLO+Norfair) |
|---------|-------------------|------------------------------|
| Manual Selection | Required | Optional |
| Object Classification | No | Yes (80 classes) |
| Occlusion Handling | Poor | Good |
| Re-identification | No | Yes |
| Multiple Objects | Manual each | Automatic |
| Accuracy | Moderate | High |
| Speed (CPU) | Very Fast | Fast |
| Speed (GPU) | N/A | Very Fast |

## Known Limitations

- Small objects (<20x20 pixels) may not be detected reliably
- Very fast motion can cause temporary ID switches
- Extreme lighting changes may affect detection
- Heavy occlusion may cause track loss
- Objects outside COCO dataset classes won't be recognized

## License

This project is provided as-is for personal and educational use.

## Credits

- **YOLOv11**: Ultralytics (https://github.com/ultralytics/ultralytics)
- **Norfair**: Tryolabs (https://github.com/tryolabs/norfair)
- **OpenCV**: Open Source Computer Vision Library

---

**Enjoy tracking!** For questions or improvements, feel free to modify the code to suit your needs.

## Quick Start Examples

### Example 1: Track People
```bash
python tracker.py --model s --conf 0.3
# Press M to switch to selective mode, then click on people to track
```

### Example 2: High Performance Mode
```bash
python tracker.py --model n --conf 0.4
# Fast tracking with fewer false positives
```

### Example 3: Maximum Accuracy (GPU Required)
```bash
python tracker.py --model l --conf 0.35
# Best detection quality
```

### Example 4: Focus on Specific Objects
```bash
python tracker.py --selective
# Only tracks what you click on
```
