# Multi-Object Webcam Tracker

A real-time object tracking application that allows you to track multiple objects simultaneously using your webcam. Simply click and drag to select objects, and the application will track them as they move around in the video feed.

## Features

- **Multi-object tracking**: Track multiple objects simultaneously
- **Click-to-select**: Easy click-and-drag interface to select objects
- **Real-time performance**: ~50-60 FPS with KCF tracker
- **Color-coded tracking**: Each tracked object gets a unique colored bounding box
- **Flexible tracking algorithms**: Choose between CSRT, KCF, MOSSE, or MIL trackers
- **Live FPS counter**: Monitor performance in real-time
- **Simple controls**: Easy keyboard shortcuts for management

## Installation

### Prerequisites

- Python 3.7 or higher
- Webcam connected to your computer

### Setup

1. Navigate to the project directory:
```bash
cd object-tracker
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

That's it! You're ready to run the tracker.

## Usage

### Basic Usage

Run the tracker with default settings (KCF algorithm):
```bash
python tracker.py
```

### Advanced Usage

Choose a specific tracking algorithm:
```bash
python tracker.py --tracker CSRT   # Most accurate, slower (~25-30 FPS)
python tracker.py --tracker KCF    # Balanced (default, ~50-60 FPS)
python tracker.py --tracker MOSSE  # Fastest (~100+ FPS)
python tracker.py --tracker MIL    # Good with occlusions
```

## Controls

### Mouse Controls
- **Click and Drag**: Select an object to track
  - Click at one corner of the object
  - Drag to the opposite corner
  - Release to start tracking
- **Repeat**: Click and drag again to add more tracked objects

### Keyboard Controls
- **D**: Delete the most recently added tracker
- **R**: Reset and remove all trackers
- **Q**: Quit the application

## Tracking Algorithms

The application supports four different tracking algorithms, each with different performance characteristics:

| Algorithm | Speed | Accuracy | Best For |
|-----------|-------|----------|----------|
| **KCF** (default) | Fast (~50-60 FPS) | Good | General purpose, balanced performance |
| **CSRT** | Slower (~25-30 FPS) | Highest | Accurate tracking, handles occlusions |
| **MOSSE** | Very Fast (~100+ FPS) | Lower | Fast-moving objects, low-power devices |
| **MIL** | Medium (~40-50 FPS) | Good | Objects with partial occlusions |

### Recommendations

- **Personal projects / General use**: Use **KCF** (default) - great balance
- **Need maximum accuracy**: Use **CSRT** - best for careful tracking
- **Low-power device / Many objects**: Use **MOSSE** - fastest option
- **Objects frequently hidden**: Use **MIL** - handles occlusions well

## How It Works

The application uses OpenCV's built-in tracking algorithms to follow objects through video frames:

1. **Initialization**: When you select an object, the tracker analyzes the visual features within the selected region
2. **Tracking**: On each new frame, the tracker searches for the object based on those features
3. **Update**: The bounding box position is updated to follow the object's movement
4. **Multi-tracking**: Each object is tracked independently with its own tracker instance

## Tips for Best Results

1. **Object Selection**:
   - Select a distinctive object with clear boundaries
   - Include the entire object in your selection
   - Avoid selecting background or overlapping objects

2. **Lighting**:
   - Ensure good, consistent lighting
   - Avoid extreme shadows or glare
   - Try to maintain stable lighting conditions

3. **Object Motion**:
   - Slow to moderate motion works best
   - Very fast movements may cause tracking loss
   - Smooth movements are easier to track than erratic ones

4. **Performance**:
   - Track fewer objects for better performance
   - Use MOSSE tracker if you need to track many objects
   - Close other applications to free up resources

## Troubleshooting

### Webcam Not Found
**Error**: "Cannot open webcam"

**Solutions**:
- Check that your webcam is connected
- Ensure no other application is using the webcam
- Try a different webcam index: modify line 20 in `tracker.py` to `cv2.VideoCapture(1)` or higher

### Tracking Lost
If tracking fails (shows "LOST"):
- Press **R** to reset and try again
- Select a more distinctive object
- Ensure better lighting
- Try a different tracker algorithm (CSRT for better accuracy)

### Low FPS
If performance is slow:
- Use MOSSE tracker: `python tracker.py --tracker MOSSE`
- Track fewer objects simultaneously
- Reduce webcam resolution (modify lines 26-27 in `tracker.py`)
- Close other resource-intensive applications

### Import Errors
**Error**: "No module named 'cv2'"

**Solution**:
```bash
pip install opencv-contrib-python
```

## Technical Details

- **Language**: Python 3
- **Main Libraries**: OpenCV (cv2), NumPy
- **Tracking Methods**: KCF, CSRT, MOSSE, MIL
- **Default Resolution**: 1280x720
- **Window System**: OpenCV HighGUI

## Project Structure

```
object-tracker/
├── tracker.py          # Main application
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Future Enhancements

Possible improvements for future versions:
- Save tracking data to file
- Record video with tracking overlays
- Deep learning-based tracking (YOLO + DeepSort)
- Object classification labels
- Tracking persistence across temporary occlusions
- Configuration file for custom settings

## License

This project is provided as-is for personal use.

## Credits

Built using OpenCV's tracking API and Python.

---

**Enjoy tracking!** If you encounter any issues or have suggestions, feel free to modify the code to suit your needs.
