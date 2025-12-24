# JM‑PoseMatcher (Core v1.0)

![Version](https://img.shields.io/badge/version-1.0-blue.svg)
![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)
![Python](https://img.shields.io/badge/python-3.7+-blue.svg)

**Lightweight real-time pose and facial expression matching against your own custom reference images**

JM-Pose-Matcher (Joint Match - Pose Matcher) is a Python-based computer vision tool that compares your live webcam image, including body pose and facial expressions against a set of reference template images. Perfect for a range of applications such as fitness form checking, dance choreography, yoga pose validation, animation reference matching, and interactive installations.

---

## 🎯 Features
- **Real-time Pose Matching** - Compare body positions against reference images with instant, real time match rates
- **Facial Expression Tracking** - Match facial expressions with configurable detail levels to match needed performance
- **Hybrid Scoring** - Combines pose and face matching with customizable weights for your use case
- **Multiple Profiles** - Pre-configured tracking profiles for different use cases, easy to add more!
- **Dynamic & Single Modes** - Auto-switch to the best match, or lock onto a specific template
- **Quality Indicators** - Visual feedback on tracking data quality and visibility to aid accuracy and performance
- **Skeleton Overlays** - Toggle landmark connections on both webcam and templates
- **Mirror Mode** - Easy toggle for mirrored display for easier, more natural matching
- **Multi-Camera Support** - Automatic detection and selection of available cameras
- **Image Compression** - To optimise memory for large template sets
- **Easy Config** - Both GUI and CMD arguments for any device

---

## 💭 What it does

JM‑PoseMatcher loads a set of **template images** (JPG or PNG) from a user defined folder. Then, based of your live camera feed, it matches the feed to calculate a match score to an image based of your position in the webcam. 

It supports two modes:
- **Dynamic mode**: automatically selects the best-matching template in real time.
- **Single mode**: locks to one template and lets you move through templates manually, showing scores for each.

and includes:
- **Pose landmark matching** (body position / posture)
- **Face expression matching** (simple expression features derived from face mesh)
- **Hybrid scoring** (weighted combination of pose + face, depending on selected profile)
- **On‑screen skeleton overlays**
- **Optional face mesh overlays**
- **Config GUI** for config
- **Command line / no‑GUI mode** for direct launching

---

## 📋 Potential use cases

After a quick prompt to a range of AI agents, and people I know in the industry, here are a list of potential use cases for a system like this one:

### Training and movement
- **Yoga / stretching**: match a reference pose photo and try to reproduce it accurately.
- **Fitness form checks**: compare posture against a known good reference (with some tinkering to support video, this could be a good next project to analyse gym form).
- **Dance practice**: store key frames as templates and match them live.

### Content creation
- **Photography / self‑timer posing**: match a desired stance/expression in multiple frames.
- **Cosplay / character posing**: maintain consistent poses and positions across shots.

### Acting / expression work
- **Expression tracking** (Face Only profiles) to rehearse facial expressions (smile, mouth open, brow raise, etc.) and get real time match rates to how accurate they are.

### Accessibility / interaction
- Hands‑free “pose selection” (dynamic mode) for interactive experiences and installations. Fun and lightweight for all systems.

---

## 📖 How it works

The app compares your webcam frame to each template image using **landmark geometry**.

### 1) Template preprocessing
When you load a templates folder, JM‑PoseMatcher:
1. Reads each image, checking its valid
2. Runs **MediaPipe Pose** to extract selected pose landmarks, which depends on profile selection
3. Runs **MediaPipe Face Mesh** and computes compact facial “expression features” if enabled
4. Stores this geometry:
   - landmarks (x, y, visibility)
   - image diagonal (for normalization)
   - facial feature vector (if available)

Templates with insufficient visible pose landmarks are skipped to avoid inaccurate results.

### 2) Live detection
For every webcam frame:
- Pose landmarks are detected with MediaPipe Pose
- Face mesh landmarks are detected with MediaPipe Face Mesh
- Facial features are computed and compared

### 3) Pose matching (similarity transform)
For landmarks that are visible in both live frame and template:
- The program estimates a **similarity transform** (scale + rotation + translation) that best maps live points onto template points using Umeyama alignment.
- After transforming the live points, it computes an RMS distance to the template.

That RMS distance is normalized by the template image diagonal and turned into a score.

#### Matching Algorithm
The scoring system uses an exponential decay function:

```math
score = 100 \times e^{-k \times relative\_distance}
```

Where:
- `relative_distance` = RMS distance normalized by image diagonal
- `k` = sensitivity constant (default: 40)
- **Result**: 0-100% match score

_Note: If facial tracking is disabled, this is then used to get a match score which is returned as a percentage._

### 4) Face expression matching (feature distance)
A small facial feature vector is derived using distances such as:
- eye openness
- mouth openness
- mouth width
- brow raise

Similarity is computed from feature distance and converted to a 0–100 score!

### 5) Combined score
Depending on profile:
- **Default / Upper / Full**: pose score dominates, face adds refinement
- **Face Only**: face score is everything

If both pose and face scores exist:

`combined = pose_weight * pose_score + face_weight * face_score`

This distance between joints and meshes approach is faster, however more approximate. Hence making this a lightweight, easy to configure pose matcher!
---

## 🚦 Quality Indicators

The system provides real-time feedback on tracking quality:

- 🟢 **GREEN (Excellent)** - Sufficient landmarks detected, high confidence
- 🟡 **AMBER (Acceptable)** - Partial detection, adjust position
- 🔴 **RED (Adjust Pose)** - Insufficient landmarks, reposition or improve lighting

---

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- Webcam or USB camera
- Windows, macOS, or Linux

### Installation

1. **Clone or download the repository:**
```bash
git clone https://github.com/JoshuMain/JM-Pose-Matcher.git
cd JM-Pose-Matcher
```

2. **Install dependencies:**
```
pip install opencv-python mediapipe numpy pillow
```

### Basic Usage
#### Launch with GUI
```
python jm_pose_matcher.py
```
#### Skip GUI and start immediately:
```
python jm_pose_matcher.py /path/to/templates --no-gui
```
_See Command Line Arguments for more args_

---
## 🖥️ Command Line Arguments

### Basic Syntax
```bash
python jm_pose_matcher.py [TEMPLATES_PATH] [OPTIONS]
```

### Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `TEMPLATES_PATH` | positional | - | Path to folder with template images (JPG/PNG) |
| `--no-gui` | flag | false | Skip config window, start video immediately |
| `--profile` | choice | `default` | Profile: `face_only`, `upper_body`, `default`, `full_body` |
| `--camera` | int | `0` | Camera index to use |
| `--mode` | choice | `dynamic` | Matching mode: `dynamic` or `single` |
| `--compression` | choice | `none` | Image compression: `none`, `low`, `medium`, `high` |
| `--no-mirror` | flag | false | Disable mirror mode |
| `--no-quality` | flag | false | Hide quality indicator |
| `--no-advanced` | flag | false | Hide advanced info bar |
| `--no-skeleton` | flag | false | Hide skeleton overlays |
| `--no-face` | flag | false | Disable face matching in scoring |
| `--face-mesh` | choice | `minimal` | Face mesh detail: `minimal`, `medium`, `full` |
| `--show-face-mesh` | flag | false | Show face mesh overlay on video |

---

## 🎮 Interactive Controls

### Keyboard Shortcuts (During Video)

| Key | Action |
| :--- | :--- |
| **M** | Toggle mirror mode on/off |
| **F** | Toggle face matching on/off |
| **[** | Previous template (single mode only) |
| **]** | Next template (single mode only) |
| **R** | Return to configuration window |
| **Q / ESC** | Quit application |

### On-Screen Buttons
- **Mirror: ON/OFF** - Toggle camera mirroring
- **Face: ON/OFF** - Enable/disable facial expression scoring
- **< Prev** - Previous template (single mode)
- **Next >** - Next template (single mode)
- **Return** - Go back to config screen
- **Quit** - Exit application

---

## 📁 Template Images guide (important)

### Supported formats
- JPG / PNG

### Tips for good templates
- Use **clear, well‑lit** photos with the person visible.
- Avoid heavy motion blur.
- Try to keep the whole relevant body region in frame.
- For full‑body profiles, include ankles/knees/hips clearly.
- Tune your images to the correct profile

### Recommended folder layout
```text
templates/
  pose_01.png
  pose_02.png
  pose_03.jpg
```

You can have as many templates as you like, however larger datasets may require more computing power to run.

Use the **Image Compression** item to decrease RAM usage from the preprocessed pictures.

---

## 🔧 Creating Custom Profiles

You can create custom tracking profiles by modifying the `PROFILES` dictionary in the code.

### Profile Structure
```python
PROFILES = {
    "your_profile_name": {
        "name": "Display Name",
        "landmarks": [
            "LANDMARK_NAME_1",
            "LANDMARK_NAME_2",
            # ... more landmarks
        ],
        "pose_weight": 0.7,   # 0.0 - 1.0
        "face_weight": 0.3    # 0.0 - 1.0 (must sum to 1.0 with pose_weight)
    }
}
```

### Available Landmarks
MediaPipe provides 33 pose landmarks. Commonly used:

```python
# Face
"NOSE", "LEFT_EYE", "RIGHT_EYE", "LEFT_EAR", "RIGHT_EAR", "MOUTH_LEFT", "MOUTH_RIGHT"

# Torso
"LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP"

# Arms
"LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST"

# Hands
"LEFT_PINKY", "RIGHT_PINKY", "LEFT_INDEX", "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB"

# Legs
"LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE"

# Feet
"LEFT_HEEL", "RIGHT_HEEL", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"
```

### Example: Arms Only Profile
```python
"arms_focus": {
    "name": "Arms Focus - Upper Limb Tracking",
    "landmarks": [
        "NOSE", "LEFT_EYE", "RIGHT_EYE",  # Stability reference
        "LEFT_SHOULDER", "RIGHT_SHOULDER",
        "LEFT_ELBOW", "RIGHT_ELBOW",
        "LEFT_WRIST", "RIGHT_WRIST"
    ],
    "pose_weight": 0.9,  # Prioritize pose
    "face_weight": 0.1
}
```

### Example: Lower Body Profile
```python
"lower_body": {
    "name": "Lower Body - Legs & Core",
    "landmarks": [
        "LEFT_HIP", "RIGHT_HIP",
        "LEFT_KNEE", "RIGHT_KNEE",
        "LEFT_ANKLE", "RIGHT_ANKLE",
        "LEFT_HEEL", "RIGHT_HEEL"
    ],
    "pose_weight": 1.0,  # Pose only
    "face_weight": 0.0
}
```

### Skeleton Connections
The skeleton overlay is automatically determined by landmark count. To customize, modify `SKELETON_CONNECTIONS`:

```python
SKELETON_CONNECTIONS = {
    8: [  # Your custom 8-landmark profile
        (0, 1), (0, 2),  # Connections as (landmark_index_A, landmark_index_B)
        (3, 4),
        (3, 5), (5, 7),
        (4, 6), (6, 8)
    ]
}
```

### Weight Configuration
**Pose Weight + Face Weight must equal 1.0**
- `pose_weight: 1.0`, `face_weight: 0.0` - Pose only
- `pose_weight: 0.0`, `face_weight: 1.0` - Face only
- `pose_weight: 0.5`, `face_weight: 0.5` - Equal balance

Adjust based on your use case importance.

---

## 🎨 Advanced Configuration

### Adjustable Parameters
Key constants at the top of the script:

```python
VISIBILITY_THRESHOLD = 0.35        # Landmark confidence threshold (0-1)
MIN_TEMPLATE_LANDMARKS = 4         # Minimum landmarks to load template
MIN_MATCH_LANDMARKS = 3            # Minimum landmarks for matching
TEMPLATE_DISPLAY_WIDTH = 360       # Template panel width (pixels)
SCORE_K = 40                       # Scoring sensitivity (higher = stricter)
FPS_SMOOTH_ALPHA = 0.08           # FPS smoothing factor
HYSTERESIS_THRESHOLD = 5           # Score difference for template switching
```
---

## 📊 Performance Tips

### For Large Template Sets (50+ images)
```bash
python jm_pose_matcher.py ./large_set \
  --compression medium \
  --no-advanced \
  --face-mesh minimal
```
Use compression factors and decrease face mesh quality

### For Low-End Hardware
**Modify in code:**
```python
model_complexity=0                   # Use lite model
FPS_SMOOTH_ALPHA = 0.2              # Less FPS smoothing
SCORE_K = 30                        # More lenient matching
```

**Command line:**
```bash
--compression high --no-skeleton --no-face
```
Use a lighter model, remove enhancements such as FPS smoothing and use compression, alongside lighter meshes and K scores

### For Maximum Accuracy
```bash
python jm_pose_matcher.py ./templates \
  --profile full_body \
  --face-mesh full \
  --show-face-mesh \
  --compression none
```
Max everything, also increase values in code if you wish!

---

## 📄 License
```
Copyright 2025 Joshua Main
SPDX-License-Identifier: Apache-2.0
```
Licensed under the Apache License, Version 2.0. See LICENSE file for details.

---

## 🙏 Acknowledgments
- **MediaPipe** - Google's cross-platform ML framework for pose/face detection
- **OpenCV** - Computer vision and image processing
- **NumPy** - Numerical computing for landmark calculations

---

## ❓ Need help or have a request?
- Open an issue on GitHub with:
  - your OS (Windows/Linux/macOS)
  - Python version
  - `opencv-python` / `mediapipe` versions
  - steps to reproduce + screenshots/logs if possible
 
---

## 🫶 Support me!

If JM‑PoseMatcher helps you, inspires your work, or saves you time, consider supporting/staring the project. Your support helps me make new cool projects, keep this one up to date; improving accuracy and performance, while shiping new features.

### Ways to support
- **Star the repository** — it’s the quickest way to help the project get discovered.
- **Share it** — post a demo video, write a short thread, or show it to friends/classes/communities that would use it.
- **Contribute** — bug reports, feature requests, documentation improvements, and pull requests are welcome.
- **Sponsor / donate** — if you’d like to support development financially, you can use one of the options below:
  - Buy Me a Coffee: https://www.buymeacoffee.com/devjoshu13
- **Check out my other projects!**
  - Games and Apps: [https://mainsoftworks.com](https://sites.google.com/view/mainsoftworks)
  - Personal Portfolio: [joshmain.dev](https://joshmain.dev)
  - GitHub Profile: https://github.com/JoshuMain
