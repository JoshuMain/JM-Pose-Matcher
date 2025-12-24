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

That RMS distance is normalized by the template image diagonal and turned into a score:

- Small error ⇒ score near **100**
- Large error ⇒ score near **0**

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


## Template Images guide (important)

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


## License
Apache‑2.0 © 2025 Joshua Main

---

## Tech Used
- MediaPipe (Pose, FaceMesh)
- OpenCV
- NumPy
