#!/usr/bin/env python3
"""
Licenced:
Copyright 2025 Joshua Main
SPDX-License-Identifier: Apache-2.0

jm_pose_matcher.py
Core Version 1.0

Lightweight Python script designed towards matching body positions to user supplied reference images.
Supports pose tracking, facial expression matching, and hybrid modes.
See Github Readme for more details!

Install dependencies:
    pip install opencv-python mediapipe numpy pillow

Run CMD:
    1. Basic launch (with GUI):
        python jm_pose_matcher.py

    2. Preload templates folder:
        python jm_pose_matcher.py /path/to/templates

    3. Skip GUI and start directly:
        python jm_pose_matcher.py /path/to/templates --no-gui

CMD Arguments:
    Syntax:
        python jm_pose_matcher.py [TEMPLATES_PATH] [OPTIONS]

    Positional Arguments:
        TEMPLATES_PATH          Path to folder containing template images (JPG/PNG)

    Optional Arguments:
        --no-gui                Skip configuration window, start video graphics immediately
        --profile PROFILE       Set profile:  face_only, upper_body, default, full_body
                                (default: default)
        --camera INDEX          Camera index to use (default: 0)
        --mode MODE             Matching mode: dynamic or single (default: dynamic)
        --compression LEVEL     Image compression:  none, low, medium, high (default: none)
        --no-mirror             Disable mirror mode (default: enabled)
        --no-quality            Hide quality indicator (default: shown)
        --no-advanced           Hide advanced info bar (default: shown)
        --no-skeleton           Hide skeleton overlays (default: shown)
        --no-face               Disable face matching (default: enabled)
        --face-mesh MODE        Face mesh detail: minimal, medium, full (default: minimal)
        --show-face-mesh        Show face mesh overlay on video (default: hidden)
        --help, -h              Show this help message

Keybinds when in camera mode (simulation)
    M - Toggle mirror
    F - Toggle face matching
    [ - Previous template (single mode)
    ] - Next template (single mode)
    R - Return to config
    Q or ESC - Quit

Template Supported Image Types
    JPG, PNG

Author and Source:
    Joshua Main(2025)
    https://github.com/JoshuMain/JM-Pose-Matcher
    Last Mod - 23/12/25
    
"""

import sys
import os
import glob
import time
import threading
import json
import argparse
from math import exp
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import platform

import cv2
import numpy as np
import mediapipe as mp

# --------------------------- Config ---------------------------
VISIBILITY_THRESHOLD = 0.35  # Landmark confidence threshold (0-1)
MIN_TEMPLATE_LANDMARKS = 4 # Minimum landmarks to load template
MIN_MATCH_LANDMARKS = 3 # Minimum landmarks for matching
TEMPLATE_DISPLAY_WIDTH = 360 # Template panel width (pixels)
SCORE_K = 40  # Scoring sensitivity (higher = stricter)
FPS_SMOOTH_ALPHA = 0.08 # FPS smoothing factor
HYSTERESIS_THRESHOLD = 5  # Score difference for template switching
INSUFFICIENT_FRAMES_THRESHOLD = 40 # Frames of low data before warning

# Profile definitions
PROFILES = {
    "default": {
        "name": "Default",
        "landmarks": ["NOSE", "LEFT_EYE", "RIGHT_EYE", "LEFT_SHOULDER", "RIGHT_SHOULDER",
                     "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST",
                     "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE",
                     "LEFT_ANKLE", "RIGHT_ANKLE"],
        "pose_weight":  0.7,
        "face_weight":  0.3
    },
    "upper_body": {
        "name": "Upper Body - Sat Down",
        "landmarks": ["NOSE", "LEFT_EYE", "RIGHT_EYE", "LEFT_SHOULDER", "RIGHT_SHOULDER",
                     "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST",
                     "LEFT_HIP", "RIGHT_HIP"],
        "pose_weight": 0.6,
        "face_weight":  0.4
    },
    "full_body": {
        "name": "Full Body - Stood Up",
        "landmarks": ["NOSE", "LEFT_EYE", "RIGHT_EYE", "LEFT_SHOULDER", "RIGHT_SHOULDER",
                     "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST",
                     "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE",
                     "LEFT_ANKLE", "RIGHT_ANKLE"],
        "pose_weight": 0.75,
        "face_weight":  0.25
    },
    "face_only": { 
        "name": "Face Only - Expression Tracking",
        "landmarks": ["NOSE", "LEFT_EYE", "RIGHT_EYE","LEFT_SHOULDER", "RIGHT_SHOULDER"],
        "pose_weight":  0.0, 
        "face_weight": 1.0 # 100% facial tracking for match rates
    }
}

# Skeleton connections based on landmark count
SKELETON_CONNECTIONS = {
    3: [  # Face only mode (minimal skeleton)
        (0, 1), (0, 2), (3, 4),# nose to eyes only
    ],
    11: [  # Sat down mode (torso + arms)
        (0, 1), (0, 2),  # nose to eyes
        (3, 4),  # shoulders
        (3, 5), (5, 7),  # left arm
        (4, 6), (6, 8),  # right arm
        (9, 10),  # hips
        (3, 9), (4, 10),  # shoulders to hips
    ],
    15: [  # Full body
        (0, 1), (0, 2),  # nose to eyes
        (3, 4),  # shoulders
        (3, 5), (5, 7),  # left arm
        (4, 6), (6, 8),  # right arm
        (9, 10),  # hips
        (3, 9), (4, 10),  # shoulders to hips
        (9, 11), (11, 13),  # left leg
        (10, 12), (12, 14)  # right leg
    ]
}

# MediaPipe setup
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh

# Face mesh indices
FM_MINIMAL = {
    "left_eye_top": 386, "left_eye_bottom": 374,
    "right_eye_top": 159, "right_eye_bottom":  145,
    "mouth_left": 61, "mouth_right": 291,
    "mouth_top": 13, "mouth_bottom": 14,
    "left_brow":  223, "right_brow": 443, "nose_tip": 1
}

FM_MEDIUM = list(range(0, 468, 7))

CONFIG_PATH = Path. home() / ".jm_pose_matcher_config.json"

# --------------------------- Camera Detection ---------------------------
def detect_available_cameras(max_test=10):
    """Detect available cameras and return list of (index, name) tuples."""
    available = []
    
    for i in range(max_test):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Try to get camera name
            backend = cap.getBackendName()
            name = f"Camera {i}"
            
            # Platform-specific name detection
            if platform.system() == "Windows":
                # Windows: Try to get friendly name via DirectShow
                name = f"Camera {i} ({backend})"
            elif platform. system() == "Linux":
                # Linux: Check /sys/class/video4linux/
                try:
                    device_path = f"/sys/class/video4linux/video{i}/name"
                    if os.path.exists(device_path):
                        with open(device_path, 'r') as f:
                            device_name = f.read().strip()
                            name = f"{device_name} (Camera {i})"
                except: 
                    pass
            elif platform.system() == "Darwin":
                # macOS: Use AVFoundation info
                name = f"Camera {i} ({backend})"
            
            available.append((i, name))
            cap.release()
        else:
            # If camera doesn't open, stop searching
            if i > 0 and len(available) == 0:
                continue
            elif len(available) > 0:
                break
    
    if not available:
        available. append((0, "Default Camera (0)"))
    
    return available

# --------------------------- Utilities ---------------------------
def get_landmarks_for_profile(profile_name):
    """Convert profile landmark names to MediaPipe enum list."""
    landmark_names = PROFILES[profile_name]["landmarks"]
    return [getattr(mp_pose.PoseLandmark, name) for name in landmark_names]

def load_config():
    """Load saved configuration."""
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
    return {}

def save_config(config):
    """Save configuration."""
    try:
        with open(CONFIG_PATH, 'w') as f:
            json. dump(config, f, indent=2)
    except Exception as e:
        print(f"Error saving config: {e}")

def compress_image(img, level="none"):
    """Compress image for memory optimization."""
    if level == "none":
        return img
    
    scale_map = {"low": 0.9, "medium": 0.7, "high": 0.5}
    scale = scale_map. get(level, 1.0)
    
    h, w = img.shape[:2]
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

# --------------------------- Pose Utilities ---------------------------
def detect_pose_landmarks_bgr(image_bgr, pose, use_landmarks):
    """Return numpy array of specified landmarks or None."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if not results.pose_landmarks:
        return None
    
    h, w = image_bgr. shape[:2]
    lm = results.pose_landmarks.landmark
    pts = []
    for idx in use_landmarks:
        p = lm[idx]
        pts.append((p.x * w, p.y * h, p.visibility if hasattr(p, "visibility") else 1.0))
    return np.array(pts, dtype=np.float32)

def filter_valid_points(pts):
    """Return indices, coords, vis for landmarks with visibility > threshold."""
    if pts is None:
        return [], np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)
    vis = pts[:, 2]
    valid_idx = [i for i, v in enumerate(vis) if v > VISIBILITY_THRESHOLD]
    if not valid_idx:
        return [], np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)
    coords = pts[valid_idx, :2]
    vis_vals = vis[valid_idx]
    return valid_idx, coords, vis_vals

def umeyama_similarity(src, dst, with_scaling=True):
    """Estimate similarity transform mapping src -> dst."""
    src = np.asarray(src, dtype=np. float64)
    dst = np.asarray(dst, dtype=np. float64)
    n = src.shape[0]
    if n < 2:
        raise ValueError("At least two points required")
    
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    cov = (dst_c.T @ src_c) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[1, 1] = -1
    R = U @ S @ Vt
    
    if with_scaling:
        var_src = (src_c ** 2).sum() / n
        s = 1.0 / var_src * np.sum(D * np.diag(S))
    else:
        s = 1.0
    
    t = mu_dst - s * R @ mu_src
    return float(s), R, t

def transform_points(pts, s, R, t):
    pts = np.asarray(pts, dtype=np. float64)
    return (s * (pts @ R.T)) + t

def match_score(src_trans, dst_pts, diag):
    """RMS distance normalized by diag mapped to 0-100 via exponential."""
    if src_trans.shape[0] == 0:
        return 0.0
    d = np.linalg.norm(src_trans - dst_pts, axis=1)
    rms = np.sqrt((d ** 2).mean())
    rel = rms / (diag + 1e-9)
    score = 100.0 * exp(-SCORE_K * rel)
    return float(np.clip(score, 0.0, 100.0))

def draw_landmark_skeleton_cv(img, pts, landmark_count, color=(0, 255, 0), radius=4):
    """Draw skeleton with proper connections based on landmark count."""
    if pts is None or len(pts) == 0:
        return img
    
    pts_i = np.array(pts).astype(int)
    
    # Draw points
    for (x, y) in pts_i:
        cv2.circle(img, (int(x), int(y)), radius, color, -1)
    
    # Get appropriate connections for landmark count
    connections = SKELETON_CONNECTIONS.get(landmark_count, SKELETON_CONNECTIONS[15])
    
    # Draw connections
    for a, b in connections:
        if a < len(pts_i) and b < len(pts_i):
            cv2.line(img, tuple(pts_i[a]), tuple(pts_i[b]), color, 2)
    
    return img

# --------------------------- Face Utilities ---------------------------
def detect_face_mesh_bgr(image_bgr, face_mesh):
    """Return array of face mesh landmarks or None."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    if not results.multi_face_landmarks:
        return None
    
    lm = results.multi_face_landmarks[0]
    h, w = image_bgr. shape[:2]
    pts = []
    for p in lm.landmark:
        pts.append((p.x * w, p.y * h, p.z))
    return np.array(pts, dtype=np.float32)

def compute_face_expression_features(landmarks):
    """Compute normalized expression features."""
    if landmarks is None: 
        return None
    
    def dist(a, b):
        return np. linalg.norm(landmarks[a][:2] - landmarks[b][:2])
    
    try:
        iod = dist(FM_MINIMAL["left_eye_top"], FM_MINIMAL["right_eye_top"]) + 1e-6
        
        mouth_open = dist(FM_MINIMAL["mouth_top"], FM_MINIMAL["mouth_bottom"]) / iod
        mouth_width = dist(FM_MINIMAL["mouth_left"], FM_MINIMAL["mouth_right"]) / iod
        mouth_open_ratio = mouth_open / (mouth_width + 1e-6)
        
        left_eye_open = dist(FM_MINIMAL["left_eye_top"], FM_MINIMAL["left_eye_bottom"]) / iod
        right_eye_open = dist(FM_MINIMAL["right_eye_top"], FM_MINIMAL["right_eye_bottom"]) / iod
        eye_open_mean = 0.5 * (left_eye_open + right_eye_open)
        
        left_brow_y = landmarks[FM_MINIMAL["left_brow"], 1]
        right_brow_y = landmarks[FM_MINIMAL["right_brow"], 1]
        left_eye_y = landmarks[FM_MINIMAL["left_eye_top"], 1]
        right_eye_y = landmarks[FM_MINIMAL["right_eye_top"], 1]
        brow_raise = ((left_eye_y - left_brow_y) + (right_eye_y - right_brow_y)) / (2 * iod)
        
        feats = np.array([
            mouth_open_ratio,
            mouth_width,
            eye_open_mean,
            brow_raise,
            mouth_open
        ], dtype=np.float32)
        return feats
    except Exception: 
        return None

def expression_similarity_score(feat_live, feat_tpl, k=20):
    """Convert L2 distance between features to 0-100 score."""
    if feat_live is None or feat_tpl is None:
        return None
    d = np.linalg.norm(feat_live - feat_tpl)
    rel = d / (np.linalg.norm(feat_tpl) + 1e-6)
    score = 100.0 * np.exp(-k * rel)
    return float(np.clip(score, 0.0, 100.0))

def draw_face_mesh_overlay(img, face_landmarks, mode="minimal", color=(0, 255, 255)):
    """Draw face mesh overlay."""
    if face_landmarks is None:
        return img
    
    if mode == "minimal":
        for name, idx in FM_MINIMAL.items():
            x, y = int(face_landmarks[idx, 0]), int(face_landmarks[idx, 1])
            cv2.circle(img, (x, y), 2, color, -1)
    elif mode == "medium":
        for idx in FM_MEDIUM:
            if idx < len(face_landmarks):
                x, y = int(face_landmarks[idx, 0]), int(face_landmarks[idx, 1])
                cv2.circle(img, (x, y), 1, color, -1)
    elif mode == "full":
        for pt in face_landmarks:
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(img, (x, y), 1, color, -1)
    
    return img

# --------------------------- Template Loading ---------------------------
def load_templates_from_folder(folder, pose_detector, face_mesh_detector, 
                               use_landmarks, compression="none", 
                               progress_callback=None):
    """Load all templates with progress updates."""
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    paths = []
    for e in exts:
        paths. extend(glob.glob(os.path.join(folder, e)))
    paths = sorted(paths)
    
    templates = []
    total = len(paths)
    
    for i, p in enumerate(paths):
        if progress_callback:
            if progress_callback(i, total, os.path.basename(p)):
                # Cancelled
                break
        
        img = cv2.imread(p)
        if img is None: 
            print(f"Warning: could not read {p}")
            continue
        
        img = compress_image(img, compression)
        
        lms_pose = detect_pose_landmarks_bgr(img, pose_detector, use_landmarks)
        if lms_pose is None: 
            print(f"Skipping {os.path.basename(p)}: no pose detected")
            continue
        
        idxs, coords, vis = filter_valid_points(lms_pose)
        if len(idxs) < MIN_TEMPLATE_LANDMARKS:
            print(f"Skipping {os.path.basename(p)}: only {len(idxs)} landmarks")
            continue
        
        try:
            lms_face = detect_face_mesh_bgr(img, face_mesh_detector)
            face_feats = compute_face_expression_features(lms_face)
        except Exception: 
            lms_face = None
            face_feats = None
        
        h, w = img.shape[:2]
        diag = float(np.linalg.norm([w, h]))
        
        tpl = {
            "path": p,
            "name": os.path.basename(p),
            "image": img,
            "landmarks": lms_pose,
            "valid_idx": idxs,
            "valid_coords": coords,
            "diag": diag,
            "face_landmarks": lms_face,
            "face_feats": face_feats,
            "landmark_count": len(use_landmarks)  # Store landmark count for validation
        }
        templates.append(tpl)
        print(f"Loaded '{tpl['name']}' ({w}x{h}), "
              f"pose: {len(idxs)}, face: {'yes' if face_feats is not None else 'no'}")
    
    if progress_callback:
        progress_callback(total, total, "Complete")
    
    return templates

# --------------------------- Worker Thread ---------------------------
class PoseMatcherWorker(threading.Thread):
    """Background worker for pose/face detection and matching."""
    
    def __init__(self, templates, matching_mode, selected_template_idx,
                 cam_index, use_landmarks, pose_weight, face_weight,
                 face_match_enabled, mirror_mode):
        super().__init__(daemon=True)
        self.templates = templates
        self.matching_mode = matching_mode
        self.selected_template_idx = selected_template_idx
        self.cam_index = cam_index
        self.use_landmarks = use_landmarks
        self.landmark_count = len(use_landmarks)
        self.pose_weight = pose_weight
        self. face_weight = face_weight
        self.face_match_enabled = face_match_enabled
        self.mirror_mode = mirror_mode
        
        self._stop = threading.Event()
        self.lock = threading.Lock()
        
        # Shared outputs
        self.latest_frame_bgr = None
        self.latest_template = None
        self.latest_template_idx = 0
        self.latest_pose_score = 0.0
        self.latest_face_score = None
        self.latest_combined_score = 0.0
        self.latest_matched_pts = 0
        self.latest_live_lms = None
        self.latest_live_face_lms = None
        self.latest_quality_state = "amber"
        self.fps = 0.0
        
        # Popup management
        self._popup_shown = False
        self._last_quality = "amber"
        
        # Hysteresis
        self._current_best_idx = None
        
        # Detectors
        self. pose_live = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.face_mesh_live = mp_face. FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5
        )
    
    def stop(self):
        self._stop.set()
    
    def stopped(self):
        return self._stop.is_set()
    
    def update_settings(self, **kwargs):
        """Update settings mid-run."""
        with self.lock:
            if 'matching_mode' in kwargs:
                self.matching_mode = kwargs['matching_mode']
            if 'selected_template_idx' in kwargs:
                self.selected_template_idx = kwargs['selected_template_idx']
            if 'mirror_mode' in kwargs:
                self.mirror_mode = kwargs['mirror_mode']
            if 'face_match_enabled' in kwargs: 
                self.face_match_enabled = kwargs['face_match_enabled']
    
    def compute_combined_score(self, tpl, live_pose, live_face_feats):
        """Compute combined pose + face score for a template."""
        # Validate landmark count matches
        if tpl. get("landmark_count", self.landmark_count) != self.landmark_count:
            print(f"Warning: Template {tpl['name']} has mismatched landmark count")
            return 0.0, None, None, 0
        
        pose_score = None
        matched_count = 0
        
        if live_pose is not None: 
            src_list = []
            dst_list = []
            for i in range(len(self.use_landmarks)):
                if i < len(tpl["landmarks"]) and i < len(live_pose):
                    if (tpl["landmarks"][i, 2] > VISIBILITY_THRESHOLD and 
                        live_pose[i, 2] > VISIBILITY_THRESHOLD):
                        dst_list.append(tpl["landmarks"][i, :2])
                        src_list.append(live_pose[i, :2])
            
            matched_count = len(src_list)
            if len(src_list) >= MIN_MATCH_LANDMARKS:
                try:
                    src_arr = np.array(src_list, dtype=np.float64)
                    dst_arr = np.array(dst_list, dtype=np.float64)
                    s, R, tvec = umeyama_similarity(src_arr, dst_arr, with_scaling=True)
                    src_trans = transform_points(src_arr, s, R, tvec)
                    pose_score = match_score(src_trans, dst_arr, tpl["diag"])
                except Exception as e:
                    print(f"Pose matching error: {e}")
                    pose_score = None
        
        face_score = None
        if self.face_match_enabled and tpl.get("face_feats") is not None and live_face_feats is not None:
            try:
                face_score = expression_similarity_score(live_face_feats, tpl["face_feats"], k=20.0)
            except Exception:
                face_score = None
        
        if pose_score is not None and face_score is not None:
            combined = self.pose_weight * pose_score + self.face_weight * face_score
        elif pose_score is not None: 
            combined = pose_score
        elif face_score is not None:
            combined = face_score
        else:
            combined = 0.0
        
        return combined, pose_score, face_score, matched_count
    
    def run(self):
        cap = cv2.VideoCapture(self.cam_index)
        if not cap.isOpened():
            print("PoseMatcherWorker: cannot open camera")
            return
        
        prev_time = time.time()
        fps_smooth = 0.0
        
        while not self. stopped():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            
            # Apply mirror mode
            if self.mirror_mode: 
                frame = cv2.flip(frame, 1)
            
            # FPS
            now = time.time()
            dt = now - prev_time if now > prev_time else 1e-6
            prev_time = now
            fps_inst = 1.0 / dt if dt > 0 else 0.0
            fps_smooth = (1.0 - FPS_SMOOTH_ALPHA) * fps_smooth + FPS_SMOOTH_ALPHA * fps_inst if fps_smooth > 0 else fps_inst
            
            # Detect
            live_pose = detect_pose_landmarks_bgr(frame, self.pose_live, self.use_landmarks)
            live_face_mesh = detect_face_mesh_bgr(frame, self.face_mesh_live)
            live_face_feats = compute_face_expression_features(live_face_mesh)
            
            # Quality state
            if live_pose is not None:
                visible_count = len([lm for lm in live_pose if lm[2] > VISIBILITY_THRESHOLD])
                
                # Face-only mode: quality based on face detection
                if self.landmark_count == 3:  # Face-only profile
                    if live_face_feats is not None:
                        quality_state = "green"  # Face detected
                    else:
                        quality_state = "red"  # No face detected
                else:  # Normal pose-based quality
                    if visible_count >= MIN_MATCH_LANDMARKS: 
                        quality_state = "green"
                    elif visible_count >= MIN_TEMPLATE_LANDMARKS // 2:
                        quality_state = "amber"
                    else: 
                        quality_state = "red"
            else: 
                quality_state = "red"
            
            # Popup management
            show_popup = False
            if quality_state == "red" and not self._popup_shown:
                self._popup_shown = True
                show_popup = True
            elif quality_state == "amber" and self._last_quality == "red":
                self._popup_shown = False
            self._last_quality = quality_state
            
            # Matching
            best_tpl = None
            best_tpl_idx = 0
            best_combined = -1.0
            best_pose_score = None
            best_face_score = None
            best_matched = 0
            
            if self.matching_mode == "dynamic":
                for idx, tpl in enumerate(self. templates):
                    combined, pose_sc, face_sc, matched = self.compute_combined_score(
                        tpl, live_pose, live_face_feats)
                    
                    threshold = best_combined
                    if self._current_best_idx is not None:
                        if idx != self._current_best_idx:
                            threshold = best_combined + HYSTERESIS_THRESHOLD
                    
                    if combined > threshold: 
                        best_combined = combined
                        best_tpl = tpl
                        best_tpl_idx = idx
                        best_pose_score = pose_sc
                        best_face_score = face_sc
                        best_matched = matched
                
                if best_tpl: 
                    self._current_best_idx = best_tpl_idx
            
            else:  # single mode
                if 0 <= self.selected_template_idx < len(self.templates):
                    best_tpl = self. templates[self.selected_template_idx]
                    best_tpl_idx = self.selected_template_idx
                    best_combined, best_pose_score, best_face_score, best_matched = \
                        self.compute_combined_score(best_tpl, live_pose, live_face_feats)
            
            # Store
            with self.lock:
                self.latest_frame_bgr = frame. copy()
                self.latest_template = best_tpl
                self.latest_template_idx = best_tpl_idx
                self.latest_pose_score = float(best_pose_score) if best_pose_score is not None else 0.0
                self.latest_face_score = float(best_face_score) if best_face_score is not None else None
                self.latest_combined_score = float(best_combined) if best_combined >= 0 else 0.0
                self.latest_matched_pts = int(best_matched)
                self.latest_live_lms = live_pose. copy() if live_pose is not None else None
                self.latest_live_face_lms = live_face_mesh.copy() if live_face_mesh is not None else None
                self.latest_quality_state = quality_state
                self.fps = float(fps_smooth)
        
        cap.release()
        self.pose_live.close()
        self.face_mesh_live.close()

# --------------------------- Progress Dialog ---------------------------
class ProgressDialog: 
    """Modal progress dialog."""
    
    def __init__(self, parent, title="Loading Templates"):
        self.top = tk.Toplevel(parent)
        self.top.title(title)
        self.top.transient(parent)
        self.top.grab_set()
        
        parent.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() // 2) - 150
        y = parent.winfo_y() + (parent.winfo_height() // 2) - 50
        self.top.geometry(f"300x100+{x}+{y}")
        
        self.label = ttk.Label(self.top, text="Initializing...")
        self.label.pack(pady=10)
        
        self.progress = ttk. Progressbar(self.top, length=250, mode='determinate')
        self.progress.pack(pady=10)
        
        self. cancel_btn = ttk.Button(self. top, text="Cancel", command=self.cancel)
        self.cancel_btn.pack(pady=5)
        
        self.cancelled = False
    
    def update(self, current, total, message):
        self.label.config(text=f"{message} ({current}/{total})")
        self.progress['value'] = (current / total * 100) if total > 0 else 0
        self.top.update()
        return self.cancelled
    
    def cancel(self):
        self.cancelled = True
        self.top.destroy()
    
    def close(self):
        try:
            self.top.destroy()
        except:
            pass

# --------------------------- Config Window ---------------------------
class ConfigWindow: 
    """Enhanced configuration window."""
    
    def __init__(self, root, initial_path=None):
        self.root = root
        self.root. title("JM-PoseMatcher - Configuration")
        self.initial_path = initial_path
        
        self.config = load_config()
        
        # Detect cameras
        self.available_cameras = detect_available_cameras()
        
        # Variables
        self.var_path = tk.StringVar(value=self.config.get("last_folder", initial_path or ""))
        self.var_matching_mode = tk.StringVar(value=self.config.get("matching_mode", "dynamic"))
        self.var_selected_template_idx = tk.IntVar(value=0)
        self.var_profile = tk.StringVar(value=self.config.get("profile", "default"))
        self.var_camera_idx = tk.IntVar(value=self.config.get("camera_index", 0))
        self.var_compression = tk.StringVar(value=self.config.get("compression", "none"))
        
        self.var_mirror = tk.BooleanVar(value=self.config.get("mirror_mode", True))
        self.var_quality_indicator = tk.BooleanVar(value=self.config.get("show_quality_indicator", True))
        self.var_advanced_info = tk.BooleanVar(value=self.config.get("show_advanced_info", True))
        self.var_skel_video = tk.BooleanVar(value=self.config.get("skeleton_video", True))
        self.var_skel_photo = tk.BooleanVar(value=self.config.get("skeleton_photo", True))
        
        self.var_face_match_enabled = tk.BooleanVar(value=self.config.get("face_match_enabled", True))
        self.var_face_mesh_video = tk.BooleanVar(value=self.config.get("face_mesh_video", False))
        self.var_face_mesh_photo = tk.BooleanVar(value=self. config.get("face_mesh_photo", False))
        self.var_face_mesh_mode = tk.StringVar(value=self.config. get("face_mesh_mode", "minimal"))
        
        self.var_templates_count = tk.IntVar(value=0)
        
        self.templates = []
        self.build_ui()
    
    def build_ui(self):
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        row = 0
        
        # Path
        ttk.Label(main_frame, text="Image set path:").grid(row=row, column=0, sticky=tk.W)
        self.entry_path = ttk.Entry(main_frame, textvariable=self.var_path, width=50)
        self.entry_path.grid(row=row, column=1, columnspan=2, sticky=(tk.W, tk.E), padx=(5, 5))
        ttk.Button(main_frame, text="Find", command=self.on_find_folder).grid(row=row, column=3, sticky=tk.W)
        ttk.Button(main_frame, text="Reload", command=self.on_reload).grid(row=row, column=4, sticky=tk.W, padx=(5, 0))
        row += 1
        
        ttk.Label(main_frame, text="Supported:  JPG, PNG", foreground="gray").grid(
            row=row, column=1, columnspan=3, sticky=tk.W, pady=(2, 10))
        row += 1
        
        # Matching mode
        mode_frame = ttk.LabelFrame(main_frame, text="Matching Mode", padding=8)
        mode_frame.grid(row=row, column=0, columnspan=5, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Radiobutton(
            mode_frame,
            text="Dynamic - Auto-switch to best match",
            variable=self.var_matching_mode,
            value="dynamic",
            command=self.on_mode_change
        ).grid(row=0, column=0, sticky=tk.W, pady=2)
        
        single_frame = ttk.Frame(mode_frame)
        single_frame. grid(row=1, column=0, sticky=tk.W, padx=(20, 0))
        
        ttk.Radiobutton(
            single_frame,
            text="Single - Lock to one template (use Prev/Next in video)",
            variable=self.var_matching_mode,
            value="single"
        ).pack(side=tk.LEFT)
        
        row += 1
        
        # Settings row
        settings_frame = ttk.Frame(main_frame)
        settings_frame.grid(row=row, column=0, columnspan=5, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(settings_frame, text="Profile:").grid(row=0, column=0, sticky=tk. W, padx=(0, 5))
        self.profile_combo = ttk.Combobox(
            settings_frame,
            textvariable=self.var_profile,
            values=[PROFILES[k]["name"] for k in ["default", "upper_body", "full_body", "face_only"]],
            state="readonly",
            width=20
        )
        self.profile_combo.grid(row=0, column=1, sticky=tk.W, padx=(0, 15))
        self.profile_combo.bind("<<ComboboxSelected>>", self. on_profile_change)
        
        self.profile_name_to_key = {v["name"]: k for k, v in PROFILES.items()}
        
        ttk.Label(settings_frame, text="Camera: ").grid(row=0, column=2, sticky=tk.W, padx=(0, 5))
        camera_names = [name for idx, name in self.available_cameras]
        self.camera_combo = ttk.Combobox(
            settings_frame,
            values=camera_names,
            state="readonly",
            width=25
        )
        if camera_names:
            self.camera_combo.current(0)
        self.camera_combo.grid(row=0, column=3, sticky=tk.W, padx=(0, 15))
        
        ttk.Label(settings_frame, text="Compression:").grid(row=0, column=4, sticky=tk.W, padx=(0, 5))
        ttk.Combobox(
            settings_frame,
            textvariable=self.var_compression,
            values=["none", "low", "medium", "high"],
            state="readonly",
            width=10
        ).grid(row=0, column=5, sticky=tk.W)
        
        row += 1
        
        # Display options
        display_frame = ttk.LabelFrame(main_frame, text="Display Options", padding=8)
        display_frame.grid(row=row, column=0, columnspan=5, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Checkbutton(display_frame, text="Mirror camera", 
                       variable=self.var_mirror).grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Checkbutton(display_frame, text="Show data quality indicator", 
                       variable=self.var_quality_indicator).grid(row=0, column=1, sticky=tk.W, pady=2, padx=(15, 0))
        ttk.Checkbutton(display_frame, text="Advanced info", 
                       variable=self.var_advanced_info).grid(row=0, column=2, sticky=tk.W, pady=2, padx=(15, 0))
        
        row += 1
        
        # Skeleton
        skeleton_frame = ttk. LabelFrame(main_frame, text="Skeleton Overlays", padding=8)
        skeleton_frame.grid(row=row, column=0, columnspan=5, sticky=(tk.W, tk.E), pady=5)
        
        ttk. Checkbutton(skeleton_frame, text="Show on webcam", 
                       variable=self.var_skel_video).grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Checkbutton(skeleton_frame, text="Show on templates", 
                       variable=self.var_skel_photo).grid(row=0, column=1, sticky=tk.W, pady=2, padx=(15, 0))
        
        row += 1
        
        # Face mesh
        face_frame = ttk.LabelFrame(main_frame, text="Face Mesh Options", padding=8)
        face_frame.grid(row=row, column=0, columnspan=5, sticky=(tk.W, tk.E), pady=5)
        
        ttk. Checkbutton(face_frame, text="Enable face scoring", 
                       variable=self.var_face_match_enabled).grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Checkbutton(face_frame, text="Show on webcam", 
                       variable=self.var_face_mesh_video).grid(row=0, column=1, sticky=tk.W, pady=2, padx=(15, 0))
        ttk.Checkbutton(face_frame, text="Show on templates", 
                       variable=self.var_face_mesh_photo).grid(row=0, column=2, sticky=tk.W, pady=2, padx=(15, 0))
        
        ttk.Label(face_frame, text="Detail: ").grid(row=1, column=0, sticky=tk.W, pady=(5, 2))
        ttk. Combobox(
            face_frame,
            textvariable=self.var_face_mesh_mode,
            values=["minimal", "medium", "full"],
            state="readonly",
            width=10
        ).grid(row=1, column=1, sticky=tk. W, pady=(5, 2), padx=(15, 0))
        
        row += 1
        
        # Templates count
        ttk.Label(main_frame, text="Loaded templates:").grid(row=row, column=0, sticky=tk.W, pady=(10, 0))
        ttk.Label(main_frame, textvariable=self.var_templates_count).grid(row=row, column=1, sticky=tk.W, pady=(10, 0))
        
        row += 1
        
        # Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=row, column=0, columnspan=5, pady=(15, 0))
        ttk.Button(btn_frame, text="Start", command=self.on_start).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(btn_frame, text="Quit", command=self.on_quit).pack(side=tk.LEFT)
        
        self.entry_path.bind("<Return>", lambda e: self.on_reload())
        self.on_mode_change()
    
    def on_mode_change(self):
        pass  # No dropdown needed, using Prev/Next buttons in video
    
    def on_profile_change(self, event=None):
        """Warn user that templates need reloading."""
        if self.templates:
            messagebox.showinfo("Profile Changed", 
                              "Profile changed. Please click 'Reload' to update templates with new landmark configuration.")
    
    def on_find_folder(self):
        folder = filedialog.askdirectory(title="Select templates folder")
        if folder:
            self.var_path.set(folder)
            self.on_reload()
    
    def on_reload(self):
        path = self.var_path.get().strip()
        if not path: 
            messagebox.showinfo("Select folder", "Please select a folder.")
            return
        if not os.path.isdir(path):
            messagebox.showerror("Invalid path", f"Path not found: {path}")
            return
        
        profile_display_name = self.var_profile.get()
        profile_key = self.profile_name_to_key.get(profile_display_name, "default")
        use_landmarks = get_landmarks_for_profile(profile_key)
        
        progress = ProgressDialog(self.root, "Loading Templates")
        
        def progress_callback(current, total, message):
            return progress.update(current, total, message)
        
        def load_thread():
            try:
                pose_static = mp_pose.Pose(
                    static_image_mode=True,
                    model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=0.5
                )
                face_static = mp_face. FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=False,
                    min_detection_confidence=0.5
                )
                
                templates = load_templates_from_folder(
                    path, pose_static, face_static, use_landmarks,
                    self.var_compression.get(), progress_callback
                )
                
                pose_static.close()
                face_static.close()
                
                self.templates = templates
                self.var_templates_count. set(len(templates))
                
                progress.close()
                
                if not templates:
                    messagebox. showwarning("No templates", "No usable templates found.")
            
            except Exception as e:
                progress.close()
                if "Cancelled" not in str(e):
                    messagebox.showerror("Error", f"Failed:  {e}")
        
        thread = threading.Thread(target=load_thread, daemon=True)
        thread.start()
    
    def on_start(self):
        if not self.templates:
            self.on_reload()
            if not self.templates:
                return
        
        # Get selected camera index
        camera_selection = self.camera_combo.current()
        if camera_selection >= 0:
            camera_idx = self.available_cameras[camera_selection][0]
        else:
            camera_idx = 0
        
        profile_display_name = self.var_profile.get()
        profile_key = self.profile_name_to_key.get(profile_display_name, "default")
        
        config = {
            "last_folder": self.var_path.get(),
            "matching_mode":  self.var_matching_mode. get(),
            "profile": profile_key,
            "camera_index": camera_idx,
            "compression": self.var_compression.get(),
            "mirror_mode": self.var_mirror.get(),
            "show_quality_indicator": self.var_quality_indicator.get(),
            "show_advanced_info":  self.var_advanced_info. get(),
            "skeleton_video": self.var_skel_video. get(),
            "skeleton_photo": self.var_skel_photo.get(),
            "face_match_enabled": self.var_face_match_enabled.get(),
            "face_mesh_video": self.var_face_mesh_video.get(),
            "face_mesh_photo": self.var_face_mesh_photo.get(),
            "face_mesh_mode": self.var_face_mesh_mode.get()
        }
        save_config(config)
        
        self.root.withdraw()
        
        run_video_loop(
            root=self.root,
            templates=self.templates,
            matching_mode=self.var_matching_mode.get(),
            selected_template_idx=self.var_selected_template_idx.get(),
            profile_key=profile_key,
            camera_index=camera_idx,
            mirror_mode=self.var_mirror.get(),
            show_quality_indicator=self.var_quality_indicator.get(),
            show_advanced_info=self.var_advanced_info.get(),
            show_skel_video=self.var_skel_video.get(),
            show_skel_photo=self.var_skel_photo.get(),
            face_match_enabled=self.var_face_match_enabled.get(),
            show_face_mesh_video=self.var_face_mesh_video.get(),
            show_face_mesh_photo=self.var_face_mesh_photo.get(),
            face_mesh_mode=self.var_face_mesh_mode.get()
        )
    
    def on_quit(self):
        self.root.quit()
        self.root.destroy()

# --------------------------- Video Loop ---------------------------
def run_video_loop(root, templates, matching_mode, selected_template_idx,
                  profile_key, camera_index, mirror_mode, show_quality_indicator,
                  show_advanced_info, show_skel_video, show_skel_photo,
                  face_match_enabled, show_face_mesh_video, show_face_mesh_photo,
                  face_mesh_mode):
    """Run main video matching loop."""
    
    profile = PROFILES[profile_key]
    use_landmarks = get_landmarks_for_profile(profile_key)
    landmark_count = len(use_landmarks)
    pose_weight = profile["pose_weight"]
    face_weight = profile["face_weight"]
    
    # Start worker
    worker = PoseMatcherWorker(
        templates=templates,
        matching_mode=matching_mode,
        selected_template_idx=selected_template_idx,
        cam_index=camera_index,
        use_landmarks=use_landmarks,
        pose_weight=pose_weight,
        face_weight=face_weight,
        face_match_enabled=face_match_enabled,
        mirror_mode=mirror_mode
    )
    worker.start()
    
    window_name = "JM-PoseMatcher - Webcam (left) | Template (right)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1200, 650)
    
    # Button tracking
    btn_return_rect = None
    btn_quit_rect = None
    btn_prev_rect = None
    btn_next_rect = None
    btn_mirror_rect = None
    btn_face_rect = None
    
    requested_return = False
    requested_quit = False
    popup_scheduled = False
    
    # Runtime state
    runtime_mirror = mirror_mode
    runtime_show_quality = show_quality_indicator
    runtime_show_advanced = show_advanced_info
    runtime_show_skel_video = show_skel_video
    runtime_show_skel_photo = show_skel_photo
    runtime_face_enabled = face_match_enabled
    runtime_show_face_video = show_face_mesh_video
    runtime_show_face_photo = show_face_mesh_photo
    runtime_face_mode = face_mesh_mode
    runtime_matching_mode = matching_mode
    runtime_selected_idx = selected_template_idx
    
    def on_mouse(event, x, y, flags, param):
        nonlocal requested_return, requested_quit, runtime_selected_idx, runtime_mirror, runtime_face_enabled
        
        if event == cv2.EVENT_LBUTTONUP:
            if btn_return_rect: 
                x1, y1, x2, y2 = btn_return_rect
                if x1 <= x <= x2 and y1 <= y <= y2:
                    requested_return = True
                    return
            
            if btn_quit_rect: 
                x1, y1, x2, y2 = btn_quit_rect
                if x1 <= x <= x2 and y1 <= y <= y2:
                    requested_quit = True
                    return
            
            # Single mode navigation
            if runtime_matching_mode == "single": 
                if btn_prev_rect:
                    x1, y1, x2, y2 = btn_prev_rect
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        runtime_selected_idx = (runtime_selected_idx - 1) % len(templates)
                        worker.update_settings(selected_template_idx=runtime_selected_idx)
                        return
                
                if btn_next_rect:
                    x1, y1, x2, y2 = btn_next_rect
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        runtime_selected_idx = (runtime_selected_idx + 1) % len(templates)
                        worker.update_settings(selected_template_idx=runtime_selected_idx)
                        return
            
            # Toggle buttons
            if btn_mirror_rect:
                x1, y1, x2, y2 = btn_mirror_rect
                if x1 <= x <= x2 and y1 <= y <= y2:
                    runtime_mirror = not runtime_mirror
                    worker.update_settings(mirror_mode=runtime_mirror)
                    return
            
            if btn_face_rect:
                x1, y1, x2, y2 = btn_face_rect
                if x1 <= x <= x2 and y1 <= y <= y2:
                    runtime_face_enabled = not runtime_face_enabled
                    worker. update_settings(face_match_enabled=runtime_face_enabled)
                    return
    
    cv2.setMouseCallback(window_name, on_mouse)
    
    try:
        while True:
            try:
                root.update()
            except tk.TclError:
                requested_quit = True
            except Exception: 
                pass
            
            # Get worker data
            with worker.lock:
                bgr = worker.latest_frame_bgr. copy() if worker.latest_frame_bgr is not None else None
                tpl = worker.latest_template
                tpl_idx = worker.latest_template_idx
                pose_score = worker.latest_pose_score
                face_score = worker. latest_face_score
                combined_score = worker.latest_combined_score
                matched_pts = worker.latest_matched_pts
                fps = worker.fps
                live_lms = worker.latest_live_lms. copy() if worker.latest_live_lms is not None else None
                live_face_lms = worker.latest_live_face_lms.copy() if worker.latest_live_face_lms is not None else None
                quality_state = worker.latest_quality_state
            
            # Popup
            if quality_state == "red" and not popup_scheduled:
                popup_scheduled = True
                try:
                    root.after(0, lambda: messagebox. showwarning(
                        "Low visibility",
                        "Insufficient landmarks.  Adjust position or lighting. "))
                except Exception:
                    pass
            if quality_state != "red": 
                popup_scheduled = False
            
            if bgr is None:
                placeholder = np.full((480, 800, 3), 60, dtype=np.uint8)
                cv2.putText(placeholder, "Waiting for camera.. .", (20, 240),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
                combined = np.hstack((placeholder, np.full((480, TEMPLATE_DISPLAY_WIDTH, 3), 30, dtype=np.uint8)))
            else:
                try:
                    live_to_show = bgr.copy()
                    h_live, w_live = live_to_show.shape[:2]
                    
                    # Draw skeleton on webcam
                    if runtime_show_skel_video and live_lms is not None:
                        _, coords_live, _ = filter_valid_points(live_lms)
                        draw_landmark_skeleton_cv(live_to_show, coords_live, landmark_count, color=(0, 255, 0), radius=4)
                    
                    # Draw face mesh on webcam
                    if runtime_show_face_video and live_face_lms is not None:
                        draw_face_mesh_overlay(live_to_show, live_face_lms, runtime_face_mode, color=(0, 255, 255))
                    
                    # Quality indicator
                    if runtime_show_quality: 
                        quality_colors = {
                            "green": (0, 255, 0),
                            "amber": (0, 165, 255),
                            "red": (0, 0, 255)
                        }
                        
                        quality_color = quality_colors.get(quality_state, (128, 128, 128))
                        
                        # Calculate quality percentage
                        visible_count = 0
                        if live_lms is not None: 
                            visible_count = len([lm for lm in live_lms if lm[2] > VISIBILITY_THRESHOLD])
                        
                        quality_percent = int((visible_count / landmark_count) * 100) if landmark_count > 0 else 0
                        
                        # Custom labels
                        if quality_state == "green":
                            status_text = "EXCELLENT"
                        elif quality_state == "amber":
                            status_text = "ACCEPTABLE"
                        else:
                            status_text = "ADJUST POSE"

                        if landmark_count == 3:  # Face-only
                            if quality_state == "green": 
                                status_text = "FACE DETECTED"
                            else: 
                                status_text = "NO FACE"
                        
                        # Draw box
                        cv2.rectangle(live_to_show, (10, 10), (200, 70), quality_color, -1)
                        cv2.rectangle(live_to_show, (10, 10), (200, 70), (255, 255, 255), 2)
                        
                        # Draw text
                        cv2.putText(live_to_show, status_text, (20, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.putText(live_to_show, f"Quality: {quality_percent}%", (20, 58),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    
                    # Template panel
                    if tpl is not None:
                        tpl_img = tpl["image"]. copy()
                        
                        if runtime_show_skel_photo: 
                            draw_landmark_skeleton_cv(tpl_img, tpl["valid_coords"], landmark_count, color=(0, 128, 255), radius=4)
                        
                        if runtime_show_face_photo and tpl. get("face_landmarks") is not None:
                            draw_face_mesh_overlay(tpl_img, tpl["face_landmarks"], runtime_face_mode, color=(255, 128, 0))
                        
                        th, tw = tpl_img. shape[:2]
                        scale = TEMPLATE_DISPLAY_WIDTH / float(tw)
                        new_w = int(tw * scale)
                        new_h = int(th * scale)
                        tpl_small = cv2.resize(tpl_img, (new_w, new_h))
                        
                        if new_h < h_live:
                            pad_top = (h_live - new_h) // 2
                            pad_bottom = h_live - new_h - pad_top
                            tpl_panel = cv2.copyMakeBorder(tpl_small, pad_top, pad_bottom, 0, 0,
                                                          cv2.BORDER_CONSTANT, value=[30, 30, 30])
                        elif new_h > h_live: 
                            start = (new_h - h_live) // 2
                            tpl_panel = tpl_small[start:start + h_live, : , :]
                        else: 
                            tpl_panel = tpl_small
                        
                        if tpl_panel.shape[1] < TEMPLATE_DISPLAY_WIDTH: 
                            pad_right = TEMPLATE_DISPLAY_WIDTH - tpl_panel.shape[1]
                            tpl_panel = cv2.copyMakeBorder(tpl_panel, 0, 0, 0, pad_right,
                                                          cv2.BORDER_CONSTANT, value=[30, 30, 30])
                    else:
                        tpl_panel = np.full((h_live, TEMPLATE_DISPLAY_WIDTH, 3), 30, dtype=np.uint8)
                        cv2.putText(tpl_panel, "No template", (10, h_live // 2),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
                    
                    if tpl_panel.shape[0] != h_live:
                        if tpl_panel.shape[0] < h_live:
                            pad_top = (h_live - tpl_panel.shape[0]) // 2
                            pad_bottom = h_live - tpl_panel.shape[0] - pad_top
                            tpl_panel = cv2.copyMakeBorder(tpl_panel, pad_top, pad_bottom, 0, 0,
                                                          cv2.BORDER_CONSTANT, value=[30, 30, 30])
                        else:
                            tpl_panel = tpl_panel[: h_live, : , :]
                    
                    combined = np.hstack((live_to_show, tpl_panel))
                    
                    # Advanced info
                    if runtime_show_advanced:
                        ch, cw = combined.shape[:2]
                        
                        # Create info bar
                        info_bar_height = 90
                        info_bar = np.zeros((info_bar_height, cw, 3), dtype=np.uint8)
                        info_bar[:] = (30, 30, 30)  # Dark gray
                        
                        # Border
                        cv2.rectangle(info_bar, (0, 0), (cw-1, info_bar_height-1), (100, 100, 100), 2)
                        
                        # Text content
                        best_name = tpl['name'] if tpl is not None else "None"
                        if runtime_matching_mode == "dynamic":
                            mode_indicator = "AUTO"
                            header_text = f"Best:  {best_name} - Score:  {combined_score:.1f}%"
                        else: 
                            mode_indicator = "LOCKED"
                            header_text = f"Target: {best_name} ({tpl_idx+1}/{len(templates)}) - Score: {combined_score:.1f}% [{mode_indicator}]"
                        
                        pose_text = f"Pose: {pose_score:.1f}%" if pose_score else "Pose: N/A"
                        face_text = f"Face: {face_score:.1f}%" if face_score else "Face: N/A"
                        info_text = f"{pose_text}  {face_text}  Matched: {matched_pts}/{landmark_count}  FPS: {fps:.1f}"
                        
                        # Draw text
                        cv2.putText(info_bar, header_text, (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.putText(info_bar, info_text, (10, 60),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)
                        
                        # Mode badge
                        badge_color = (0, 150, 255) if runtime_matching_mode == "dynamic" else (255, 100, 0)
                        cv2.rectangle(info_bar, (cw - 150, 10), (cw - 10, 70), badge_color, 2)
                        cv2.putText(info_bar, mode_indicator, (cw - 135, 47),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, badge_color, 2, cv2.LINE_AA)
                        
                        # Stack info bar above video
                        combined = np.vstack((info_bar, combined))
                    
                    # Draw control buttons at bottom
                    ch, cw = combined.shape[: 2]  # Update dimensions after stacking
                    btn_w, btn_h = 110, 35
                    spacing = 10
                    y_bottom = ch - btn_h - 10
                    
                    # Right side:  Return and Quit
                    btn_quit_x2 = cw - 10
                    btn_quit_x1 = btn_quit_x2 - btn_w
                    btn_return_x2 = btn_quit_x1 - spacing
                    btn_return_x1 = btn_return_x2 - btn_w
                    
                    cv2.rectangle(combined, (btn_return_x1, y_bottom), (btn_return_x2, y_bottom + btn_h),
                                (70, 70, 70), -1)
                    cv2.rectangle(combined, (btn_return_x1, y_bottom), (btn_return_x2, y_bottom + btn_h),
                                (200, 200, 200), 2)
                    cv2.putText(combined, "Return", (btn_return_x1 + 18, y_bottom + 23),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    cv2.rectangle(combined, (btn_quit_x1, y_bottom), (btn_quit_x2, y_bottom + btn_h),
                                (50, 50, 150), -1)
                    cv2.rectangle(combined, (btn_quit_x1, y_bottom), (btn_quit_x2, y_bottom + btn_h),
                                (100, 100, 200), 2)
                    cv2.putText(combined, "Quit", (btn_quit_x1 + 32, y_bottom + 23),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    btn_return_rect = (btn_return_x1, y_bottom, btn_return_x2, y_bottom + btn_h)
                    btn_quit_rect = (btn_quit_x1, y_bottom, btn_quit_x2, y_bottom + btn_h)
                    
                    # Left side: Runtime toggles
                    x_start = 10
                    
                    # Mirror toggle
                    mirror_text = "Mirror: ON" if runtime_mirror else "Mirror: OFF"
                    mirror_color = (0, 200, 0) if runtime_mirror else (100, 100, 100)
                    cv2.rectangle(combined, (x_start, y_bottom), (x_start + btn_w, y_bottom + btn_h),
                                mirror_color, -1)
                    cv2.rectangle(combined, (x_start, y_bottom), (x_start + btn_w, y_bottom + btn_h),
                                (200, 200, 200), 2)
                    cv2.putText(combined, mirror_text, (x_start + 8, y_bottom + 23),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                    btn_mirror_rect = (x_start, y_bottom, x_start + btn_w, y_bottom + btn_h)
                    x_start += btn_w + spacing
                    
                    # Face toggle
                    face_text = "Face: ON" if runtime_face_enabled else "Face: OFF"
                    face_color = (0, 200, 0) if runtime_face_enabled else (100, 100, 100)
                    cv2.rectangle(combined, (x_start, y_bottom), (x_start + btn_w, y_bottom + btn_h),
                                face_color, -1)
                    cv2.rectangle(combined, (x_start, y_bottom), (x_start + btn_w, y_bottom + btn_h),
                                (200, 200, 200), 2)
                    cv2.putText(combined, face_text, (x_start + 12, y_bottom + 23),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                    btn_face_rect = (x_start, y_bottom, x_start + btn_w, y_bottom + btn_h)
                    x_start += btn_w + spacing
                    
                    # navigation
                    if runtime_matching_mode == "single":
                        center_x = cw // 2
                        prev_x1 = center_x - btn_w - spacing // 2
                        prev_x2 = prev_x1 + btn_w
                        next_x1 = center_x + spacing // 2
                        next_x2 = next_x1 + btn_w
                        
                        # Prev button
                        cv2.rectangle(combined, (prev_x1, y_bottom), (prev_x2, y_bottom + btn_h),
                                    (150, 100, 50), -1)
                        cv2.rectangle(combined, (prev_x1, y_bottom), (prev_x2, y_bottom + btn_h),
                                    (200, 200, 200), 2)
                        cv2.putText(combined, "< Prev", (prev_x1 + 18, y_bottom + 23),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                        btn_prev_rect = (prev_x1, y_bottom, prev_x2, y_bottom + btn_h)
                        
                        # Next button
                        cv2.rectangle(combined, (next_x1, y_bottom), (next_x2, y_bottom + btn_h),
                                    (150, 100, 50), -1)
                        cv2.rectangle(combined, (next_x1, y_bottom), (next_x2, y_bottom + btn_h),
                                    (200, 200, 200), 2)
                        cv2.putText(combined, "Next >", (next_x1 + 18, y_bottom + 23),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                        btn_next_rect = (next_x1, y_bottom, next_x2, y_bottom + btn_h)
                    else:
                        btn_prev_rect = None
                        btn_next_rect = None
                
                except Exception as e:
                    print(f"Frame composition error: {e}")
                    combined = np.full((480, w_live + TEMPLATE_DISPLAY_WIDTH, 3), 0, dtype=np.uint8)
                    cv2.putText(combined, "Frame error - continuing.. .", (20, 40),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            cv2.imshow(window_name, combined)
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27 or key == ord('q'):
                requested_quit = True
            elif key == ord('r'):
                requested_return = True
            elif key == ord('m'):
                runtime_mirror = not runtime_mirror
                worker.update_settings(mirror_mode=runtime_mirror)
            elif key == ord('f'):
                runtime_face_enabled = not runtime_face_enabled
                worker.update_settings(face_match_enabled=runtime_face_enabled)
            elif key == ord('[') and runtime_matching_mode == "single":
                runtime_selected_idx = (runtime_selected_idx - 1) % len(templates)
                worker.update_settings(selected_template_idx=runtime_selected_idx)
            elif key == ord(']') and runtime_matching_mode == "single":
                runtime_selected_idx = (runtime_selected_idx + 1) % len(templates)
                worker.update_settings(selected_template_idx=runtime_selected_idx)
            
            # Detect window closed
            try:
                win_prop = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
                if win_prop < 1:
                    requested_return = True
            except Exception:
                pass
            
            if requested_return:
                worker.stop()
                worker.join(timeout=2)
                cv2.destroyWindow(window_name)
                try:
                    root. deiconify()
                except Exception:
                    pass
                return
            
            if requested_quit:
                worker.stop()
                worker. join(timeout=2.0)
                cv2.destroyAllWindows()
                try: 
                    root.quit()
                    root.destroy()
                except Exception:
                    pass
                return
    
    except Exception as ex:
        try:
            worker.stop()
            worker.join(timeout=2.0)
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            root. after(0, lambda: messagebox.showerror("Error", f"Unexpected error: {ex}"))
            root. deiconify()
        except Exception:
            print(f"Fatal error: {ex}")

# --------------------------- Main Entry Point ---------------------------

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='JM-PoseMatcher:  Lightweight pose matching against reference images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        """
    )
    
    # Positional
    parser.add_argument(
        'templates_path',
        nargs='?',
        default=None,
        help='Path to folder containing template images (JPG/PNG)'
    )
    
    # Mode control
    parser.add_argument(
        '--no-gui',
        action='store_true',
        help='Skip configuration window and start video immediately'
    )
    
    # Profile settings
    parser.add_argument(
        '--profile',
        choices=['face_only', 'upper_body', 'default', 'full_body'],
        default='default',
        help='Pose profile to use (default: default)'
    )
    
    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        metavar='INDEX',
        help='Camera index to use (default: 0)'
    )
    
    parser.add_argument(
        '--mode',
        choices=['dynamic', 'single'],
        default='dynamic',
        help='Matching mode: dynamic (auto-switch) or single (manual) (default: dynamic)'
    )
    
    parser.add_argument(
        '--compression',
        choices=['none', 'low', 'medium', 'high'],
        default='none',
        help='Image compression level (default: none)'
    )
    
    # Display toggles
    parser.add_argument(
        '--no-mirror',
        action='store_true',
        help='Disable mirror mode (default: enabled)'
    )
    
    parser.add_argument(
        '--no-quality',
        action='store_true',
        help='Hide quality indicator (default: shown)'
    )
    
    parser.add_argument(
        '--no-advanced',
        action='store_true',
        help='Hide advanced info bar (default: shown)'
    )
    
    parser.add_argument(
        '--no-skeleton',
        action='store_true',
        help='Hide skeleton overlays (default: shown)'
    )
    
    parser.add_argument(
        '--no-face',
        action='store_true',
        help='Disable face matching in scoring (default: enabled)'
    )
    
    parser.add_argument(
        '--face-mesh',
        choices=['minimal', 'medium', 'full'],
        default='minimal',
        metavar='MODE',
        help='Face mesh detail level (default: minimal)'
    )
    
    parser.add_argument(
        '--show-face-mesh',
        action='store_true',
        help='Show face mesh overlay on video (default:  hidden)'
    )
    
    return parser.parse_args()

def main():
    """Main entry point with CLI argument support."""
    args = parse_arguments()
    
    # If --no-gui, skip config window and load directly
    if args.no_gui:
        if not args.templates_path:
            print("Error: --no-gui requires TEMPLATES_PATH argument")
            print("Example: python jm_pose_matcher.py /path/to/templates --no-gui")
            sys.exit(1)
        
        if not os.path.isdir(args.templates_path):
            print(f"Error:  Templates path not found: {args.templates_path}")
            sys.exit(1)
        
        print(f"Loading templates from: {args.templates_path}")
        print(f"Profile: {args.profile}")
        print(f"Mode: {args.mode}")
        print(f"Camera: {args.camera}")
        print("Loading...  (this may take a moment)\n")
        
        # Load templates without GUI
        use_landmarks = get_landmarks_for_profile(args.profile)
        
        pose_static = mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )
        face_static = mp_face. FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5
        )
        
        templates = load_templates_from_folder(
            args.templates_path,
            pose_static,
            face_static,
            use_landmarks,
            args.compression,
            progress_callback=None
        )
        
        pose_static.close()
        face_static.close()
        
        if not templates:
            print("Error: No usable templates found in folder")
            print("Ensure images contain detectable poses/faces")
            sys.exit(1)
        
        print(f"\n✓ Loaded {len(templates)} templates")
        print("Starting video matching.. .\n")
        
        # Create minimal Tk root for messagebox support
        root = tk.Tk()
        root.withdraw()
        
        # Start video loop directly
        run_video_loop(
            root=root,
            templates=templates,
            matching_mode=args.mode,
            selected_template_idx=0,
            profile_key=args.profile,
            camera_index=args.camera,
            mirror_mode=not args.no_mirror,
            show_quality_indicator=not args.no_quality,
            show_advanced_info=not args.no_advanced,
            show_skel_video=not args.no_skeleton,
            show_skel_photo=not args.no_skeleton,
            face_match_enabled=not args.no_face,
            show_face_mesh_video=args.show_face_mesh,
            show_face_mesh_photo=args.show_face_mesh,
            face_mesh_mode=args.face_mesh
        )
        
        root. destroy()
    
    else:
        # Normal GUI mode
        root = tk.Tk()
        app = ConfigWindow(root, initial_path=args.templates_path)
        root.mainloop()


if __name__ == "__main__":
    main()