#!/usr/bin/env python3
"""
FreiHAND Dataset Evaluator - Windows Paths Fixed
"""

import sys
import os

# === IMPORT SETUP ===
def setup_imports():
    """Setup Python path to find project modules"""
    current_file = os.path.abspath(__file__)  # src/evaluation/freihand_evaluator.py
    evaluation_dir = os.path.dirname(current_file)  # src/evaluation
    src_dir = os.path.dirname(evaluation_dir)       # src
    project_root = os.path.dirname(src_dir)         # D:\Hand-gesture-controlled
    
    print(f"[DATA] Path setup:")
    print(f"   Project root: {project_root}")
    print(f"   Src directory: {src_dir}")
    
    # Add to Python path
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    
    return project_root, src_dir

project_root, src_dir = setup_imports()

try:
    from detectors.hand_detector import HandDetector
    print("[OK] HandDetector imported successfully")
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    print(f"   Looking in: {src_dir}")
    sys.exit(1)

import json
import cv2
import numpy as np
from pathlib import Path, PureWindowsPath
import glob

class FreiHandEvaluator:
    def __init__(self, data_dir=None):
        """Initialize evaluator with absolute Windows paths"""
        if data_dir is None:
            # ABSOLUTE PATH to your data
            self.data_dir = Path(r"D:\Hand-gesture-controlled\data\freihand\FreiHAND_pub_v2_eval")
        else:
            self.data_dir = Path(data_dir)
        
        self.rgb_dir = self.data_dir / "evaluation" / "rgb"
        self.anno_dir = self.data_dir / "evaluation" / "anno"
        
        print(f"\n[DATA] Data directories:")
        print(f"   Data root: {self.data_dir}")
        print(f"   RGB images: {self.rgb_dir}")
        print(f"   Annotations: {self.anno_dir}")
        
        # Check if directories exist
        if not self.rgb_dir.exists():
            print(f"[ERROR] ERROR: RGB directory not found: {self.rgb_dir}")
            print("   Please check the path above is correct")
            sys.exit(1)
        
        # Get files (Windows path handling)
        self.image_paths = sorted([str(p) for p in self.rgb_dir.glob("*.jpg")])
        self.anno_paths = sorted([str(p) for p in self.anno_dir.glob("*.json")])
        
        print(f"\n[INFO] Dataset loaded:")
        print(f"   Images: {len(self.image_paths)}")
        print(f"   Annotations: {len(self.anno_paths)}")
        
        # Display first few files
        if self.image_paths:
            print(f"   First image: {os.path.basename(self.image_paths[0])}")
            print(f"   First annotation: {os.path.basename(self.anno_paths[0])}")
        
        # CORRECTED: MANO to MediaPipe vertex mapping for FreiHAND
        # These indices map MANO vertices (778 total) to 21 MediaPipe keypoints
        self.mano_to_mediapipe = [
            0,    # WRIST
            5,    # THUMB_CMC
            9,    # THUMB_MCP  
            10,   # THUMB_IP
            11,   # THUMB_TIP
            17,   # INDEX_FINGER_MCP
            18,   # INDEX_FINGER_PIP
            19,   # INDEX_FINGER_DIP
            20,   # INDEX_FINGER_TIP
            25,   # MIDDLE_FINGER_MCP
            26,   # MIDDLE_FINGER_PIP
            27,   # MIDDLE_FINGER_DIP
            28,   # MIDDLE_FINGER_TIP
            33,   # RING_FINGER_MCP
            34,   # RING_FINGER_PIP
            35,   # RING_FINGER_DIP
            36,   # RING_FINGER_TIP
            41,   # PINKY_MCP
            42,   # PINKY_PIP
            43,   # PINKY_DIP
            44    # PINKY_TIP
        ]
        
        # Initialize detector with conservative settings
        self.detector = HandDetector(
            max_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("[INFO] HandDetector initialized")
    
    def debug_first_annotation(self):
        """Debug the first annotation to understand structure"""
        print(f"\n[DEBUG] Debugging first annotation...")
        
        if not self.anno_paths:
            print("   No annotation files found!")
            return
        
        anno_path = self.anno_paths[0]
        print(f"   Reading: {anno_path}")
        
        with open(anno_path, 'r') as f:
            data = json.load(f)
        
        print(f"\n   Keys in JSON: {list(data.keys())}")
        
        K = np.array(data['K'])
        vertices = np.array(data['verts'])
        
        print(f"   Camera matrix K shape: {K.shape}")
        print(f"   K matrix:\n{K}")
        print(f"   Vertices shape: {vertices.shape}")
        print(f"   First 3 vertices:")
        for i in range(3):
            print(f"     [{i}] {vertices[i]}")
        
        # Check if vertices look reasonable
        print(f"\n   Vertex statistics:")
        print(f"     Min: {vertices.min():.3f}, Max: {vertices.max():.3f}")
        print(f"     Mean: {vertices.mean():.3f}, Std: {vertices.std():.3f}")
        
        return data
    
    def project_3d_to_2d(self, vertices_3d, K):
        """
        Correct 3D to 2D projection for FreiHAND data
        
        vertices_3d: (N, 3) - 3D coordinates in camera space (meters)
        K: (3, 3) - Camera intrinsic matrix
        returns: (N, 2) - 2D pixel coordinates
        """
        # Ensure correct shape
        if vertices_3d.shape[1] != 3:
            print(f"[WARNING] Reshaping vertices from {vertices_3d.shape} to (N, 3)")
            vertices_3d = vertices_3d[:, :3]
        
        # Convert to homogeneous coordinates (N, 4)
        N = vertices_3d.shape[0]
        ones = np.ones((N, 1))
        vertices_hom = np.hstack([vertices_3d, ones])
        
        # Project: (3, 4) @ (4, N) -> (3, N)  [Actually K @ vertices_hom.T]
        # K is (3, 3), but we need to handle homogeneous coords
        # Actually: K @ vertices_3d.T gives (3, N) then we need to handle z
        
        # Correct way: K @ vertices_3d.T, then divide by z
        points_2d_h = K @ vertices_3d.T  # (3, N)
        
        # Divide by z (depth) to get pixel coordinates
        points_2d = points_2d_h[:2, :] / points_2d_h[2:, :]  # (2, N)
        
        # Transpose to get (N, 2)
        points_2d = points_2d.T
        
        return points_2d
    
    def get_mediapipe_keypoints(self, vertices_3d, K):
        """Extract 21 MediaPipe keypoints from MANO vertices"""
        # Select the 21 vertices corresponding to MediaPipe keypoints
        selected_indices = self.mano_to_mediapipe
        
        # Ensure indices are valid
        max_idx = vertices_3d.shape[0]
        valid_indices = [idx for idx in selected_indices if idx < max_idx]
        
        if len(valid_indices) != 21:
            print(f"[WARNING] Warning: Only {len(valid_indices)} valid indices (need 21)")
            print(f"   Using first 21 vertices as fallback")
            valid_indices = list(range(21))
        
        selected_vertices = vertices_3d[valid_indices]
        kp_2d = self.project_3d_to_2d(selected_vertices, K)
        
        print(f"   Selected {len(valid_indices)} vertices for keypoints")
        print(f"   First projected point: {kp_2d[0]}")
        
        return kp_2d
    
    def evaluate_single(self, idx=0, visualize=True):
        """Evaluate a single image"""
        print(f"\n{'='*60}")
        print(f"EVALUATING Image {idx}")
        print(f"{'='*60}")
        
        # Check bounds
        if idx >= len(self.image_paths):
            print(f"[ERROR] Index {idx} out of range (max: {len(self.image_paths)-1})")
            return None
        
        # 1. Load and debug annotation
        print(f"\n[1/3] Loading annotation...")
        anno_path = self.anno_paths[idx]
        with open(anno_path, 'r') as f:
            data = json.load(f)
        
        K = np.array(data['K'])
        vertices = np.array(data['verts'])
        
        print(f"   Camera K:\n{K}")
        print(f"   Vertices: {vertices.shape} (min={vertices.min():.3f}, max={vertices.max():.3f})")
        
        # 2. Get ground truth keypoints
        print(f"\n[2/3] Computing ground truth keypoints...")
        kp_gt = self.get_mediapipe_keypoints(vertices, K)
        print(f"   Ground truth keypoints shape: {kp_gt.shape}")
        
        # 3. Load image and run detector
        print(f"\n[3/3] Running MediaPipe detector...")
        img_path = self.image_paths[idx]
        frame = cv2.imread(img_path)
        
        if frame is None:
            print(f"[ERROR] Failed to load image: {img_path}")
            return None
        
        h, w, _ = frame.shape
        print(f"   Image size: {w}x{h}")
        
        # Process with your detector
        processed_frame = self.detector.find_hands(frame, draw=False)
        landmark_list = self.detector.get_position(frame, hand_no=0)
        
        result = {
            "image_idx": idx,
            "image_path": img_path,
            "image_size": (w, h),
            "detected": landmark_list is not None and len(landmark_list) > 0
        }
        
        if result["detected"]:
            # Convert detector output to numpy array
            kp_pred = np.array([[lm[1], lm[2]] for lm in landmark_list])
            print(f"   Detected {len(landmark_list)} landmarks")
            
            # Calculate errors
            errors = np.linalg.norm(kp_pred - kp_gt, axis=1)
            result["mean_error_px"] = np.mean(errors)
            result["median_error_px"] = np.median(errors)
            result["max_error_px"] = np.max(errors)
            result["errors"] = errors.tolist()
            
            # PCK metrics
            thresholds = [5, 10, 15, 20, 30]
            for thresh in thresholds:
                pck = np.mean(errors < thresh) * 100
                result[f"pck_{thresh}px"] = pck
            
            print(f"\n[RESULTS]:")
            print(f"   Mean error: {result['mean_error_px']:.1f} px")
            print(f"   Median error: {result['median_error_px']:.1f} px")
            print(f"   Max error: {result['max_error_px']:.1f} px")
            print(f"   PCK@20px: {result['pck_20px']:.1f} %")
            
            if visualize:
                self.visualize_comparison(frame, kp_pred, kp_gt, result)
        else:
            print(f"[ERROR] No hand detected in image")
        
        return result
    
    def visualize_comparison(self, frame, kp_pred, kp_gt, result):
        """Visualize predictions vs ground truth"""
        vis_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw ground truth (RED)
        for x, y in kp_gt:
            cv2.circle(vis_frame, (int(x), int(y)), 5, (0, 0, 255), -1)
        
        # Draw predictions (GREEN)
        for x, y in kp_pred:
            cv2.circle(vis_frame, (int(x), int(y)), 3, (0, 255, 0), -1)
        
        # Draw lines between corresponding points
        for (x1, y1), (x2, y2) in zip(kp_pred, kp_gt):
            cv2.line(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                    (255, 255, 0), 1)
        
        # Add text overlay
        text_lines = [
            f"Image {result['image_idx']} - {w}x{h}",
            f"Mean Error: {result['mean_error_px']:.1f} px",
            f"PCK@20px: {result['pck_20px']:.1f}%",
            "Red: Ground Truth (FreiHAND)",
            "Green: MediaPipe Detector"
        ]
        
        for i, line in enumerate(text_lines):
            y_pos = 30 + i * 25
            cv2.putText(vis_frame, line, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(vis_frame, line, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Show image
        cv2.imshow("Evaluation: Ground Truth (RED) vs MediaPipe (GREEN)", vis_frame)
        print("\n[VISUALIZATION]:")
        print("   Red circles: Ground truth from FreiHAND")
        print("   Green circles: Your MediaPipe detector")
        print("   Yellow lines: Error distances")
        print("\n   Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def batch_evaluate(self, num_samples=10):
        """Evaluate multiple images"""
        print(f"\n{'='*60}")
        print(f"BATCH EVALUATION - {num_samples} samples")
        print(f"{'='*60}")
        
        results = []
        for i in range(min(num_samples, len(self.image_paths))):
            print(f"\n[{i+1}/{min(num_samples, len(self.image_paths))}] ", end="")
            result = self.evaluate_single(i, visualize=False)
            if result:
                results.append(result)
        
        # Analyze results
        return self.analyze_results(results)
    
    def analyze_results(self, results):
        """Analyze batch results"""
        if not results:
            return {"error": "No results"}
        
        detected = [r for r in results if r["detected"]]
        detection_rate = len(detected) / len(results) * 100
        
        stats = {
            "total_samples": len(results),
            "detection_rate": f"{detection_rate:.1f}%",
            "detected_count": len(detected),
            "failed_count": len(results) - len(detected)
        }
        
        if detected:
            # Error statistics
            mean_errors = [r["mean_error_px"] for r in detected]
            stats["mean_error_px"] = f"{np.mean(mean_errors):.1f}"
            stats["error_std_px"] = f"{np.std(mean_errors):.1f}"
            stats["min_error_px"] = f"{np.min(mean_errors):.1f}"
            stats["max_error_px"] = f"{np.max(mean_errors):.1f}"
            
            # PCK statistics
            for thresh in [10, 20, 30]:
                pck_values = [r[f"pck_{thresh}px"] for r in detected]
                stats[f"pck_{thresh}px_avg"] = f"{np.mean(pck_values):.1f}%"
        
        return stats

def main():
    print("="*60)
    print("FREIHAND DATASET EVALUATION")
    print("MediaPipe HandDetector Accuracy Measurement")
    print("="*60)
    
    # Initialize evaluator
    evaluator = FreiHandEvaluator()
    
    # Debug first annotation
    evaluator.debug_first_annotation()
    
    # Test single image
    print(f"\n{'='*60}")
    print("SINGLE IMAGE EVALUATION")
    print(f"{'='*60}")
    
    result = evaluator.evaluate_single(idx=0, visualize=True)
    
    if result and result["detected"]:
        print(f"\n[RESULTS] SINGLE IMAGE RESULTS:")
        print(f"   Mean Error: {result['mean_error_px']:.1f} px")
        print(f"   PCK@20px: {result['pck_20px']:.1f} %")
        
        # Interpretation
        error = result['mean_error_px']
        if error < 15:
            print(f"\n[EXCELLENT] Research-grade accuracy!")
        elif error < 25:
            print(f"\n[GOOD] Suitable for real-time applications")
        elif error < 40:
            print(f"\n[FAIR] May need parameter tuning")
        else:
            print(f"\n[NEEDS WORK] Check vertex mapping and parameters")
    
    # Ask about batch evaluation
    print(f"\n{'='*60}")
    print("NEXT STEPS")
    print(f"{'='*60}")
    print("To run batch evaluation on more images:")
    print("  evaluator.batch_evaluate(num_samples=20)")
    print("\nExample:")
    print("  >>> from src.evaluation.freihand_evaluator import FreiHandEvaluator")
    print("  >>> evaluator = FreiHandEvaluator()")
    print("  >>> stats = evaluator.batch_evaluate(20)")
    print("  >>> print(stats)")

if __name__ == "__main__":
    main()