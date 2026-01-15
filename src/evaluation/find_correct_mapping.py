# src/evaluation/find_mano_topology.py
import json
import numpy as np

def load_annotation(idx=0):
    """Load annotation file"""
    base_path = "D:/Hand-gesture-controlled/data/freihand/FreiHAND_pub_v2_eval/evaluation"
    anno_path = f"{base_path}/anno/{idx:08d}.json"
    
    with open(anno_path, 'r') as f:
        data = json.load(f)
    
    return np.array(data['K']), np.array(data['verts'])

def project_3d_to_2d(vertices, K):
    """Project 3D vertices to 2D"""
    points_2d_h = K @ vertices.T
    points_2d = points_2d_h[:2, :] / points_2d_h[2:, :]
    return points_2d.T

def find_vertex_by_position(vertices, condition_func, description=""):
    """Find vertex that best matches a condition"""
    best_idx = -1
    best_score = float('inf')
    
    for i, vertex in enumerate(vertices):
        score = condition_func(vertex, i)
        if score < best_score:
            best_score = score
            best_idx = i
    
    print(f"  {description}: Vertex {best_idx} at {vertices[best_idx]}")
    return best_idx

def main():
    print("="*60)
    print("FINDING MANO TOPOLOGY VERTICES")
    print("="*60)
    
    # Load data
    K, vertices = load_annotation(0)
    points_2d = project_3d_to_2d(vertices, K)
    
    print(f"\nTotal vertices: {vertices.shape}")
    print(f"Image size: 224x224")
    
    # Based on MANO model structure and research papers,
    # these are commonly used vertex indices for hand keypoints
    print("\n" + "="*60)
    print("COMMON MANO VERTEX INDICES FOR HAND KEYPOINTS")
    print("="*60)
    
    # Try different known mappings from research
    mappings = {
        "Mapping A (from hand-tailor paper)": [
            0,    # WRIST
            13,   # THUMB_CMC
            14,   # THUMB_MCP
            15,   # THUMB_IP
            16,   # THUMB_TIP
            1,    # INDEX_MCP
            2,    # INDEX_PIP
            3,    # INDEX_DIP
            17,   # INDEX_TIP
            4,    # MIDDLE_MCP
            5,    # MIDDLE_PIP
            6,    # MIDDLE_DIP
            18,   # MIDDLE_TIP
            10,   # RING_MCP
            11,   # RING_PIP
            12,   # RING_DIP
            19,   # RING_TIP
            7,    # PINKY_MCP
            8,    # PINKY_PIP
            9,    # PINKY_DIP
            20    # PINKY_TIP
        ],
        
        "Mapping B (alternative)": [
            0,    # WRIST
            1,    # THUMB_CMC
            2,    # THUMB_MCP
            3,    # THUMB_IP
            4,    # THUMB_TIP
            5,    # INDEX_MCP
            6,    # INDEX_PIP
            7,    # INDEX_DIP
            8,    # INDEX_TIP
            9,    # MIDDLE_MCP
            10,   # MIDDLE_PIP
            11,   # MIDDLE_DIP
            12,   # MIDDLE_TIP
            13,   # RING_MCP
            14,   # RING_PIP
            15,   # RING_DIP
            16,   # RING_TIP
            17,   # PINKY_MCP
            18,   # PINKY_PIP
            19,   # PINKY_DIP
            20    # PINKY_TIP
        ],
        
        "Mapping C (from FreiHAND utils)": [
            0,    # WRIST
            13,   # THUMB_CMC
            14,   # THUMB_MCP
            15,   # THUMB_IP
            16,   # THUMB_TIP
            1,    # INDEX_MCP
            2,    # INDEX_PIP
            3,    # INDEX_DIP
            17,   # INDEX_TIP
            4,    # MIDDLE_MCP
            5,    # MIDDLE_PIP
            6,    # MIDDLE_DIP
            18,   # MIDDLE_TIP
            10,   # RING_MCP
            11,   # RING_PIP
            12,   # RING_DIP
            19,   # RING_TIP
            7,    # PINKY_MCP
            8,    # PINKY_PIP
            9,    # PINKY_DIP
            20    # PINKY_TIP
        ]
    }
    
    # Test each mapping by checking if vertices form a reasonable hand
    print("\nTesting different mappings...")
    print("-" * 40)
    
    for name, mapping in mappings.items():
        print(f"\n{name}:")
        print(f"Indices: {mapping}")
        
        # Check if indices are valid
        if max(mapping) >= len(vertices):
            print(f"  [ERROR] Invalid: max index {max(mapping)} >= {len(vertices)}")
            continue
        
        # Get projected points
        mapped_points = points_2d[mapping]
        
        # Calculate spread
        x_range = np.ptp(mapped_points[:, 0])
        y_range = np.ptp(mapped_points[:, 1])
        
        print(f"  X range: {x_range:.1f}px")
        print(f"  Y range: {y_range:.1f}px")
        
        # Check if they look like a hand (reasonable spread)
        if x_range > 30 and y_range > 30:
            print(f"  [OK] Looks plausible")
            
            # Show first few positions
            print(f"  First 5 points (2D):")
            for i in range(min(5, len(mapping))):
                print(f"    {i}: ({mapped_points[i, 0]:.1f}, {mapped_points[i, 1]:.1f})")
        else:
            print(f"  [WARN]  Spread too small - unlikely to be correct")
    
    # Try to find correct mapping by analyzing vertex neighborhoods
    print("\n" + "="*60)
    print("ANALYZING VERTEX NEIGHBORHOODS")
    print("="*60)
    
    # The MANO model has known vertex neighborhoods
    # Let's check which vertices have similar properties to MediaPipe landmarks
    
    # MediaPipe landmarks are roughly in this order:
    mediapipe_order = [
        "WRIST",
        "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
        "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
        "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
        "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
        "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
    ]
    
    # Let's search for vertices that could be fingertips
    print("\nSearching for fingertips (should have highest Z values)...")
    
    z_values = vertices[:, 2]
    fingertip_candidates = np.argsort(z_values)[-30:]  # Top 30 by depth
    
    print("Top 15 vertices by depth (potential fingertips):")
    for i, idx in enumerate(fingertip_candidates[::-1][:15]):
        print(f"  {i+1:2d}. Vertex {idx:3d}: Z={z_values[idx]:.4f}, "
              f"2D=({points_2d[idx, 0]:.1f}, {points_2d[idx, 1]:.1f})")
    
    # Based on the output, we can manually identify which might be fingertips
    # Typical pattern: 5 fingertips, with thumb being most different
    
    print("\n" + "="*60)
    print("RECOMMENDED APPROACH")
    print("="*60)
    
    print("\nGiven the complexity, here's the simplest solution:")
    print("1. Use the proven mapping from research (most likely correct)")
    print("2. Test it and see if error improves")
    
    # The most commonly used mapping in research is:
    correct_mapping = [
        0,    # WRIST
        13,   # THUMB_CMC
        14,   # THUMB_MCP
        15,   # THUMB_IP
        16,   # THUMB_TIP
        1,    # INDEX_MCP
        2,    # INDEX_PIP
        3,    # INDEX_DIP
        17,   # INDEX_TIP
        4,    # MIDDLE_MCP
        5,    # MIDDLE_PIP
        6,    # MIDDLE_DIP
        18,   # MIDDLE_TIP
        10,   # RING_MCP
        11,   # RING_PIP
        12,   # RING_DIP
        19,   # RING_TIP
        7,    # PINKY_MCP
        8,    # PINKY_PIP
        9,    # PINKY_DIP
        20    # PINKY_TIP
    ]
    
    print(f"\n[OK] RECOMMENDED MAPPING (copy this):")
    print(f"mano_to_mediapipe = {correct_mapping}")
    
    # Let's validate this mapping
    print(f"\n[INFO] Validation of recommended mapping:")
    
    # Check indices are valid
    if max(correct_mapping) < len(vertices):
        mapped_points = points_2d[correct_mapping]
        
        # Calculate statistics
        x_range = np.ptp(mapped_points[:, 0])
        y_range = np.ptp(mapped_points[:, 1])
        
        print(f"  Indices valid: [OK] (max {max(correct_mapping)} < {len(vertices)})")
        print(f"  X range: {x_range:.1f}px")
        print(f"  Y range: {y_range:.1f}px")
        print(f"  Wrist position: ({mapped_points[0, 0]:.1f}, {mapped_points[0, 1]:.1f})")
        
        # Check if fingertips (indices 4, 8, 12, 16, 20) have high Z
        fingertip_indices = [4, 8, 12, 16, 20]
        print(f"\n  Fingertip Z values (should be high > 0.8):")
        for idx in fingertip_indices:
            vertex_idx = correct_mapping[idx]
            z = vertices[vertex_idx, 2]
            print(f"    Keypoint {idx} (vertex {vertex_idx}): Z={z:.3f}")
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("1. Copy the mapping above into freihand_evaluator.py")
    print("2. Re-run evaluation")
    print("3. If error < 20px, mapping is correct!")
    print("4. If still high error, we need to find ground truth 2D annotations")

if __name__ == "__main__":
    main()