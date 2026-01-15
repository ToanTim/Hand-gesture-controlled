# debug_vertices.py
import json
import numpy as np

# Load first annotation
with open('./data/freihand/FreiHAND_pub_v2_eval/evaluation/anno/00000000.json', 'r') as f:
    data = json.load(f)

vertices = np.array(data['verts'])
print(f"Total vertices: {vertices.shape}")  # Should be (778, 3)
print(f"First 10 vertices:\n{vertices[:10]}")

# Check our mapping
mano_to_mediapipe = [0, 5, 9, 10, 11, 17, 18, 19, 20, 25, 26, 27, 28, 
                     33, 34, 35, 36, 41, 42, 43, 44]

print(f"\nSelected vertices (indices {mano_to_mediapipe}):")
for i, idx in enumerate(mano_to_mediapipe):
    print(f"Keypoint {i}: vertex {idx} = {vertices[idx]}")