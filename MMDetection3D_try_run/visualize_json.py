import numpy as np
import json
import matplotlib

matplotlib.use('Agg')  # Fixes Qt plugin error by using a non-GUI backend
import matplotlib.pyplot as plt
import logging
import os
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--bin-file',
        type=str,
        default='./000008.bin',
        help='Input bin file.')
    parser.add_argument(
        '--json-file',
        type=str,
        default='./000008.json',
        help='Input json predicted file.')
    parser.add_argument(
        '--out-file',
        type=str,
        default='./output.png',
        help='Output visualization result.')
    return vars(parser.parse_args())

# Load arguments
args = parse_args()

# Join path to get full file locations
pcd_path = args['bin_file']
json_path = args['json_file']
out_file = args['out_file']
print(f"pcd_path = {pcd_path}")
print(f"json_path = {json_path}")

# Load Point Cloud
point_cloud = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 4)[:, :3]  # XYZ only

# Load Predictions
with open(json_path, "r") as f:
    predictions = json.load(f)

# Debug: Print JSON Keys
print("JSON Keys:", predictions.keys())

# Fix key mismatch (use 'bboxes_3d' instead of 'boxes_3d')
if "bboxes_3d" not in predictions or "scores_3d" not in predictions or "labels_3d" not in predictions:
    print("Error: JSON file does not contain required keys ('bboxes_3d', 'scores_3d', 'labels_3d').")
    print("JSON Content:", predictions)  # Debugging output
    exit()

# Extract data
boxes = np.array(predictions["bboxes_3d"])  # FIX: Use 'bboxes_3d'
scores = np.array(predictions["scores_3d"])
labels = np.array(predictions["labels_3d"])

# Matplotlib Plot (2D Projection)
fig, ax = plt.subplots(figsize=(8, 6))

# Scatter plot for point cloud
ax.scatter(point_cloud[:, 0], point_cloud[:, 1], s=0.5, c="gray", label="Point Cloud")

# Draw bounding boxes
for i in range(len(boxes)):
    if scores[i] < 0.3:  # Adjust confidence threshold if needed
        continue

    x, y, z, w, h, l, _ = boxes[i]  # Extract bounding box params
    rect = plt.Rectangle((x - w / 2, y - l / 2), w, l, linewidth=1.5, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.text(x, y, f"ID: {int(labels[i])} ({scores[i]:.2f})", color="red", fontsize=6)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.legend()
ax.set_title("2D Projection of 3D Bounding Boxes")
plt.savefig(f"{out_file}", dpi=300)  # Save instead of showing (no GUI required)
print(f"Visualization saved as '{out_file}'")
