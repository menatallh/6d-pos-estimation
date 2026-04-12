import numpy as np
import cv2
import os
from glob import glob
from PIL import Image
import open3d as o3d

# --- Camera Intrinsics ---
K = np.array([
    [797.1086, 0.0, 426.5295],
    [0.0, 735.0294, 198.7779],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

# --- Paths ---
train_dir = 'train'
mask_dir = 'masks_with_debug'
depth_dir = 'output_depths'
ply_model = '../FoundationPose/ImageToStl.com_140922_RevB_3U_CubeSat_ISIS_Advised_Max_Dimensions(1).ply'

# --- Load 3D CAD model and sample keypoints ---
mesh = o3d.io.read_triangle_mesh(ply_model)
mesh.compute_vertex_normals()
model_vertices = np.asarray(mesh.vertices)
num_keypoints = 100
indices = np.linspace(0, len(model_vertices) - 1, num=num_keypoints, dtype=int)
model_keypoints = model_vertices[indices]

# --- Loop through images ---
results = []
#image_paths = sorted(glob(os.path.join(train_dir, '*.*')))
image_paths = sorted(
    glob(os.path.join(train_dir, '*.jpg')) + glob(os.path.join(train_dir, '*.jpeg'))
)

for img_path in image_paths:
    name = os.path.splitext(os.path.basename(img_path))[0]
    
    depth_path = os.path.join(depth_dir, f"{name}_depth.png")
    #if not os.path.exists(depth_path):
    #    continue

    # Load image, mask(s), and depth
    rgb = np.array(Image.open(img_path).convert('RGB'))
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    depth /= 65535.0  # normalize if 16-bit PNG

    mask_paths = glob(os.path.join(mask_dir, f"{name}_*.png"))
    for mask_path in mask_paths:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = mask > 0

        ys, xs = np.where(mask)
        #if len(xs) < num_keypoints:
        #    continue
        sel_idx = np.random.choice(len(xs), size=num_keypoints, replace=False)
        img_pts = np.stack([xs[sel_idx], ys[sel_idx]], axis=-1).astype(np.float32)

        # Depth for 2D keypoints
        d = depth[ys[sel_idx], xs[sel_idx]]
        valid = d > 0
        #if np.count_nonzero(valid) < 6:
        #    continue
        img_pts = img_pts[valid]
        d = d[valid]

        # Back-project to 3D
        x = (img_pts[:, 0] - K[0, 2]) * d / K[0, 0]
        y = (img_pts[:, 1] - K[1, 2]) * d / K[1, 1]
        z = d
        cam_pts = np.stack((x, y, z), axis=1)

        #if cam_pts.shape[0] != model_keypoints.shape[0]:
        #    continue

        # Solve PnP
        success, rvec, tvec = cv2.solvePnP(model_keypoints, cam_pts, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
        if success:
            results.append({
                "image": name,
                "mask_id": os.path.basename(mask_path),
                "rvec": rvec.flatten().tolist(),
                "tvec": tvec.flatten().tolist()
            })

# Output first result
if results:
    print("Sample 6D pose result:\n", results[0])
else:
    print("No valid poses found.")



import json
import matplotlib.pyplot as plt

# --- Visualization Function ---
def visualize_pose(rgb, rvec, tvec, model_points, K):
    projected, _ = cv2.projectPoints(model_points, rvec, tvec, K, None)
    projected = projected.squeeze().astype(np.int32)

    vis_img = rgb.copy()
    for pt in projected:
        cv2.circle(vis_img, tuple(pt), 3, (0, 255, 0), -1)

    plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
    plt.title("Projected 3D model over image")
    plt.axis('off')
    plt.show()

# --- Visualize First Result ---
if results:
    first_result = results[0]
    image_path = os.path.join(train_dir, first_result["image"] + ".png")
    if not os.path.exists(image_path):
        image_path = image_path.replace(".png", ".jpg")  # fallback if jpg
    rgb = cv2.imread(image_path)
    rvec = np.array(first_result["rvec"]).reshape(3, 1)
    tvec = np.array(first_result["tvec"]).reshape(3, 1)
    visualize_pose(rgb, rvec, tvec, model_keypoints, K)

# --- Save all poses to JSON ---
with open("6d_pose_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"Saved {len(results)} poses to 6d_pose_results.json")
