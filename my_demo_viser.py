import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from glob import glob
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
import os
import yaml
from demo_viser import viser_wrapper
import struct
import numpy as np
def pack_rgb(r, g, b):
    """Pack r, g, b into a single float32"""
    rgb_int = (int(r) << 16) | (int(g) << 8) | int(b)
    return struct.unpack('f', struct.pack('I', rgb_int))[0]


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

    # Load and preprocess example images (replace with your own image paths)
    # image_names = glob("airsim_single/rgb/*.png")  # Replace with your image paths
    with open("demo_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    cam_id_list = [0,1,2,3]
    image_names_ori = [f"{idx:06d}_camera{cam_idx}.png" for idx in config['frame_idx'] for cam_idx in cam_id_list]

    image_names = [os.path.join(config["data_path"], config["current_time"], str(config["ego_id"]), img_name) for img_name in image_names_ori]
    image_names += [os.path.join(config["data_path"], config["current_time"], str(config["contributer_id"]), img_name) for img_name in image_names_ori]
    
    os.makedirs("./skymasker_input", exist_ok=True)
    print(image_names)
    for img_path in image_names:
        path_parts = img_path.split('/')
        dest_path = os.path.join("./skymasker_input", f"{path_parts[-2]}_{path_parts[-1]}")
        if not os.path.exists(dest_path):
            with open(img_path, "rb") as src, open(dest_path, "wb") as dst:
                dst.write(src.read())
    
    images = load_and_preprocess_images(image_names).to(device)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)
            # images = images[None]  # add batch dimension
            # aggregated_tokens_list, ps_idx = model.aggregator(images)

    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    print("Processing model outputs...")
    for key in predictions.keys():
        print(key)
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension and convert to numpy
    

    # Camera Alignment
    print(predictions["extrinsic"].shape)

    # Output Point3D
    print("Making PCD file")

    points_3d = unproject_depth_map_to_point_map(predictions["depth"], predictions["extrinsic"], predictions["intrinsic"])

    points_rgb = (images.cpu().numpy() * 255).astype(np.uint8)
    points_rgb = points_rgb.transpose(0, 2, 3, 1)

    points_3d = points_3d.reshape(-1, 3)
    points_rgb = points_rgb.reshape(-1, 3)

    conf_thres = 3.0
    conf_mask = (predictions["depth_conf"] >= conf_thres).reshape(-1)
    points_3d = points_3d[conf_mask]*100
    points_rgb = points_rgb[conf_mask]

    print(points_3d.shape, points_rgb.shape)
    
    with open('output.pcd', 'w') as f:
        # Header
        f.write("""# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z rgb
SIZE 4 4 4 4
TYPE F F F F
COUNT 1 1 1 1
WIDTH {}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {}
DATA ascii
""".format(len(points_3d), len(points_3d)))
        # Data
        for pt, color in zip(points_3d, points_rgb):
            rgb_float = pack_rgb(*color)
            f.write(f"{pt[0]} {pt[1]} {pt[2]} {rgb_float}\n")

    # Viser Visualization
    print("Starting viser visualization...")
    viser_server = viser_wrapper(
        predictions,
        port=8080,
        init_conf_threshold=50,
        use_point_map=False,
        background_mode=False,
        mask_sky=True,
        image_folder="./skymasker_input",
    )
    print("Visualization complete")