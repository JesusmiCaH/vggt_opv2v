import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from torch.utils.data import DataLoader
from data_utils.opv2v.opv2v_img_dataset import OPV2VImageDataset
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from demo_viser import viser_wrapper
import time
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

datapath = "datasets/tum/rgbd_dataset_freiburg1_desk2/rgb"
img_list = os.listdir(datapath)
ori_image_names = img_list[4:120:14]
print(len(ori_image_names))
image_names = [os.path.join(datapath, img) for img in ori_image_names]

images = load_and_preprocess_images(image_names).to(device)

start_time = time.time()
with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        images = images[None]  # add batch dimension
        # aggregated_tokens_list, ps_idx = model.aggregator(images)
        predictions = model(images)
end_time = time.time()
print(f"Spent time: {end_time - start_time:.4f} seconds")


print("Converting pose encoding to extrinsic and intrinsic matrices...")
extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
predictions["extrinsic"] = extrinsic
predictions["intrinsic"] = intrinsic

print("Processing model outputs...")
for key in predictions.keys():
    if isinstance(predictions[key], torch.Tensor):
        predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension and convert to numpy

# Viser Visualization
print("Starting viser visualization...")
viser_server = viser_wrapper(
    predictions,
    port=8080,
    init_conf_threshold=50,
    use_point_map=False,
    background_mode=False,
    mask_sky=False,
    image_folder="./skymasker_input",
)
print("Visualization complete")