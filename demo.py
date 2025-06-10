import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from torch.utils.data import DataLoader
from data_utils.opv2v.opv2v_img_dataset import OPV2VImageDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

dataset = OPV2VImageDataset(data_root="/data/opv2v/2021_08_16_22_26_54", agent_id="641", transform=None)
train_loader = DataLoader(dataset, batch_size=10, shuffle=True)

# load the first batch of images from the dataset
for batch in train_loader:
    img_batch = batch['front_image'] + batch['rear_image']
    images = load_and_preprocess_images(img_batch).to(device)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            # Predict attributes including cameras, depth maps, and point maps.
            predictions = model(images)
    print(predictions)
    break