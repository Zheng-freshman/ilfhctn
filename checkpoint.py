import os
import torch
from torch.hub import load_state_dict_from_url

checkpoint_path = "E:/Github/Semi-supervised-learning/saved_models/usb_hl/semi-semiheal/del2000-197540/latest_model.pth"
checkpoint_path = "https://github.com/microsoft/Semi-supervised-learning/releases/download/v.0.0.0/vit_small_patch16_224_mlp_im_1k_224.pth"
if checkpoint_path and os.path.isfile(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
else:
    checkpoint = load_state_dict_from_url(checkpoint_path, map_location='cpu')
print(checkpoint["model"].keys())
