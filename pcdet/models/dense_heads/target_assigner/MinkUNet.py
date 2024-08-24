import MinkowskiEngine as ME
import numpy as np
import torch
import os
from pcdet.models.dense_heads.target_assigner.res16unet import Res16UNet34C as MinkUNet

def load_state_with_same_shape(model, weights):
    """
    Load common weights in two similar models
    (for instance between a pretraining and a downstream training)
    """
    model_state = model.state_dict()
    if list(weights.keys())[0].startswith("model."):
        weights = {k.partition("model.")[2]: weights[k] for k in weights.keys()}

    if list(weights.keys())[0].startswith("model_points."):
        weights = {k.partition("model_points.")[2]: weights[k] for k in weights.keys()}

    if list(weights.keys())[0].startswith("module."):
        print("Loading multigpu weights with module. prefix...")
        weights = {k.partition("module.")[2]: weights[k] for k in weights.keys()}

    if list(weights.keys())[0].startswith("encoder."):
        print("Loading multigpu weights with encoder. prefix...")
        weights = {k.partition("encoder.")[2]: weights[k] for k in weights.keys()}

    filtered_weights = {
        k: v
        for k, v in weights.items()
        if (k in model_state and v.size() == model_state[k].size())
    }
    removed_weights = {
        k: v
        for k, v in weights.items()
        if not (k in model_state and v.size() == model_state[k].size())
    }
    print("Loading weights:" + ", ".join(filtered_weights.keys()))
    print("")
    print("Not loading weights:" + ", ".join(removed_weights.keys()))
    return filtered_weights

me_cfg = {
    'kernel_size': 3,
    'normalize_features': True,
    'model_n_out': 64,
    'bn_momentum': 0.05
}
load_path = '/path/of/the/seal/model.pt'

model_points = MinkUNet(1, me_cfg["model_n_out"], me_cfg)
# load pretrain model
checkpoint = torch.load(load_path, map_location="cpu")
key = "state_dict"
filtered_weights = load_state_with_same_shape(model_points, checkpoint[key])
model_dict = model_points.state_dict()
model_dict.update(filtered_weights)
model_points.load_state_dict(model_dict)

model_points = model_points.to('cuda')
model_points.eval()



current_path = os.path.abspath(__file__)
save_path = os.path.dirname(current_path) + '/'

feats = np.load(save_path + 'feats.npy')
coords = np.load(save_path + 'coords.npy')

feats = torch.tensor(feats).to('cuda')
coords = torch.tensor(coords).to('cuda')

sparse_input = ME.SparseTensor(feats, coords)

output_points = model_points(sparse_input).F

np.save(save_path + 'output_points.npy', output_points.detach().cpu().numpy())

print('MinkUNet processed point clouds!')
