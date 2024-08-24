import MinkowskiEngine as ME
import numpy as np
import torch
import os
from pcdet.models.dense_heads.target_assigner.res16unet import Res16UNet34C as MinkUNet
import pickle


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

input_path = '/path/of/seal/input/'
output_path = '/path/of/seal/output/'

files = os.listdir(input_path)


for file_name in files:
    file_path = input_path + file_name
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    feats = data['feats']
    coords = data['coords']
    # coords = np.concatenate((np.zeros((coords.shape[0], 1), dtype=np.int32), coords), axis=1)
    feats = torch.tensor(feats).to('cuda')
    coords = torch.tensor(coords).to('cuda')
    sparse_input = ME.SparseTensor(feats, coords)
    output_points = model_points(sparse_input).F
    np.save(output_path + file_name.split('.pkl')[0] + '.npy', output_points.detach().cpu().numpy())
    print('processed point clouds: ' + file_name)
