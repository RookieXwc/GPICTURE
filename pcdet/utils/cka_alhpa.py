import numpy as np
import torch
import os
import math
import random
import argparse
import pickle
from pathlib import Path
from pcdet.models import build_network
from pcdet.datasets import build_dataloader
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg

args, cfg = parse_config()

def linear_cka(X, Y):
    # Ensure the data is centered (centering operation)
    X = X - np.mean(X, axis=0)
    Y = Y - np.mean(Y, axis=0)

    # Compute the Gram matrices
    K = np.dot(X, X.T)
    L = np.dot(Y, Y.T)

    # Compute the Frobenius norms
    norm_K = np.linalg.norm(K, 'fro')
    norm_L = np.linalg.norm(L, 'fro')

    # Calculate Linear CKA
    cka = np.dot(K.flatten(), L.flatten()) / (norm_K * norm_L)
    
    return cka

def compute_ckas(weights_path_1, weights_path_2):
    # Initialize two models
    model_t_1 = build_network(model_cfg=cfg.MODEL)
    model_t_2 = build_network(model_cfg=cfg.MODEL)

    model_t_1 = model_t_1.load_state_dict(torch.load(weights_path_1))
    model_t_2 = model_t_2.load_state_dict(torch.load(weights_path_2))

    test_set, test_loader, sampler = build_dataloader(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            batch_size=1,
            training=False
    )
    test_sample = test_loader[random.randint(0, len(test_loader))]

    output_batch_dict_t_1 = model_t_1(test_sample)
    output_batch_dict_t_2 = model_t_2(test_sample)

    # Calculate the CKA of all encoder layers
    CKAs = []
    for i in range(8):
        activation_t_1 = output_batch_dict_t_1[f'block_layer_{i}']
        activation_t_2 = output_batch_dict_t_2[f'block_layer_{i}']
        CKA = linear_cka(activation_t_1, activation_t_2)
        CKAs.append(CKA)
    
    return CKAs

weights_path_ssl = 'path/to/ssl/weights'
weights_path_sl_object_detection = 'path/to/sl/od/weights'
weights_path_sl_semantic_segmentation = 'path/to/sl/ss/weights'
weights_path_sl_occupancy_prediction = 'path/to/sl/op/weights'

CKAs_ssl_od = compute_ckas(weights_path_ssl, weights_path_sl_object_detection)
CKAs_ssl_ss = compute_ckas(weights_path_ssl, weights_path_sl_semantic_segmentation)
CKAs_ssl_op = compute_ckas(weights_path_ssl, weights_path_sl_occupancy_prediction)

cka_alhpa = [(a + b + c) / 3 for a, b, c in zip(CKAs_ssl_od, CKAs_ssl_ss, CKAs_ssl_op)]

with open('cka_alhpa.pkl', 'wb') as file:
    pickle.dump(cka_alhpa, file)