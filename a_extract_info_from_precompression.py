import random
import torch
from b_pattern_pruning_CPU import weight_padding, weight_depadding


#extract original precompression model's whole pattern for every layer
def extract_original_layers_whole_pattern(model,device):
    original_whole_pattern = {}
    for name,weight in model.named_parameters():
        if ('in_proj_weight' in name) or ('out_proj.weight' in name) \
                or ('linear1.weight' in name) or ('linear2.weight' in name) \
                or ('encoder.weight' in name) or ('decoder.weight' in name):
            weight_replica = weight.detach().cpu()
            whole_pattern = torch.ones(weight_replica.shape[0],weight_replica.shape[1], dtype=torch.int)
            zero_location = (weight_replica == 0.)
            whole_pattern[zero_location] = 0
            whole_pattern.to(device)
            original_whole_pattern[name] = whole_pattern
    return original_whole_pattern