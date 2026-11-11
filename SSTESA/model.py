from pathlib import Path
import random
import numpy as np
import torch
from models.sstesa_v2 import SSTESAConfig, SSTESARUL
from models.sstesa_grouped_summary_compact import replace_with_compact_grouped_summary
ARCHITECTURES = {'FD001': dict(input_patch=5, patch_count=6, conv_out=7, cnn_out=32, hidden_dim=8), 'FD002': dict(input_patch=5, patch_count=10, conv_out=7, cnn_out=12, hidden_dim=8), 'FD003': dict(input_patch=2, patch_count=25, conv_out=4, cnn_out=6, hidden_dim=24), 'FD004': dict(input_patch=5, patch_count=10, conv_out=7, cnn_out=6, hidden_dim=8)}
FIXED_SEEDS = {'FD001': 0, 'FD002': 1, 'FD003': 0, 'FD004': 1528475823}
SPLIT_SEED = 20261027

def seed_everything(dataset):
    value = FIXED_SEEDS[dataset]
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False

def build_model(dataset='FD001', variant='compact_esa', checkpoint=None, device='cpu'):
    if dataset not in ARCHITECTURES:
        raise ValueError('Unknown dataset')
    if variant not in ('compact_esa', 'grouped_summary'):
        raise ValueError('Unknown variant')
    if variant == 'grouped_summary' and dataset != 'FD001':
        raise ValueError('Unsupported dataset for grouped_summary')
    architecture = dict(ARCHITECTURES[dataset], num_sensors=14, cnn_hidden=8, conv_kernel=2, moving_window=[2, 2], stride=[1, 2], decay=0.7)
    seed_everything(dataset)
    reference = SSTESARUL(**architecture, config=SSTESAConfig(mode='mpnn', selector_stats_every=0, window_chunk=512, checkpoint_edges=False))
    seed_everything(dataset)
    config = SSTESAConfig(mode='esa', sparse=True, sparse_k=8, sparse_mode='typed_budget', layer_types=('M', 'S'), edge_dim=8, heads=2, dropout=0.1, edge_attrs=False, readout='fcst_flatten', window_chunk=512, checkpoint_edges=False, selector_stats_every=0)
    model = SSTESARUL(**architecture, config=config)
    own = model.state_dict()
    model.load_state_dict({key: value for key, value in reference.state_dict().items() if key in own and own[key].shape == value.shape}, strict=False)
    del reference
    if dataset == 'FD003':
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261111 + FIXED_SEEDS[dataset])
            model.head = torch.nn.Sequential(torch.nn.Linear(12096, 12), torch.nn.ReLU(), torch.nn.Linear(12, 12), torch.nn.ReLU(), torch.nn.Linear(12, 6), torch.nn.ReLU(), torch.nn.Linear(6, 1))
    if variant == 'grouped_summary':
        model = replace_with_compact_grouped_summary(model)
    if checkpoint is not None:
        data = torch.load(Path(checkpoint), map_location='cpu', weights_only=True)
        if 'variant' in data and data['variant'] != variant:
            raise ValueError('Checkpoint variant mismatch')
        if 'dataset' in data and data['dataset'] != dataset:
            raise ValueError('Checkpoint dataset mismatch')
        identity = data.get('model_id', '')
        if 'grouped' in identity and variant != 'grouped_summary':
            raise ValueError('Checkpoint variant mismatch')
        model.load_state_dict(data.get('model', data), strict=True)
    return model.to(device)
