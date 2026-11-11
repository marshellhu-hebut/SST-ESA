import torch
from .sstesa_grouped_summary import ReceiverGroupedSummaryBlock

def replace_with_compact_grouped_summary(model):
    c = model.config
    if not (c.mode == 'esa' and c.sparse and (c.sparse_k == 8) and (c.sparse_mode == 'typed_budget') and (tuple(c.layer_types) == ('M', 'S')) and (c.edge_dim == 8) and (c.heads == 2) and (not c.edge_attrs) and (c.patching == 'fixed') and (c.readout == 'fcst_flatten') and (c.projection_placement == 'per_edge') and (c.edge_output_norm == 'layernorm') and (not c.encoder_outer_residual)):
        raise ValueError('This experiment requires fixed-patch d8/h2 typed-k8')
    for name in ['block1', 'block2']:
        old = getattr(model, name)
        if old.window != 2:
            raise ValueError('The fixed grouping requires two-patch windows')
        with torch.random.fork_rng(devices=[]):
            new = ReceiverGroupedSummaryBlock(old.input_dim, old.output_dim, old.num_sensors, old.window, old.stride, old.decay, pool=old.pool, mode='esa', edge_dim=8, heads=2, layer_types=('M', 'S'), dropout=c.dropout, sparse_k=8, sparse_mode='typed_budget', sparse=True, window_chunk=old.window_chunk, checkpoint_edges=old.checkpoint_edges, selector_stats_every=old.selector_stats_every)
        new.load_state_dict(old.state_dict(), strict=True)
        setattr(model, name, new)
    return model
