from __future__ import annotations
import torch
from .sstesa_v2 import EdgeWindowBlock, SSTESARUL, edge_tokenize

class ReceiverGroupedSummaryBlock(EdgeWindowBlock):

    def _local_groups(self, z):
        b, e, d = z.shape
        n = self.num_sensors
        if e != n * 19:
            raise ValueError(f'Expected {n * 19} sorted typed-k8 edges, got {e}')
        split = n * 9
        first = self.edge_encoder.blocks[0](z[:, :split].reshape(b * n, 9, d))
        second = self.edge_encoder.blocks[0](z[:, split:].reshape(b * n, 10, d))
        return (first.reshape(b, n, 9, d), second.reshape(b, n, 10, d))

    def _edge_chunk(self, nodes, src, dst, weights, valid):
        b, m, _ = nodes.shape
        if m != 2 * self.num_sensors:
            raise ValueError('This frozen experiment requires two-patch windows')
        z = edge_tokenize(nodes, src, dst, weights, self.edge_embed)
        first, second = self._local_groups(z)
        n = self.num_sensors
        split = n * 9
        summaries = []
        for values, ww in ((first, weights[:, :split].reshape(b, n, 9)), (second, weights[:, split:].reshape(b, n, 10))):
            wf = ww.float()
            summary = (values.float() * wf.unsqueeze(-1)).sum(dim=2)
            summary = summary / wf.sum(dim=2, keepdim=True).clamp_min(1e-08)
            summaries.append(summary.to(z.dtype))
        summary = torch.cat(summaries, dim=1)
        summary = self.edge_encoder.blocks[1](summary)
        summary = self.edge_encoder.out_norm(summary)
        self.last_group_stats = {'edges_per_window': z.size(1), 'receivers_per_window': m, 'local_group_sizes': [9, 10], 'local_groups_each_size': n, 'local_candidate_pairs_per_head': n * (9 ** 2 + 10 ** 2), 'global_summary_tokens': m, 'global_candidate_pairs_per_head': m * m, 'pool': 'graph-weighted mean; FP32 accumulation; no new parameters'}
        return self.edge_to_node(summary)

def replace_with_grouped_summary(model: SSTESARUL) -> SSTESARUL:
    c = model.config
    if not (c.mode == 'esa' and c.sparse and (c.sparse_k == 8) and (c.sparse_mode == 'typed_budget') and (tuple(c.layer_types) == ('M', 'S')) and (c.edge_dim == 32) and (c.heads == 4) and (not c.edge_attrs) and (c.patching == 'fixed') and (c.readout == 'fcst_flatten') and (c.projection_placement == 'per_edge') and (c.edge_output_norm == 'layernorm') and (not c.encoder_outer_residual)):
        raise ValueError('Grouped-summary pilot is frozen to the d32/h4 typed-k8 baseline')
    for name in ('block1', 'block2'):
        old = getattr(model, name)
        if old.window != 2:
            raise ValueError('Grouped-summary pilot requires window=2')
        with torch.random.fork_rng(devices=[]):
            replacement = ReceiverGroupedSummaryBlock(old.input_dim, old.output_dim, old.num_sensors, old.window, old.stride, old.decay, pool=old.pool, mode='esa', edge_dim=32, heads=4, layer_types=('M', 'S'), dropout=c.dropout, sparse_k=8, sparse_mode='typed_budget', sparse=True, window_chunk=old.window_chunk, checkpoint_edges=old.checkpoint_edges, selector_stats_every=old.selector_stats_every)
        replacement.load_state_dict(old.state_dict(), strict=True)
        setattr(model, name, replacement)
    return model
