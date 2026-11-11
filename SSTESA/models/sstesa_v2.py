from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from .sstesa_extensions import BinaryGaussianPatching, OfficialHardPatchSelection, InducedEdgeAttention

class SafeBatchNorm1d(nn.BatchNorm1d):

    def forward(self, x):
        if self.training and x.numel() // x.size(1) == 1:
            return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, training=False, momentum=0.0, eps=self.eps)
        return super().forward(x)

def conv_graph_st(x: Tensor, window: int, stride: int) -> Tensor:
    if x.ndim != 4:
        raise ValueError(f'expected [B,T,N,D], got {tuple(x.shape)}')
    b, t, n, d = x.shape
    if window < 1 or stride < 1 or window > t:
        raise ValueError(f'invalid window={window}, stride={stride}, T={t}')
    j = (t - window) // stride + 1
    return torch.stack([x[:, start:start + window] for start in range(0, j * stride, stride)], dim=1)

def decay_relation(num_sensors: int, window: int, decay: float, *, device=None, dtype=None) -> Tensor:
    if not 0 < decay <= 1:
        raise ValueError('decay must be in (0, 1]')
    t = torch.arange(window, device=device)
    lag = (t[:, None] - t[None, :]).abs()
    relation_t = torch.pow(torch.as_tensor(decay, device=device, dtype=dtype or torch.get_default_dtype()), lag)
    return relation_t.repeat_interleave(num_sensors, 0).repeat_interleave(num_sensors, 1)

class DotGraphConstruction(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.mapping = nn.Linear(dim, dim)

    def forward(self, node_features: Tensor) -> Tensor:
        if node_features.ndim != 3:
            raise ValueError('node_features must be [batch,nodes,dim]')
        h = self.mapping(node_features)
        logits = h @ h.transpose(1, 2)
        original_dtype = logits.dtype
        if original_dtype == torch.float16:
            logits = logits.float()
        n = h.size(1)
        eye = torch.eye(n, dtype=logits.dtype, device=h.device).unsqueeze(0)
        logits = F.leaky_relu(logits - eye * 100000000.0)
        return (F.softmax(logits, dim=-1) + eye).to(original_dtype)

class MPNNPropagation(nn.Module):

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.theta = nn.Linear(input_dim, output_dim)
        self.bn = SafeBatchNorm1d(output_dim)

    def forward(self, x: Tensor, a: Tensor) -> Tensor:
        y = self.theta(a @ x)
        y = self.bn(y.transpose(1, 2)).transpose(1, 2)
        return F.leaky_relu(y)

def edge_intersection_mask(src: Tensor, dst: Tensor) -> Tensor:
    return (src[..., :, None] == src[..., None, :]) | (src[..., :, None] == dst[..., None, :]) | (dst[..., :, None] == src[..., None, :]) | (dst[..., :, None] == dst[..., None, :])

def edge_tokenize(nodes, src, dst, weights, embedding, attrs=None):
    d = nodes.size(-1)
    hu = nodes.gather(1, src.unsqueeze(-1).expand(-1, -1, d))
    hv = nodes.gather(1, dst.unsqueeze(-1).expand(-1, -1, d))
    pieces = [hu, hv, weights.unsqueeze(-1)]
    if attrs is not None:
        pieces.append(attrs)
    return embedding(torch.cat(pieces, dim=-1))

def weighted_edge_to_node(z, weights, dst, num_nodes, projection, valid=None, projection_placement='per_edge'):
    if projection_placement not in {'per_edge', 'post_aggregate'}:
        raise ValueError('unknown projection_placement')
    values = (projection(z) if projection_placement == 'per_edge' else z) * weights.unsqueeze(-1)
    if valid is not None:
        values = values * valid.unsqueeze(-1)
    out = values.new_zeros(values.size(0), num_nodes, values.size(-1))
    out = out.scatter_add(1, dst.unsqueeze(-1).expand_as(values), values)
    return projection(out) if projection_placement == 'post_aggregate' else out

class EdgeAttentionBlock(nn.Module):

    def __init__(self, dim: int, heads: int=4, dropout: float=0.1, expansion: int=2):
        super().__init__()
        if dim % heads:
            raise ValueError(f'edge dim {dim} must be divisible by heads {heads}')
        self.dim, self.heads, self.dropout = (dim, heads, float(dropout))
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        hidden = expansion * dim
        self.ffn = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: Tensor, allow: Optional[Tensor]=None, valid: Optional[Tensor]=None) -> Tensor:
        b, e, d = x.shape
        if e == 0:
            return x
        h = self.norm1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        dh = d // self.heads
        q = q.view(b, e, self.heads, dh).transpose(1, 2)
        k = k.view(b, e, self.heads, dh).transpose(1, 2)
        v = v.view(b, e, self.heads, dh).transpose(1, 2)
        if allow is None and valid is not None:
            allow = valid[:, :, None] & valid[:, None, :]
        if allow is not None:
            allow = allow.to(device=x.device)
            if allow.ndim == 2:
                allow = allow.unsqueeze(0)
            if allow.size(0) == 1 and b != 1:
                allow = allow.expand(b, -1, -1)
            if valid is not None:
                if allow.dtype != torch.bool:
                    raise ValueError('padding with additive bias requires explicit masking')
                vv = valid[:, :, None] & valid[:, None, :]
                allow = allow & vv
                eye = torch.eye(e, device=x.device, dtype=torch.bool).unsqueeze(0)
                allow = allow | eye & valid[:, :, None]
            allow = allow.unsqueeze(1)
        p = self.dropout if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=allow, dropout_p=p)
        y = y.transpose(1, 2).reshape(b, e, d)
        x = x + self.proj(y)
        x = x + self.ffn(self.norm2(x))
        if valid is not None:
            x = x * valid.unsqueeze(-1).to(x.dtype)
        return x

class EdgeSetEncoder(nn.Module):

    def __init__(self, dim: int, heads: int=4, layers: Sequence[str]=('M', 'S'), dropout: float=0.1, expansion: int=2, inducing: int=16, edge_output_norm: str='layernorm', encoder_outer_residual: bool=False):
        super().__init__()
        if not layers or any((layer not in {'M', 'S', 'I'} for layer in layers)):
            raise ValueError('layers must contain M, S and/or I')
        self.layer_types = tuple(layers)
        self.blocks = nn.ModuleList([InducedEdgeAttention(dim, heads, inducing, dropout) if typ == 'I' else EdgeAttentionBlock(dim, heads, dropout, expansion) for typ in layers])
        if edge_output_norm not in {'layernorm', 'identity'}:
            raise ValueError('edge_output_norm must be layernorm or identity')
        self.out_norm = nn.LayerNorm(dim) if edge_output_norm == 'layernorm' else nn.Identity()
        self.encoder_outer_residual = encoder_outer_residual

    def forward(self, edge_tokens: Tensor, src: Tensor, dst: Tensor, valid: Optional[Tensor]=None) -> Tensor:
        if src.ndim == 1:
            src = src.unsqueeze(0).expand(edge_tokens.size(0), -1)
            dst = dst.unsqueeze(0).expand(edge_tokens.size(0), -1)
        m = edge_intersection_mask(src, dst) if 'M' in self.layer_types else None
        x = edge_tokens
        for block, typ in zip(self.blocks, self.layer_types):
            x = block(x, m if typ == 'M' else None, valid)
        if self.encoder_outer_residual:
            x = x + edge_tokens
        x = self.out_norm(x)
        if valid is not None:
            x = x * valid.unsqueeze(-1).to(x.dtype)
        return x

class PerEdgeResidualMLP(nn.Module):

    def __init__(self, dim: int, hidden: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.up = nn.Linear(dim, hidden)
        self.down = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, valid: Optional[Tensor]=None) -> Tensor:
        update = self.down(self.drop(F.gelu(self.up(self.norm(x)))))
        out = x + self.drop(update)
        if valid is not None:
            out = out * valid.unsqueeze(-1).to(out.dtype)
        return out

class _EdgeSelector:

    def __init__(self, num_sensors: int, window: int, k: int, mode: str):
        if mode not in {'topk_all', 'typed_budget'}:
            raise ValueError('sparse mode must be topk_all or typed_budget')
        self.n, self.w, self.k, self.mode = (num_sensors, window, max(0, int(k)), mode)

    def __call__(self, a: Tensor, collect_stats: bool=True):
        b, m, _ = a.shape
        idx = torch.arange(m, device=a.device)
        times, sensors = (idx // self.n, idx % self.n)
        eye = torch.eye(m, device=a.device, dtype=torch.bool)
        nonself = ~eye
        same_time = (times[:, None] == times[None, :]) & nonself
        cross = (times[:, None] != times[None, :]) & (sensors[:, None] != sensors[None, :])
        chain = (times[:, None] == times[None, :] + 1) & (sensors[:, None] == sensors[None, :])
        mandatory = eye | chain
        learned = torch.zeros_like(a, dtype=torch.bool)
        k = min(self.k, m - 1)

        def select(pool, number):
            if number:
                ranking = a.masked_fill(~pool, -torch.inf)
                top = ranking.topk(number, dim=-1).indices
                learned.scatter_(2, top, True)
        if self.mode == 'topk_all':
            select(nonself, k)
        else:
            ks = min(k // 2, self.n - 1)
            kc = min(k - k // 2, (self.w - 1) * (self.n - 1))
            select(same_time, ks)
            select(cross, kc)
            select(nonself & ~learned, k - ks - kc)
        selected = learned | mandatory
        flat = selected.flatten(1)
        lengths = flat.sum(-1)
        if self.mode == 'typed_budget' and ks + kc == k:
            e = m * k + m + (self.w - 1) * self.n
        else:
            e = int(lengths.max())
        positions = torch.arange(m * m, device=a.device).expand(b, -1)
        packed = positions.masked_fill(~flat, m * m).sort(dim=-1).values[:, :e]
        valid = torch.arange(e, device=a.device)[None] < lengths[:, None]
        packed = packed.clamp_max(m * m - 1)
        src, dst = (packed % m, packed // m)
        weights = a.flatten(1).gather(1, packed) * valid
        if not collect_stats:
            return (src, dst, weights, valid, {})
        incoming, outgoing = (selected.sum(-1), selected.sum(-2))
        incident = incoming + outgoing - selected.diagonal(dim1=-2, dim2=-1).long()
        overlap = selected.to(torch.int32) + selected.transpose(-1, -2).to(torch.int32)
        degree = incident.gather(1, src) + incident.gather(1, dst) - overlap.flatten(1).gather(1, packed)
        degree = torch.where(src == dst, incident.gather(1, src), degree)
        selected_total = lengths.sum()
        stats = {'mean_edges_per_window': lengths.float().mean(), 'min_edges_per_window': lengths.min(), 'max_edges_per_window': lengths.max(), 'mandatory_edges_per_window': mandatory.sum(), 'forced_extra_edges_per_window': (mandatory & ~learned).sum((1, 2)).float().mean(), 'cross_sensor_cross_time_fraction': (selected & cross).sum() / selected_total, 'max_node_indegree': incoming.max(), 'max_node_outdegree': outgoing.max(), 'mean_node_indegree': incoming.float().mean(), 'mean_node_outdegree': outgoing.float().mean(), 'mean_M_neighborhood': (degree * valid).sum() / selected_total, 'max_M_neighborhood': (degree * valid).max(), 'M_allowed_pairs_mean_per_window': (degree * valid).sum(-1).float().mean()}
        values = torch.stack([v.float() for v in stats.values()]).detach().cpu().tolist()
        return (src, dst, weights, valid, dict(zip(stats, values)))

class EdgeWindowBlock(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, num_sensors: int, window: int, stride: int, decay: float, pool: str='mean', mode: str='esa', edge_dim: int=32, heads: int=4, layer_types: Sequence[str]=('M', 'S'), dropout: float=0.1, sparse_k: int=0, sparse_mode: str='topk_all', edge_attrs: bool=False, sparse: bool=False, window_chunk: int=16, checkpoint_edges: bool=True, inducing: int=16, projection_placement: str='per_edge', edge_output_norm: str='layernorm', encoder_outer_residual: bool=False, selector_stats_every: int=1, edge_mlp_hidden: int=128):
        super().__init__()
        if mode not in {'mpnn', 'esa', 'edge_mlp', 'nsa'}:
            raise ValueError('mode must be mpnn, esa, edge_mlp or nsa')
        if pool not in {'mean', 'max'}:
            raise ValueError('pool must be mean or max')
        self.input_dim, self.output_dim = (input_dim, output_dim)
        self.num_sensors, self.window, self.stride = (num_sensors, window, stride)
        self.decay, self.pool, self.mode = (float(decay), pool, mode)
        self.edge_attrs = edge_attrs
        if projection_placement not in {'per_edge', 'post_aggregate'}:
            raise ValueError('unknown projection_placement')
        if selector_stats_every < 0:
            raise ValueError('selector_stats_every must be nonnegative')
        self.projection_placement = projection_placement
        self.selector_stats_every = selector_stats_every
        self._selector_calls = 0
        self.window_chunk, self.checkpoint_edges = (window_chunk, checkpoint_edges)
        self.bn_in = SafeBatchNorm1d(input_dim)
        self.graph = DotGraphConstruction(input_dim)
        self.mpnn = MPNNPropagation(input_dim, output_dim) if mode == 'mpnn' else None
        attr_dim = 4 if edge_attrs else 0
        self.edge_embed = nn.Sequential(nn.Linear(2 * input_dim + 1 + attr_dim, edge_dim), nn.GELU(), nn.Linear(edge_dim, edge_dim)) if mode in {'esa', 'edge_mlp'} else None
        self.edge_encoder = EdgeSetEncoder(edge_dim, heads, layer_types, dropout, inducing=inducing, edge_output_norm=edge_output_norm, encoder_outer_residual=encoder_outer_residual) if mode == 'esa' else None
        self.edge_mlp = nn.ModuleList([PerEdgeResidualMLP(edge_dim, edge_mlp_hidden, dropout) for _ in layer_types]) if mode == 'edge_mlp' else None
        self.edge_to_node = nn.Linear(edge_dim, output_dim) if mode in {'esa', 'edge_mlp'} else None
        self.node_bn = SafeBatchNorm1d(output_dim) if mode in {'esa', 'edge_mlp', 'nsa'} else None
        if mode == 'nsa':
            if any((typ not in {'M', 'S'} for typ in layer_types)):
                raise ValueError('NSA control supports M/S; M+I is the explicit edge variant')
            self.nsa_input = nn.Linear(input_dim, edge_dim)
            self.nsa_layers = nn.ModuleList([EdgeAttentionBlock(edge_dim, heads, dropout) for _ in layer_types])
            self.nsa_types = tuple(layer_types)
            self.nsa_output = nn.Linear(edge_dim, output_dim)
        self.register_buffer('relation', decay_relation(num_sensors, window, decay), persistent=False)
        self.selector = _EdgeSelector(num_sensors, window, sparse_k, sparse_mode) if sparse or sparse_k > 0 else None
        self.last_stats: Dict[str, float] = {}

    def _attrs(self, src: Tensor, dst: Tensor, *, dtype: torch.dtype) -> Tensor:
        ts, ss = (src.div(self.num_sensors, rounding_mode='floor'), src.remainder(self.num_sensors))
        td, sd = (dst.div(self.num_sensors, rounding_mode='floor'), dst.remainder(self.num_sensors))
        lag = (td - ts).to(dtype) / max(1, self.window - 1)
        same_s = (ss == sd).to(dtype)
        same_t = (ts == td).to(dtype)
        rel_type = (same_s * 1.0 + same_t * 2.0) / 3.0
        return torch.stack((lag, same_s, same_t, rel_type), dim=-1)

    def forward(self, x: Tensor, patch_centers: Optional[Tensor]=None, reference_patch_length: float=1.0) -> Tensor:
        windows = conv_graph_st(x, self.window, self.stride)
        b, j, w, n, d = windows.shape
        if n != self.num_sensors:
            raise ValueError(f'configured N={self.num_sensors}, input N={n}')
        m = w * n
        nodes = windows.permute(0, 1, 2, 3, 4).reshape(b * j, m, d)
        relation = self.relation.to(device=x.device, dtype=nodes.dtype)
        if patch_centers is not None:
            centers = torch.stack([patch_centers[:, start:start + w] for start in range(0, j * self.stride, self.stride)], dim=1)
            centers = centers[:, :, :, None].expand(-1, -1, -1, n).reshape(b * j, m)
            distances = (centers[:, :, None] - centers[:, None, :]).abs() / reference_patch_length
            relation = self.decay ** distances
        a = self.graph(nodes) * relation
        x_bn = self.bn_in(nodes.transpose(1, 2)).transpose(1, 2)
        collect_stats = self.selector_stats_every > 0 and self._selector_calls % max(1, self.selector_stats_every) == 0
        self._selector_calls += 1
        if self.mode == 'mpnn':
            stats = {}
            if self.selector is not None:
                src, dst, weight, valid, stats = self.selector(a, collect_stats)
                a = a.new_zeros(b * j, m * m).scatter_add(1, dst * m + src, weight).view(b * j, m, m)
            out = self.mpnn(x_bn, a)
            self.last_stats = {'nodes': float(m), 'edges': stats.get('mean_edges_per_window', float(src.size(1)) if self.selector is not None else float(m * m)), 'mode': 'sparse_mpnn' if self.selector is not None else 'mpnn', **stats}
            if self.selector is not None:
                self.last_stats['packed_edges'] = src.size(1)
        elif self.mode == 'nsa':
            stats = {}
            if self.selector is not None:
                src, dst, weight, valid, stats = self.selector(a, collect_stats)
                a = a.new_zeros(b * j, m * m).scatter_add(1, dst * m + src, weight).view(b * j, m, m)
            h = self.nsa_input(x_bn)
            bias = torch.where(a > 0, a.clamp_min(1e-30).log(), -torch.inf)
            for layer, typ in zip(self.nsa_layers, self.nsa_types):
                h = layer(h, bias if typ == 'M' else None)
            out = self.nsa_output(h)
            out = F.leaky_relu(self.node_bn(out.transpose(1, 2)).transpose(1, 2))
            self.last_stats = {'nodes': float(m), 'mode': 'weighted_nsa', 'node_score_pairs': float(m * m), **stats}
        else:
            if self.selector is None:
                src = torch.arange(m, device=x.device).repeat_interleave(m).view(1, -1).expand(b * j, -1)
                dst = torch.arange(m, device=x.device).repeat(m).view(1, -1).expand(b * j, -1)
                weights = a[torch.arange(b * j, device=x.device)[:, None], dst, src]
                valid = None
                self.last_stats = {'nodes': float(m), 'edges': float(m * m), 'interaction_pairs': float((m * m) ** 2), 'mode': 'esa_full', 'M_allowed_pairs': float(m * (2 * m - 1) + m * (m - 1) * (4 * m - 4)), 'node_indegree': float(m), 'node_outdegree': float(m), 'cross_sensor_cross_time_fraction': float((w - 1) * (n - 1) / m)}
                stats = {'mean_edges_per_window': float(src.size(1)), 'interaction_pairs': float(src.size(1) ** 2), 'M_allowed_pairs': float(m * (2 * m - 1) + m * (m - 1) * (4 * m - 4)), 'node_indegree': float(m), 'node_outdegree': float(m), 'cross_sensor_cross_time_fraction': float((w - 1) * (n - 1) / m)}
            else:
                src, dst, weights, valid, stats = self.selector(a, collect_stats)
            self.last_stats = {'nodes': float(m), 'edges': stats.get('mean_edges_per_window', float(src.size(1))), 'interaction_pairs': float(src.size(1) ** 2), 'mode': 'edge_mlp_sparse' if self.mode == 'edge_mlp' else 'esa_sparse', **stats}
            chunks = []
            for start in range(0, b * j, self.window_chunk):
                end = min(start + self.window_chunk, b * j)
                valid_c = valid[start:end] if valid is not None else None
                args = (x_bn[start:end], src[start:end], dst[start:end], weights[start:end], valid_c)
                if self.checkpoint_edges and self.training and torch.is_grad_enabled():
                    chunks.append(checkpoint(self._edge_chunk, *args, use_reentrant=False))
                else:
                    chunks.append(self._edge_chunk(*args))
            out = torch.cat(chunks, dim=0)
            out = self.node_bn(out.transpose(1, 2)).transpose(1, 2)
            out = F.leaky_relu(out)
        out = out.reshape(b, j, w, n, self.output_dim)
        out = out.mean(dim=2) if self.pool == 'mean' else out.max(dim=2).values
        return out

    def _edge_chunk(self, nodes, src, dst, weights, valid):
        b, m, d = nodes.shape
        attrs = self._attrs(src, dst, dtype=nodes.dtype) if self.edge_attrs else nodes.new_zeros(b, src.size(1), 0)
        z = edge_tokenize(nodes, src, dst, weights, self.edge_embed, attrs)
        if self.mode == 'esa':
            z = self.edge_encoder(z, src[:1] if valid is None else src, dst[:1] if valid is None else dst, valid)
        else:
            for layer in self.edge_mlp:
                z = layer(z, valid)
        return weighted_edge_to_node(z, weights, dst, m, self.edge_to_node, valid, self.projection_placement)

class PatchCNN(nn.Module):

    def __init__(self, hidden: int, out_dim: int, kernel: int=2, dropout: float=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(1, hidden, kernel, padding=kernel // 2, bias=False), SafeBatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout), nn.Conv1d(hidden, out_dim, kernel, padding=1, bias=False), SafeBatchNorm1d(out_dim), nn.ReLU())

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x.transpose(-1, -2))

class SinusoidalEncoding(nn.Module):

    def __init__(self, dim: int, max_len: int=5000, dropout: float=0.1):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(100.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:pe[:, 1::2].shape[1]])
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop(x + self.pe[:, :x.size(1)].to(dtype=x.dtype, device=x.device))

@dataclass
class SSTESAConfig:
    mode: str = 'esa'
    sparse: bool = False
    sparse_k: int = 8
    sparse_mode: str = 'topk_all'
    layer_types: Tuple[str, ...] = ('M', 'S')
    edge_dim: int = 32
    heads: int = 4
    dropout: float = 0.1
    edge_attrs: bool = False
    readout: str = 'fcst_flatten'
    window_chunk: int = 16
    checkpoint_edges: bool = True
    inducing: int = 16
    patching: str = 'fixed'
    gap_patches: Optional[int] = None
    patch_selection: str = 'none'
    gap_center_decay: bool = False
    projection_placement: str = 'per_edge'
    edge_output_norm: str = 'layernorm'
    encoder_outer_residual: bool = False
    selector_stats_every: int = 1
    edge_mlp_hidden: int = 128

class SSTESARUL(nn.Module):

    def __init__(self, input_patch: int, num_sensors: int, patch_count: int, conv_out: int, cnn_hidden: int=8, cnn_out: int=32, conv_kernel: int=2, hidden_dim: int=8, moving_window: Sequence[int]=(2, 2), stride: Sequence[int]=(1, 2), decay: float=0.7, n_class: int=1, config: Optional[SSTESAConfig]=None):
        super().__init__()
        self.config = config or SSTESAConfig()
        if self.config.patching not in {'fixed', 'gap'}:
            raise ValueError('patching must be fixed or gap')
        if self.config.patch_selection not in {'none', 'official_hard'}:
            raise ValueError('patch_selection must be none or official_hard')
        if self.config.patching == 'fixed' and (self.config.gap_center_decay or self.config.patch_selection != 'none'):
            raise ValueError('GAP options require patching=gap')
        self.input_patch, self.input_patch_count = (input_patch, patch_count)
        self.raw_length = input_patch * patch_count
        self.last_patch_metadata = None
        if self.config.patching == 'gap':
            output_patch_count = self.config.gap_patches or patch_count
            self.gap = BinaryGaussianPatching(output_patch_count, self.raw_length, num_sensors)
            self.patch_selector = OfficialHardPatchSelection() if self.config.patch_selection == 'official_hard' else None
            conv_out = self.raw_length + 2 * (conv_kernel // 2) + 4 - 2 * conv_kernel
            patch_count = output_patch_count
        elif self.config.gap_patches is not None:
            raise ValueError('gap_patches requires patching=gap')
        self.num_sensors, self.patch_count = (num_sensors, patch_count)
        self.frontend = PatchCNN(cnn_hidden, cnn_out, conv_kernel)
        self.front_proj = nn.Sequential(nn.Linear(cnn_out * conv_out, 2 * hidden_dim), SafeBatchNorm1d(2 * hidden_dim))
        self.pos = SinusoidalEncoding(2 * hidden_dim)
        kwargs = dict(num_sensors=num_sensors, decay=decay, pool='mean', mode=self.config.mode, edge_dim=self.config.edge_dim, heads=self.config.heads, layer_types=self.config.layer_types, dropout=self.config.dropout, sparse_k=self.config.sparse_k if self.config.sparse else 0, sparse_mode=self.config.sparse_mode, edge_attrs=self.config.edge_attrs, sparse=self.config.sparse, window_chunk=self.config.window_chunk, checkpoint_edges=self.config.checkpoint_edges, inducing=self.config.inducing, projection_placement=self.config.projection_placement, edge_output_norm=self.config.edge_output_norm, encoder_outer_residual=self.config.encoder_outer_residual, selector_stats_every=self.config.selector_stats_every, edge_mlp_hidden=self.config.edge_mlp_hidden)
        self.block1 = EdgeWindowBlock(2 * hidden_dim, hidden_dim, window=moving_window[0], stride=stride[0], **kwargs)
        self.block2 = EdgeWindowBlock(2 * hidden_dim, hidden_dim, window=moving_window[1], stride=stride[1], **kwargs)
        self.windows = tuple(((patch_count - w) // s + 1 for w, s in zip(moving_window, stride)))
        self.readout = self.config.readout
        flat_dim = hidden_dim * num_sensors * sum(self.windows)
        if self.readout == 'fcst_flatten':
            self.head = nn.Sequential(nn.Linear(flat_dim, 2 * hidden_dim), nn.ReLU(), nn.Linear(2 * hidden_dim, 2 * hidden_dim), nn.ReLU(), nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_class))
        elif self.readout == 'pma':
            self.branch_embedding = nn.Embedding(2, hidden_dim)
            self.sensor_embedding = nn.Embedding(num_sensors, hidden_dim)
            self.readout_position = SinusoidalEncoding(hidden_dim, dropout=0.0)
            self.query_tokens = nn.Parameter(torch.randn(1, 4, hidden_dim) * 0.02)
            self.pma = nn.MultiheadAttention(hidden_dim, self.config.heads, batch_first=True, dropout=self.config.dropout)
            self.head = nn.Sequential(nn.Linear(4 * hidden_dim, 2 * hidden_dim), nn.ReLU(), nn.Linear(2 * hidden_dim, n_class))
        else:
            raise ValueError('readout must be fcst_flatten or pma')

    def _features(self, x: Tensor) -> Tensor:
        b, t, n, raw = x.shape
        z = self.frontend(x.reshape(b * t * n, raw, 1)).reshape(b * t * n, -1)
        expected = self.front_proj[0].in_features
        if z.size(-1) != expected:
            raise ValueError(f'frontend produced {z.size(-1)} values, expected conv_out*cnn_out={expected}; check input_patch/conv_out')
        z = self.front_proj(z).reshape(b, t, n, -1)
        z = z.transpose(1, 2).reshape(b * n, t, -1)
        z = self.pos(z).reshape(b, n, t, -1).transpose(1, 2).contiguous()
        return z

    def forward(self, x: Tensor, return_dict: bool=False):
        if x.shape[1:3] != (self.input_patch_count, self.num_sensors) or x.size(-1) != self.input_patch:
            raise ValueError(f'expected [B,{self.input_patch_count},{self.num_sensors},{self.input_patch}], got {tuple(x.shape)}')
        centers = None
        if self.config.patching == 'gap':
            b, t, n, p = x.shape
            raw = x.transpose(2, 3).reshape(b, t * p, n).transpose(1, 2)
            x, metadata = self.gap(raw)
            if self.patch_selector is not None:
                x = self.patch_selector(x)
            self.last_patch_metadata = {k: v.detach() if isinstance(v, Tensor) else v for k, v in metadata.items()}
            if self.config.gap_center_decay:
                centers = metadata['centers']
        z = self._features(x)
        y1, y2 = (self.block1(z, centers, self.input_patch), self.block2(z, centers, self.input_patch))
        if self.readout == 'fcst_flatten':
            features = torch.cat((y1.flatten(1), y2.flatten(1)), dim=-1)
            pred = self.head(features)
        else:
            branches = []
            for branch, values in enumerate((y1, y2)):
                b, j, n, d = values.shape
                pos = self.readout_position(values.new_zeros(1, j, d))
                sensor = self.sensor_embedding(torch.arange(n, device=x.device))
                identified = values + pos[:, :, None, :] + sensor[None, None] + self.branch_embedding.weight[branch]
                branches.append(identified.flatten(1, 2))
            tokens = torch.cat(branches, dim=1)
            q = self.query_tokens.expand(tokens.size(0), -1, -1)
            pooled = self.pma(q, tokens, tokens, need_weights=False)[0]
            pred = self.head(pooled.flatten(1))
            features = tokens
        if return_dict:
            return {'pred': pred, 'features': features, 'branch1': y1, 'branch2': y2, 'stats1': self.block1.last_stats, 'stats2': self.block2.last_stats, 'patch_metadata': self.last_patch_metadata}
        return pred

def parameter_count(model: nn.Module) -> int:
    return sum((p.numel() for p in model.parameters() if p.requires_grad))
__all__ = ['SSTESAConfig', 'SSTESARUL', 'EdgeWindowBlock', 'EdgeSetEncoder', 'EdgeAttentionBlock', 'DotGraphConstruction', 'MPNNPropagation', 'conv_graph_st', 'decay_relation', 'edge_intersection_mask', 'parameter_count', 'edge_tokenize', 'weighted_edge_to_node']
