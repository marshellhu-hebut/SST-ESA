import torch
from torch import nn
import torch.nn.functional as F

class CrossAttentionBlock(nn.Module):

    def __init__(self, dim, heads, dropout=0.1):
        super().__init__()
        self.heads, self.dropout = (heads, dropout)
        self.qnorm, self.knorm = (nn.LayerNorm(dim), nn.LayerNorm(dim))
        self.q, self.k, self.v, self.proj = [nn.Linear(dim, dim) for _ in range(4)]
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, query, keys, key_valid=None):
        b, lq, d = query.shape
        lk = keys.size(1)
        h = self.heads
        qn, kn = (self.qnorm(query), self.knorm(keys))
        q = self.q(qn).view(b, lq, h, d // h).transpose(1, 2)
        k = self.k(kn).view(b, lk, h, d // h).transpose(1, 2)
        v = self.v(kn).view(b, lk, h, d // h).transpose(1, 2)
        allow = key_valid[:, None, None, :] if key_valid is not None else None
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=allow, dropout_p=self.dropout if self.training else 0.0)
        out = out.transpose(1, 2).reshape(b, lq, d)
        query = query + self.proj(out)
        return query + self.ffn(self.norm(query))

class InducedEdgeAttention(nn.Module):

    def __init__(self, dim, heads=4, inducing=16, dropout=0.1):
        super().__init__()
        self.inducing = nn.Parameter(torch.empty(1, inducing, dim))
        nn.init.xavier_uniform_(self.inducing)
        self.read = CrossAttentionBlock(dim, heads, dropout)
        self.write = CrossAttentionBlock(dim, heads, dropout)

    def forward(self, x, allow=None, valid=None):
        if x.size(1) == 0:
            return x
        summaries = self.read(self.inducing.expand(x.size(0), -1, -1), x, valid)
        out = self.write(x, summaries)
        return out if valid is None else out * valid.unsqueeze(-1)

class BinaryGaussianPatching(nn.Module):

    def __init__(self, patches, raw_length, channels):
        super().__init__()
        self.patches, self.raw_length = (patches, raw_length)
        self.channel_proj = nn.Conv1d(channels, channels, 3, padding=1)
        self.temp_attn = nn.Sequential(nn.Conv1d(channels, 16, 5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(16, patches * 2))
        with torch.no_grad():
            self.temp_attn[-1].bias[:patches].copy_(torch.linspace(-2, 2, patches))
        self.threshold = nn.Parameter(torch.tensor(0.3))
        self.register_buffer('pos', torch.linspace(0, 1, raw_length))

    def forward(self, x):
        params = self.temp_attn(self.channel_proj(x))
        mu_raw, sigma_raw = params.chunk(2, dim=1)
        mu = mu_raw.sigmoid()
        sigma = 0.1 + 0.1 * sigma_raw.exp()
        weights = torch.exp(-0.5 * ((self.pos[None, None] - mu[:, :, None]) / sigma[:, :, None].clamp(min=0.0001)) ** 2)
        hard = (weights > self.threshold).to(weights.dtype)
        soft = torch.sigmoid(100 * (weights - self.threshold))
        mask = soft + (hard - soft).detach()
        patches = x[:, None] * mask[:, :, None, :]
        indices = torch.arange(self.raw_length, device=x.device)
        count = hard.sum(-1)
        lo = torch.where(hard.bool(), indices, self.raw_length).amin(-1)
        hi = torch.where(hard.bool(), indices, -1).amax(-1)
        lo = torch.where(count > 0, lo, -1)
        metadata = {'centers': mu * (self.raw_length - 1), 'sigma_cycles': sigma * (self.raw_length - 1), 'coverage_start': lo, 'coverage_end': hi, 'coverage_count': count, 'mask': hard, 'slot_order': 'unsorted learned slots; no automatic reordering'}
        return (patches, metadata)

class OfficialHardPatchSelection(nn.Module):

    def __init__(self, d=8):
        super().__init__()
        self.feature_extractor = nn.Sequential(nn.Conv2d(1, d, (3, 5), padding=(1, 2)), nn.ReLU(), nn.MaxPool2d((2, 1)), nn.Conv2d(d, 2 * d, (3, 3), padding=1), nn.ReLU(), nn.AdaptiveMaxPool2d(1))
        self.query = nn.Parameter(torch.randn(2 * d) * 0.1)
        self.key_proj = nn.Linear(2 * d, 2 * d)
        self.threshold_generator = nn.Sequential(nn.Linear(2 * d, 1), nn.Tanh())
        self.decay_generator = nn.Sequential(nn.Linear(2 * d, 1), nn.Sigmoid())
        self.saliency_scale = nn.Parameter(torch.tensor(1.0))
        self.threshold_base = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        b, w, c, l = x.shape
        features = self.feature_extractor(x.reshape(b * w, 1, c, l)).reshape(b, w, -1)
        saliency = torch.einsum('bwd,d->bw', self.key_proj(features), self.query) * self.saliency_scale
        global_features = features.mean(1)
        threshold = self.threshold_base + self.threshold_generator(global_features).squeeze(1)
        decay = self.decay_generator(global_features).squeeze(1)
        above = (saliency - saliency.mean(1, keepdim=True) > threshold[:, None]).to(x.dtype)
        weights = above + (1 - above) * decay[:, None]
        return x * weights[:, :, None, None]
