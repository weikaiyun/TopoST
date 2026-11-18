"""Topology encoder and node-specialized forecasting backbone."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, width, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)


class TopologyEncoder(nn.Module):
    """GIN: MLP((1 + epsilon) * h + sum of neighbor features)."""
    def __init__(self, descriptor_dim=16, hidden_dim=64, code_dim=8, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(width, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
                          nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim))
            for width in (descriptor_dim, hidden_dim, hidden_dim)
        ])
        self.register_buffer('epsilon', torch.zeros(3))
        self.readout = nn.Sequential(nn.Linear(hidden_dim, code_dim), nn.LayerNorm(code_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph):
        features = graph.features
        for index, layer in enumerate(self.layers):
            features = layer((1 + self.epsilon[index]) * features + graph.support @ features)
            if index < 2:
                features = self.dropout(F.relu(features))
        return self.readout(features)


class GeneratedLinear(nn.Module):
    """W_i = sum_gamma z[i, gamma] * B[gamma]."""
    def __init__(self, width, code_dim, bias=True):
        super().__init__()
        self.pool = nn.Parameter(torch.empty(code_dim, width, width))
        self.bias_pool = nn.Parameter(torch.empty(code_dim, width)) if bias else None
        nn.init.xavier_uniform_(self.pool)
        if self.bias_pool is not None:
            nn.init.xavier_uniform_(self.bias_pool)

    def forward(self, x, codes):
        weights = torch.einsum('nk,kio->nio', codes, self.pool)
        output = torch.einsum('bpni,nio->bpno', x, weights)
        if self.bias_pool is not None:
            output = output + (codes @ self.bias_pool)[None, None]
        return output


class PatchEmbedding(nn.Module):
    """One-hour non-overlapping patches, padded to twelve measurement slots."""
    def __init__(self, width, reference_steps=288, patch_length=12, channels=1):
        super().__init__()
        self.patch_length = patch_length
        self.num_patches = reference_steps // patch_length
        self.projection = nn.Linear(patch_length * channels, width, bias=False)
        positions = torch.arange(self.num_patches).float()[:, None]
        frequency = torch.exp(torch.arange(0, width, 2).float() * (-math.log(10000) / width))
        encoding = torch.zeros(self.num_patches, width)
        encoding[:, 0::2] = torch.sin(positions * frequency)
        encoding[:, 1::2] = torch.cos(positions * frequency)
        self.register_buffer('positions', encoding)

    def to_patches(self, flow):
        steps = flow.shape[1]
        if steps % self.num_patches or not self.num_patches <= steps <= self.num_patches * self.patch_length:
            raise ValueError('History must contain a whole number of samples per one-hour patch.')
        length = steps // self.num_patches
        patches = flow.permute(0, 2, 3, 1).unfold(-1, length, length)
        patches = F.pad(patches, (0, self.patch_length - length))
        return patches.permute(0, 3, 1, 2, 4).flatten(-2)

    def forward(self, flow, keep=None):
        patches = self.to_patches(flow)
        if keep is not None:
            patches = patches * keep[:, :, None, None]
        return self.projection(patches) + self.positions[None, :, None]


class CalendarEmbedding(nn.Module):
    """Embed minute-of-day and weekday at the start of each temporal patch."""
    def __init__(self, width, num_patches):
        super().__init__()
        self.num_patches = num_patches
        self.minute_embedding = nn.Embedding(1441, width // 2)
        self.weekday_embedding = nn.Embedding(8, width // 2)

    def forward(self, calendar):
        if calendar.ndim != 4 or calendar.shape[-1] != 2:
            raise ValueError('Calendar must have shape [batch, steps, nodes, 2].')
        if calendar.shape[1] < self.num_patches or calendar.shape[1] % self.num_patches:
            raise ValueError('Calendar length must match complete temporal patches.')
        patch_steps = calendar.shape[1] // self.num_patches
        starts = calendar[:, ::patch_steps].long()
        return torch.cat([self.minute_embedding(starts[..., 0]),
                          self.weekday_embedding(starts[..., 1])], dim=-1)


class FeedForward(nn.Module):
    def __init__(self, width, ratio):
        super().__init__()
        self.gate = nn.Linear(width, width * ratio, bias=False)
        self.value = nn.Linear(width, width * ratio, bias=False)
        self.output = nn.Linear(width * ratio, width, bias=False)

    def forward(self, x):
        return self.output(F.silu(self.gate(x)) * self.value(x))


class SpatioTemporalBlock(nn.Module):
    def __init__(self, width, code_dim, heads, mlp_ratio, attention_dropout, dropout, alpha):
        super().__init__()
        if width % heads:
            raise ValueError('Embedding dimension must be divisible by attention heads.')
        self.heads = heads
        self.head_dim = width // heads
        self.alpha = alpha
        self.input_norm = RMSNorm(width)
        self.calendar_norm = RMSNorm(width)
        self.temporal_norm = RMSNorm(width)
        self.output_norm = RMSNorm(width)
        self.query = GeneratedLinear(width, code_dim)
        self.key = GeneratedLinear(width, code_dim)
        self.value = nn.Linear(width, width)
        self.calendar_query = nn.Linear(width, width)
        self.calendar_key = nn.Linear(width, width)
        self.calendar_value = nn.Linear(width, width)
        self.calendar_attention_dropout = nn.Dropout(attention_dropout)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.graph_projection = GeneratedLinear(width, code_dim, bias=False)
        self.fusion_weight = nn.Parameter(torch.ones(()))
        self.spatial_output = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)
        self.ffn = FeedForward(width, mlp_ratio)

    def forward(self, x, codes, graph, historical_calendar, future_calendar):
        batch, patches, nodes, width = x.shape
        normalized = self.input_norm(x)

        def split_heads(values):
            return values.reshape(batch, patches, nodes, self.heads, self.head_dim).permute(0, 2, 3, 1, 4)

        calendar_query = split_heads(self.calendar_query(future_calendar))
        calendar_key = split_heads(self.calendar_key(historical_calendar))
        calendar_value = split_heads(self.calendar_value(normalized))
        calendar_attention = (calendar_query @ calendar_key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        calendar_attention = self.calendar_attention_dropout(calendar_attention.softmax(-1))
        context = (calendar_attention @ calendar_value).permute(0, 3, 1, 2, 4).reshape(batch, patches, nodes, width)
        conditioned = self.calendar_norm(normalized + context)

        query = split_heads(self.query(conditioned, codes))
        key = split_heads(self.key(conditioned, codes))
        value = split_heads(self.value(conditioned))
        attention = (query @ key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attention = self.attention_dropout(attention.softmax(-1))
        attended = (attention @ value).permute(0, 3, 1, 2, 4).reshape(batch, patches, nodes, width)
        temporal = self.temporal_norm(conditioned + attended)
        logits = F.relu(codes @ codes.T).masked_fill(~graph.mask, float('-inf'))
        mixed = graph.fixed + self.fusion_weight * logits.softmax(-1)
        messages = torch.einsum('ij,bpjd->bpid', mixed, temporal)
        messages = self.graph_projection(messages, codes)
        spatial = x + self.dropout(self.spatial_output(self.alpha * temporal + (1 - self.alpha) * messages))
        return spatial + self.ffn(self.output_norm(spatial))


class TopoST(nn.Module):
    def __init__(self, embed_dim=128, code_dim=8, gin_hidden_dim=64, gin_dropout=0.1,
                 depth=2, heads=16, mlp_ratio=2, attention_dropout=0.3, dropout=0.1,
                 alpha=0.05, reference_steps=288, forecast_steps=288, patch_length=12,
                 use_baseline_ratio=False, use_reconstruction=False, reconstruction_mask_ratio=0.25):
        super().__init__()
        if embed_dim % 2 or reference_steps % patch_length or not 0 < alpha < 1:
            raise ValueError('Invalid embedding dimension, patch length, or spatial mixing coefficient.')
        if not 0 <= reconstruction_mask_ratio < 1:
            raise ValueError('Reconstruction mask ratio must be in [0, 1).')
        self.use_baseline_ratio = use_baseline_ratio
        self.use_reconstruction = use_reconstruction
        self.reconstruction_mask_ratio = reconstruction_mask_ratio
        self.forecast_steps = forecast_steps
        self.topology = TopologyEncoder(16, gin_hidden_dim, code_dim, gin_dropout)
        self.embedding = PatchEmbedding(embed_dim, reference_steps, patch_length,
                                        channels=1 + int(use_baseline_ratio))
        self.calendar_embedding = CalendarEmbedding(embed_dim, self.embedding.num_patches)
        self.reconstruction_head = nn.Linear(embed_dim, patch_length) if use_reconstruction else None
        self.blocks = nn.ModuleList([
            SpatioTemporalBlock(embed_dim, code_dim, heads, mlp_ratio, attention_dropout, dropout, alpha)
            for _ in range(depth)
        ])
        self.head = nn.Linear(embed_dim * self.embedding.num_patches, forecast_steps)

    def forward(self, history, graph, future_calendar, output_steps=None, return_aux=False):
        """History channels: measurement, minute, weekday, and optional ratio.

        future_calendar contains only known minute/weekday indices, never measurements.
        """
        if history.ndim != 4 or history.shape[-1] != 3 + int(self.use_baseline_ratio) or history.shape[2] != len(graph.nodes):
            raise ValueError('Input channels or local node order do not match the model.')
        if (future_calendar.ndim != 4 or future_calendar.shape[-1] != 2
                or future_calendar.shape[0] != history.shape[0]
                or future_calendar.shape[2] != history.shape[2]):
            raise ValueError('Provide future minute/weekday indices for the same batch and nodes.')
        output_steps = future_calendar.shape[1] if output_steps is None else output_steps
        if future_calendar.shape[1] != output_steps:
            raise ValueError('Future calendar and forecast lengths must match.')
        if not 0 < output_steps <= self.forecast_steps:
            raise ValueError('Forecast length exceeds the shared prediction head.')
        flow = history[..., :1]
        mean = flow.mean(1, keepdim=True).detach()
        scale = (flow.var(1, keepdim=True, unbiased=False) + 1e-5).sqrt().detach()
        normalized = (flow - mean) / scale
        inputs = torch.cat([normalized, history[..., 3:4]], dim=-1) if self.use_baseline_ratio else normalized
        keep = None
        if self.training and self.use_reconstruction:
            keep = (torch.rand(len(history), self.embedding.num_patches, device=history.device)
                    >= self.reconstruction_mask_ratio).to(history.dtype)
        hidden = self.embedding(inputs, keep)
        codes = self.topology(graph)
        historical_time = self.calendar_embedding(history[..., 1:3])
        future_time = self.calendar_embedding(future_calendar)
        for block in self.blocks:
            hidden = block(hidden, codes, graph, historical_time, future_time)
        auxiliary = hidden.new_zeros(())
        if keep is not None:
            masked = (1 - keep)[:, :, None, None]
            targets = self.embedding.to_patches(normalized)
            errors = (self.reconstruction_head(hidden) - targets).abs()
            # Sum over nodes; average over masked temporal patch slots.
            auxiliary = (errors * masked).sum() / (masked.sum() * self.embedding.patch_length + 1e-6)
        prediction = self.head(hidden.permute(0, 2, 3, 1).flatten(-2))
        prediction = prediction[..., :output_steps].transpose(1, 2).unsqueeze(-1)
        prediction = prediction * scale + mean
        return (prediction, auxiliary) if return_aux else prediction
