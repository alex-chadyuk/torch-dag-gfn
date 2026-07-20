"""GFlowNet policy network (faithful PyTorch port of the Linear Transformer).

The policy operates over the N**2 candidate edges as tokens. Each edge (s -> t)
is embedded from its endpoint ids, a shared body of Transformer blocks mixes the
edge tokens, and two heads produce (a) a per-edge "add this edge" logit and
(b) a single pooled "stop" logit. The blocks use the O(N**2) linear attention of
Katharopoulos et al. (feature map elu(x)+1), and re-inject an embedding of the
raw adjacency matrix at every block. The module is written natively batched
(leading batch dimension B) -- no vmap.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def variance_scaling_(weight, scale):
    """Haiku-style VarianceScaling init (fan_in, truncated normal)."""
    fan_in = weight.shape[-1]
    std = math.sqrt(scale / max(fan_in, 1)) / 0.87962566103423978
    with torch.no_grad():
        nn.init.trunc_normal_(weight, mean=0., std=std, a=-2. * std, b=2. * std)


def _linear(in_features, out_features, scale):
    layer = nn.Linear(in_features, out_features)
    variance_scaling_(layer.weight, scale)
    nn.init.zeros_(layer.bias)
    return layer


class LinearMultiHeadAttention(nn.Module):
    """Linear self-attention with feature map elu(x) + 1 (O(T) in tokens)."""
    def __init__(self, model_dim, num_heads, key_size, w_init_scale):
        super().__init__()
        self.num_heads = num_heads
        self.key_size = key_size
        self.value_size = key_size
        self.model_size = num_heads * self.value_size
        self.eps = 1e-6

        self.q_proj = _linear(model_dim, num_heads * key_size, w_init_scale)
        self.k_proj = _linear(model_dim, num_heads * key_size, w_init_scale)
        self.v_proj = _linear(model_dim, num_heads * self.value_size, w_init_scale)
        self.out_proj = _linear(num_heads * self.value_size, self.model_size, w_init_scale)

    def forward(self, x):
        B, T, _ = x.shape
        H, P, Q = self.num_heads, self.key_size, self.value_size

        q = self.q_proj(x).view(B, T, H, P)
        k = self.k_proj(x).view(B, T, H, P)
        v = self.v_proj(x).view(B, T, H, Q)

        q = F.elu(q) + 1.
        k = F.elu(k) + 1.

        kv = torch.einsum('bthp,bthq->bhqp', k, v)          # (B, H, Q, P)
        k_sum = k.sum(dim=1)                                # (B, H, P)
        z = 1. / (torch.einsum('bthp,bhp->bth', q, k_sum) + self.eps)
        attn = torch.einsum('bthp,bhqp,bth->bthq', q, kv, z)  # (B, T, H, Q)

        attn_vec = attn.reshape(B, T, H * Q)
        return self.out_proj(attn_vec)


class DenseBlock(nn.Module):
    def __init__(self, in_dim, output_size, init_scale, widening_factor=2):
        super().__init__()
        hidden = widening_factor * output_size
        self.linear_1 = _linear(in_dim, hidden, init_scale)
        self.linear_2 = _linear(hidden, output_size, init_scale)

    def forward(self, x):
        return self.linear_2(F.gelu(self.linear_1(x)))


class TransformerBlock(nn.Module):
    """Pre-LayerNorm block that re-injects an embedding of the raw input at each
    sublayer (matches the original `TransformerBlock`)."""
    def __init__(self, input_dim, hidden_dim, num_heads, key_size,
                 embedding_size, init_scale, widening_factor=2):
        super().__init__()
        norm_dim = embedding_size + hidden_dim

        self.linear_1 = _linear(input_dim, embedding_size, init_scale)
        self.layernorm_1 = nn.LayerNorm(norm_dim)
        self.attn = LinearMultiHeadAttention(norm_dim, num_heads, key_size, init_scale)

        self.linear_2 = _linear(input_dim, embedding_size, init_scale)
        self.layernorm_2 = nn.LayerNorm(norm_dim)
        self.dense = DenseBlock(norm_dim, hidden_dim, init_scale, widening_factor)

    def forward(self, hiddens, inputs):
        ie = self.linear_1(inputs)
        h_norm = self.layernorm_1(torch.cat((ie, hiddens), dim=-1))
        hiddens = hiddens + self.attn(h_norm)

        ie = self.linear_2(inputs)
        h_norm = self.layernorm_2(torch.cat((ie, hiddens), dim=-1))
        hiddens = hiddens + self.dense(h_norm)
        return hiddens


class _MLP(nn.Module):
    """3-layer MLP [256, 128, 1] with ReLU between layers (matches hk.nets.MLP)."""
    def __init__(self, in_dim, sizes=(256, 128, 1)):
        super().__init__()
        layers, dim = [], in_dim
        for size in sizes:
            layers.append(_linear(dim, size, 1.0))
            dim = size
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x


class GFlowNetPolicy(nn.Module):
    """Edge-token Linear Transformer policy.

    forward(adjacency, mask) -> (logits (B, N**2), stop (B, 1)); the normalised
    log-policy over N**2 + 1 actions is assembled in `gflownet.log_policy`.
    """
    def __init__(self, num_variables, embed_dim=128, num_heads=4, key_size=64,
                 num_backbone=3, num_head_layers=2):
        super().__init__()
        self.num_variables = num_variables
        hidden_dim = num_heads * key_size          # 256
        init_scale = 2. / 5.                        # original uses num_layers = 5

        self.embed = nn.Embedding(2 * num_variables, embed_dim)

        indices = torch.arange(num_variables ** 2)
        sources = indices // num_variables
        targets = indices % num_variables
        edges = torch.stack((sources, num_variables + targets), dim=1)  # (N**2, 2)
        self.register_buffer('edges', edges)

        def block():
            return TransformerBlock(
                input_dim=1, hidden_dim=hidden_dim, num_heads=num_heads,
                key_size=key_size, embedding_size=embed_dim,
                init_scale=init_scale, widening_factor=2)

        self.backbone = nn.ModuleList(block() for _ in range(num_backbone))
        self.logits_blocks = nn.ModuleList(block() for _ in range(num_head_layers))
        self.stop_blocks = nn.ModuleList(block() for _ in range(num_head_layers))
        self.logits_mlp = _MLP(hidden_dim)
        self.stop_mlp = _MLP(hidden_dim)

    def forward(self, adjacency, mask):
        B = adjacency.shape[0]
        num_edges = self.num_variables ** 2

        embeddings = self.embed(self.edges).reshape(num_edges, -1)   # (N**2, 256)
        embeddings = embeddings.unsqueeze(0).expand(B, -1, -1)       # (B, N**2, 256)
        inputs = adjacency.reshape(B, num_edges, 1)                  # (B, N**2, 1)

        h = embeddings
        for blk in self.backbone:
            h = blk(h, inputs)

        logits = h
        for blk in self.logits_blocks:
            logits = blk(logits, inputs)
        logits = self.logits_mlp(logits).squeeze(-1)                # (B, N**2)

        stop = h
        for blk in self.stop_blocks:
            stop = blk(stop, inputs)
        stop = self.stop_mlp(stop.mean(dim=1))                      # (B, 1)

        return logits, stop
