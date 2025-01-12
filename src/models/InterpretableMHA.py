import torch.nn as nn
import torch

class InterpretableMHA(nn.Module):
    """
    Very similar mechanism compared to simple MHA, but share the same values acrross all the attention heads.

    See "Temporal Fusion Transformer" paper page 9.
    """

    def __init__(self, d_model, n_heads, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if d_model % n_heads != 0:
            raise ValueError("d_model must be a multiple of n_heads")
        
        self.n_heads = n_heads
        self.d_attention = d_model // n_heads
        self.d_model = d_model

        self.key_weights = torch.nn.Parameter(torch.rand((n_heads, d_model, self.d_attention)))
        self.key_weights.requires_grad = True
        self.query_weights = torch.nn.Parameter(torch.rand((n_heads, d_model, self.d_attention)))
        self.query_weights.requires_grad = True
        self.value_weights = torch.nn.Parameter(torch.rand((d_model, d_model)))
        self.value_weights.requires_grad = True

        self.attention_heads_weights = torch.nn.Parameter(torch.rand((d_model, d_model)))
        self.attention_heads_weights.requires_grad = True

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):

        V = torch.matmul(v, self.value_weights)

        q = q.repeat(self.n_heads, 1, 1, 1)
        query_weights = self.query_weights.unsqueeze(1).expand(-1, q.size(1), -1, -1)
        Q = torch.matmul(q, query_weights)

        k = k.repeat(self.n_heads, 1, 1, 1)
        key_weights = self.key_weights.unsqueeze(1).expand(-1, k.size(1), -1, -1)
        K = torch.matmul(k, key_weights)

        attention_scores = nn.functional.softmax(Q*K / torch.sqrt(torch.Tensor([self.d_attention]).to(K.device)), dim=-1)

        attention_scores = attention_scores.reshape(attention_scores.shape[1], attention_scores.shape[2], self.d_model)

        H = (1/self.n_heads) * attention_scores * V

        res = torch.bmm(H, self.attention_heads_weights.unsqueeze(0).expand(H.shape[0], -1, -1))

        return res, attention_scores
