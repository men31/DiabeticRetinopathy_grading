import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(
        self, embed_size, num_heads, dropout: float = 0.3, get_attention: bool = False
    ):
        super(MultiHeadAttention, self).__init__()
        assert (
            embed_size % num_heads == 0
        ), "Embedding size must be divisible by number of heads"

        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads  # Dimension per head
        self.get_attention = get_attention

        # Linear layers for Query, Key, Value projections
        self.qkv_proj = nn.Linear(embed_size, embed_size * 3)
        self.fc_out = nn.Linear(embed_size, embed_size)  # Output projection layer

        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32)
        )  # Scaling factor

    def forward(self, x, mask=None):
        batch_size, seq_length, embed_size = x.shape

        # Compute Q, K, V
        qkv = self.qkv_proj(x)  # (batch_size, seq_length, embed_size * 3)
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        q, k, v = qkv.chunk(3, dim=-1)  # Split into Q, K, V

        # Transpose for multi-head processing (batch_size, num_heads, seq_length, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        attention_scores = (
            torch.matmul(q, k.transpose(-2, -1)) / self.scale
        )  # (batch_size, num_heads, seq_length, seq_length)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(
                mask == 0, float("-inf")
            )  # Masking for padding

        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to V
        attention_output = torch.matmul(
            attention_weights, v
        )  # (batch_size, num_heads, seq_length, head_dim)

        # Reshape and concatenate heads
        attention_output = attention_output.transpose(1, 2).reshape(
            batch_size, seq_length, embed_size
        )

        # Final linear layer
        output = self.fc_out(attention_output)

        if self.get_attention:
            return (
                output,
                attention_weights,
            )  # Returning attention weights for visualization if needed

        return output
