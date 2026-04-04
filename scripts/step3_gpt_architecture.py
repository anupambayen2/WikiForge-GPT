"""
WikiForge-GPT Model Architecture
=================================
GPT transformer built from scratch for Stage 0 (Tiny Model).

Architecture:
- 4 layers, 4 heads, 256 d_model
- ~12M parameters
- Optimized for RTX 4060 (8GB VRAM)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

@dataclass
class GPTConfig:
    """Configuration for GPT model."""
    
    # Model architecture
    vocab_size: int = 8000
    n_layers: int = 4
    n_heads: int = 4
    d_model: int = 256
    d_ff: int = 1024  # 4 * d_model
    max_seq_length: int = 256
    dropout: float = 0.1
    
    # Special tokens
    pad_token_id: int = 1
    eos_token_id: int = 0
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    Allows model to attend to different parts of the sequence.
    """
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_heads
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        
        # Output projection
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        
        # Dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Causal mask (lower triangular)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_seq_length, config.max_seq_length))
            .view(1, 1, config.max_seq_length, config.max_seq_length)
        )
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: [batch_size, seq_len, d_model]
            attention_mask: [batch_size, seq_len] - 1 for real tokens, 0 for padding
        
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)  # [B, T, d_model]
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        # [B, T, d_model] -> [B, T, n_heads, head_dim] -> [B, n_heads, T, head_dim]
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        # [B, n_heads, T, head_dim] @ [B, n_heads, head_dim, T] -> [B, n_heads, T, T]
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask (prevent attending to future tokens)
        attn_scores = attn_scores.masked_fill(
            self.causal_mask[:, :, :seq_len, :seq_len] == 0,
            float('-inf')
        )
        
        # Apply padding mask if provided
        if attention_mask is not None:
            # [B, T] -> [B, 1, 1, T]
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        # [B, n_heads, T, T] @ [B, n_heads, T, head_dim] -> [B, n_heads, T, head_dim]
        attn_output = attn_weights @ v
        
        # Reshape back
        # [B, n_heads, T, head_dim] -> [B, T, n_heads, head_dim] -> [B, T, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # Output projection
        output = self.out_proj(attn_output)
        output = self.resid_dropout(output)
        
        return output


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    Two linear layers with GELU activation.
    """
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        x = self.fc1(x)
        x = F.gelu(x)  # Gaussian Error Linear Unit
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Single transformer block.
    Layer norm -> Attention -> Add & Norm -> FFN -> Add & Norm
    """
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        
        # Attention and FFN
        self.attn = MultiHeadAttention(config)
        self.ffn = FeedForward(config)
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: [batch_size, seq_len, d_model]
            attention_mask: [batch_size, seq_len]
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        # Pre-LayerNorm architecture (like GPT-2)
        # Attention block with residual connection
        x = x + self.attn(self.ln1(x), attention_mask)
        
        # FFN block with residual connection
        x = x + self.ffn(self.ln2(x))
        
        return x


class GPTModel(nn.Module):
    """
    Full GPT model built from scratch.
    
    Stage 0 (Tiny): 4 layers, 4 heads, 256 d_model = ~12M parameters
    """
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        
        self.config = config
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_length, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.d_model)
        
        # Language model head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Tie weights (token embedding and lm_head share weights)
        self.lm_head.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Count parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"GPT Model initialized with {n_params:,} parameters")
    
    def _init_weights(self, module):
        """Initialize weights using GPT-2 strategy."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass.
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len] - 1 for real, 0 for padding
            labels: [batch_size, seq_len] - for computing loss
        
        Returns:
            logits: [batch_size, seq_len, vocab_size]
            loss: scalar (if labels provided)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Create position IDs
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)  # [B, T, d_model]
        position_embeds = self.position_embedding(position_ids)
        
        x = token_embeds + position_embeds
        x = self.dropout(x)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Language model head
        logits = self.lm_head(x)  # [B, T, vocab_size]
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Shift labels and logits for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten for cross-entropy
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=self.config.pad_token_id,
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ):
        """
        Generate text autoregressively.
        
        Args:
            input_ids: [batch_size, seq_len] - prompt tokens
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature (higher = more random)
            top_k: if set, only sample from top k tokens
        
        Returns:
            generated: [batch_size, seq_len + max_new_tokens]
        """
        for _ in range(max_new_tokens):
            # Crop to max_seq_length
            idx_cond = input_ids[:, -self.config.max_seq_length:]
            
            # Forward pass
            logits, _ = self(idx_cond)
            
            # Get last token's logits
            logits = logits[:, -1, :] / temperature
            
            # Optional top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Softmax and sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, idx_next], dim=1)
            
            # Stop if EOS token generated
            if idx_next.item() == self.config.eos_token_id:
                break
        
        return input_ids


def create_stage0_model():
    """Create Stage 0 (Tiny) model configuration."""
    
    config = GPTConfig(
        vocab_size=8000,
        n_layers=4,
        n_heads=4,
        d_model=256,
        d_ff=1024,
        max_seq_length=256,
        dropout=0.1,
        pad_token_id=1,
        eos_token_id=0,
    )
    
    model = GPTModel(config)
    
    return model, config


if __name__ == "__main__":
    # Test the model
    print("="*80)
    print("Testing GPT Model Architecture")
    print("="*80)
    
    # Create model
    model, config = create_stage0_model()
    
    # Test forward pass
    batch_size = 2
    seq_len = 64
    
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = input_ids.clone()
    
    print(f"\nInput shape: {input_ids.shape}")
    
    # Forward pass
    logits, loss = model(input_ids, attention_mask, labels)
    
    print(f"Output logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    
    # Test generation
    print("\nTesting generation...")
    prompt = input_ids[:1, :10]  # First 10 tokens
    generated = model.generate(prompt, max_new_tokens=20, temperature=0.8)
    
    print(f"Prompt shape: {prompt.shape}")
    print(f"Generated shape: {generated.shape}")
    
    print("\n" + "="*80)
    print("✅ Model architecture verified!")
    print("="*80)