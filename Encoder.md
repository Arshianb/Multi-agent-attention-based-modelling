# Multi-agent-attention-based-modelling
the model contains two parts encoder and decoder but also you can just use encoder or decoder. it's depends on your decision.
## explaining `model/algorithm/mat_encoder.py`
### Attention Mechanism Configuration

First, define the following hyperparameters:

1. **Hidden‑layer dimension**  
2. **Number of attention heads**  
3. **Boolean mask flag** (whether to apply the attention mask)

---

#### How the Mask Works

Imagine two agents whose decisions were made in the previous time step. When computing their next decisions, we measure attention in three directions:

- **Agent 1 → Agent 1**  
- **Agent 1 → Agent 2**  
- **Agent 2 → Agent 2**  

Because Agent 1’s action has already occurred:

1. We redirect all of Agent 1’s self‑attention toward Agent 2.  
2. We zero out the attention from Agent 2 back to Agent 1, since that influence is now irrelevant.  

The **mask flag** toggles this redirect-and-zero behavior on or off during the attention computation.  

![alt text](<img/Screenshot from 2025-04-18 19-45-08.png>) ![alt text](<img/Screenshot from 2025-04-18 19-45-16.png>)

```python 
def forward(self, key, value, query):
    B, L, D = query.size()

    k = self.key(key).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
    q = self.query(query).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
    v = self.value(value).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)

    # causal attention: (B, nh, L, hs) x (B, nh, hs, L) -> (B, nh, L, L)
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

    if self.masked:
        att = att.masked_fill(self.mask[:, :, :L, :L] == 0, float('-inf'))
    att = F.softmax(att, dim=-1)

    y = att @ v  # (B, nh, L, L) x (B, nh, L, hs) -> (B, nh, L, hs)
    y = y.transpose(1, 2).contiguous().view(B, L, D) # re-assemble all head outputs side by side

    y = self.proj(y)
return y
```
Shapes and Terminology
- **B: Batch size**

- **L: Sequence length (e.g., number of agents)**

- **D: Embedding dimension (e.g., observation size)**

- **nh: Number of attention heads**

- **hs: Head size = D / nh (must divide evenly)**

By default, `key`, `value`, and `query` all have shape (B, L, D). After the linear projection and reshape, each becomes (B, nh, L, hs).

#### Processing Steps
1. **Linear Projections & Reshape**
Each of `key`, `value`, and `query` is passed through its own `nn.Linear` layer, then reshaped into (B, nh, L, hs) and transposed to put the head dimension first.

2. **Scaled Dot‑Product Attention**

- **Compute raw attention scores:**

$$ score = \frac{Q \times K^T}{\sqrt{hs}} $$
resulting in a tensor of shape (B, nh, L, L), where each entry $(i,j)$ measures how much position $i$ attends to position $j$.

3. **Causal Masking**
if `self.masked` is `True`, apply an upper‑triangular mask (zeros above the diagonal) so each position can only attend to itself and previous positions.

4. **Softmax Normalization**
Convert scores to probabilities along the last axis:
$$A = softmax(scores)$$
maintaining shape (B, nh, L, L).

5. **Weighted Sum of Values**
- **Multiply attention weights $A$ by $V$:**
$$Y = A \times V$$
giving (B, nh, L, hs).

6. **Reassemble & Project**
- **Transpose and reshape back to (B, L, D) by concatenating all heads.**
- **Apply a final linear layer (self.proj) to mix head information.**

#### Causal Mask Intuition
When `masked=True`, any position $i$ can only attend to positions $≤i$. This ensures that future information does not leak into the current timestep—critical for autoregressive models and sequential decision-making.

### EncodeBlock explanation
This document describes the `EncodeBlock` module, which applies self‐attention followed by a simple feed‑forward network, with layer normalization and residual (“skip”) connections.

![encode block image](<img/Screenshot from 2025-04-18 19-46-26.png>)

The `EncodeBlock` integrates three main components:

1. **Self‑Attention**  
   Captures contextual relationships between tokens by computing attention scores for each pair of positions in the input sequence.

2. **Feed‑Forward Network (MLP)**  
   Applies two linear transformations with a non‑linear activation in between (GELU), projecting from and back to the embedding dimension.

3. **Layer Normalization & Residual Connections**  
   Each sublayer (attention and MLP) is preceded by layer normalization and followed by a residual connection to stabilize training and improve gradient flow.

#### Architecture

1. **LayerNorm + Attention + Residual**  
   ```python
   x = x + SelfAttention(x, x, x)
   x = LayerNorm(x)

2. **LayerNorm + MLP + Residual**
```python
x = x + MLP(x)
x = LayerNorm(x)
```
#### Implementation
```python
import torch.nn as nn

class EncodeBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int, n_agent: int):
        """
        Args:
            n_embd (int): Dimensionality of token embeddings.
            n_head (int): Number of attention heads.
            n_agent (int): Number of agents (for multi‑agent setups).
        """
        super().__init__()

        # First normalization layer
        self.ln1 = nn.LayerNorm(n_embd)
        # Second normalization layer
        self.ln2 = nn.LayerNorm(n_embd)

        # Self-attention (unmasked)
        self.attn = SelfAttention(
            embed_dim=n_embd,
            num_heads=n_head,
            num_agents=n_agent,
            masked=False
        )

        # Two-layer MLP with GELU activation
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(n_embd, n_embd))
        )

    def forward(self, x):
        # Attention block with residual connection
        attn_out = self.attn(x, x, x)
        x = self.ln1(x + attn_out)

        # MLP block with residual connection
        mlp_out = self.mlp(x)
        x = self.ln2(x + mlp_out)

        return x
```
- **Note:** Use unmasked attention (`masked=False`) when encoding full sequences; set `masked=True` (causal) for autoregressive decoding.

### Encoder Explanation

We can have multiple `EncodeBlock` layers for the Encoder part of our agents, as shown below:

```python
self.blocks = nn.Sequential(*[EncodeBlock(n_embd, n_head, n_agent) for _ in range(n_block)])
```

Thus, we construct a feed-forward encoder as follows:

```python
def forward(self, state, obs):
    # state: (batch, n_agent, state_dim)
    # obs: (batch, n_agent, obs_dim)
    if self.encode_state:
        state_embeddings = self.state_encoder(state)
        x = state_embeddings
    else:
        obs_embeddings = self.obs_encoder(obs)
        x = obs_embeddings

    rep = self.blocks(self.ln(x))
    v_loc = self.head(rep)
    logit = self.act_head(rep)

    return v_loc, rep, logit
```

This encoder takes either the environment state or observations as input and passes these through sequential `EncodeBlock` layers to generate representations (`rep`).

If we later want to include state-value function or action policy predictions, our model can output multiple tensors:

- `logit`: shape `(BatchSize × agents × ActionSpace)` for policy predictions.
- `v_loc`: shape `(BatchSize × agents × 1)` for state-value predictions.
