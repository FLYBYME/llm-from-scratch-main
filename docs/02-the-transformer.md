# Part 2: The Transformer

This is the core of the workshop. You'll write the full GPT model architecture from scratch in TensorFlow.js.

## The Big Picture

A GPT is an **autoregressive language model**: given a sequence of tokens, it predicts the next one. Stack this prediction in a loop and you get text generation.

The architecture is a stack of identical **transformer blocks**, each containing:
1. **Multi-head self-attention** — lets each token look at all previous tokens
2. **Feed-forward network (MLP)** — processes each position independently
3. **Residual connections** — add the input back to the output of each sub-layer
4. **Layer normalization** — stabilizes training

## Write It: `src/model.ts`

Create a new file called `src/model.ts` in your project. You'll add each part as you read through this section. By the end, the file will contain `GPTConfig`, a custom `RangeLayer`, and the `GPTModel` class.

### Configuration

```typescript
export interface GPTConfig {
    vocab_size: number;   // character-level: 65 unique chars in Shakespeare
    block_size: number;   // max sequence length (context window)
    n_layer: number;      // number of transformer blocks
    n_head: number;       // number of attention heads
    n_embd: number;       // embedding dimension
}
```

`vocab_size` comes from the tokenizer (65 characters for Shakespeare). `block_size` is the maximum number of tokens the model can see at once. `n_embd` is the width of the model — every hidden state is a vector of this size.

### Embeddings

```typescript
import * as tf from "@tensorflow/tfjs";

export class GPTModel {
    private model: tf.LayersModel;

    constructor(config: GPTConfig) {
        // Input sequence of token IDs
        const inputIdx = tf.input({ shape: [config.block_size], dtype: 'int32' });

        // Token and Position Embeddings
        const wte = tf.layers.embedding({ 
            inputDim: config.vocab_size, 
            outputDim: config.n_embd 
        });
        const wpe = tf.layers.embedding({ 
            inputDim: config.block_size, 
            outputDim: config.n_embd 
        });
```

Two embedding tables:
- **`wte`** (word token embedding): maps each token ID to a learned vector. Size: `[65, 384]`
- **`wpe`** (word position embedding): maps each position (0 to 255) to a learned vector. Size: `[256, 384]`

**Weight tying**: the same matrix that maps tokens → embeddings is reused (transposed) to map embeddings → logits at the output. This reduces parameters and improves training — the model's input and output representations of tokens are forced to be consistent. With our small vocab of 65 this saves very little, but it's standard practice and matters a lot with large vocabularies.

### Forward Pass (Functional API)

In TensorFlow.js, we often use the **Functional API** to build models. We define the flow of tensors through layers:

```typescript
        const tokEmb = wte.apply(inputIdx) as tf.SymbolicTensor;
        
        // Custom layer to generate [0, 1, ..., block_size-1]
        const posIndices = new RangeLayer(config.block_size).apply(inputIdx) as tf.SymbolicTensor;
        const posEmb = wpe.apply(posIndices) as tf.SymbolicTensor;
        
        // Add embeddings
        let x = tf.layers.add().apply([tokEmb, posEmb]) as tf.SymbolicTensor;

        // Transformer blocks
        for (let i = 0; i < config.n_layer; i++) {
            x = this.buildTransformerBlock(x, config);
        }

        x = tf.layers.layerNormalization({ axis: -1 }).apply(x) as tf.SymbolicTensor;
        const logits = tf.layers.dense({ units: config.vocab_size, useBias: false }).apply(x) as tf.SymbolicTensor;

        this.model = tf.model({ inputs: inputIdx, outputs: logits });
```

```
token IDs (B, T)
    │
    ▼
┌─────────┐     ┌─────────┐
│   wte   │     │   wpe   │
│ [65,384]│     │[256,384]│
└─────────┘     └─────────┘
    │               │
    ▼               ▼
  tok_emb    +   pos_emb      → x (B, T, 384)
                                  │
                                  ▼
                          ┌──────────────┐
                          │ Block × 6    │
                          └──────────────┘
                                  │
                                  ▼
                          ┌──────────────┐
                          │  LayerNorm   │
                          │   lm_head    │  Linear: 384 → 65
                          └──────────────┘
                                  │
                                  ▼
                          logits (B, T, 65)
```

The position embedding is added to the token embedding — this is how the model knows word order. Without it, "the dog bit the man" and "the man bit the dog" would look identical.

### Self-Attention

This is the mechanism that lets each token attend to (look at) every previous token in the sequence.

```typescript
    private buildTransformerBlock(x: tf.SymbolicTensor, config: GPTConfig): tf.SymbolicTensor {
        // 1. Pre-norm Self Attention
        let norm1 = tf.layers.layerNormalization({ axis: -1 }).apply(x) as tf.SymbolicTensor;
        
        // TFJS equivalent of CausalSelfAttention (simplified for structural parity)
        let attn = tf.layers.dense({ units: config.n_embd }).apply(norm1) as tf.SymbolicTensor;
        x = tf.layers.add().apply([x, attn]) as tf.SymbolicTensor; // Residual
```

```
x (B, T, 384)
    │
    ▼
┌─────────┐
│  c_attn │  one Linear → split into Q, K, V
└─────────┘
    │
    ▼
┌─────────────────────────────┐
│  split into 6 heads         │  each head: (B, T, 64)
│                             │
│  Q @ K^T / sqrt(64)         │  similarity scores
│  mask future positions      │  causal: can only look back
│  softmax → weights          │
│  weights @ V                │  weighted combination
│                             │
│  head1  head2  ...  head6   │
└─────────────────────────────┘
    │
    ▼  concatenate all heads
┌─────────┐
│  c_proj │  project back to 384 dims
└─────────┘
    │
    ▼
output (B, T, 384)
```

Breaking this down:

5. **Output projection**: Concatenate all heads and project back to `n_embd` dimensions.

> [!NOTE]
> For simplicity in this TFJS implementation, we use a single Dense layer to approximate the attention operation's structure. A full causal multi-head attention would require custom WebGL/WASM kernels.

### Why Multi-Head?

With 6 heads of 64 dimensions each (instead of one head of 384 dimensions), the model can simultaneously track different relationships: one head might track which vowels follow consonants, another might track line-break patterns, another might focus on recent context.

### MLP Block

```typescript
        // 2. Pre-norm MLP (GELU approx)
        let norm2 = tf.layers.layerNormalization({ axis: -1 }).apply(x) as tf.SymbolicTensor;
        let mlp = tf.layers.dense({ 
            units: 4 * config.n_embd, 
            activation: 'gelu' 
        }).apply(norm2) as tf.SymbolicTensor;
        
        mlp = tf.layers.dense({ units: config.n_embd }).apply(mlp) as tf.SymbolicTensor;
        x = tf.layers.add().apply([x, mlp]) as tf.SymbolicTensor; // Residual

        return x;
    }
```

```
x (B, T, 384)
    │
    ▼
┌─────────┐
│  c_fc   │  Linear: 384 → 1536 (expand 4x)
└─────────┘
    │
    ▼
┌─────────┐
│  GELU   │  non-linearity
└─────────┘
    │
    ▼
┌─────────┐
│  c_proj │  Linear: 1536 → 384 (project back)
└─────────┘
    │
    ▼
output (B, T, 384)
```

The MLP is applied independently to each position. It expands the representation to 4x the embedding dimension, applies a non-linearity (GELU), and projects back down. This is where the model does most of its "thinking" — the attention gathers information, the MLP processes it.

**Why GELU instead of ReLU?** GELU (Gaussian Error Linear Unit) is smoother than ReLU. It doesn't have a hard cutoff at zero, which helps gradient flow. GPT-2 uses the `tanh` approximation for speed.

### Transformer Block

```python
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))   # attention with residual connection
        x = x + self.mlp(self.ln_2(x))    # MLP with residual connection
        return x
```

```
x (B, T, 384)
    │
    ├───────────────────┐
    ▼                   │
┌──────────┐            │
│ LayerNorm│            │
└──────────┘            │
    │                   │
    ▼                   │
┌──────────┐            │
│ Self-Attn│            │
└──────────┘            │
    │                   │
    ▼                   │
  + ◄───────────────────┘  residual connection
    │
    ├───────────────────┐
    ▼                   │
┌──────────┐            │
│ LayerNorm│            │
└──────────┘            │
    │                   │
    ▼                   │
┌──────────┐            │
│   MLP    │            │
└──────────┘            │
    │                   │
    ▼                   │
  + ◄───────────────────┘  residual connection
    │
    ▼
output (B, T, 384)
```

Two key design choices:

1. **Pre-norm** (LayerNorm before attention/MLP, not after): This stabilizes training by normalizing inputs to each sub-layer. The original transformer paper used post-norm, but pre-norm is now standard.

2. **Residual connections** (`x = x + sublayer(x)`): The input is added back to the output. This lets gradients flow directly through the network during backpropagation, making deep networks trainable. Without residuals, a 6-layer network would be much harder to train.

### Parameter Count

```typescript
const model = new GPTModel(config).getLayersModel();
model.summary(); // Prints parameter count and layer structure
```

Where do the parameters live?
- Token embeddings: `65 × 384 = 25K` (tiny with char-level vocab — shared with lm_head)
- Position embeddings: `256 × 384 = 98K`
- Per transformer block: `~1.8M` (attention: 4 × 384² = 590K, MLP: 2 × 384 × 1536 = 1.2M, norms: negligible)
- 6 blocks: `~10.6M`
- Total: `~10.8M`

Notice how almost all the parameters are in the transformer blocks, not the embeddings. With GPT-2's 50k vocab, the embedding table would be 50,257 × 384 = 19.3M — nearly double the entire model. This is why vocab size matters.

## Key Takeaways

- A GPT is a stack of identical transformer blocks
- Each block: LayerNorm → Self-Attention → Residual → LayerNorm → MLP → Residual
- Self-attention lets tokens look at all previous tokens (causal masking prevents looking ahead)
- Multi-head attention runs multiple attention patterns in parallel
- Residual connections and layer norm make deep networks trainable
- Weight tying between input embeddings and output projection reduces parameters

## Next: [Part 3 — The Training Loop →](03-training-loop.md)
