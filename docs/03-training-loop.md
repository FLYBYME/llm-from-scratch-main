# Part 3: The Training Loop

You have a model. Now you need to teach it language. The training loop is where the model actually learns, and every decision here affects whether your model converges or diverges into nonsense.

## The Training Objective

GPT is trained with **next-token prediction**: given tokens `[t0, t1, ..., tn]`, predict `[t1, t2, ..., tn+1]`. The loss function is cross-entropy between the model's predicted probability distribution and the actual next token.

This is a self-supervised task — the labels come from the data itself. Every piece of text is simultaneously input and target, just shifted by one position.

## Write It: `src/train.ts`

Create a new file called `src/train.ts` in your project. This file handles the data loading, batching, and the main training loop using `model.fit()`.

Add each piece below to `src/train.ts` as you read through this section.

### Step 1: Data Loading (Character-Level)

```typescript
import * as tf from "@tensorflow/tfjs";

export async function train(textData: string, config: GPTConfig, batchSize: number = 64, maxSteps: number = 5000) {
    const tokenizer = new CharTokenizer(textData);
    const tokens = tokenizer.encode(textData);

    const getBatch = () => {
        return tf.tidy(() => {
            const xBatch: number[][] = [];
            const yBatch: number[][] = [];

            for (let i = 0; i < batchSize; i++) {
                const ix = Math.floor(Math.random() * (tokens.length - config.block_size - 1));
                xBatch.push(tokens.slice(ix, ix + config.block_size));
                yBatch.push(tokens.slice(ix + 1, ix + config.block_size + 1));
            }

            const x = tf.tensor2d(xBatch, [batchSize, config.block_size], 'int32');
            const y = tf.oneHot(tf.tensor2d(yBatch, [batchSize, config.block_size], 'int32'), config.vocab_size);
            return { x, y: y as tf.Tensor3D };
        });
    };
```

Each batch:
- Sample `batch_size` random starting positions
- `x`: characters from position `i` to `i + block_size` (input)
- `y`: characters from position `i+1` to `i + block_size + 1` (target — shifted by one)

The function returns `stoi`/`itos` mappings alongside the batch generators — you'll need these for text generation.

### Step 2: Backend Setup

In Node.js, we use the high-performance C++ backend:

```typescript
import "@tensorflow/tfjs-node";
```

This is significantly faster than the default JavaScript backend.

### Step 3: Optimizer and Loss

```typescript
    const model = new GPTModel(config).getLayersModel();
    const optimizer = tf.train.adam(1e-3);

    model.compile({
        optimizer: optimizer,
        loss: tf.losses.softmaxCrossEntropy,
    });
```

We use the Adam optimizer and Softmax Cross Entropy loss. In TensorFlow.js, `model.fit()` handles the training loop, including the forward pass, loss calculation, and backpropagation.

### Step 4: The Full Training Loop

```typescript
    for (let step = 0; step < maxSteps; step++) {
        const batch = getBatch();
        
        const history = await model.fit(batch.x, batch.y, {
            batchSize: batchSize,
            epochs: 1,
            verbose: 0
        });

        tf.dispose([batch.x, batch.y]); // Crucial to avoid memory leaks

        if (step % 100 === 0) {
            const losses = history.history['loss'];
            const loss = losses?.[0];
            if (loss !== undefined) {
                console.log(`Step ${step} | loss: ${(loss as number).toFixed(4)}`);
            }
        }
    }
    
    return { model, tokenizer };
}
```

### Step 5: Entry Point

In Node.js, we use an `async main` function as our entry point:

```typescript
async function main() {
    const dataPath = "./data/shakespeare.txt";
    const textData = fs.readFileSync(dataPath, "utf-8");

    const config: GPTConfig = {
        vocab_size: 0, // Set by tokenizer
        block_size: 64,
        n_layer: 4,
        n_head: 4,
        n_embd: 128,
    };

    const { model, tokenizer } = await train(textData, config);
    // ... generation code
}

main().catch(console.error);
```

### What Each Part Does

**Validation loss**: Every 100 steps, evaluate on held-out data. If train loss goes down but val loss goes up, you're overfitting.

**Gradient clipping** (`clip_grad_norm_`): Caps the total gradient magnitude at 1.0. Prevents occasional large gradients from blowing up the weights.

**Sample generation**: Every 100 steps, generate text so you can watch the model learn. You'll see it go from random characters → random words → Shakespeare-like text.

**Checkpointing**: Save model state periodically. Checkpoints include `stoi`/`itos` so you can generate text from a saved model without the original data. A final checkpoint is saved as `checkpoint_final.pt` at the end of training.

**Loss log**: Training and validation losses are saved to `loss_log.json` so you can plot loss curves after training (see Part 5).

## What Loss Numbers Mean (Character-Level, vocab=65)

- **~4.2**: Random (untrained). `ln(65) ≈ 4.17`
- **~3.3**: Learned character frequencies (which letters are common)
- **~2.5**: Learned common bigrams ("th", "he", "in")
- **~1.5-2.0**: Generates recognizable words and Shakespeare-like structure
- **~1.0-1.2**: Good quality — generates verse with character names, line breaks
- **<1.0**: Likely memorizing the training data

## Watching the Model Learn

Here's what a real training run looks like (6L/6H/384D, batch_size=64, M3 Pro). The prompt is always "To be or not":

**Step 200** (val loss: ~3.5) — Random characters, no words:
```
To be or notis p ce mei odorethleedetire'ilethed ye m arkesothir fnon b tigb'i.
```

**Step 800** (val loss: ~1.8) — Words forming, character names appearing:
```
To be or not men, and my lord.

ROMEO:
Thou sir, do content the he, stray, there ir;
```

**Step 1000** (val loss: 1.64) — Coherent phrases, Shakespeare structure:
```
To be or nothing are good men,
The profent of little, our actory.

CORIOLANUS:
Is it now of your many death?
```

**Step 2400** (val loss: ~1.60) — Peak quality. Plausible Shakespeare:
```
To be or not to be some of you shall know
That everlature by Romeo: what news,
Which you had knock'd my part to speak
```

**Step 3500** (val loss: 2.34) — Overfitting. Still fluent but less creative:
```
To be or nothing, take me but most profane,
That offer them not amish. If I defeath
Is not a puggival and self,
```

## Overfitting: Train Loss vs Val Loss

With 10M parameters and only ~1M characters of Shakespeare, the model will **overfit** — it memorizes the training data instead of learning general patterns. You'll see this clearly:

```
Step   500 | val loss: 2.14   ← dropping fast, learning structure
Step  1000 | val loss: 1.64   ← still improving
Step  1500 | val loss: 1.57   ← best region
Step  2000 | val loss: 1.59   ← starting to plateau
Step  2500 | val loss: 1.71   ← val loss going UP — overfitting
Step  3000 | val loss: 1.98   ← getting worse
Step  3500 | val loss: 2.34   ← fully memorizing (train loss is 0.54)
```

The **best model** is around step 1500-2000 (val loss ~1.57), not step 5000. After that, every step makes the model *worse* at generating novel text — it's just getting better at reciting the training data.

### What Causes Overfitting?

The model has **10M parameters** learning from **~1M characters**. That's a 10:1 parameter-to-data ratio — the model has more than enough capacity to memorize every character in the training set. The fix is always the same: **more data** or **smaller model**.

### What to Do About It

For this workshop, overfitting is expected and fine — it demonstrates an important concept. In practice you would:

1. **Use more data** — TinyStories (476M tokens) would keep a 10M model learning much longer
2. **Use a smaller model** — a 2L/2H/128D model (~0.5M params) overfits much slower on Shakespeare
3. **Add dropout** — randomly zeroing activations during training acts as regularization
4. **Stop early** — save the checkpoint with the lowest val loss and use that

## Typical Training on MacBook (Character-Level Shakespeare)

| Model | Params | Batch Size | Steps | Time (M3 Pro) | Best Val Loss | Overfits At |
|-------|--------|-----------|-------|---------------|---------------|-------------|
| 6L/6H/384D | ~10M | 64 | 5,000 | ~45 min | ~1.7 (step 2500) | ~step 1500 |
| 4L/4H/256D | ~4M | 64 | 5,000 | ~20 min | ~1.6 (step 3000) | ~step 2000 |
| 2L/2H/128D | ~0.5M | 64 | 5,000 | ~5 min | ~1.8 (step 5000) | barely |

The 6L model gets the best samples fastest but overfits soonest. The 2L model trains quickly and barely overfits, but the output quality is lower. This is the fundamental tradeoff: **model capacity vs data size**.

Start with the 6L/6H/384D config. With batch_size=64 on an M3 Pro, you get ~1.9 it/s.

## Key Takeaways

- The objective is next-character prediction with cross-entropy loss
- Character-level tokenization works best for small datasets — BPE vocab is too sparse
- AdamW with lr=1e-3 and cosine decay is a good starting point
- Gradient clipping at 1.0 prevents training instability
- Generate samples during training — it's the best way to see progress
- **Watch the gap between train and val loss** — when val loss starts rising, you're overfitting
- The best model isn't the one with the lowest train loss — it's the one with the lowest val loss

## Next: [Part 4 — Text Generation →](04-text-generation.md)
