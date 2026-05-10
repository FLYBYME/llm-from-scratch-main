# Part 4: Text Generation

Your model is trained. Now let's make it write. Text generation with a GPT is **autoregressive**: generate one token at a time, append it to the input, and repeat.

Create a new file called `src/generate.ts` in your project. Once you've written it, you can import it in `src/main.ts` to generate text from your trained model.

## The Naive Approach: Greedy Decoding

Always pick the most probable next token.

```typescript
async function generateGreedy(model, tokenizer, config, prompt, maxNewTokens) {
    let currentSequence = tokenizer.encode(prompt);
    for (let i = 0; i < maxNewTokens; i++) {
        const inputTensor = tf.tensor2d([currentSequence.slice(-config.block_size)], [1, config.block_size]);
        const logits = model.predict(inputTensor);
        const nextToken = logits.argMax(-1).dataSync()[0];
        currentSequence.push(nextToken);
    }
    return tokenizer.decode(currentSequence);
}
```

This is deterministic — the same prompt always produces the same output. It tends to be repetitive and boring because the highest-probability continuation reinforces itself.

## Temperature

Scale the logits before applying softmax. Higher temperature = more random, lower = more deterministic.

```typescript
const scaledLogits = finalStepLogits.div(tf.scalar(temperature));
```

The math: softmax computes `exp(logit_i) / sum(exp(logit_j))`. Dividing all logits by temperature changes the distribution:
- **T = 1.0**: Normal probabilities
- **T → 0**: Approaches greedy (argmax)
- **T > 1.0**: Flattens the distribution, giving rare tokens more chance
- **T = 0.7-0.9**: The typical sweet spot for coherent but varied text

## Top-k Sampling

Only consider the k most probable tokens. Set everything else to `-inf`.

```typescript
// Not natively in TFJS core, but you can zero out all but top-k before softmax
```

This prevents the model from sampling extremely unlikely tokens. With a character-level model (vocab=65), `top_k=40` is reasonable — it still considers most characters but excludes the very unlikely ones.

## The Full Generate Function

```typescript
export async function generate(
    model: tf.LayersModel, 
    tokenizer: CharTokenizer, 
    config: GPTConfig, 
    prompt: string, 
    maxNewTokens: number = 200, 
    temperature: number = 0.8
): Promise<string> {
    let currentSequence = tokenizer.encode(prompt);

    for (let i = 0; i < maxNewTokens; i++) {
        const nextTokenId = tf.tidy(() => {
            const context = currentSequence.slice(-config.block_size);
            const inputTensor = tf.tensor2d([context], [1, config.block_size]);
            
            const logits = model.predict(inputTensor) as tf.Tensor3D;
            const finalStepLogits = logits.slice([0, config.block_size - 1, 0], [1, 1, config.vocab_size])
                                          .squeeze([0, 1]) as tf.Tensor1D;

            const scaledLogits = finalStepLogits.div(tf.scalar(temperature));
            const probs = tf.softmax(scaledLogits);
            
            return tf.multinomial(probs as tf.Tensor1D, 1, undefined, true).dataSync()[0];
        });

        if (nextTokenId !== undefined) {
            currentSequence.push(nextTokenId);
        }
    }

    return tokenizer.decode(currentSequence);
}
```

The pipeline for each token:
1. Run the model on the current sequence → get logits for next position
2. Apply temperature scaling
3. Filter with top-k (remove very unlikely tokens)
4. Convert to probabilities with softmax
5. Sample from the distribution with `multinomial`
6. Append the sampled token and repeat

`@torch.no_grad()` disables gradient computation — we don't need it for inference and it saves memory.

The function takes `stoi`/`itos` mappings from the training data — these define how characters map to token IDs and back.

### Using the Generate Function

In your `src/main.ts`, you can call it like this:

```typescript
const prompt = "ROMEO:";
const generatedText = await generate(model, tokenizer, config, prompt, 100);
console.log(generatedText);
```

## Reproducibility with Seeds

Generation involves random sampling (`torch.multinomial`), so the same prompt produces different output each time. To get reproducible results, set a seed before generating:

```python
torch.manual_seed(42)
print(generate(model, "To be or not", stoi, itos, temperature=0.8))
# same output every time with seed=42
```

From the command line:
```bash
python generate.py checkpoint_final.pt --prompt "To be or not" --seed 42
```

## Try Different Settings

```python
checkpoint = torch.load("checkpoint_final.pt", weights_only=False)
config = checkpoint["config"]
stoi = checkpoint["stoi"]
itos = checkpoint["itos"]

model = GPT(config)
model.load_state_dict(checkpoint["model_state_dict"])

# deterministic, repetitive
print(generate(model, "To be or not to be", stoi, itos, temperature=0.1))

# balanced
print(generate(model, "To be or not to be", stoi, itos, temperature=0.8))

# creative, potentially incoherent
print(generate(model, "To be or not to be", stoi, itos, temperature=1.5))
```

## What to Expect

Here are real samples from a training run (6L/6H/384D on Shakespeare):

### Step 200 (val loss ~3.5) — Random characters
```
To be or notis p ce mei odorethleedetire'ilethed ye m arkesothir fnon b tigb'i.
```

### Step 1000 (val loss 1.64) — Words and structure emerging
```
To be or nothing are good men,
The profent of little, our actory.

CORIOLANUS:
Is it now of your many death?
```

### Step 2400 (val loss ~1.60) — Peak quality, plausible Shakespeare
```
To be or not to be some of you shall know
That everlature by Romeo: what news,
Which you had knock'd my part to speak
```

Note: the best output is around step 1500-2500. After that, the model overfits and starts regurgitating memorized training data (see Part 3 for details).

## Key Takeaways

- Autoregressive generation: predict one token, append, repeat
- Greedy decoding is deterministic and repetitive
- Temperature controls randomness (0.7-0.9 is usually good)
- Top-k removes extremely unlikely tokens
- With character-level models, generate samples during training to watch the model learn

## Next: [Part 5 — Putting It All Together →](05-putting-it-together.md)
