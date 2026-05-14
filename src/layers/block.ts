import * as tf from "@tensorflow/tfjs";
import { type GPTConfig } from "../tokenizer.js";
import { CausalSelfAttention } from "./attention.js";
import { FeedForward } from "./mlp.js";
import { ensureTensor } from "../type-guards.js";

/**
 * Standard Transformer Block.
 * Composes Self-Attention and MLP with LayerNorm and Residual connections.
 * This is the fundamental unit of the GPT architecture.
 */
export class TransformerBlock extends tf.layers.Layer {
    private layerNormBeforeAttention: tf.layers.Layer;
    private selfAttention: CausalSelfAttention;
    private layerNormBeforeMLP: tf.layers.Layer;
    private feedForwardMLP: FeedForward;

    constructor(config: GPTConfig) {
        super({});
        
        // 1. Layer normalization before the attention mechanism
        this.layerNormBeforeAttention = tf.layers.layerNormalization({ axis: -1, epsilon: 1e-5, name: 'ln_1' });
        this.selfAttention = new CausalSelfAttention(config);
        
        // 2. Layer normalization before the MLP (Feed-Forward) mechanism
        this.layerNormBeforeMLP = tf.layers.layerNormalization({ axis: -1, epsilon: 1e-5, name: 'ln_2' });
        this.feedForwardMLP = new FeedForward(config);
    }

    static get className() { return 'TransformerBlock'; }

    call(inputs: tf.Tensor | tf.Tensor[], kwargs: any): tf.Tensor {
        return tf.tidy(() => {
            const inputTensor = ensureTensor(inputs);

            // Path 1: Attention with Residual connection
            // x = x + SelfAttention(LayerNorm(x))
            // This allows the model to refine token representations based on their context.
            const norm1 = ensureTensor(this.layerNormBeforeAttention.apply(inputTensor));
            const attentionOutput = ensureTensor(this.selfAttention.apply(norm1, kwargs));
            const x1 = tf.add(inputTensor, attentionOutput);

            // Path 2: MLP with Residual connection
            // x = x1 + FeedForward(LayerNorm(x1))
            // This allows the model to process each token position independently.
            const norm2 = ensureTensor(this.layerNormBeforeMLP.apply(x1));
            const mlpOutput = ensureTensor(this.feedForwardMLP.apply(norm2, kwargs));
            const x2 = tf.add(x1, mlpOutput);

            return x2;
        });
    }

    computeOutputShape(inputShape: tf.Shape | tf.Shape[]): tf.Shape | tf.Shape[] {
        return inputShape;
    }
}

tf.serialization.registerClass(TransformerBlock);
