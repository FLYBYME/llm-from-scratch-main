import * as tf from "@tensorflow/tfjs";
import { type GPTConfig } from "../tokenizer.js";
import { ensureTensor } from "../type-guards.js";

/**
 * High-fidelity Causal Multi-Head Self-Attention.
 * Implements the core mechanism of the Transformer architecture.
 */
export class CausalSelfAttention extends tf.layers.Layer {
    private config: GPTConfig;
    private combinedAttentionProjection: tf.layers.Layer;
    private outputProjection: tf.layers.Layer;
    private attentionDropout: tf.layers.Layer;
    private residualDropout: tf.layers.Layer;

    constructor(config: GPTConfig) {
        super({});
        this.config = config;

        // Key, Query, Value projections in a single dense layer for efficiency
        this.combinedAttentionProjection = tf.layers.dense({
            units: 3 * config.embeddingDim,
            useBias: false,
            name: 'attention_projection'
        });

        // Output projection
        this.outputProjection = tf.layers.dense({
            units: config.embeddingDim,
            useBias: false,
            name: 'output_projection'
        });

        // Regularization
        this.attentionDropout = tf.layers.dropout({ rate: config.dropout });
        this.residualDropout = tf.layers.dropout({ rate: config.dropout });
    }

    static get className() { return 'CausalSelfAttention'; }

    /**
     * The core attention computation logic.
     * Uses tf.tidy for aggressive memory management.
     */
    call(inputs: tf.Tensor | tf.Tensor[], kwargs: any): tf.Tensor {
        return tf.tidy(() => {
            const inputTensor = ensureTensor(inputs);

            const { shape } = inputTensor;
            if (shape.length < 3) throw new Error(`Expected rank 3 input, got rank ${shape.length}`);

            const batchSize = shape[0];
            const sequenceLength = shape[1];
            const embeddingDimension = shape[2];

            if (typeof batchSize !== 'number' || typeof sequenceLength !== 'number' || typeof embeddingDimension !== 'number') {
                throw new Error("Batch size, sequence length, and embedding dimension must be numbers");
            }

            if (embeddingDimension !== this.config.embeddingDim) {
                throw new Error(`Expected embedding dimension ${this.config.embeddingDim}, got ${embeddingDimension}`);
            }

            const numHeads = this.config.numHeads;
            const headSize = embeddingDimension / numHeads;

            // 1. QKV Projections: Map input to Query, Key, and Value spaces
            // Output shape: [BatchSize, SequenceLength, 3 * EmbeddingDim]
            const qkv = ensureTensor(this.combinedAttentionProjection.apply(inputTensor));

            // Split into Q, K, V (Query, Key, Value)
            // Each shape: [BatchSize, SequenceLength, EmbeddingDim]
            const [query, key, value] = tf.split(qkv, 3, -1);
            if (!query || !key || !value) throw new Error("Split failed");

            // 2. Multi-head split
            // Reshape to [Batch, Seq, NumHeads, HeadSize] then transpose to [Batch, NumHeads, Seq, HeadSize]
            // This allows us to perform attention on each head independently.
            const query_heads = query.reshape([batchSize, sequenceLength, numHeads, headSize]).transpose([0, 2, 1, 3]);
            const key_heads = key.reshape([batchSize, sequenceLength, numHeads, headSize]).transpose([0, 2, 1, 3]);
            const value_heads = value.reshape([batchSize, sequenceLength, numHeads, headSize]).transpose([0, 2, 1, 3]);

            // 3. Scaled Dot-Product Attention
            // attention_scores = (Query @ Key^T) / sqrt(HeadSize)
            // Shape: [Batch, NumHeads, Seq, Seq]
            let attention_scores = tf.matMul(query_heads, key_heads, false, true).div(tf.sqrt(tf.scalar(headSize)));

            // 4. Causal Mask
            // Apply mask so that tokens can only attend to current and previous tokens (not future tokens)
            const mask = tf.tidy(() => {
                const ones = tf.ones([sequenceLength, sequenceLength]);
                // Create lower triangular matrix (1s on and below diagonal)
                return tf.linalg.bandPart(ones, -1, 0);
            });

            // Fill upper triangle with very large negative number to effectively zero out those scores after softmax
            const negativeInfinity = tf.scalar(-1e9);
            attention_scores = tf.where(mask.asType('bool'), attention_scores, negativeInfinity);

            // 5. Softmax & Dropout: Normalize scores to sum to 1 and apply dropout for regularization
            attention_scores = tf.softmax(attention_scores, -1);
            attention_scores = ensureTensor(this.attentionDropout.apply(attention_scores, kwargs));

            // 6. Attention Weighted Values: Multiply attention scores by Value to get refined token representations
            // Output shape: [Batch, NumHeads, Seq, HeadSize]
            let output = tf.matMul(attention_scores, value_heads);

            // 7. Re-assemble heads
            // Transpose back to [Batch, Seq, NumHeads, HeadSize] then flatten to [Batch, Seq, EmbeddingDim]
            output = output.transpose([0, 2, 1, 3]).reshape([batchSize, sequenceLength, embeddingDimension]);
            // 8. Output projection & residual dropout
            output = ensureTensor(this.outputProjection.apply(output));
            output = ensureTensor(this.residualDropout.apply(output, kwargs));

            return output;
        });
    }

    computeOutputShape(inputShape: tf.Shape | tf.Shape[]): tf.Shape | tf.Shape[] {
        return inputShape;
    }

    getConfig(): tf.serialization.ConfigDict {
        const config = super.getConfig();
        Object.assign(config, { gptConfig: this.config });
        return config;
    }
}

tf.serialization.registerClass(CausalSelfAttention);
