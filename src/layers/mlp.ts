import * as tf from "@tensorflow/tfjs";
import { type GPTConfig } from "../tokenizer.js";
import { ensureTensor } from "../type-guards.js";

/**
 * High-fidelity Feed-Forward Multi-Layer Perceptron (MLP).
 * Implements the point-wise feed-forward part of the Transformer block.
 * This layer processes each token position independently.
 */
export class FeedForward extends tf.layers.Layer {
    private config: GPTConfig;
    private linearFullyConnected: tf.layers.Layer;
    private linearProjection: tf.layers.Layer;
    private dropoutLayer: tf.layers.Layer;

    constructor(config: GPTConfig) {
        super({});
        this.config = config;

        // 1. Expanded dimension: Map embeddingDim to 4 * embeddingDim
        // This allows the model to learn more complex features by projecting into a higher-dimensional space.
        this.linearFullyConnected = tf.layers.dense({ 
            units: 4 * config.embeddingDim, 
            activation: 'gelu', 
            name: 'linear_fc' 
        });

        // 2. Projection back to original embeddingDim
        this.linearProjection = tf.layers.dense({ 
            units: config.embeddingDim, 
            name: 'linear_projection' 
        });

        this.dropoutLayer = tf.layers.dropout({ rate: config.dropout });
    }

    static get className() { return 'FeedForward'; }

    call(inputs: tf.Tensor | tf.Tensor[], kwargs: any): tf.Tensor {
        return tf.tidy(() => {
            const inputTensor = ensureTensor(inputs);
            
            // Apply the first linear layer with GELU activation
            let hiddenState = ensureTensor(this.linearFullyConnected.apply(inputTensor));
            
            // Project back to the original embedding dimension
            hiddenState = ensureTensor(this.linearProjection.apply(hiddenState));
            
            // Apply dropout for regularization
            hiddenState = ensureTensor(this.dropoutLayer.apply(hiddenState, kwargs));
            
            return hiddenState;
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

tf.serialization.registerClass(FeedForward);
