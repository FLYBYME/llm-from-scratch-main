import * as tf from "@tensorflow/tfjs";
import { type GPTConfig } from "./tokenizer.js";
import { TransformerBlock } from "./layers/block.js";
import { LearnedPositionEmbedding } from "./layers/embeddings.js";
import { ensureSymbolic } from "./type-guards.js";

/**
 * High-fidelity GPT Model builder.
 * Uses custom layers to construct a real Transformer architecture.
 */
export class GPTModel {
    private model: tf.LayersModel;

    constructor(config: GPTConfig) {
        // 1. Input: Sequence of token IDs [Batch, SeqLen]
        const tokenInputs = tf.input({ shape: [config.block_size], dtype: 'int32' });

        // 2. Token Embedding [Batch, SeqLen, embeddingDim]
        // Map token IDs to vectors.
        const tokenEmbeddings = ensureSymbolic(tf.layers.embedding({ 
            inputDim: config.vocab_size, 
            outputDim: config.embeddingDim,
            name: 'token_embeddings'
        }).apply(tokenInputs));

        // 3. Position Embedding [1, SeqLen, embeddingDim] (broadcasted to batch)
        // Map token positions to vectors.
        const positionEmbeddings = ensureSymbolic(new LearnedPositionEmbedding(config).apply(tokenInputs));

        // 4. Combined Embeddings + Dropout
        // Sum the token and position vectors to get the final representation of each token.
        let x = ensureSymbolic(tf.layers.add().apply([tokenEmbeddings, positionEmbeddings]));
        x = ensureSymbolic(tf.layers.dropout({ rate: config.dropout }).apply(x));

        // 5. Transformer Stack
        // Pass the representations through a series of Transformer blocks.
        for (let i = 0; i < config.numLayers; i++) {
            x = ensureSymbolic(new TransformerBlock(config).apply(x));
        }

        // 6. Final LayerNorm
        x = ensureSymbolic(tf.layers.layerNormalization({ axis: -1, epsilon: 1e-5, name: 'final_layer_norm' }).apply(x));
        
        // 7. Language Model Head
        // Project the final representations back to the vocabulary space to get logits for each token.
        const logits = ensureSymbolic(tf.layers.dense({ 
            units: config.vocab_size, 
            useBias: false,
            name: 'language_model_head'
        }).apply(x));

        this.model = tf.model({ inputs: tokenInputs, outputs: logits });
    }

    /**
     * Compiles the model with the Adam optimizer and sparse categorical crossentropy.
     */
    public compile(learningRate: number = 3e-4) {
        this.model.compile({
            optimizer: tf.train.adam(learningRate),
            loss: 'sparseCategoricalCrossentropy',
            metrics: ['accuracy']
        });
    }

    public getLayersModel(): tf.LayersModel {
        return this.model;
    }

    public summary() {
        this.model.summary();
    }
}
