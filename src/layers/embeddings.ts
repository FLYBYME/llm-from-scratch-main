import * as tf from "@tensorflow/tfjs";
import { type GPTConfig } from "../tokenizer.js";
import { ensureTensor } from "../type-guards.js";

/**
 * Learned Position Embedding Layer.
 * Unlike sinusoidal embeddings, we learn a specific vector for each position index.
 * This allows the model to learn its own representation of "sequence order".
 */
export class LearnedPositionEmbedding extends tf.layers.Layer {
    private positionEmbeddingLayer: tf.layers.Layer;
    private config: GPTConfig;

    constructor(config: GPTConfig) {
        super({});
        this.config = config;
        this.positionEmbeddingLayer = tf.layers.embedding({ 
            inputDim: config.block_size, 
            outputDim: config.embeddingDim,
            name: 'position_embeddings'
        });
    }

    static get className() { return 'LearnedPositionEmbedding'; }

    /**
     * Maps input sequence to learned position embeddings.
     */
    call(inputs: tf.Tensor | tf.Tensor[]): tf.Tensor {
        return tf.tidy(() => {
            const inputTensor = ensureTensor(inputs);
            const { shape } = inputTensor;
            if (shape.length < 2) throw new Error(`Expected at least rank 2 input, got rank ${shape.length}`);
            
            const sequenceLength = shape[1];
            if (typeof sequenceLength !== 'number') {
                throw new Error("Sequence length must be a number");
            }
            
            // Create position indices: [0, 1, ..., sequenceLength-1]
            const positionIndices = tf.range(0, sequenceLength, 1, 'int32').expandDims(0);
            
            return ensureTensor(this.positionEmbeddingLayer.apply(positionIndices));
        });
    }

    /**
     * Calculates the output shape of the layer.
     */
    computeOutputShape(inputShape: tf.Shape | tf.Shape[]): tf.Shape | tf.Shape[] {
        // If the first element is an array, we are dealing with tf.Shape[]
        const first = inputShape[0];
        if (Array.isArray(first)) {
            const batchDim = first[0];
            const seqDim = first[1];
            return [typeof batchDim === 'number' ? batchDim : null, typeof seqDim === 'number' ? seqDim : null, this.config.embeddingDim];
        }
        
        // Otherwise it's a single tf.Shape
        const batchDim = inputShape[0];
        const seqDim = inputShape[1];
        return [typeof batchDim === 'number' ? batchDim : null, typeof seqDim === 'number' ? seqDim : null, this.config.embeddingDim];
    }
}

tf.serialization.registerClass(LearnedPositionEmbedding);
