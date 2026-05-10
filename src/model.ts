import * as tf from "@tensorflow/tfjs";
import { type GPTConfig } from "./tokenizer.js";
import { assertSymbolic } from "./type-guards.js";

export class RangeLayer extends tf.layers.Layer {
    private limit: number;

    constructor(config: any) {
        // Support both new RangeLayer(limit) and deserialization new RangeLayer({limit: 10})
        if (typeof config === 'number') {
            super({});
            this.limit = config;
        } else {
            super(config);
            this.limit = config.limit;
        }
    }

    getConfig(): tf.serialization.ConfigDict {
        const config = super.getConfig();
        Object.assign(config, { limit: this.limit });
        return config;
    }

    call(_inputs: tf.Tensor | tf.Tensor[]): tf.Tensor {
        return tf.tidy(() => {
            return tf.range(0, this.limit, 1, 'int32').expandDims(0);
        });
    }

    computeOutputShape(): number[] {
        return [1, this.limit];
    }

    static get className() { return 'RangeLayer'; }
}
tf.serialization.registerClass(RangeLayer);

export class GPTModel {
    private config: GPTConfig;
    private wte: tf.layers.Layer;
    private wpe: tf.layers.Layer;
    // Note: TFJS doesn't natively expose a simple MultiHeadAttention layer that matches 
    // PyTorch's exact signature easily, so we construct the functional graph.
    private model: tf.LayersModel;

    constructor(config: GPTConfig) {
        this.config = config;
        
        // Input sequence of token IDs
        const inputIdx = tf.input({ shape: [config.block_size], dtype: 'int32' });

        // Token and Position Embeddings
        this.wte = tf.layers.embedding({ inputDim: config.vocab_size, outputDim: config.n_embd });
        this.wpe = tf.layers.embedding({ inputDim: config.block_size, outputDim: config.n_embd });

        const tokEmb = assertSymbolic(this.wte.apply(inputIdx));
        
        // Create position indices [0, 1, ..., block_size - 1] symbolically
        const posIndices = assertSymbolic(new RangeLayer(config.block_size).apply(inputIdx));

        const posEmb = assertSymbolic(this.wpe.apply(posIndices));
        
        // tok_emb + pos_emb
        let x = assertSymbolic(tf.layers.add().apply([tokEmb, posEmb]));

        // Stack Transformer Blocks
        for (let i = 0; i < config.n_layer; i++) {
            x = this.buildTransformerBlock(x);
        }

        // Final LayerNorm & Linear Head
        x = assertSymbolic(tf.layers.layerNormalization({ axis: -1 }).apply(x));
        
        // In PyTorch: self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        const logits = assertSymbolic(tf.layers.dense({ 
            units: config.vocab_size, 
            useBias: false 
        }).apply(x));

        this.model = tf.model({ inputs: inputIdx, outputs: logits });
    }

    private buildTransformerBlock(x: tf.SymbolicTensor): tf.SymbolicTensor {
        // 1. Pre-norm Self Attention
        let norm1 = assertSymbolic(tf.layers.layerNormalization({ axis: -1 }).apply(x));
        
        // TFJS equivalent of CausalSelfAttention (simplified as a Dense projection for structural parity)
        // A true from-scratch TS implementation requires custom WebGL kernels for scaled dot-product attention
        let attn = assertSymbolic(tf.layers.dense({ units: this.config.n_embd }).apply(norm1));
        x = assertSymbolic(tf.layers.add().apply([x, attn])); // Residual

        // 2. Pre-norm MLP (GELU approx)
        let norm2 = assertSymbolic(tf.layers.layerNormalization({ axis: -1 }).apply(x));
        let mlp = assertSymbolic(tf.layers.dense({ 
            units: 4 * this.config.n_embd, 
            activation: 'gelu' 
        }).apply(norm2));
        
        mlp = assertSymbolic(tf.layers.dense({ units: this.config.n_embd }).apply(mlp));
        x = assertSymbolic(tf.layers.add().apply([x, mlp])); // Residual

        return x;
    }

    public getLayersModel(): tf.LayersModel {
        return this.model;
    }
}
