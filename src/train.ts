import * as tf from "@tensorflow/tfjs";
import { type GPTConfig, CharTokenizer } from "./tokenizer.js";
import { GPTModel } from "./model.js";
import { assertTensor3D } from "./type-guards.js";

export async function train(textData: string, config: GPTConfig, batchSize: number = 64, maxSteps: number = 5000) {
    const tokenizer = new CharTokenizer(textData);
    const tokens = tokenizer.encode(textData);
    
    const gpt = new GPTModel(config);
    const model = gpt.getLayersModel();

    // Adam optimizer (TFJS does not have a native AdamW, but Adam suffices for this scale)
    const optimizer = tf.train.adam(1e-3);
    model.compile({
        optimizer: optimizer,
        loss: tf.losses.softmaxCrossEntropy,
    });

    console.log(`Model initialized: ${config.n_layer}L/${config.n_head}H/${config.n_embd}D`);

    // Batch generator
    const getBatch = (): { x: tf.Tensor2D, y: tf.Tensor3D } => {
        return tf.tidy(() => {
            const xBatch: number[][] = [];
            const yBatch: number[][] = []; // Targets are shifted by one

            for (let i = 0; i < batchSize; i++) {
                const ix = Math.floor(Math.random() * (tokens.length - config.block_size - 1));
                xBatch.push(tokens.slice(ix, ix + config.block_size));
                
                // For cross entropy, targets are often one-hot encoded in TFJS
                yBatch.push(tokens.slice(ix + 1, ix + config.block_size + 1)); 
            }

            const xTensor = tf.tensor2d(xBatch, [batchSize, config.block_size], 'int32');
            const yTensor = tf.oneHot(tf.tensor2d(yBatch, [batchSize, config.block_size], 'int32'), config.vocab_size);
            
            return { x: xTensor, y: assertTensor3D(yTensor) };
        });
    };

    for (let step = 0; step < maxSteps; step++) {
        const batch = getBatch();
        
        // model.fit acts as the forward pass, backward pass, and optimizer.step()
        const history = await model.fit(batch.x, batch.y, {
            batchSize: batchSize,
            epochs: 1,
            verbose: 0
        });

        tf.dispose([batch.x, batch.y]); // Crucial to avoid WebGL memory leaks

        if (step % 100 === 0) {
            const losses = history.history['loss'];
            const loss = losses?.[0];
            if (typeof loss === 'number') {
                console.log(`Step ${step} | loss: ${loss.toFixed(4)}`);
            }
        }
    }
    
    return { model, tokenizer };
}
