import * as tf from "@tensorflow/tfjs";
import { type GPTConfig, CharTokenizer } from "./tokenizer.js";
import { GPTModel } from "./model.js";
import { assertTensor3D } from "./type-guards.js";

export async function train(textData: string, config: GPTConfig, batchSize: number = 64, maxSteps: number = 5000) {
    const tokenizer = new CharTokenizer(textData);
    const tokensArray = tokenizer.encode(textData);
    
    // Move the entire dataset to the GPU once to eliminate Thunderbolt transfer bottlenecks
    const allTokens = tf.tensor1d(tokensArray, 'int32');
    
    const gpt = new GPTModel(config);
    const model = gpt.getLayersModel();

    // Adam optimizer
    const optimizer = tf.train.adam(1e-3);
    model.compile({
        optimizer: optimizer,
        loss: tf.losses.softmaxCrossEntropy,
    });

    console.log(`Model initialized: ${config.n_layer}L/${config.n_head}H/${config.n_embd}D`);

    // High-performance batch generator (Slices directly on the GPU)
    const getBatch = (): { x: tf.Tensor2D, y: tf.Tensor3D } => {
        return tf.tidy(() => {
            const xIndices: tf.Tensor2D[] = [];
            const yIndices: tf.Tensor2D[] = [];

            for (let i = 0; i < batchSize; i++) {
                const ix = Math.floor(Math.random() * (tokensArray.length - config.block_size - 1));
                
                // These slices now happen directly in VRAM
                xIndices.push(allTokens.slice([ix], [config.block_size]).expandDims(0) as tf.Tensor2D);
                yIndices.push(allTokens.slice([ix + 1], [config.block_size]).expandDims(0) as tf.Tensor2D);
            }

            const xTensor = tf.concat(xIndices, 0);
            const yIndicesTensor = tf.concat(yIndices, 0);
            const yTensor = tf.oneHot(yIndicesTensor, config.vocab_size);
            
            return { x: xTensor as tf.Tensor2D, y: assertTensor3D(yTensor) };
        });
    };

    console.log("Entering training loop...");
    let lastTime = performance.now();
    
    for (let step = 0; step < maxSteps; step++) {
        const stepStartTime = performance.now();
        
        const batch = getBatch();
        const lossTensor = await model.trainOnBatch(batch.x, batch.y) as tf.Tensor;
        const loss = (await lossTensor.data())[0];

        tf.dispose([batch.x, batch.y, lossTensor]);

        if (step % 10 === 0 && step > 0) {
            const currentTime = performance.now();
            const timePerStep = (currentTime - lastTime) / 10;
            const stepsPerSec = (1000 / timePerStep).toFixed(2);
            
            if (typeof loss === 'number') {
                process.stdout.write(`\rStep ${step} | loss: ${loss.toFixed(4)} | speed: ${stepsPerSec} steps/s (${timePerStep.toFixed(0)}ms/step)`);
            }
            lastTime = currentTime;
        }

        if (step === maxSteps - 1) console.log("\nTraining complete.");
    }
    
    return { model, tokenizer };
}
