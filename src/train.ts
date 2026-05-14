import * as tf from "@tensorflow/tfjs";
import { type GPTConfig, CharTokenizer } from "./tokenizer.js";
import { GPTModel } from "./model.js";
import { assertTensor3D } from "./type-guards.js";

export async function train(textData: string, config: GPTConfig, batchSize: number = 32, maxSteps: number = 5000) {
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

    console.log("Entering training loop...");
    let lastTime = performance.now();
    
    for (let step = 0; step < maxSteps; step++) {
        tf.engine().startScope();

        try {
            // 1. Generate random starting indices
            const startIndices = Array.from({ length: batchSize }, () => 
                Math.floor(Math.random() * (tokensArray.length - config.block_size - 1))
            );

            // 2. High-speed GPU gathering
            const { xs, ys } = tf.tidy(() => {
                const startTensor = tf.tensor1d(startIndices, 'int32');
                const range = tf.range(0, config.block_size, 1, 'int32');
                
                const xIndices = startTensor.expandDims(1).add(range).flatten().toInt();
                const batchXs = allTokens.gather(xIndices).reshape([batchSize, config.block_size]);
                
                const yIndices = startTensor.add(1).expandDims(1).add(range).flatten().toInt();
                const batchYs = tf.oneHot(
                    allTokens.gather(yIndices).reshape([batchSize, config.block_size]).toInt(), 
                    config.vocab_size
                );
                
                return { xs: batchXs, ys: batchYs };
            });

            // 3. Train
            const result = await model.trainOnBatch(xs, ys);
            
            // 4. Safely extract the loss value synchronously
            let loss: number;
            if (typeof result === 'number') {
                loss = result;
            } else if (Array.isArray(result)) {
                const first = result[0];
                loss = typeof first === 'number' ? first : first.dataSync()[0];
            } else {
                loss = result.dataSync()[0];
            }

            if (step % 10 === 0 && step > 0) {
                const currentTime = performance.now();
                const timePerStep = (currentTime - lastTime) / 10;
                const stepsPerSec = (1000 / timePerStep).toFixed(2);
                
                process.stdout.write(`\rStep ${step} | loss: ${loss.toFixed(4)} | speed: ${stepsPerSec} steps/s (${timePerStep.toFixed(0)}ms/step)`);
                lastTime = currentTime;
            }

            if (step % 50 === 0) {
                console.log(`\n[Step ${step}] Active Tensors: ${tf.memory().numTensors}`);
            }

        } finally {
            tf.engine().endScope();
        }

        // Prevent the JS event loop from starving
        await tf.nextFrame();

        if (step === maxSteps - 1) console.log("\nTraining complete.");
    }
    
    return { model, tokenizer };
}
