import * as tf from "@tensorflow/tfjs";
import { type GPTConfig, CharTokenizer } from "./tokenizer.js";
import { GPTModel } from "./model.js";


export async function train(textData: string, config: GPTConfig, batchSize: number = 32, maxSteps: number = 5000) {
    const tokenizer = new CharTokenizer(textData);
    const tokensArray = tokenizer.encode(textData);
    
    const gpt = new GPTModel(config);
    const model = gpt.getLayersModel();

    // 1. Compile with the highly-optimized sparse loss kernel
    model.compile({
        optimizer: tf.train.adam(1e-3),
        loss: 'sparseCategoricalCrossentropy',
    });

    // --- Model Metrics ---
    const totalParams = model.countParams();
    const weightBytes = totalParams * 4; // Assuming float32
    const mem = tf.memory();
    
    console.log(`\n--- Model Configuration ---`);
    console.log(`Backend: ${tf.getBackend()}`);
    console.log(`Architecture: ${config.n_layer}L / ${config.n_head}H / ${config.n_embd}D`);
    console.log(`Context Window: ${config.block_size} tokens`);
    console.log(`Vocabulary Size: ${config.vocab_size}`);
    console.log(`Total Parameters: ${(totalParams / 1e6).toFixed(2)}M`);
    
    console.log(`\n--- VRAM Metrics ---`);
    console.log(`Weight Memory: ${(weightBytes / (1024 * 1024)).toFixed(2)} MB`);
    console.log(`Active Tensors: ${mem.numTensors}`);
    console.log(`Current Allocation: ${(mem.numBytes / (1024 * 1024)).toFixed(2)} MB`);
    console.log(`---------------------------\n`);

    console.log("Entering training loop...");
    let lastTime = performance.now();
    
    for (let step = 0; step < maxSteps; step++) {
        const memBeforeStep = tf.memory().numTensors;
        
        // 2. Fast JS-based batch generator: Only uploads 4KB per step
        const { xs, ys } = tf.tidy(() => {
            const xBatch: number[][] = [];
            const yBatch: number[][] = [];

            for (let i = 0; i < batchSize; i++) {
                const ix = Math.floor(Math.random() * (tokensArray.length - config.block_size - 1));
                xBatch.push(tokensArray.slice(ix, ix + config.block_size));
                yBatch.push(tokensArray.slice(ix + 1, ix + config.block_size + 1));
            }

            return {
                xs: tf.tensor2d(xBatch, [batchSize, config.block_size], 'int32'),
                ys: tf.tensor2d(yBatch, [batchSize, config.block_size], 'float32').reshape([batchSize, config.block_size, 1])
            };
        });

        tf.engine().startScope();
        try {
            // 3. Optimized Fit: Bypasses the sync locks of trainOnBatch in Node.js
            const history = await model.fit(xs, ys, {
                batchSize: batchSize,
                epochs: 1,
                verbose: 0
            });

            const loss = (history.history.loss?.[0] as number) ?? 0;

            if (step % 10 === 0 && step > 0) {
                const currentTime = performance.now();
                const timePerStep = (currentTime - lastTime) / 10;
                const stepsPerSec = (1000 / timePerStep).toFixed(2);
                
                process.stdout.write(`\rStep ${step} | loss: ${loss.toFixed(4)} | speed: ${stepsPerSec} steps/s (${timePerStep.toFixed(0)}ms/step)`);
                lastTime = currentTime;
            }

            if (step % 50 === 0) {
                const memInfo = tf.memory();
                const leakAmount = memInfo.numTensors - memBeforeStep;
                console.log(`\n--- [Step ${step}] ---`);
                console.log(`Tensors: ${memInfo.numTensors} (${leakAmount >= 0 ? '+' : ''}${leakAmount} step leak)`);
                console.log(`VRAM: ${(memInfo.numBytes / (1024 * 1024)).toFixed(2)} MB`);
                console.log(`---------------------`);
            }
        } finally {
            tf.engine().endScope();
            tf.dispose([xs, ys]); // Clean up the batch tensors
        }

        // Prevent the JS event loop from starving
        await tf.nextFrame();

        if (step === maxSteps - 1) console.log("\nTraining complete.");
    }
    
    return { model, tokenizer };
}
