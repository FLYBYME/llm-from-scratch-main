import { GPT, type EvalResult, type TrainStepResult } from "./gpt.js";
import * as fs from "fs/promises";
import * as path from "path";
import { fileURLToPath } from "url";
import { setupEnv } from "./env.js";

// Setup environment for GPU acceleration and Node polyfills
await setupEnv();

async function demo() {
    const __filename = fileURLToPath(import.meta.url);
    const __dirname = path.dirname(__filename);

    const inputPath = path.join(__dirname, "..", "data", "shakespeare.txt");
    const outPutPath = path.join(__dirname, "..", "output");

    const inputText = await fs.readFile(inputPath, "utf-8");
    const model = new GPT({
        input: inputText,
        outputFolder: outPutPath,
        params: {
            /**
             * @field seed 
             * Seed for random number generator
             */
            seed: 42,

            /**
             * @field tokensPerBlock 
             * Max sequence length the model can handle
             * 
             * WHY: Determines the context window of the model.
             * The larger the context window, the more text the model can "remember" 
             * from previous tokens, but the more computationally expensive it is to train.
             * 
             * WHAT: 
             * Example: If tokensPerBlock is 32, the model can process sequences of up to 32 tokens
             * at a time. This means the model can "remember" up to 32 previous tokens when generating
             * the next token.
             */
            tokensPerBlock: 32,

            /**
             * @field batchSize 
             * Number of sequences processed in parallel
             * 
             * WHY: Larger batch sizes can lead to more stable training 
             * and better GPU utilization, but they require more memory.
             */
            batchSize: 80,

            /**
             * @field numLayers
             * Number of transformer blocks
             * 
             * WHY: Determines the depth of the model.
             * The deeper the model, the more text it can "understand" and the more complex
             * patterns it can learn, but the more computationally expensive it is to train.
             * 
             * WHAT: 
             * Example: If numLayers is 1, the model has 1 transformer block.
             * This means the model can process sequences of up to 1 token at a time.
             */
            numLayers: 1,

            /**
             * @field numHeads
             * Number of attention heads
             * 
             * WHY: The number of attention heads determines how many different
             * representation subspaces the model learns simultaneously.
             * Each head can focus on different aspects of the input (e.g., syntax, semantics,
             * positional relationships), allowing the model to capture complex dependencies.
             * Increasing the number of heads generally improves model performance 
             * but also increases computational cost.
             * 
             * WHAT: 
             * Example: If numHeads is 1, the model has 1 attention head.
             * This means the model can process sequences of up to 1 token at a time.
             */
            numHeads: 1,

            /**
             * @field embeddingDim
             * Dimension of token embeddings
             * 
             * WHY: The embedding dimension determines the size of the vector space
             * in which token representations are learned. A larger embedding dimension
             * allows the model to capture more complex relationships between tokens,
             * but it also increases the number of parameters and computational cost.
             * 
             * WHAT: 
             * Example: If embeddingDim is 64, the model has 64 dimensions for each token.
             * This means the model can process sequences of up to 64 tokens at a time.
             */
            embeddingDim: 64,

            /**
             * @field dropout
             * Dropout rate for regularization
             * 
             * WHY: Dropout is a regularization technique that randomly sets a fraction of
             * neuron activations to zero during training. This prevents overfitting
             * by forcing the model to learn redundant representations and reduces
             * the model's reliance on any single neuron.
             * 
             * WHAT: 
             * Example: If dropout is 0.1, the model drops 10% of neuron activations during training.
             */
            dropout: 0.1
        }
    });

    model.on('eval', (data: EvalResult) => {
        console.log(data);
    });

    model.on('eval_completion', (data: string) => {
        console.log(data);
    });

    model.on('train_step', (data: TrainStepResult) => {
        console.log(data);
    });

    model.on('train_completion', (data: string) => {
        console.log(data);
    });

    await model.initializeWeights()
    await model.initializeTokenizer()
    await model.train({
        steps: 100,
        evalEvery: 50,
        evalBatch: 32,
        evalSteps: 50,
        logEvery: 10,
    })
}

demo()
