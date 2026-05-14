import * as tf from "@tensorflow/tfjs";
import { EventEmitter } from "events";
import { CharTokenizer, type GPTConfig } from "./tokenizer.js";
import { GPTModel } from "./model.js";

export interface EvalResult {
    step: number;
    loss: number;
    accuracy: number;
    completion: string;
}

export interface TrainStepResult {
    step: number;
    loss: number;
    accuracy: number;
    msPerStep: number;
}

export interface GPTParams {
    input: string;
    outputFolder: string;
    params: {
        seed: number;
        tokensPerBlock: number;
        batchSize: number;
        numLayers: number;
        numHeads: number;
        embeddingDim: number;
        dropout: number;
    }
}

/**
 * High-level GPT API for easy training and interaction.
 * Acts as an orchestrator for the model, tokenizer, and training loop.
 */
export class GPT extends EventEmitter {
    private config: GPTConfig;
    private tokenizer!: CharTokenizer;
    private gptModel!: GPTModel;
    private inputText: string;
    private trainingParams: GPTParams['params'];

    constructor(options: GPTParams) {
        super();
        this.inputText = options.input;
        this.trainingParams = options.params;

        // Map demo params to internal GPTConfig
        this.config = {
            vocab_size: 0, // Set after tokenizer initialization
            block_size: options.params.tokensPerBlock,
            numLayers: options.params.numLayers,
            numHeads: options.params.numHeads,
            embeddingDim: options.params.embeddingDim,
            dropout: options.params.dropout
        };
    }

    async initializeTokenizer() {
        this.tokenizer = new CharTokenizer(this.inputText);
        this.config.vocab_size = this.tokenizer.vocab_size;
        console.log(`Tokenizer initialized with vocab size: ${this.config.vocab_size}`);
        this.emit('log', `Tokenizer initialized with vocab size: ${this.config.vocab_size}`);
    }

    async initializeWeights() {
        if (!this.tokenizer) await this.initializeTokenizer();
        this.gptModel = new GPTModel(this.config);
        this.gptModel.compile();
        console.log(`Model initialized and compiled.`);
        this.emit('log', `Model initialized and compiled.`);
    }

    async train(options: {
        steps: number;
        evalEvery: number;
        evalBatch: number;
        evalSteps: number;
        logEvery: number;
    }) {
        const model = this.gptModel.getLayersModel();
        const tokensArray = this.tokenizer.encode(this.inputText);

        this.emit('log', `Starting training for ${options.steps} steps...`);

        for (let step = 1; step <= options.steps; step++) {
            const start = performance.now();

            // Generate batch
            const { xs, ys } = tf.tidy(() => {
                const xBatch: number[][] = [];
                const yBatch: number[][] = [];

                for (let i = 0; i < this.trainingParams.batchSize; i++) {
                    const ix = Math.floor(Math.random() * (tokensArray.length - this.config.block_size - 1));
                    xBatch.push(tokensArray.slice(ix, ix + this.config.block_size));
                    yBatch.push(tokensArray.slice(ix + 1, ix + this.config.block_size + 1));
                }

                return {
                    xs: tf.tensor2d(xBatch, [this.trainingParams.batchSize, this.config.block_size], 'int32'),
                    ys: tf.tensor2d(yBatch, [this.trainingParams.batchSize, this.config.block_size], 'float32').reshape([this.trainingParams.batchSize, this.config.block_size, 1])
                };
            });

            // Train step
            const history = await model.trainOnBatch(xs, ys);
            
            const getVal = (v: any): number => {
                if (typeof v === 'number') return v;
                if (v instanceof tf.Tensor) {
                    const data = v.dataSync()[0];
                    return typeof data === 'number' ? data : 0;
                }
                return 0;
            };

            const loss = Array.isArray(history) ? getVal(history[0]) : getVal(history);
            const accuracy = Array.isArray(history) && history.length > 1 ? getVal(history[1]) : 0;
            const end = performance.now();

            tf.dispose([xs, ys]);

            if (step % options.logEvery === 0) {
                const result: TrainStepResult = {
                    step,
                    loss,
                    accuracy,
                    msPerStep: end - start
                };
                this.emit('train_step', result);
            }

            if (step % options.evalEvery === 0) {
                const completion = await this.generate("ROMEO:", 50);
                const result: EvalResult = {
                    step,
                    loss,
                    accuracy,
                    completion
                };
                this.emit('eval', result);
            }
        }

        this.emit('train_completion', 'Training finished successfully.');
    }

    async generate(prompt: string, maxNewTokens: number = 100): Promise<string> {
        const model = this.gptModel.getLayersModel();
        let currentSequence = this.tokenizer.encode(prompt);

        for (let i = 0; i < maxNewTokens; i++) {
            const seqLength = currentSequence.length;
            const context = seqLength > this.config.block_size
                ? currentSequence.slice(seqLength - this.config.block_size)
                : currentSequence;

            // Pad if necessary
            const paddedContext = context.length < this.config.block_size
                ? [...new Array(this.config.block_size - context.length).fill(0), ...context]
                : context;

            const nextTokenId = await tf.tidy(() => {
                const inputTensor = tf.tensor2d([paddedContext], [1, this.config.block_size], 'int32');
                const logits = model.predict(inputTensor);
                const logitsTensor = Array.isArray(logits) ? logits[0] : logits;
                if (!(logitsTensor instanceof tf.Tensor)) return 0;

                // Get last token logits
                const lastLogits = logitsTensor.slice([0, this.config.block_size - 1, 0], [1, 1, this.config.vocab_size]).flatten();

                // Sample (greedy for now or multinomial)
                const res = tf.argMax(lastLogits).dataSync()[0];
                return typeof res === 'number' ? res : 0;
            });

            currentSequence.push(nextTokenId);
        }

        return this.tokenizer.decode(currentSequence);
    }
}
