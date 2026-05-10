import { setupEnv } from "./env.js";
setupEnv();
import "@tensorflow/tfjs-node";
import * as fs from "fs";
import * as path from "path";
import { train } from "./train.js";
import { generate } from "./generate.js";
import { type GPTConfig } from "./tokenizer.js";
import { fileURLToPath } from 'url';

async function main() {
    // 1. Load data
    const __filename = fileURLToPath(import.meta.url);
    const __dirname = path.dirname(__filename);
    const dataPath = path.join(__dirname, "..", "data", "shakespeare.txt");
    let textData = "Hello, world! This is a test string for training.";
    
    if (fs.existsSync(dataPath)) {
        textData = fs.readFileSync(dataPath, "utf-8");
        console.log(`Loaded ${textData.length} characters from ${dataPath}`);
    } else {
        console.log("Shakespeare data not found, using fallback string.");
    }

    // 2. Define configuration (Tiny config for quick demo)
    const config: GPTConfig = {
        vocab_size: 0, // Will be set by tokenizer
        block_size: 64,
        n_layer: 4,
        n_head: 4,
        n_embd: 128,
    };

    // 3. Train the model
    const batchSize = 32;
    const maxSteps = 500; // Small number of steps for demo
    
    console.log("Starting training...");
    const { model, tokenizer } = await train(textData, { ...config, vocab_size: new Set(textData.split('')).size }, batchSize, maxSteps);

    // 4. Generate text
    console.log("\nGenerating text...");
    const prompt = "ROMEO:";
    const generatedText = await generate(model, tokenizer, config, prompt, 100);
    
    console.log("--- Generated Text ---");
    console.log(generatedText);
    console.log("----------------------");
}

main().catch(console.error);
