import { setupEnv } from "./env.js";
setupEnv();
import * as fs from "fs";
import * as path from "path";
import { train } from "./train.js";
import { generate } from "./generate.js";
import { type GPTConfig } from "./tokenizer.js";
import { fileURLToPath } from 'url';

async function main() {
    // 0. Ensure GPU bindings are loaded after environment setup
    await import("@tensorflow/tfjs-node-gpu");

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

    // 2. Define configuration
    // PERFORMANCE CONFIG: Optimized for a dedicated 16GB eGPU.
    const config: GPTConfig = {
        vocab_size: 0,    
        block_size: 256,  
        n_layer: 8,       
        n_head: 8,        
        n_embd: 512,      // Balanced at 512 to fit in VRAM with large batches
    };

    // 3. Train the model
    const batchSize = 128;  
    const maxSteps = 5000; 

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
