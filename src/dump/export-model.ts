import { setupEnv } from "./env.js";
setupEnv();
import * as fs from "fs";
import * as path from "path";
import { fileURLToPath } from 'url';
import { train } from "./train.js";
import { type GPTConfig } from "./tokenizer.js";

async function exportModel() {
    await import("@tensorflow/tfjs-node-gpu");
    const __filename = fileURLToPath(import.meta.url);
    const __dirname = path.dirname(__filename);
    const dataPath = path.join(__dirname, "..", "data", "shakespeare.txt");

    if (!fs.existsSync(dataPath)) {
        throw new Error("Shakespeare data not found!");
    }

    const textData = fs.readFileSync(dataPath, "utf-8");
    console.log(`Loaded ${textData.length} characters.`);

    const config: GPTConfig = {
        vocab_size: new Set(textData.split('')).size,
        block_size: 64,
        n_layer: 4,
        n_head: 4,
        n_embd: 128,
    };

    console.log("Training model for export (ultra-tiny run)...");
    const { model, tokenizer: _tokenizer } = await train(textData, config, 32, 10);

    const exportPath = path.join(__dirname, "..", "public", "model");
    console.log(`Saving model to ${exportPath}...`);

    // Save model topology and weights
    await model.save(`file://${exportPath}`);

    // Also save tokenizer metadata (the characters it knows)
    const tokenizerMeta = {
        chars: Array.from(new Set(textData.split(''))).sort()
    };
    fs.writeFileSync(path.join(exportPath, "tokenizer.json"), JSON.stringify(tokenizerMeta));

    console.log("Export complete!");
}

exportModel().catch(console.error);
