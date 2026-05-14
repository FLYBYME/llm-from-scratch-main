import * as tf from "@tensorflow/tfjs";
import { type GPTConfig, CharTokenizer } from "./tokenizer.js";
import { assertTensor1D, assertTensor3D } from "./type-guards.js";

export async function generate(
    model: tf.LayersModel, 
    tokenizer: CharTokenizer, 
    config: GPTConfig, 
    prompt: string, 
    maxNewTokens: number = 200, 
    temperature: number = 0.8
): Promise<string> {
    
    let currentSequence = tokenizer.encode(prompt);

    for (let i = 0; i < maxNewTokens; i++) {
        // Truncate to block_size
        const seqLength = currentSequence.length;
        const context = seqLength > config.block_size 
            ? currentSequence.slice(seqLength - config.block_size) 
            : currentSequence;

        // Pad sequence if it's smaller than block_size (TFJS dense models expect fixed input shapes)
        const paddedContext = context.length < config.block_size
            ? [...new Array(config.block_size - context.length).fill(0), ...context]
            : context;

        const nextTokenIdTensor = tf.tidy(() => {
            const inputTensor = tf.tensor2d([paddedContext], [1, config.block_size], 'int32');
            
            // Get logits: shape (1, block_size, vocab_size)
            const logits = assertTensor3D(model.predict(inputTensor));
            
            // Pluck the logits for the final time step
            const finalStepLogits = assertTensor1D(logits.slice([0, config.block_size - 1, 0], [1, 1, config.vocab_size])
                                          .squeeze([0, 1]));

            // Apply temperature
            const scaledLogits = finalStepLogits.div(tf.scalar(temperature));
            
            // Softmax to get probabilities
            const probs = tf.softmax(scaledLogits);
            
            // Return the sampler tensor to be read asynchronously
            return tf.multinomial(assertTensor1D(probs), 1, undefined, true);
        });

        const nextTokenIdArray = await nextTokenIdTensor.data();
        nextTokenIdTensor.dispose();
        const nextTokenId = nextTokenIdArray[0];

        if (nextTokenId !== undefined) {
            currentSequence.push(nextTokenId);
        }
    }

    return tokenizer.decode(currentSequence);
}
