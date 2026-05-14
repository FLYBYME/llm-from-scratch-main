import util from 'util';
import path from 'path';
import { env, platform } from 'process';

/**
 * Polyfill for Node v25+ compatibility with tfjs-node.
 * Also ensures GPU DLLs are found on Windows.
 */
export function setupEnv() {
    if (!('isNullOrUndefined' in util)) {
        Object.defineProperty(util, 'isNullOrUndefined', {
            value: (val: unknown) => val === null || val === undefined,
            writable: true,
            configurable: true
        });
    }

    // Windows GPU Path Fix
    if (platform === 'win32') {
        const tfLibPath = path.resolve('node_modules/@tensorflow/tfjs-node-gpu/deps/lib');
        if (!env.PATH?.includes(tfLibPath)) {
            env.PATH = `${tfLibPath};${env.PATH}`;
        }
        
        // Prevent TF from hogging 100% of VRAM immediately. 
        env.TF_FORCE_GPU_ALLOW_GROWTH = 'true';

        // TARGET NVIDIA GPU: If you have an AMD iGPU + NVIDIA GPU, 
        // this ensures TF only looks at the NVIDIA one.
        env.CUDA_VISIBLE_DEVICES = '0';

        // HARD LIMIT: Force TensorFlow to only ever touch 80% of your VRAM.
        // Since you have two GPUs, you can move your browser/UI to the AMD one
        // to free up even more of the 3080!
        env.TF_PER_PROCESS_GPU_MEMORY_FRACTION = '0.8';
    }
}
