import util from 'util';
import path from 'path';
import { env, platform } from 'process';

/**
 * Polyfill for Node compatibility and GPU environment setup.
 */
export async function setupEnv() {
    // Import GPU backend
    try {
        await import('@tensorflow/tfjs-node-gpu');
        console.log('Successfully loaded tfjs-node-gpu');
    } catch (e) {
        console.warn('Failed to load tfjs-node-gpu, falling back to CPU. Error:', e);
    }
    // Polyfill for older/newer Node versions if needed
    if (!('isNullOrUndefined' in util)) {
        Object.defineProperty(util, 'isNullOrUndefined', {
            value: (val: unknown) => val === null || val === undefined,
            writable: true,
            configurable: true
        });
    }

    // Performance and GPU configuration
    // Allow memory growth instead of pre-allocating a large fraction
    env.TF_FORCE_GPU_ALLOW_GROWTH = 'true';

    // If FORCE_CPU is set, disable GPU by hiding devices
    if (env.FORCE_CPU === 'true') {
        env.CUDA_VISIBLE_DEVICES = '-1';
        console.log('FORCE_CPU is set. Disabling GPU initialization.');
    }

    // Windows-specific DLL path fixes
    if (platform === 'win32') {
        const tfLibPath = path.resolve('node_modules/@tensorflow/tfjs-node-gpu/deps/lib');
        if (!env.PATH?.includes(tfLibPath)) {
            env.PATH = `${tfLibPath};${env.PATH}`;
        }
    }
}
