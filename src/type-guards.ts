import * as tf from "@tensorflow/tfjs";

/**
 * Narrow a value to a single Tensor.
 */
export function ensureTensor(val: tf.Tensor | tf.Tensor[] | tf.SymbolicTensor | tf.SymbolicTensor[]): tf.Tensor {
    if (val instanceof tf.Tensor) {
        return val;
    }
    if (Array.isArray(val) && val[0] instanceof tf.Tensor) {
        return val[0];
    }
    throw new Error(`Expected Tensor, but received ${Array.isArray(val) ? 'Array' : typeof val}`);
}

/**
 * Narrow a value to a single SymbolicTensor.
 */
export function ensureSymbolic(val: tf.Tensor | tf.Tensor[] | tf.SymbolicTensor | tf.SymbolicTensor[]): tf.SymbolicTensor {
    if (val instanceof tf.SymbolicTensor) {
        return val;
    }
    if (Array.isArray(val) && val[0] instanceof tf.SymbolicTensor) {
        return val[0];
    }
    throw new Error(`Expected SymbolicTensor, but received ${Array.isArray(val) ? 'Array' : typeof val}`);
}

/**
 * Asserts that a value is a Tensor of a specific rank.
 */
export function ensureRank(val: tf.Tensor | tf.Tensor[], rank: number): tf.Tensor {
    const t = ensureTensor(val);
    if (t.rank === rank) {
        return t;
    }
    throw new Error(`Expected Tensor rank ${rank}, but received Tensor rank ${t.rank}`);
}
