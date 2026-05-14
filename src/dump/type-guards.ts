import * as tf from "@tensorflow/tfjs";

/**
 * Asserts that a value is a single SymbolicTensor.
 */
export function assertSymbolic(val: tf.Tensor | tf.Tensor[] | tf.SymbolicTensor | tf.SymbolicTensor[]): tf.SymbolicTensor {
    if (val instanceof tf.SymbolicTensor) {
        return val;
    }
    throw new Error(`Expected SymbolicTensor, but received ${Array.isArray(val) ? 'Array' : typeof val}`);
}

/**
 * Asserts that a value is a Tensor1D.
 */
export function assertTensor1D(val: tf.Tensor | tf.Tensor[]): tf.Tensor1D {
    if (!Array.isArray(val) && val.rank === 1) {
        return val as tf.Tensor1D; // This 'as' is safe because of the rank check
    }
    throw new Error(`Expected Tensor1D, but received ${Array.isArray(val) ? 'Array' : 'Tensor' + val.rank}`);
}

/**
 * Asserts that a value is a Tensor2D.
 */
export function assertTensor2D(val: tf.Tensor | tf.Tensor[]): tf.Tensor2D {
    if (!Array.isArray(val) && val.rank === 2) {
        return val as tf.Tensor2D;
    }
    throw new Error(`Expected Tensor2D, but received ${Array.isArray(val) ? 'Array' : 'Tensor' + val.rank}`);
}

/**
 * Asserts that a value is a Tensor3D.
 */
export function assertTensor3D(val: tf.Tensor | tf.Tensor[]): tf.Tensor3D {
    if (!Array.isArray(val) && val.rank === 3) {
        return val as tf.Tensor3D;
    }
    throw new Error(`Expected Tensor3D, but received ${Array.isArray(val) ? 'Array' : 'Tensor' + val.rank}`);
}
