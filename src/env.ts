import util from 'util';

/**
 * Polyfill for Node v25+ compatibility with tfjs-node.
 * tfjs-node internally relies on util.isNullOrUndefined which was removed in recent Node versions.
 */
export function setupEnv() {
    if (!('isNullOrUndefined' in util)) {
        Object.defineProperty(util, 'isNullOrUndefined', {
            value: (val: unknown) => val === null || val === undefined,
            writable: true,
            configurable: true
        });
    }
}
