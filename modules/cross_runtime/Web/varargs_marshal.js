// varargs_marshal.js – ES module
// Marshals JS values into a heap-allocated Variant** array for WASM varargs calls.
// All allocations are pushed into tempPtrs for the caller to free after the call.

import { constructVariant } from './variant_construct.js';

export function marshalVarargs(M, a, tempPtrs, variantPtrs) {
    const items = Array.isArray(a) ? a : (a != null ? [a] : []);
    const ptrsPtr = M._malloc(items.length * 4);
    const ptrsView = new Uint32Array(M.HEAPU8.buffer, ptrsPtr, items.length);
    for (let j = 0; j < items.length; j++) {
        const vptr = constructVariant(M, items[j]);
        ptrsView[j] = vptr;
        variantPtrs.push(vptr);
    }
    tempPtrs.push(ptrsPtr);
    return [ptrsPtr, items.length];
}