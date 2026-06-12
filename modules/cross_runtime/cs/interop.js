// This is the js interop layer
/*
- obtain a view of Godot WASM heap layer
- expose bulk read and write to csharp
- keep those bytes aligned with memory contract
*/

//window is used in browswer main thread and self in workers - if windows exist it uses it, otherwise it falls bback to self
const globalScope = typeof window !== 'undefined' ? window : self;
const heap = globalScope.godotHeapF32; //accesses the heap
if (!heap) throw new Error("Godot heap not available for interop"); //guards against missing memory

export function bulkRead(srcOffset, destArray, length) {
    const src = new Uint8Array(heap.buffer, heap.byteOffset + srcOffset, length);
    destArray.set(src);
}

export function bulkWrite(srcArray, destOffset, length) {
    const dest = new Uint8Array(heap.buffer, heap.byteOffset + destOffset, length);
    dest.set(srcArray);
}