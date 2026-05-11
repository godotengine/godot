export function bulkRead(srcOffset, length) {
	if (typeof self.__heapBulkRead !== 'function') {
		throw new Error('[interop] bulkRead not ready - heap not attached');
	}

	const offset = srcOffset | 0;
	const size = length | 0;
	return self.__heapBulkRead(offset, size);
}

export function bulkWrite(srcArray, destOffset, length) {
	if (typeof self.__heapBulkWrite !== 'function') {
		throw new Error('[interop] bulkWrite not ready - heap not attached');
	}

	self.__heapBulkWrite(srcArray, destOffset, length);
}

export function atomicWriteInt32(offset, value) {
	const buffer = self.__heapBuffer;

	if (!(buffer instanceof SharedArrayBuffer)) {
		throw new Error('[interop] atomicWriteInt32 requires SharedArrayBuffer heap');
	}

	if ((offset & 3) !== 0) {
		throw new Error('[interop] atomicWriteInt32 offset must be 4-byte aligned');
	}

	const view = new Int32Array(buffer, offset, 1);
	Atomics.store(view, 0, value | 0);
}

export function atomicReadInt32(offset) {
	const buffer = self.__heapBuffer;

	if (!(buffer instanceof SharedArrayBuffer)) {
		throw new Error('[interop] atomicReadInt32 requires SharedArrayBuffer heap');
	}

	if ((offset & 3) !== 0) {
		throw new Error('[interop] atomicReadInt32 offset must be 4-byte aligned');
	}

	const view = new Int32Array(buffer, offset, 1);
	return Atomics.load(view, 0);
}
