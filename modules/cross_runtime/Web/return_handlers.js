// return_handlers.js – ES module

/**
 * Decode the raw return value from a WASM call.
 * @param {object}   TheGodotModule           – Emscripten Module
 * @param {any}      raw         – raw return value
 * @param {string}   returnKind  – the expected return kind
 * @param {function} readUTF8    – UTF‑8 decoder
 * @param {function} decodeVariant – variant decoder
 * @returns {any}                – decoded value
 */
export function decodeReturn(TheGodotModule, raw, returnKind, readUTF8, decodeVariant) {
	switch (returnKind) {
	case 'void': return null;
	case 'i64': {
		const result = typeof raw === 'bigint' ? Number(raw) : raw;
		return result;
	}
	case 'i32': return Math.trunc(Number(raw));
	case 'f64': return Number(raw);
	case 'f32': return Math.fround(Number(raw));
	case 'cstring': return readUTF8(Number(raw));
	case 'variant*': {
		if (!raw) {
			return null;
		}
		return decodeVariant(raw);
	}
	// This has not yet been fully implemented in the methods in C++
	case 'Callable': {
		const ptr = Number(raw);
		const targetId = TheGodotModule._callable_get_target_id(ptr);
		const methodPtr = TheGodotModule._callable_get_method(ptr);
		const method = methodPtr ? readUTF8(methodPtr) : '';
		if (methodPtr) {
			TheGodotModule._free(methodPtr);
		}
		const result = { targetId: Number(targetId), method };
		return result;
	}
	default: return raw;
	}
}
