// marshal_cases.js – ES module
import { constructVariant } from './variant_construct.js';
/**
 * Shared bulk-copy helper: copies bytes from the .NET wasm heap into the
 * Godot wasm heap via a single TypedArray.set() (native memcpy), and
 * registers the destination pointer for later freeing.
 * @returns {{ptr: number, byteLength: number}}
 */
function bulkCopy(TheGodotModule, dotnetModule, a, tempPtrs) {
	const byteLength = a._length;
	const ptr = TheGodotModule._malloc(byteLength);
	TheGodotModule.HEAPU8.set(dotnetModule.HEAPU8.subarray(a._pointer, a._pointer + byteLength), ptr);
	tempPtrs.push(ptr);
	return { ptr, byteLength };
}

/**
 * Marshal a single argument according to its kind.
 * @param {object}   TheGodotModule             – Emscripten Module
 * @param {object}   sm            – StringMarshaller instance
 * @param {number[]} tempPtrs      – array of heap pointers to free later
 * @param {any[]}    marshaledArgs – the final argument list being built
 * @param {string}   kind          – marshaling kind
 * @param {any}      a             – the raw argument value
 * @param {number}   i             – argument index (for errors)
 * @param {function} marshalVarargs– imported varargs helper
 */
export function marshalArg(TheGodotModule, sm, tempPtrs, variantPtrs, marshaledArgs, kind, a, i, marshalVarargs, nextArg) {
	switch (kind) {
	case 'varargs': {
		const [ptr, count] = marshalVarargs(TheGodotModule, a, tempPtrs, variantPtrs);
		marshaledArgs.push(ptr, count);
		break;
	}
	case 'cstring':
		marshaledArgs.push(sm.marshal(a == null ? '' : String(a)));
		break;
	case 'ptr':
		if (a == null) {
			marshaledArgs.push(0);
		} else if (typeof a === 'bigint') {
			marshaledArgs.push(Number(a));
		} else {
			marshaledArgs.push(Math.trunc(Number(a)));
		}
		break;
	case 'i64':
		if (a == null) {
			marshaledArgs.push(0n);
		} else if (typeof a === 'bigint') {
			marshaledArgs.push(a);
		} else {
			marshaledArgs.push(BigInt(Math.trunc(Number(a))));
		}
		break;
	case 'i32':
		marshaledArgs.push(Math.trunc(Number(a ?? 0)));
		break;
	case 'f32':
		marshaledArgs.push(Math.fround(Number(a ?? 0)));
		break;
	case 'f64':
		marshaledArgs.push(Number(a ?? 0));
		break;
	case 'bool':
		marshaledArgs.push(a ? 1 : 0);
		break;

		// ---- Packed*Array: all use the same bulk-copy path ----
		// Span<byte> from CallGodotPacked / CallGodotPacked1-6.
		// The element-count divisor differs per type; PackedStringArray and
		// PackedByteArray use the raw byte count (1).

	case 'PackedByteArray': {
		const dotnetModule = globalThis.__dotnetModule;
		const { ptr, byteLength } = bulkCopy(TheGodotModule, dotnetModule, a, tempPtrs);
		marshaledArgs.push(ptr, byteLength); // count = byte length
		break;
	}
	case 'PackedStringArray': {
		// Unlike every other Packed*Array case, this does NOT come from a
		// Span<byte> MemoryView over .NET's heap — JSType.Array<JSType.String>
		// produces a genuine JS array of JS strings with no underlying
		// contiguous .NET buffer to bulkCopy from. So instead, marshal each
		// string individually via the StringMarshaller (same path as
		// "cstring"), and build a pointer table in Godot's heap.
		const strings = a ?? [];
		const count = strings.length;

		const ptrTable = TheGodotModule._malloc(count * 4); // count * sizeof(char*)
		for (let j = 0; j < count; j++) {
			const cstr = sm.marshal(strings[j] == null ? '' : String(strings[j]));
			TheGodotModule.HEAPU32[(ptrTable >> 2) + j] = cstr;
		}
		tempPtrs.push(ptrTable);

		marshaledArgs.push(ptrTable, count); // (const char** strs, int count)
		break;
	}

	case 'PackedInt32Array':
	case 'PackedFloat32Array': {
		const dotnetModule = globalThis.__dotnetModule;
		const { ptr, byteLength } = bulkCopy(TheGodotModule, dotnetModule, a, tempPtrs);
		marshaledArgs.push(ptr, byteLength / 4);
		break;
	}

	case 'PackedInt64Array':
	case 'PackedFloat64Array': {
		const dotnetModule = globalThis.__dotnetModule;
		const { ptr, byteLength } = bulkCopy(TheGodotModule, dotnetModule, a, tempPtrs);
		marshaledArgs.push(ptr, byteLength / 8);
		break;
	}

	case 'PackedVector2Array': {
		const dotnetModule = globalThis.__dotnetModule;
		const { ptr, byteLength } = bulkCopy(TheGodotModule, dotnetModule, a, tempPtrs);
		marshaledArgs.push(ptr, byteLength / 8); // 2 floats/element
		break;
	}

	case 'PackedVector3Array': {
		const dotnetModule = globalThis.__dotnetModule;
		const { ptr, byteLength } = bulkCopy(TheGodotModule, dotnetModule, a, tempPtrs);
		marshaledArgs.push(ptr, byteLength / 12); // 3 floats/element
		break;
	}

	case 'PackedVector4Array':
	case 'PackedColorArray': {
		const dotnetModule = globalThis.__dotnetModule;
		const { ptr, byteLength } = bulkCopy(TheGodotModule, dotnetModule, a, tempPtrs);
		marshaledArgs.push(ptr, byteLength / 16); // 4 floats/element
		break;
	}

	case 'Array': {
		const [values, count] = a;
		const n = Math.trunc(Number(count));
		const ptrsPtr = TheGodotModule._malloc(n * 4);
		const ptrsView = new Uint32Array(TheGodotModule.HEAPU8.buffer, ptrsPtr, n);
		for (let j = 0; j < n; j++) {
			const vptr = constructVariant(TheGodotModule, values[j]);
			ptrsView[j] = vptr;
			variantPtrs.push(vptr);
		}
		tempPtrs.push(ptrsPtr);
		marshaledArgs.push(ptrsPtr, n);
		break;
	}

	case 'Dictionary': {
		const [keys, values, count] = a;
		const n = Math.trunc(Number(count));
		const ptrTable = TheGodotModule._malloc(n * 4);
		for (let j = 0; j < n; j++) {
			TheGodotModule.HEAPU32[(ptrTable >> 2) + j] = sm.marshal(String(keys[j] ?? ''));
		}
		tempPtrs.push(ptrTable);
		const ptrsPtr = TheGodotModule._malloc(n * 4);
		const ptrsView = new Uint32Array(TheGodotModule.HEAPU8.buffer, ptrsPtr, n);
		for (let j = 0; j < n; j++) {
			const vptr = constructVariant(TheGodotModule, values[j]);
			ptrsView[j] = vptr;
			variantPtrs.push(vptr);
		}
		tempPtrs.push(ptrsPtr);
		marshaledArgs.push(ptrTable, ptrsPtr, n);
		break;
	}

	default:
		if (a == null) {
			marshaledArgs.push(0);
		} else if (typeof a === 'bigint') {
			marshaledArgs.push(a);
		} else if (typeof a === 'number') {
			marshaledArgs.push(a);
		} else if (typeof a === 'boolean') {
			marshaledArgs.push(a ? 1 : 0);
		} else if (typeof a === 'string') {
			marshaledArgs.push(sm.marshal(a));
		} else {
			throw new TypeError(`[marshal] arg[${i}] unknown kind "${kind}"`);
		}
	}
}
