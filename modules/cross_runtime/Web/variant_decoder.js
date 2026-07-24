// variant_decoder.js
// Converts Godot Variant pointers from the WASM heap into JS-friendly values.
// This decoder is the read-side companion to the argument marshaling layer.
// It uses Godot's Variant type tags to decide how each value should be decoded.
// Some types are returned as plain JS primitives.
// Some types are expanded into temporary JS objects.
// Packed arrays are copied out of WASM memory to avoid dangling typed-array views.
export function createVariantDecoder(TheGodotModule, readUTF8) {
	// Reads a homogeneous packed array from a memalloc'd buffer.
	// If convertToArray is true, copies into a plain JS Array and frees the buffer.
	// If false, returns a typed-array view directly into WASM memory (no free —
	// caller must not use the view after any WASM allocation that could resize memory).
	function readPackedArray(vPtr, getterName, sizeGetter, TypedArray, convertToArray = false) {
		if (typeof TheGodotModule[getterName] !== 'function' || typeof sizeGetter !== 'function') {
			return convertToArray ? [] : new TypedArray();
		}
		const ptr = TheGodotModule[getterName](vPtr);
		const size = sizeGetter(vPtr);
		if (!ptr) {
			return convertToArray ? [] : new TypedArray();
		}
		const arr = new TypedArray(TheGodotModule.HEAPU8.buffer, ptr, size);
		const result = convertToArray ? Array.from(arr) : arr;
		if (convertToArray) {
			TheGodotModule._variant_free_packed(ptr);
		}
		return result;
	}

	// Reads a PackedInt64Array from a memalloc'd buffer.
	// BigInt64Array is used to preserve 64-bit precision; values are converted
	// to JS number on the way out (precision loss possible for large values).
	function readPackedInt64Array(vPtr) {
		if (typeof TheGodotModule._variant_as_packed_int64_array !== 'function'
			|| typeof TheGodotModule._variant_packed_int64_size !== 'function') {
			return [];
		}
		const ptr = TheGodotModule._variant_as_packed_int64_array(vPtr);
		const size = TheGodotModule._variant_packed_int64_size(vPtr);
		if (!ptr) {
			return [];
		}
		const arr = new BigInt64Array(TheGodotModule.HEAPU8.buffer, ptr, size);
		const result = Array.from(arr, (v) => Number(v));
		TheGodotModule._variant_free_packed(ptr);
		return result;
	}

	// Reads a packed struct array (Vector2/3/4, Color) from a memalloc'd float buffer.
	// Each element is fieldsPerElement consecutive float32 values, unpacked into
	// a plain JS object keyed by fieldNames. Buffer is freed after copying.
	function readPackedStructArray(vPtr, getterName, sizeGetter, fieldsPerElement, fieldNames) {
		if (typeof TheGodotModule[getterName] !== 'function' || typeof sizeGetter !== 'function') {
			return [];
		}
		const ptr = TheGodotModule[getterName](vPtr);
		const count = sizeGetter(vPtr);
		if (!ptr) {
			return [];
		}
		const arr = new Float32Array(TheGodotModule.HEAPU8.buffer, ptr, count * fieldsPerElement);
		const result = [];
		for (let i = 0; i < count; i++) {
			const obj = {};
			for (let j = 0; j < fieldsPerElement; j++) {
				obj[fieldNames[j]] = arr[i * fieldsPerElement + j];
			}
			result.push(obj);
		}
		TheGodotModule._variant_free_packed(ptr);
		return result;
	}

	// Decode one Godot Variant pointer into a JavaScript value.
	// The Variant type tag (0–38) determines which native helper to call.
	// All memnew'd Variant* and memalloc'd buffers produced during decoding
	// are freed before returning — the caller owns only the returned JS value.
	return function decodeVariant(vPtr) {
		if (!vPtr) {
			return null;
		}
		if (typeof TheGodotModule._variant_get_type !== 'function') {
			return Number(vPtr);
		}

		const type = TheGodotModule._variant_get_type(vPtr);

		switch (type) {
		case 0: return null; // NIL
		case 1: return TheGodotModule._variant_as_bool(vPtr) !== 0; // BOOL
		case 2: return Number(TheGodotModule._variant_as_int64(vPtr)); // INT
		case 3: return TheGodotModule._variant_as_double(vPtr); // FLOAT
		case 4: { // STRING
			const ptr = TheGodotModule._variant_as_string(vPtr);
			const str = readUTF8(ptr);
			TheGodotModule._variant_free_packed(ptr);
			return str;
		}
		case 5: { // VECTOR2
			const out = TheGodotModule._malloc(16);
			if (!out) {
				throw new Error('[decodeVariant] malloc failed for Vector2');
			}
			try {
				TheGodotModule._variant_as_vector2(vPtr, out);
				const view = new DataView(TheGodotModule.HEAPU8.buffer);
				return { X: view.getFloat64(out, true), Y: view.getFloat64(out + 8, true) };
			} finally {
				TheGodotModule._free(out);
			}
		}
		case 6: { // VECTOR2I
			const out = TheGodotModule._malloc(8);
			if (!out) {
				throw new Error('[decodeVariant] malloc failed for Vector2i');
			}
			try {
				TheGodotModule._variant_as_vector2i(vPtr, out);
				const view = new DataView(TheGodotModule.HEAPU8.buffer);
				return { X: view.getInt32(out, true), Y: view.getInt32(out + 4, true) };
			} finally {
				TheGodotModule._free(out);
			}
		}
		case 7: { // RECT2
			const out = TheGodotModule._malloc(16);
			if (!out) {
				throw new Error('malloc failed for Rect2');
			}
			try {
				TheGodotModule._variant_as_rect2(vPtr, out);
				return {
					X: TheGodotModule.HEAPF32[out >> 2], Y: TheGodotModule.HEAPF32[(out + 4) >> 2],
					Width: TheGodotModule.HEAPF32[(out + 8) >> 2], Height: TheGodotModule.HEAPF32[(out + 12) >> 2],
				};
			} finally {
				TheGodotModule._free(out);
			}
		}
		case 8: { // RECT2I
			const out = TheGodotModule._malloc(16);
			if (!out) {
				throw new Error('malloc failed for Rect2i');
			}
			try {
				TheGodotModule._variant_as_rect2i(vPtr, out);
				return {
					X: TheGodotModule.HEAP32[out >> 2], Y: TheGodotModule.HEAP32[(out + 4) >> 2],
					Width: TheGodotModule.HEAP32[(out + 8) >> 2], Height: TheGodotModule.HEAP32[(out + 12) >> 2],
				};
			} finally {
				TheGodotModule._free(out);
			}
		}
		case 9: { // VECTOR3
			const out = TheGodotModule._malloc(24);
			if (!out) {
				throw new Error('[decodeVariant] malloc failed for Vector3');
			}
			try {
				TheGodotModule._variant_as_vector3(vPtr, out);
				const view = new DataView(TheGodotModule.HEAPU8.buffer);
				return {
					X: view.getFloat64(out, true),
					Y: view.getFloat64(out + 8, true),
					Z: view.getFloat64(out + 16, true),
				};
			} finally {
				TheGodotModule._free(out);
			}
		}
		case 10: { // VECTOR3I
			const out = TheGodotModule._malloc(12);
			if (!out) {
				throw new Error('[decodeVariant] malloc failed for Vector3i');
			}
			try {
				TheGodotModule._variant_as_vector3i(vPtr, out);
				const view = new DataView(TheGodotModule.HEAPU8.buffer);
				return {
					X: view.getInt32(out, true),
					Y: view.getInt32(out + 4, true),
					Z: view.getInt32(out + 8, true),
				};
			} finally {
				TheGodotModule._free(out);
			}
		}
		case 11: { // TRANSFORM2D
			const out = TheGodotModule._malloc(48);
			if (!out) {
				throw new Error('malloc failed for Transform2D');
			}
			try {
				TheGodotModule._variant_as_transform2d(vPtr, out);
				const view = new DataView(TheGodotModule.HEAPU8.buffer);
				return {
					'Column0.X': view.getFloat64(out, true), 'Column0.Y': view.getFloat64(out + 8, true),
					'Column1.X': view.getFloat64(out + 16, true), 'Column1.Y': view.getFloat64(out + 24, true),
					'Origin.X': view.getFloat64(out + 32, true), 'Origin.Y': view.getFloat64(out + 40, true),
				};
			} finally {
				TheGodotModule._free(out);
			}
		}
		case 12: { // VECTOR4
			const out = TheGodotModule._malloc(32);
			if (!out) {
				throw new Error('[decodeVariant] malloc failed for Vector4');
			}
			try {
				TheGodotModule._variant_as_vector4(vPtr, out);
				const view = new DataView(TheGodotModule.HEAPU8.buffer);
				return {
					X: view.getFloat64(out, true), Y: view.getFloat64(out + 8, true),
					Z: view.getFloat64(out + 16, true), W: view.getFloat64(out + 24, true),
				};
			} finally {
				TheGodotModule._free(out);
			}
		}
		case 13: { // VECTOR4I
			const out = TheGodotModule._malloc(16);
			if (!out) {
				throw new Error('[decodeVariant] malloc failed for Vector4i');
			}
			try {
				TheGodotModule._variant_as_vector4i(vPtr, out);
				const view = new DataView(TheGodotModule.HEAPU8.buffer);
				return {
					X: view.getInt32(out, true), Y: view.getInt32(out + 4, true),
					Z: view.getInt32(out + 8, true), W: view.getInt32(out + 12, true),
				};
			} finally {
				TheGodotModule._free(out);
			}
		}
		case 14: { // PLANE
			const out = TheGodotModule._malloc(32);
			if (!out) {
				throw new Error('malloc failed for Plane');
			}
			try {
				TheGodotModule._variant_as_plane(vPtr, out);
				const view = new DataView(TheGodotModule.HEAPU8.buffer);
				return {
					'Normal.X': view.getFloat64(out, true),
					'Normal.Y': view.getFloat64(out + 8, true),
					'Normal.Z': view.getFloat64(out + 16, true),
					'D': view.getFloat64(out + 24, true),
				};
			} finally {
				TheGodotModule._free(out);
			}
		}
		case 15: { // QUATERNION
			const out = TheGodotModule._malloc(32);
			if (!out) {
				throw new Error('malloc failed for Quaternion');
			}
			try {
				TheGodotModule._variant_as_quaternion(vPtr, out);
				const view = new DataView(TheGodotModule.HEAPU8.buffer);
				return {
					X: view.getFloat64(out, true), Y: view.getFloat64(out + 8, true),
					Z: view.getFloat64(out + 16, true), W: view.getFloat64(out + 24, true),
				};
			} finally {
				TheGodotModule._free(out);
			}
		}
		case 16: { // AABB
			const out = TheGodotModule._malloc(48);
			if (!out) {
				throw new Error('malloc failed for AABB');
			}
			try {
				TheGodotModule._variant_as_aabb(vPtr, out);
				const view = new DataView(TheGodotModule.HEAPU8.buffer);
				return {
					'Position.X': view.getFloat64(out, true),
					'Position.Y': view.getFloat64(out + 8, true),
					'Position.Z': view.getFloat64(out + 16, true),
					'Size.X': view.getFloat64(out + 24, true),
					'Size.Y': view.getFloat64(out + 32, true),
					'Size.Z': view.getFloat64(out + 40, true),
				};
			} finally {
				TheGodotModule._free(out);
			}
		}
		case 17: { // BASIS
			const out = TheGodotModule._malloc(72);
			if (!out) {
				throw new Error('malloc failed for Basis');
			}
			try {
				TheGodotModule._variant_as_basis(vPtr, out);
				const view = new DataView(TheGodotModule.HEAPU8.buffer);
				return {
					'Column0.X': view.getFloat64(out, true), 'Column0.Y': view.getFloat64(out + 8, true), 'Column0.Z': view.getFloat64(out + 16, true),
					'Column1.X': view.getFloat64(out + 24, true), 'Column1.Y': view.getFloat64(out + 32, true), 'Column1.Z': view.getFloat64(out + 40, true),
					'Column2.X': view.getFloat64(out + 48, true), 'Column2.Y': view.getFloat64(out + 56, true), 'Column2.Z': view.getFloat64(out + 64, true),
				};
			} finally {
				TheGodotModule._free(out);
			}
		}
		case 18: { // TRANSFORM3D
			const out = TheGodotModule._malloc(96);
			if (!out) {
				throw new Error('malloc failed for Transform3D');
			}
			try {
				TheGodotModule._variant_as_transform3d(vPtr, out);
				const view = new DataView(TheGodotModule.HEAPU8.buffer);
				return {
					'Basis.Column0.X': view.getFloat64(out, true), 'Basis.Column0.Y': view.getFloat64(out + 8, true), 'Basis.Column0.Z': view.getFloat64(out + 16, true),
					'Basis.Column1.X': view.getFloat64(out + 24, true), 'Basis.Column1.Y': view.getFloat64(out + 32, true), 'Basis.Column1.Z': view.getFloat64(out + 40, true),
					'Basis.Column2.X': view.getFloat64(out + 48, true), 'Basis.Column2.Y': view.getFloat64(out + 56, true), 'Basis.Column2.Z': view.getFloat64(out + 64, true),
					'Origin.X': view.getFloat64(out + 72, true), 'Origin.Y': view.getFloat64(out + 80, true), 'Origin.Z': view.getFloat64(out + 88, true),
				};
			} finally {
				TheGodotModule._free(out);
			}
		}
		case 19: { // COLOR
			const out = TheGodotModule._malloc(16);
			if (!out) {
				throw new Error('malloc failed for Color');
			}
			try {
				TheGodotModule._variant_as_color(vPtr, out);
				return {
					R: TheGodotModule.HEAPF32[out >> 2], G: TheGodotModule.HEAPF32[(out + 4) >> 2],
					B: TheGodotModule.HEAPF32[(out + 8) >> 2], A: TheGodotModule.HEAPF32[(out + 12) >> 2],
				};
			} finally {
				TheGodotModule._free(out);
			}
		}
		case 21: { // STRING_NAME
			const ptr = TheGodotModule._variant_as_string_name(vPtr);
			const str = readUTF8(ptr);
			TheGodotModule._variant_free_packed(ptr);
			return str;
		}
		case 22: { // NODE_PATH
			const ptr = TheGodotModule._variant_as_node_path(vPtr);
			const str = readUTF8(ptr);
			TheGodotModule._variant_free_packed(ptr);
			return str;
		}
		case 23: return TheGodotModule._variant_as_rid(vPtr); // RID
		case 24: return TheGodotModule._variant_as_object(vPtr); // OBJECT
		case 25: return null; // CALLABLE – not yet supported
		case 26: { // SIGNAL
			const idRaw = TheGodotModule._variant_signal_target_id(vPtr);
			const namePtr = TheGodotModule._variant_signal_name(vPtr);
			const name = readUTF8(namePtr);
			TheGodotModule._variant_free_packed(namePtr);
			return { __type: 'Signal', targetId: Number(idRaw), name };
		}
		case 27: { // DICTIONARY
			const dictVariant = TheGodotModule._variant_as_dictionary(vPtr);
			if (!dictVariant) {
				return [];
			}
			const size = TheGodotModule._variant_dictionary_size(dictVariant);
			const result = [];
			for (let i = 0; i < size; i++) {
				const keyPtr = TheGodotModule._variant_dictionary_get_key(dictVariant, i);
				const valPtr = TheGodotModule._variant_dictionary_get_value(dictVariant, i);
				result.push(decodeVariant(keyPtr));
				result.push(decodeVariant(valPtr));
				TheGodotModule._variant_destroy(keyPtr);
				TheGodotModule._variant_destroy(valPtr);
			}
			TheGodotModule._variant_destroy(dictVariant);
			return result;
		}
		case 28: { // ARRAY
			const arrVariant = TheGodotModule._variant_as_array(vPtr);
			if (!arrVariant) {
				return [];
			}
			const size = TheGodotModule._variant_array_size(arrVariant);
			const result = [];
			for (let i = 0; i < size; i++) {
				const elemPtr = TheGodotModule._variant_array_get(arrVariant, i);
				result.push(decodeVariant(elemPtr));
				TheGodotModule._variant_destroy(elemPtr);
			}
			TheGodotModule._variant_destroy(arrVariant);
			return result;
		}
		case 29: return readPackedArray(vPtr, '_variant_as_packed_byte_array', TheGodotModule._variant_packed_byte_size, Uint8Array);
		case 30: return readPackedArray(vPtr, '_variant_as_packed_int32_array', TheGodotModule._variant_packed_int32_size, Int32Array, true);
		case 31: return readPackedInt64Array(vPtr);
		case 32: return readPackedArray(vPtr, '_variant_as_packed_float32_array', TheGodotModule._variant_packed_float32_size, Float32Array, true);
		case 33: return readPackedArray(vPtr, '_variant_as_packed_float64_array', TheGodotModule._variant_packed_float64_size, Float64Array, true);
		case 34: { // PACKED_STRING_ARRAY
			const ptr = TheGodotModule._variant_as_packed_string_array(vPtr);
			const count = TheGodotModule._variant_packed_string_count(vPtr);
			if (!ptr || count === 0) {
				return [];
			}
			const view = new DataView(TheGodotModule.HEAPU8.buffer);
			const result = [];
			const dataStart = ptr + 4 + count * 4;
			for (let i = 0; i < count; i++) {
				const offset = view.getInt32(ptr + 4 + i * 4, true);
				result.push(readUTF8(dataStart + offset));
			}
			TheGodotModule._variant_free_packed(ptr);
			return result;
		}
		case 35: return readPackedStructArray(vPtr, '_variant_as_packed_vector2_array', TheGodotModule._variant_packed_vector2_size, 2, ['X', 'Y']);
		case 36: return readPackedStructArray(vPtr, '_variant_as_packed_vector3_array', TheGodotModule._variant_packed_vector3_size, 3, ['X', 'Y', 'Z']);
		case 37: return readPackedStructArray(vPtr, '_variant_as_packed_color_array', TheGodotModule._variant_packed_color_size, 4, ['R', 'G', 'B', 'A']);
		case 38: return readPackedStructArray(vPtr, '_variant_as_packed_vector4_array', TheGodotModule._variant_packed_vector4_size, 4, ['X', 'Y', 'Z', 'W']);
		default: return Number(vPtr); // fallback for unknown types
		}
	};
}
