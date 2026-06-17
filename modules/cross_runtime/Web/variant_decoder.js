// variant_decoder.js
// Converts Godot Variant pointers from the WASM heap into JS-friendly values.
// This decoder is the read-side companion to the argument marshalling layer.
// It uses Godot's Variant type tags to decide how each value should be decoded.
// Some types are returned as plain JS primitives.
// Some types are expanded into temporary JS objects.
// Packed arrays are copied out of WASM memory to avoid dangling typed-array views.
export function createVariantDecoder(M, readUTF8) {

  // Decode one Godot Variant pointer into a JavaScript value.
  // The Variant type tag (0–38) determines which native helper to call.
  // All memnew'd Variant* and memalloc'd buffers produced during decoding
  // are freed before returning — the caller owns only the returned JS value.
  return function decodeVariant(vPtr) {
    if (!vPtr) return null;
    if (typeof M._variant_get_type !== "function") return Number(vPtr);

    const type = M._variant_get_type(vPtr);

    switch (type) {

      case 0: return null;    // NIL

      case 1: return M._variant_as_bool(vPtr) !== 0;   // BOOL

      // INT — decoded via signed int64; returned as JS number.
      // Note: values beyond Number.MAX_SAFE_INTEGER lose precision.
      case 2: return Number(M._variant_as_int64(vPtr));

      case 3: return M._variant_as_double(vPtr);  // FLOAT

      // STRING — variant_as_string memalloc's a UTF-8 buffer; free after copying.
      case 4: {
        const ptr = M._variant_as_string(vPtr);
        const str = readUTF8(ptr);
        M._variant_free_packed(ptr);
        return str;
      }

      // ── Math / geometry types ────────────────────────────────────────────
      // Each case mallocs a temporary output buffer, asks Godot to write the
      // struct into it, reads the fields back into a plain JS object, then
      // frees the buffer. The try/finally ensures the buffer is freed even
      // if readback throws.

      case 5: { // VECTOR2 — 2 × float64
        const out = M._malloc(16);
        if (!out) throw new Error("[decodeVariant] malloc failed for Vector2");
        try {
          M._variant_as_vector2(vPtr, out);
          const view = new DataView(M.HEAPU8.buffer);
          return { X: view.getFloat64(out, true), Y: view.getFloat64(out + 8, true) };
        } finally { M._free(out); }
      }

      case 6: { // VECTOR2I — 2 × int32
        const out = M._malloc(8);
        if (!out) throw new Error("[decodeVariant] malloc failed for Vector2i");
        try {
          M._variant_as_vector2i(vPtr, out);
          const view = new DataView(M.HEAPU8.buffer);
          return { X: view.getInt32(out, true), Y: view.getInt32(out + 4, true) };
        } finally { M._free(out); }
      }

      case 7: { // RECT2 — 4 × float32 (x, y, width, height)
        const out = M._malloc(16);
        if (!out) throw new Error("malloc failed for Rect2");
        try {
          M._variant_as_rect2(vPtr, out);
          return {
            X: M.HEAPF32[out >> 2],         Y: M.HEAPF32[(out + 4) >> 2],
            Width: M.HEAPF32[(out + 8) >> 2], Height: M.HEAPF32[(out + 12) >> 2],
          };
        } finally { M._free(out); }
      }

      case 8: { // RECT2I — 4 × int32
        const out = M._malloc(16);
        if (!out) throw new Error("malloc failed for Rect2i");
        try {
          M._variant_as_rect2i(vPtr, out);
          return {
            X: M.HEAP32[out >> 2],         Y: M.HEAP32[(out + 4) >> 2],
            Width: M.HEAP32[(out + 8) >> 2], Height: M.HEAP32[(out + 12) >> 2],
          };
        } finally { M._free(out); }
      }

      case 9: { // VECTOR3 — 3 × float64
        const out = M._malloc(24);
        if (!out) throw new Error("[decodeVariant] malloc failed for Vector3");
        try {
          M._variant_as_vector3(vPtr, out);
          const view = new DataView(M.HEAPU8.buffer);
          return {
            X: view.getFloat64(out, true),
            Y: view.getFloat64(out + 8, true),
            Z: view.getFloat64(out + 16, true),
          };
        } finally { M._free(out); }
      }

      case 10: { // VECTOR3I — 3 × int32
        const out = M._malloc(12);
        if (!out) throw new Error("[decodeVariant] malloc failed for Vector3i");
        try {
          M._variant_as_vector3i(vPtr, out);
          const view = new DataView(M.HEAPU8.buffer);
          return {
            X: view.getInt32(out, true),
            Y: view.getInt32(out + 4, true),
            Z: view.getInt32(out + 8, true),
          };
        } finally { M._free(out); }
      }

      case 11: { // TRANSFORM2D — 6 × float64 (col0, col1, origin)
        const out = M._malloc(48);
        if (!out) throw new Error("malloc failed for Transform2D");
        try {
          M._variant_as_transform2d(vPtr, out);
          const view = new DataView(M.HEAPU8.buffer);
          return {
            "Column0.X": view.getFloat64(out, true),      "Column0.Y": view.getFloat64(out + 8, true),
            "Column1.X": view.getFloat64(out + 16, true), "Column1.Y": view.getFloat64(out + 24, true),
            "Origin.X":  view.getFloat64(out + 32, true), "Origin.Y":  view.getFloat64(out + 40, true),
          };
        } finally { M._free(out); }
      }

      case 12: { // VECTOR4 — 4 × float64
        const out = M._malloc(32);
        if (!out) throw new Error("[decodeVariant] malloc failed for Vector4");
        try {
          M._variant_as_vector4(vPtr, out);
          const view = new DataView(M.HEAPU8.buffer);
          return {
            X: view.getFloat64(out, true),      Y: view.getFloat64(out + 8, true),
            Z: view.getFloat64(out + 16, true), W: view.getFloat64(out + 24, true),
          };
        } finally { M._free(out); }
      }

      case 13: { // VECTOR4I — 4 × int32
        const out = M._malloc(16);
        if (!out) throw new Error("[decodeVariant] malloc failed for Vector4i");
        try {
          M._variant_as_vector4i(vPtr, out);
          const view = new DataView(M.HEAPU8.buffer);
          return {
            X: view.getInt32(out, true),      Y: view.getInt32(out + 4, true),
            Z: view.getInt32(out + 8, true),  W: view.getInt32(out + 12, true),
          };
        } finally { M._free(out); }
      }

      case 14: { // PLANE — normal (3 × float64) + d (float64)
        const out = M._malloc(32);
        if (!out) throw new Error("malloc failed for Plane");
        try {
          M._variant_as_plane(vPtr, out);
          const view = new DataView(M.HEAPU8.buffer);
          return {
            "Normal.X": view.getFloat64(out, true),
            "Normal.Y": view.getFloat64(out + 8, true),
            "Normal.Z": view.getFloat64(out + 16, true),
            "D":        view.getFloat64(out + 24, true),
          };
        } finally { M._free(out); }
      }

      case 15: { // QUATERNION — 4 × float64
        const out = M._malloc(32);
        if (!out) throw new Error("malloc failed for Quaternion");
        try {
          M._variant_as_quaternion(vPtr, out);
          const view = new DataView(M.HEAPU8.buffer);
          return {
            X: view.getFloat64(out, true),      Y: view.getFloat64(out + 8, true),
            Z: view.getFloat64(out + 16, true), W: view.getFloat64(out + 24, true),
          };
        } finally { M._free(out); }
      }

      case 16: { // AABB — position + size (6 × float64)
        const out = M._malloc(48);
        if (!out) throw new Error("malloc failed for AABB");
        try {
          M._variant_as_aabb(vPtr, out);
          const view = new DataView(M.HEAPU8.buffer);
          return {
            "Position.X": view.getFloat64(out, true),
            "Position.Y": view.getFloat64(out + 8, true),
            "Position.Z": view.getFloat64(out + 16, true),
            "Size.X":     view.getFloat64(out + 24, true),
            "Size.Y":     view.getFloat64(out + 32, true),
            "Size.Z":     view.getFloat64(out + 40, true),
          };
        } finally { M._free(out); }
      }

      case 17: { // BASIS — 3 columns × 3 × float64 = 72 bytes
        const out = M._malloc(72);
        if (!out) throw new Error("malloc failed for Basis");
        try {
          M._variant_as_basis(vPtr, out);
          const view = new DataView(M.HEAPU8.buffer);
          return {
            "Column0.X": view.getFloat64(out, true),      "Column0.Y": view.getFloat64(out + 8, true),  "Column0.Z": view.getFloat64(out + 16, true),
            "Column1.X": view.getFloat64(out + 24, true), "Column1.Y": view.getFloat64(out + 32, true), "Column1.Z": view.getFloat64(out + 40, true),
            "Column2.X": view.getFloat64(out + 48, true), "Column2.Y": view.getFloat64(out + 56, true), "Column2.Z": view.getFloat64(out + 64, true),
          };
        } finally { M._free(out); }
      }

      case 18: { // TRANSFORM3D — basis (9 × float64) + origin (3 × float64) = 96 bytes
        const out = M._malloc(96);
        if (!out) throw new Error("malloc failed for Transform3D");
        try {
          M._variant_as_transform3d(vPtr, out);
          const view = new DataView(M.HEAPU8.buffer);
          return {
            "Basis.Column0.X": view.getFloat64(out, true),      "Basis.Column0.Y": view.getFloat64(out + 8, true),  "Basis.Column0.Z": view.getFloat64(out + 16, true),
            "Basis.Column1.X": view.getFloat64(out + 24, true), "Basis.Column1.Y": view.getFloat64(out + 32, true), "Basis.Column1.Z": view.getFloat64(out + 40, true),
            "Basis.Column2.X": view.getFloat64(out + 48, true), "Basis.Column2.Y": view.getFloat64(out + 56, true), "Basis.Column2.Z": view.getFloat64(out + 64, true),
            "Origin.X":        view.getFloat64(out + 72, true), "Origin.Y":        view.getFloat64(out + 80, true), "Origin.Z":        view.getFloat64(out + 88, true),
          };
        } finally { M._free(out); }
      }

      case 19: { // COLOR — 4 × float32 (r, g, b, a)
        const out = M._malloc(16);
        if (!out) throw new Error("malloc failed for Color");
        try {
          M._variant_as_color(vPtr, out);
          return {
            R: M.HEAPF32[out >> 2],          G: M.HEAPF32[(out + 4) >> 2],
            B: M.HEAPF32[(out + 8) >> 2],    A: M.HEAPF32[(out + 12) >> 2],
          };
        } finally { M._free(out); }
      }

      // STRING_NAME / NODE_PATH — same memalloc pattern as STRING; free after copy.
      case 21: {
        const ptr = M._variant_as_string_name(vPtr);
        const str = readUTF8(ptr);
        M._variant_free_packed(ptr);
        return str;
      }

      case 22: {
        const ptr = M._variant_as_node_path(vPtr);
        const str = readUTF8(ptr);
        M._variant_free_packed(ptr);
        return str;
      }

      // RID — returned as raw uint64 number (no allocation).
      case 23: return M._variant_as_rid(vPtr);

      // OBJECT — returned as raw instance-id uint64 number (no allocation).
      case 24: return M._variant_as_object(vPtr);

      // CALLABLE — cannot be decoded across the WASM boundary; the target object
      // reference is null in the receiving module. Known limitation: Callable
      // returns are not supported. Returns null until a workaround is designed.
      case 25: return null;

      // SIGNAL — decoded into { __type, targetId, name } so .NET can reconstruct
      // a Signal struct. Rarely returned by Godot APIs but supported for completeness.
      case 26: {
        const idRaw   = M._variant_signal_target_id(vPtr);
        const namePtr = M._variant_signal_name(vPtr);
        const name    = readUTF8(namePtr);
        M._variant_free_packed(namePtr);
        return { __type: "Signal", targetId: Number(idRaw), name };
      }

      // DICTIONARY — variant_as_dictionary memnew's a Variant* copy of the dict.
      // Iterate keys/values via index helpers (each also memnew'd), recurse into
      // decodeVariant for each, then destroy all intermediates.
      // Returns a flat interleaved array [key0, val0, key1, val1, ...] which
      // the C# side reassembles into a Dictionary<object,object>.
      case 27: {
        const dictVariant = M._variant_as_dictionary(vPtr);
        if (!dictVariant) return [];
        const size = M._variant_dictionary_size(dictVariant);
        const result = [];
        for (let i = 0; i < size; i++) {
          const keyPtr = M._variant_dictionary_get_key(dictVariant, i);
          const valPtr = M._variant_dictionary_get_value(dictVariant, i);
          result.push(decodeVariant(keyPtr));
          result.push(decodeVariant(valPtr));
          M._variant_destroy(keyPtr);
          M._variant_destroy(valPtr);
        }
        M._variant_destroy(dictVariant);
        return result;
      }

      // ARRAY — variant_as_array memnew's a Variant* copy of the array.
      // Iterate elements via index helper (each also memnew'd), recurse into
      // decodeVariant for each, then destroy all intermediates.
      // Returns a plain JS array which the C# side pours into a Godot.Array.
      case 28: {
        const arrVariant = M._variant_as_array(vPtr);
        if (!arrVariant) return [];
        const size = M._variant_array_size(arrVariant);
        const result = [];
        for (let i = 0; i < size; i++) {
          const elemPtr = M._variant_array_get(arrVariant, i);
          result.push(decodeVariant(elemPtr));
          M._variant_destroy(elemPtr);
        }
        M._variant_destroy(arrVariant);
        return result;
      }

      // ── Packed arrays ────────────────────────────────────────────────────
      // Each packed type uses a C++ helper that memalloc's a contiguous buffer
      // and returns a pointer + element count. The helpers below copy the data
      // into JS and free the buffer. Typed-array views are only kept when
      // convertToArray is false (caller must not hold the view after GC).

      case 29: return readPackedArray(M, vPtr, "_variant_as_packed_byte_array",    M._variant_packed_byte_size,     Uint8Array);
      case 30: return readPackedArray(M, vPtr, "_variant_as_packed_int32_array",   M._variant_packed_int32_size,   Int32Array,   true);
      case 31: return readPackedInt64Array(M, vPtr);
      case 32: return readPackedArray(M, vPtr, "_variant_as_packed_float32_array", M._variant_packed_float32_size,  Float32Array, true);
      case 33: return readPackedArray(M, vPtr, "_variant_as_packed_float64_array", M._variant_packed_float64_size,  Float64Array, true);

      // PackedStringArray uses a custom layout: [int32 count][int32 offsets...][utf8 data...]
      // Strings are read directly from the buffer using the offset table, then the buffer is freed.
      case 34: {
        const ptr = M._variant_as_packed_string_array(vPtr);
        const count = M._variant_packed_string_count(vPtr);
        if (!ptr || count === 0) return [];
        const view = new DataView(M.HEAPU8.buffer);
        const result = [];
        const dataStart = ptr + 4 + count * 4; // skip count + offset table
        for (let i = 0; i < count; i++) {
          const offset = view.getInt32(ptr + 4 + i * 4, true);
          result.push(readUTF8(dataStart + offset));
        }
        M._variant_free_packed(ptr);
        return result;
      }

      // Struct packed arrays: each element is N floats; unpacked into plain JS objects.
      case 35: return readPackedStructArray(M, vPtr, "_variant_as_packed_vector2_array", M._variant_packed_vector2_size, 2, ["X","Y"]);
      case 36: return readPackedStructArray(M, vPtr, "_variant_as_packed_vector3_array", M._variant_packed_vector3_size, 3, ["X","Y","Z"]);
      case 37: return readPackedStructArray(M, vPtr, "_variant_as_packed_color_array",   M._variant_packed_color_size,   4, ["R","G","B","A"]);
      case 38: return readPackedStructArray(M, vPtr, "_variant_as_packed_vector4_array", M._variant_packed_vector4_size, 4, ["X","Y","Z","W"]);

      default:
        console.warn(`[decodeVariant] unknown variant type ${type}, returning raw pointer`);
        return Number(vPtr);
    }
  };

  // ── Helper functions ───────────────────────────────────────────────────────

  // Reads a homogeneous packed array from a memalloc'd buffer.
  // If convertToArray is true, copies into a plain JS Array and frees the buffer.
  // If false, returns a typed-array view directly into WASM memory (no free —
  // caller must not use the view after any WASM allocation that could resize memory).
  function readPackedArray(M, vPtr, getterName, sizeGetter, TypedArray, convertToArray = false) {
    if (typeof M[getterName] !== "function" || typeof sizeGetter !== "function") {
      console.warn(`[decodeVariant] ${getterName} helper missing`);
      return convertToArray ? [] : new TypedArray();
    }
    const ptr  = M[getterName](vPtr);
    const size = sizeGetter(vPtr);
    if (!ptr) return convertToArray ? [] : new TypedArray();
    const arr = new TypedArray(M.HEAPU8.buffer, ptr, size);
    const result = convertToArray ? Array.from(arr) : arr;
    if (convertToArray) M._variant_free_packed(ptr);
    return result;
  }

  // Reads a PackedInt64Array from a memalloc'd buffer.
  // BigInt64Array is used to preserve 64-bit precision; values are converted
  // to JS number on the way out (precision loss possible for large values).
  function readPackedInt64Array(M, vPtr) {
    if (typeof M._variant_as_packed_int64_array !== "function" || typeof M._variant_packed_int64_size !== "function") {
      console.warn("[decodeVariant] _variant_as_packed_int64_array helper missing");
      return [];
    }
    const ptr  = M._variant_as_packed_int64_array(vPtr);
    const size = M._variant_packed_int64_size(vPtr);
    if (!ptr) return [];
    const arr = new BigInt64Array(M.HEAPU8.buffer, ptr, size);
    const result = Array.from(arr, v => Number(v));
    M._variant_free_packed(ptr);
    return result;
  }

  // Reads a packed struct array (Vector2/3/4, Color) from a memalloc'd float buffer.
  // Each element is fieldsPerElement consecutive float32 values, unpacked into
  // a plain JS object keyed by fieldNames. Buffer is freed after copying.
  function readPackedStructArray(M, vPtr, getterName, sizeGetter, fieldsPerElement, fieldNames) {
    if (typeof M[getterName] !== "function" || typeof sizeGetter !== "function") {
      console.warn(`[decodeVariant] ${getterName} helper missing`);
      return [];
    }
    const ptr   = M[getterName](vPtr);
    const count = sizeGetter(vPtr);
    if (!ptr) return [];
    const arr = new Float32Array(M.HEAPU8.buffer, ptr, count * fieldsPerElement);
    const result = [];
    for (let i = 0; i < count; i++) {
      const obj = {};
      for (let j = 0; j < fieldsPerElement; j++) {
        obj[fieldNames[j]] = arr[i * fieldsPerElement + j];
      }
      result.push(obj);
    }
    M._variant_free_packed(ptr);
    return result;
  }
}