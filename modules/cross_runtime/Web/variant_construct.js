// variant_construct.js
// Constructs a heap-allocated Godot Variant* from a plain JS value.
// This is the shared per-element construction logic used by both the
// varargs marshaller and the Array/Dictionary marshal cases.
// Every returned pointer is a memnew'd Variant* that must be freed
// by the caller via M._variant_destroy() after use.

export function constructVariant(M, val) {
    // Unwrap boxed primitives (e.g. new Number(1), new Boolean(true))
    // so the type checks below see the raw primitive value.
    if (val && typeof val === "object" && !Array.isArray(val) && typeof val.valueOf === "function") {
        const u = val.valueOf();
        if (u != null && typeof u !== "object") val = u;
    }

    // ── Primitive types ───────────────────────────────────────────────────
    if (val === null || val === undefined) return M._variant_new_nil();
    if (typeof val === "boolean")          return M._variant_new_bool(val);
    if (typeof val === "bigint")           return M._variant_new_int64(val);
    if (typeof val === "number")
        // Integers are sent as int64; floats as double.
        return Number.isInteger(val) ? M._variant_new_int64(BigInt(val)) : M._variant_new_double(val);
    if (typeof val === "string")
        // stringToUTF8OnStack writes into Emscripten's stack — no heap alloc needed for the string.
        return M._variant_new_string(M.stringToUTF8OnStack(val));

    // ── Tagged array types ────────────────────────────────────────────────
    // Godot struct types are sent from C# as tagged JS arrays: [tag, ...fields].
    // VariantPacker.Flatten on the C# side produces this shape for all
    // non-primitive Godot types (Vector2, Color, Transform3D, etc.).
    if (Array.isArray(val)) {
        const tag = val[0];
        switch (tag) {
            // ── Vectors ──
            case "Vector2":     return M._variant_new_vector2(val[1], val[2]);
            case "Vector2i":    return M._variant_new_vector2i(val[1], val[2]);
            case "Vector3":     return M._variant_new_vector3(val[1], val[2], val[3]);
            case "Vector3i":    return M._variant_new_vector3i(val[1], val[2], val[3]);
            case "Vector4":     return M._variant_new_vector4(val[1], val[2], val[3], val[4]);
            case "Vector4i":    return M._variant_new_vector4i(val[1], val[2], val[3], val[4]);

            // ── Geometry ──
            case "Color":       return M._variant_new_color(val[1], val[2], val[3], val[4]);
            case "Rect2":       return M._variant_new_rect2(val[1], val[2], val[3], val[4]);
            case "Rect2i":      return M._variant_new_rect2i(val[1], val[2], val[3], val[4]);
            case "AABB":        return M._variant_new_aabb(val[1], val[2], val[3], val[4], val[5], val[6]);
            case "Plane":       return M._variant_new_plane(val[1], val[2], val[3], val[4]);

            // ── Transforms ──
            case "Quaternion":  return M._variant_new_quaternion(val[1], val[2], val[3], val[4]);
            case "Basis":       return M._variant_new_basis(val[1], val[2], val[3], val[4], val[5], val[6], val[7], val[8], val[9]);
            case "Transform2D": return M._variant_new_transform2d(val[1], val[2], val[3], val[4], val[5], val[6]);
            case "Transform3D": return M._variant_new_transform3d(val[1], val[2], val[3], val[4], val[5], val[6], val[7], val[8], val[9], val[10], val[11], val[12]);

            // ── Godot handles ──
            // RID and ObjectId carry a single uint64 encoded as a JS number.
            case "RID":         return M._variant_new_rid(BigInt(Math.trunc(val[1])));
            case "ObjectId":    return M._variant_new_object_id(BigInt(Math.trunc(val[1])));

            // ── String types ──
            case "NodePath":    return M._variant_new_node_path(M.stringToUTF8OnStack(String(val[1] ?? "")));
            case "StringName":  return M._variant_new_string_name(M.stringToUTF8OnStack(String(val[1] ?? "")));

            // ── Callable / Signal ──
            // val[1] is a raw native pointer to an existing Callable/Signal in Godot memory.
            // NOTE: Callable returns across the WASM boundary are a known limitation —
            // the target object reference will be null in the receiving module.
            case "Callable":    return M._variant_new_callable(Math.trunc(val[1]));
            case "Signal":      return M._variant_new_signal(Math.trunc(val[1]));

            // ── Empty containers ──
            // Populated Array/Dictionary args are handled by their own marshal cases;
            // these tags cover empty containers passed as varargs.
            case "Dictionary":  return M._variant_new_dictionary();
            case "Array":       return M._variant_new_array();

            // Unknown tag — fall through to nil.
            default:            return M._variant_new_nil();
        }
    }

    // Unrecognised JS value — produce a nil Variant rather than throwing.
    return M._variant_new_nil();
}