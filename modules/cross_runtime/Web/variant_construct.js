// variant_construct.js
// Constructs a heap-allocated Godot Variant* from a plain JS value.
// This is the shared per-element construction logic used by both the
// varargs marshaller and the Array/Dictionary marshal cases.
// Every returned pointer is a memnew'd Variant* that must be freed
// by the caller via TheGodotModule._variant_destroy() after use.

export function constructVariant(TheGodotModule, val) {
	// Unwrap boxed primitives (e.g. new Number(1), new Boolean(true))
	// so the type checks below see the raw primitive value.
	let v = val;
	if (v && typeof v === 'object' && !Array.isArray(v) && typeof v.valueOf === 'function') {
		const u = v.valueOf();
		if (u != null && typeof u !== 'object') {
			v = u;
		}
	}

	// ── Primitive types ───────────────────────────────────────────────────
	if (v === null || v === undefined) {
		return TheGodotModule._variant_new_nil();
	}
	if (typeof v === 'boolean') {
		return TheGodotModule._variant_new_bool(v);
	}
	if (typeof v === 'bigint') {
		return TheGodotModule._variant_new_int64(v);
	}
	if (typeof v === 'number') {
		// Integers are sent as int64; floats as double.
		return Number.isInteger(v)
			? TheGodotModule._variant_new_int64(BigInt(v))
			: TheGodotModule._variant_new_double(v);
	}
	if (typeof v === 'string') {
		// stringToUTF8OnStack writes into Emscripten's stack — no heap alloc needed for the string.
		return TheGodotModule._variant_new_string(TheGodotModule.stringToUTF8OnStack(v));
	}

	// ── Tagged array types ────────────────────────────────────────────────
	// Godot struct types are sent from C# as tagged JS arrays: [tag, ...fields].
	// VariantPacker.Flatten on the C# side produces this shape for all
	// non-primitive Godot types (Vector2, Color, Transform3D, etc.).
	if (Array.isArray(v)) {
		const tag = v[0];
		switch (tag) {
		// ── Vectors ──
		case 'Vector2': return TheGodotModule._variant_new_vector2(v[1], v[2]);
		case 'Vector2i': return TheGodotModule._variant_new_vector2i(v[1], v[2]);
		case 'Vector3': return TheGodotModule._variant_new_vector3(v[1], v[2], v[3]);
		case 'Vector3i': return TheGodotModule._variant_new_vector3i(v[1], v[2], v[3]);
		case 'Vector4': return TheGodotModule._variant_new_vector4(v[1], v[2], v[3], v[4]);
		case 'Vector4i': return TheGodotModule._variant_new_vector4i(v[1], v[2], v[3], v[4]);

			// ── Geometry ──
		case 'Color': return TheGodotModule._variant_new_color(v[1], v[2], v[3], v[4]);
		case 'Rect2': return TheGodotModule._variant_new_rect2(v[1], v[2], v[3], v[4]);
		case 'Rect2i': return TheGodotModule._variant_new_rect2i(v[1], v[2], v[3], v[4]);
		case 'AABB': return TheGodotModule._variant_new_aabb(v[1], v[2], v[3], v[4], v[5], v[6]);
		case 'Plane': return TheGodotModule._variant_new_plane(v[1], v[2], v[3], v[4]);

			// ── Transforms ──
		case 'Quaternion': return TheGodotModule._variant_new_quaternion(v[1], v[2], v[3], v[4]);
		case 'Basis': return TheGodotModule._variant_new_basis(v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9]);
		case 'Transform2D': return TheGodotModule._variant_new_transform2d(v[1], v[2], v[3], v[4], v[5], v[6]);
		case 'Transform3D': return TheGodotModule._variant_new_transform3d(v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11], v[12]);

			// ── Godot handles ──
			// RID and ObjectId carry a single uint64 encoded as a JS number.
		case 'RID': return TheGodotModule._variant_new_rid(BigInt(Math.trunc(v[1])));
		case 'ObjectId': return TheGodotModule._variant_new_object_id(BigInt(Math.trunc(v[1])));

			// ── String types ──
		case 'NodePath': return TheGodotModule._variant_new_node_path(TheGodotModule.stringToUTF8OnStack(String(v[1] ?? '')));
		case 'StringName': return TheGodotModule._variant_new_string_name(TheGodotModule.stringToUTF8OnStack(String(v[1] ?? '')));

			// ── Callable / Signal ──
			// still a work in progress
		case 'Callable': return TheGodotModule._variant_new_callable(Math.trunc(v[1]));
		case 'Signal': return TheGodotModule._variant_new_signal(Math.trunc(v[1]));

			// ── Empty containers ──
			// Populated Array/Dictionary args are handled by their own marshal cases;
			// these tags cover empty containers passed as varargs.
		case 'Dictionary': return TheGodotModule._variant_new_dictionary();
		case 'Array': return TheGodotModule._variant_new_array();

			// Unknown tag — fall through to nil.
		default: return TheGodotModule._variant_new_nil();
		}
	}

	// Unrecognized JS value — produce a nil Variant rather than throwing.
	return TheGodotModule._variant_new_nil();
}
