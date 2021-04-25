/*************************************************************************/
/*  variant.cpp                                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "gdnative/variant.h"

#include "core/object/reference.h"
#include "core/variant/variant.h"

#ifdef __cplusplus
extern "C" {
#endif

static_assert(sizeof(godot_variant) == sizeof(Variant), "Variant size mismatch");

// Workaround GCC ICE on armv7hl which was affected GCC 6.0 up to 8.0 (GH-16100).
// It was fixed upstream in 8.1, and a fix was backported to 7.4.
// This can be removed once no supported distro ships with versions older than 7.4.
#if defined(__arm__) && defined(__GNUC__) && !defined(__clang__) && \
		(__GNUC__ == 6 || (__GNUC__ == 7 && __GNUC_MINOR__ < 4) || (__GNUC__ == 8 && __GNUC_MINOR__ < 1))
#pragma GCC push_options
#pragma GCC optimize("-O0")
#endif

#define memnew_placement_custom(m_placement, m_class, m_constr) _post_initialize(new (m_placement, sizeof(m_class), "") m_constr)

#if defined(__arm__) && defined(__GNUC__) && !defined(__clang__) && \
		(__GNUC__ == 6 || (__GNUC__ == 7 && __GNUC_MINOR__ < 4) || (__GNUC__ == 8 && __GNUC_MINOR__ < 1))
#pragma GCC pop_options
#endif

// Memory

void GDAPI godot_variant_new_copy(godot_variant *p_dest, const godot_variant *p_src) {
	Variant *dest = (Variant *)p_dest;
	const Variant *src = (const Variant *)p_src;
	memnew_placement(dest, Variant(*src));
}

void GDAPI godot_variant_new_nil(godot_variant *r_dest) {
	Variant *dest = (Variant *)r_dest;
	memnew_placement(dest, Variant);
}

void GDAPI godot_variant_new_bool(godot_variant *r_dest, const godot_bool p_b) {
	Variant *dest = (Variant *)r_dest;
	memnew_placement_custom(dest, Variant, Variant(p_b));
}

void GDAPI godot_variant_new_int(godot_variant *r_dest, const godot_int p_i) {
	Variant *dest = (Variant *)r_dest;
	memnew_placement_custom(dest, Variant, Variant(p_i));
}

void GDAPI godot_variant_new_float(godot_variant *r_dest, const godot_float p_r) {
	Variant *dest = (Variant *)r_dest;
	memnew_placement_custom(dest, Variant, Variant(p_r));
}

void GDAPI godot_variant_new_string(godot_variant *r_dest, const godot_string *p_s) {
	Variant *dest = (Variant *)r_dest;
	const String *s = (const String *)p_s;
	memnew_placement_custom(dest, Variant, Variant(*s));
}

void GDAPI godot_variant_new_string_name(godot_variant *r_dest, const godot_string_name *p_s) {
	Variant *dest = (Variant *)r_dest;
	const StringName *s = (const StringName *)p_s;
	memnew_placement_custom(dest, Variant, Variant(*s));
}

void GDAPI godot_variant_new_vector2(godot_variant *r_dest, const godot_vector2 *p_v2) {
	Variant *dest = (Variant *)r_dest;
	const Vector2 *v2 = (const Vector2 *)p_v2;
	memnew_placement_custom(dest, Variant, Variant(*v2));
}

void GDAPI godot_variant_new_vector2i(godot_variant *r_dest, const godot_vector2i *p_v2) {
	Variant *dest = (Variant *)r_dest;
	const Vector2i *v2 = (const Vector2i *)p_v2;
	memnew_placement_custom(dest, Variant, Variant(*v2));
}

void GDAPI godot_variant_new_rect2(godot_variant *r_dest, const godot_rect2 *p_rect2) {
	Variant *dest = (Variant *)r_dest;
	const Rect2 *rect2 = (const Rect2 *)p_rect2;
	memnew_placement_custom(dest, Variant, Variant(*rect2));
}

void GDAPI godot_variant_new_rect2i(godot_variant *r_dest, const godot_rect2i *p_rect2) {
	Variant *dest = (Variant *)r_dest;
	const Rect2i *rect2 = (const Rect2i *)p_rect2;
	memnew_placement_custom(dest, Variant, Variant(*rect2));
}

void GDAPI godot_variant_new_vector3(godot_variant *r_dest, const godot_vector3 *p_v3) {
	Variant *dest = (Variant *)r_dest;
	const Vector3 *v3 = (const Vector3 *)p_v3;
	memnew_placement_custom(dest, Variant, Variant(*v3));
}

void GDAPI godot_variant_new_vector3i(godot_variant *r_dest, const godot_vector3i *p_v3) {
	Variant *dest = (Variant *)r_dest;
	const Vector3i *v3 = (const Vector3i *)p_v3;
	memnew_placement_custom(dest, Variant, Variant(*v3));
}

void GDAPI godot_variant_new_transform2d(godot_variant *r_dest, const godot_transform2d *p_t2d) {
	Variant *dest = (Variant *)r_dest;
	const Transform2D *t2d = (const Transform2D *)p_t2d;
	memnew_placement_custom(dest, Variant, Variant(*t2d));
}

void GDAPI godot_variant_new_plane(godot_variant *r_dest, const godot_plane *p_plane) {
	Variant *dest = (Variant *)r_dest;
	const Plane *plane = (const Plane *)p_plane;
	memnew_placement_custom(dest, Variant, Variant(*plane));
}

void GDAPI godot_variant_new_quat(godot_variant *r_dest, const godot_quat *p_quat) {
	Variant *dest = (Variant *)r_dest;
	const Quat *quat = (const Quat *)p_quat;
	memnew_placement_custom(dest, Variant, Variant(*quat));
}

void GDAPI godot_variant_new_aabb(godot_variant *r_dest, const godot_aabb *p_aabb) {
	Variant *dest = (Variant *)r_dest;
	const AABB *aabb = (const AABB *)p_aabb;
	memnew_placement_custom(dest, Variant, Variant(*aabb));
}

void GDAPI godot_variant_new_basis(godot_variant *r_dest, const godot_basis *p_basis) {
	Variant *dest = (Variant *)r_dest;
	const Basis *basis = (const Basis *)p_basis;
	memnew_placement_custom(dest, Variant, Variant(*basis));
}

void GDAPI godot_variant_new_transform(godot_variant *r_dest, const godot_transform *p_trans) {
	Variant *dest = (Variant *)r_dest;
	const Transform *trans = (const Transform *)p_trans;
	memnew_placement_custom(dest, Variant, Variant(*trans));
}

void GDAPI godot_variant_new_color(godot_variant *r_dest, const godot_color *p_color) {
	Variant *dest = (Variant *)r_dest;
	const Color *color = (const Color *)p_color;
	memnew_placement_custom(dest, Variant, Variant(*color));
}

void GDAPI godot_variant_new_node_path(godot_variant *r_dest, const godot_node_path *p_np) {
	Variant *dest = (Variant *)r_dest;
	const NodePath *np = (const NodePath *)p_np;
	memnew_placement_custom(dest, Variant, Variant(*np));
}

void GDAPI godot_variant_new_rid(godot_variant *r_dest, const godot_rid *p_rid) {
	Variant *dest = (Variant *)r_dest;
	const RID *rid = (const RID *)p_rid;
	memnew_placement_custom(dest, Variant, Variant(*rid));
}

void GDAPI godot_variant_new_callable(godot_variant *r_dest, const godot_callable *p_cb) {
	Variant *dest = (Variant *)r_dest;
	const Callable *cb = (const Callable *)p_cb;
	memnew_placement_custom(dest, Variant, Variant(*cb));
}

void GDAPI godot_variant_new_signal(godot_variant *r_dest, const godot_signal *p_signal) {
	Variant *dest = (Variant *)r_dest;
	const Signal *signal = (const Signal *)p_signal;
	memnew_placement_custom(dest, Variant, Variant(*signal));
}

void GDAPI godot_variant_new_object(godot_variant *r_dest, const godot_object *p_obj) {
	Variant *dest = (Variant *)r_dest;
	const Object *obj = (const Object *)p_obj;
	const Reference *reference = Object::cast_to<Reference>(obj);
	REF ref;
	if (reference) {
		ref = REF(reference);
	}
	if (!ref.is_null()) {
		memnew_placement_custom(dest, Variant, Variant(ref));
	} else {
#if defined(DEBUG_METHODS_ENABLED)
		if (reference) {
			ERR_PRINT("Reference object has 0 refcount in godot_variant_new_object - you lost it somewhere.");
		}
#endif
		memnew_placement_custom(dest, Variant, Variant(obj));
	}
}

void GDAPI godot_variant_new_dictionary(godot_variant *r_dest, const godot_dictionary *p_dict) {
	Variant *dest = (Variant *)r_dest;
	const Dictionary *dict = (const Dictionary *)p_dict;
	memnew_placement_custom(dest, Variant, Variant(*dict));
}

void GDAPI godot_variant_new_array(godot_variant *r_dest, const godot_array *p_arr) {
	Variant *dest = (Variant *)r_dest;
	const Array *arr = (const Array *)p_arr;
	memnew_placement_custom(dest, Variant, Variant(*arr));
}

void GDAPI godot_variant_new_packed_byte_array(godot_variant *r_dest, const godot_packed_byte_array *p_pba) {
	Variant *dest = (Variant *)r_dest;
	const PackedByteArray *pba = (const PackedByteArray *)p_pba;
	memnew_placement_custom(dest, Variant, Variant(*pba));
}

void GDAPI godot_variant_new_packed_int32_array(godot_variant *r_dest, const godot_packed_int32_array *p_pia) {
	Variant *dest = (Variant *)r_dest;
	const PackedInt32Array *pia = (const PackedInt32Array *)p_pia;
	memnew_placement_custom(dest, Variant, Variant(*pia));
}

void GDAPI godot_variant_new_packed_int64_array(godot_variant *r_dest, const godot_packed_int64_array *p_pia) {
	Variant *dest = (Variant *)r_dest;
	const PackedInt64Array *pia = (const PackedInt64Array *)p_pia;
	memnew_placement_custom(dest, Variant, Variant(*pia));
}

void GDAPI godot_variant_new_packed_float32_array(godot_variant *r_dest, const godot_packed_float32_array *p_pra) {
	Variant *dest = (Variant *)r_dest;
	const PackedFloat32Array *pra = (const PackedFloat32Array *)p_pra;
	memnew_placement_custom(dest, Variant, Variant(*pra));
}

void GDAPI godot_variant_new_packed_float64_array(godot_variant *r_dest, const godot_packed_float64_array *p_pra) {
	Variant *dest = (Variant *)r_dest;
	const PackedFloat64Array *pra = (const PackedFloat64Array *)p_pra;
	memnew_placement_custom(dest, Variant, Variant(*pra));
}

void GDAPI godot_variant_new_packed_string_array(godot_variant *r_dest, const godot_packed_string_array *p_psa) {
	Variant *dest = (Variant *)r_dest;
	const PackedStringArray *psa = (const PackedStringArray *)p_psa;
	memnew_placement_custom(dest, Variant, Variant(*psa));
}

void GDAPI godot_variant_new_packed_vector2_array(godot_variant *r_dest, const godot_packed_vector2_array *p_pv2a) {
	Variant *dest = (Variant *)r_dest;
	const PackedVector2Array *pv2a = (const PackedVector2Array *)p_pv2a;
	memnew_placement_custom(dest, Variant, Variant(*pv2a));
}

void GDAPI godot_variant_new_packed_vector3_array(godot_variant *r_dest, const godot_packed_vector3_array *p_pv3a) {
	Variant *dest = (Variant *)r_dest;
	const PackedVector3Array *pv3a = (const PackedVector3Array *)p_pv3a;
	memnew_placement_custom(dest, Variant, Variant(*pv3a));
}

void GDAPI godot_variant_new_packed_color_array(godot_variant *r_dest, const godot_packed_color_array *p_pca) {
	Variant *dest = (Variant *)r_dest;
	const PackedColorArray *pca = (const PackedColorArray *)p_pca;
	memnew_placement_custom(dest, Variant, Variant(*pca));
}

godot_bool GDAPI godot_variant_as_bool(const godot_variant *p_self) {
	const Variant *self = (const Variant *)p_self;
	return self->operator bool();
}

godot_int GDAPI godot_variant_as_int(const godot_variant *p_self) {
	const Variant *self = (const Variant *)p_self;
	return self->operator int64_t();
}

godot_float GDAPI godot_variant_as_float(const godot_variant *p_self) {
	const Variant *self = (const Variant *)p_self;
	return self->operator double();
}

godot_string GDAPI godot_variant_as_string(const godot_variant *p_self) {
	godot_string raw_dest;
	const Variant *self = (const Variant *)p_self;
	String *dest = (String *)&raw_dest;
	memnew_placement(dest, String(self->operator String())); // operator = is overloaded by String
	return raw_dest;
}

godot_string_name GDAPI godot_variant_as_string_name(const godot_variant *p_self) {
	godot_string_name raw_dest;
	const Variant *self = (const Variant *)p_self;
	StringName *dest = (StringName *)&raw_dest;
	memnew_placement(dest, StringName(self->operator StringName())); // operator = is overloaded by StringName
	return raw_dest;
}

godot_vector2 GDAPI godot_variant_as_vector2(const godot_variant *p_self) {
	godot_vector2 raw_dest;
	const Variant *self = (const Variant *)p_self;
	Vector2 *dest = (Vector2 *)&raw_dest;
	*dest = *self;
	return raw_dest;
}

godot_vector2i GDAPI godot_variant_as_vector2i(const godot_variant *p_self) {
	godot_vector2i raw_dest;
	const Variant *self = (const Variant *)p_self;
	Vector2i *dest = (Vector2i *)&raw_dest;
	*dest = *self;
	return raw_dest;
}

godot_rect2 GDAPI godot_variant_as_rect2(const godot_variant *p_self) {
	godot_rect2 raw_dest;
	const Variant *self = (const Variant *)p_self;
	Rect2 *dest = (Rect2 *)&raw_dest;
	*dest = *self;
	return raw_dest;
}

godot_rect2i GDAPI godot_variant_as_rect2i(const godot_variant *p_self) {
	godot_rect2i raw_dest;
	const Variant *self = (const Variant *)p_self;
	Rect2i *dest = (Rect2i *)&raw_dest;
	*dest = *self;
	return raw_dest;
}

godot_vector3 GDAPI godot_variant_as_vector3(const godot_variant *p_self) {
	godot_vector3 raw_dest;
	const Variant *self = (const Variant *)p_self;
	Vector3 *dest = (Vector3 *)&raw_dest;
	*dest = *self;
	return raw_dest;
}

godot_vector3i GDAPI godot_variant_as_vector3i(const godot_variant *p_self) {
	godot_vector3i raw_dest;
	const Variant *self = (const Variant *)p_self;
	Vector3i *dest = (Vector3i *)&raw_dest;
	*dest = *self;
	return raw_dest;
}

godot_transform2d GDAPI godot_variant_as_transform2d(const godot_variant *p_self) {
	godot_transform2d raw_dest;
	const Variant *self = (const Variant *)p_self;
	Transform2D *dest = (Transform2D *)&raw_dest;
	*dest = *self;
	return raw_dest;
}

godot_plane GDAPI godot_variant_as_plane(const godot_variant *p_self) {
	godot_plane raw_dest;
	const Variant *self = (const Variant *)p_self;
	Plane *dest = (Plane *)&raw_dest;
	*dest = *self;
	return raw_dest;
}

godot_quat GDAPI godot_variant_as_quat(const godot_variant *p_self) {
	godot_quat raw_dest;
	const Variant *self = (const Variant *)p_self;
	Quat *dest = (Quat *)&raw_dest;
	*dest = *self;
	return raw_dest;
}

godot_aabb GDAPI godot_variant_as_aabb(const godot_variant *p_self) {
	godot_aabb raw_dest;
	const Variant *self = (const Variant *)p_self;
	AABB *dest = (AABB *)&raw_dest;
	*dest = *self;
	return raw_dest;
}

godot_basis GDAPI godot_variant_as_basis(const godot_variant *p_self) {
	godot_basis raw_dest;
	const Variant *self = (const Variant *)p_self;
	Basis *dest = (Basis *)&raw_dest;
	*dest = *self;
	return raw_dest;
}

godot_transform GDAPI godot_variant_as_transform(const godot_variant *p_self) {
	godot_transform raw_dest;
	const Variant *self = (const Variant *)p_self;
	Transform *dest = (Transform *)&raw_dest;
	*dest = *self;
	return raw_dest;
}

godot_color GDAPI godot_variant_as_color(const godot_variant *p_self) {
	godot_color raw_dest;
	const Variant *self = (const Variant *)p_self;
	Color *dest = (Color *)&raw_dest;
	*dest = *self;
	return raw_dest;
}

godot_node_path GDAPI godot_variant_as_node_path(const godot_variant *p_self) {
	godot_node_path raw_dest;
	const Variant *self = (const Variant *)p_self;
	NodePath *dest = (NodePath *)&raw_dest;
	memnew_placement(dest, NodePath(self->operator NodePath())); // operator = is overloaded by NodePath
	return raw_dest;
}

godot_rid GDAPI godot_variant_as_rid(const godot_variant *p_self) {
	godot_rid raw_dest;
	const Variant *self = (const Variant *)p_self;
	RID *dest = (RID *)&raw_dest;
	*dest = *self;
	return raw_dest;
}

godot_callable GDAPI godot_variant_as_callable(const godot_variant *p_self) {
	godot_callable raw_dest;
	const Variant *self = (const Variant *)p_self;
	Callable *dest = (Callable *)&raw_dest;
	*dest = *self;
	return raw_dest;
}

godot_signal GDAPI godot_variant_as_signal(const godot_variant *p_self) {
	godot_signal raw_dest;
	const Variant *self = (const Variant *)p_self;
	Signal *dest = (Signal *)&raw_dest;
	*dest = *self;
	return raw_dest;
}

godot_object GDAPI *godot_variant_as_object(const godot_variant *p_self) {
	const Variant *self = (const Variant *)p_self;
	Object *dest;
	dest = *self;
	return (godot_object *)dest;
}

godot_dictionary GDAPI godot_variant_as_dictionary(const godot_variant *p_self) {
	godot_dictionary raw_dest;
	const Variant *self = (const Variant *)p_self;
	Dictionary *dest = (Dictionary *)&raw_dest;
	memnew_placement(dest, Dictionary(self->operator Dictionary())); // operator = is overloaded by Dictionary
	return raw_dest;
}

godot_array GDAPI godot_variant_as_array(const godot_variant *p_self) {
	godot_array raw_dest;
	const Variant *self = (const Variant *)p_self;
	Array *dest = (Array *)&raw_dest;
	memnew_placement(dest, Array(self->operator Array())); // operator = is overloaded by Array
	return raw_dest;
}

godot_packed_byte_array GDAPI godot_variant_as_packed_byte_array(const godot_variant *p_self) {
	godot_packed_byte_array raw_dest;
	const Variant *self = (const Variant *)p_self;
	PackedByteArray *dest = (PackedByteArray *)&raw_dest;
	memnew_placement(dest, PackedByteArray(self->operator PackedByteArray())); // operator = is overloaded by PackedByteArray
	*dest = *self;
	return raw_dest;
}

godot_packed_int32_array GDAPI godot_variant_as_packed_int32_array(const godot_variant *p_self) {
	godot_packed_int32_array raw_dest;
	const Variant *self = (const Variant *)p_self;
	PackedInt32Array *dest = (PackedInt32Array *)&raw_dest;
	memnew_placement(dest, PackedInt32Array(self->operator PackedInt32Array())); // operator = is overloaded by PackedInt32Array
	*dest = *self;
	return raw_dest;
}

godot_packed_int64_array GDAPI godot_variant_as_packed_int64_array(const godot_variant *p_self) {
	godot_packed_int64_array raw_dest;
	const Variant *self = (const Variant *)p_self;
	PackedInt64Array *dest = (PackedInt64Array *)&raw_dest;
	memnew_placement(dest, PackedInt64Array(self->operator PackedInt64Array())); // operator = is overloaded by PackedInt64Array
	*dest = *self;
	return raw_dest;
}

godot_packed_float32_array GDAPI godot_variant_as_packed_float32_array(const godot_variant *p_self) {
	godot_packed_float32_array raw_dest;
	const Variant *self = (const Variant *)p_self;
	PackedFloat32Array *dest = (PackedFloat32Array *)&raw_dest;
	memnew_placement(dest, PackedFloat32Array(self->operator PackedFloat32Array())); // operator = is overloaded by PackedFloat32Array
	*dest = *self;
	return raw_dest;
}

godot_packed_float64_array GDAPI godot_variant_as_packed_float64_array(const godot_variant *p_self) {
	godot_packed_float64_array raw_dest;
	const Variant *self = (const Variant *)p_self;
	PackedFloat64Array *dest = (PackedFloat64Array *)&raw_dest;
	memnew_placement(dest, PackedFloat64Array(self->operator PackedFloat64Array())); // operator = is overloaded by PackedFloat64Array
	*dest = *self;
	return raw_dest;
}

godot_packed_string_array GDAPI godot_variant_as_packed_string_array(const godot_variant *p_self) {
	godot_packed_string_array raw_dest;
	const Variant *self = (const Variant *)p_self;
	PackedStringArray *dest = (PackedStringArray *)&raw_dest;
	memnew_placement(dest, PackedStringArray(self->operator PackedStringArray())); // operator = is overloaded by PackedStringArray
	*dest = *self;
	return raw_dest;
}

godot_packed_vector2_array GDAPI godot_variant_as_packed_vector2_array(const godot_variant *p_self) {
	godot_packed_vector2_array raw_dest;
	const Variant *self = (const Variant *)p_self;
	PackedVector2Array *dest = (PackedVector2Array *)&raw_dest;
	memnew_placement(dest, PackedVector2Array(self->operator PackedVector2Array())); // operator = is overloaded by PackedVector2Array
	*dest = *self;
	return raw_dest;
}

godot_packed_vector3_array GDAPI godot_variant_as_packed_vector3_array(const godot_variant *p_self) {
	godot_packed_vector3_array raw_dest;
	const Variant *self = (const Variant *)p_self;
	PackedVector3Array *dest = (PackedVector3Array *)&raw_dest;
	memnew_placement(dest, PackedVector3Array(self->operator PackedVector3Array())); // operator = is overloaded by PackedVector3Array
	*dest = *self;
	return raw_dest;
}

godot_packed_color_array GDAPI godot_variant_as_packed_color_array(const godot_variant *p_self) {
	godot_packed_color_array raw_dest;
	const Variant *self = (const Variant *)p_self;
	PackedColorArray *dest = (PackedColorArray *)&raw_dest;
	memnew_placement(dest, PackedColorArray(self->operator PackedColorArray())); // operator = is overloaded by PackedColorArray
	*dest = *self;
	return raw_dest;
}

void GDAPI godot_variant_destroy(godot_variant *p_self) {
	Variant *self = (Variant *)p_self;
	self->~Variant();
}

// Dynamic interaction.

void GDAPI godot_variant_call(godot_variant *p_self, const godot_string_name *p_method, const godot_variant **p_args, const godot_int p_argcount, godot_variant *r_return, godot_variant_call_error *r_error) {
	Variant *self = (Variant *)p_self;
	const StringName *method = (const StringName *)p_method;
	const Variant **args = (const Variant **)p_args;
	Variant ret;
	Callable::CallError error;
	self->call(*method, args, p_argcount, ret, error);
	memnew_placement_custom(r_return, Variant, Variant(ret));

	if (r_error) {
		r_error->error = (godot_variant_call_error_error)error.error;
		r_error->argument = error.argument;
		r_error->expected = (godot_variant_type)error.expected;
	}
}

void GDAPI godot_variant_call_with_cstring(godot_variant *p_self, const char *p_method, const godot_variant **p_args, const godot_int p_argcount, godot_variant *r_return, godot_variant_call_error *r_error) {
	Variant *self = (Variant *)p_self;
	const StringName method(p_method);
	const Variant **args = (const Variant **)p_args;
	Variant ret;
	Callable::CallError error;
	self->call(method, args, p_argcount, ret, error);
	memnew_placement_custom(r_return, Variant, Variant(ret));

	if (r_error) {
		r_error->error = (godot_variant_call_error_error)error.error;
		r_error->argument = error.argument;
		r_error->expected = (godot_variant_type)error.expected;
	}
}

void GDAPI godot_variant_call_static(godot_variant_type p_type, const godot_string_name *p_method, const godot_variant **p_args, const godot_int p_argcount, godot_variant *r_return, godot_variant_call_error *r_error) {
	Variant::Type type = (Variant::Type)p_type;
	const StringName *method = (const StringName *)p_method;
	const Variant **args = (const Variant **)p_args;
	Variant ret;
	Callable::CallError error;
	Variant::call_static(type, *method, args, p_argcount, ret, error);
	memnew_placement_custom(r_return, Variant, Variant(ret));

	if (r_error) {
		r_error->error = (godot_variant_call_error_error)error.error;
		r_error->argument = error.argument;
		r_error->expected = (godot_variant_type)error.expected;
	}
}

void GDAPI godot_variant_call_static_with_cstring(godot_variant_type p_type, const char *p_method, const godot_variant **p_args, const godot_int p_argcount, godot_variant *r_return, godot_variant_call_error *r_error) {
	Variant::Type type = (Variant::Type)p_type;
	const StringName method(p_method);
	const Variant **args = (const Variant **)p_args;
	Variant ret;
	Callable::CallError error;
	Variant::call_static(type, method, args, p_argcount, ret, error);
	memnew_placement_custom(r_return, Variant, Variant(ret));

	if (r_error) {
		r_error->error = (godot_variant_call_error_error)error.error;
		r_error->argument = error.argument;
		r_error->expected = (godot_variant_type)error.expected;
	}
}

void GDAPI godot_variant_evaluate(godot_variant_operator p_op, const godot_variant *p_a, const godot_variant *p_b, godot_variant *r_return, bool *r_valid) {
	Variant::Operator op = (Variant::Operator)p_op;
	const Variant *a = (const Variant *)p_a;
	const Variant *b = (const Variant *)p_b;
	Variant *ret = (Variant *)r_return;
	Variant::evaluate(op, *a, *b, *ret, *r_valid);
}

void GDAPI godot_variant_set(godot_variant *p_self, const godot_variant *p_key, const godot_variant *p_value, bool *r_valid) {
	Variant *self = (Variant *)p_self;
	const Variant *key = (const Variant *)p_key;
	const Variant *value = (const Variant *)p_value;

	self->set(*key, *value, r_valid);
}

void GDAPI godot_variant_set_named(godot_variant *p_self, const godot_string_name *p_key, const godot_variant *p_value, bool *r_valid) {
	Variant *self = (Variant *)p_self;
	const StringName *key = (const StringName *)p_key;
	const Variant *value = (const Variant *)p_value;

	self->set_named(*key, *value, *r_valid);
}

void GDAPI godot_variant_set_named_with_cstring(godot_variant *p_self, const char *p_key, const godot_variant *p_value, bool *r_valid) {
	Variant *self = (Variant *)p_self;
	const StringName key(p_key);
	const Variant *value = (const Variant *)p_value;

	self->set_named(key, *value, *r_valid);
}

void GDAPI godot_variant_set_keyed(godot_variant *p_self, const godot_variant *p_key, const godot_variant *p_value, bool *r_valid) {
	Variant *self = (Variant *)p_self;
	const Variant *key = (const Variant *)p_key;
	const Variant *value = (const Variant *)p_value;

	self->set_keyed(*key, *value, *r_valid);
}

void GDAPI godot_variant_set_indexed(godot_variant *p_self, godot_int p_index, const godot_variant *p_value, bool *r_valid, bool *r_oob) {
	Variant *self = (Variant *)p_self;
	const Variant *value = (const Variant *)p_value;

	self->set_indexed(p_index, value, *r_valid, *r_oob);
}

godot_variant GDAPI godot_variant_get(const godot_variant *p_self, const godot_variant *p_key, bool *r_valid) {
	const Variant *self = (const Variant *)p_self;
	const Variant *key = (const Variant *)p_key;
	Variant ret;

	ret = self->get(*key, r_valid);
	godot_variant result;
	memnew_placement_custom(&result, Variant, Variant(ret));
	return result;
}

godot_variant GDAPI godot_variant_get_named(const godot_variant *p_self, const godot_string_name *p_key, bool *r_valid) {
	const Variant *self = (const Variant *)p_self;
	const StringName *key = (const StringName *)p_key;
	Variant ret;

	ret = self->get_named(*key, *r_valid);
	godot_variant result;
	memnew_placement_custom(&result, Variant, Variant(ret));
	return result;
}

godot_variant GDAPI godot_variant_get_named_with_cstring(const godot_variant *p_self, const char *p_key, bool *r_valid) {
	const Variant *self = (const Variant *)p_self;
	const StringName *key = (const StringName *)p_key;
	Variant ret;

	ret = self->get_named(*key, *r_valid);
	godot_variant result;
	memnew_placement_custom(&result, Variant, Variant(ret));
	return result;
}

godot_variant GDAPI godot_variant_get_keyed(const godot_variant *p_self, const godot_variant *p_key, bool *r_valid) {
	const Variant *self = (const Variant *)p_self;
	const Variant *key = (const Variant *)p_key;
	Variant ret;

	ret = self->get_keyed(*key, *r_valid);
	godot_variant result;
	memnew_placement_custom(&result, Variant, Variant(ret));
	return result;
}

godot_variant GDAPI godot_variant_get_indexed(const godot_variant *p_self, godot_int p_index, bool *r_valid, bool *r_oob) {
	const Variant *self = (const Variant *)p_self;
	Variant ret;

	ret = self->get_indexed(p_index, *r_valid, *r_oob);
	godot_variant result;
	memnew_placement_custom(&result, Variant, Variant(ret));
	return result;
}

/// Iteration.
bool GDAPI godot_variant_iter_init(const godot_variant *p_self, godot_variant *r_iter, bool *r_valid) {
	const Variant *self = (const Variant *)p_self;
	Variant *iter = (Variant *)r_iter;

	return self->iter_init(*iter, *r_valid);
}

bool GDAPI godot_variant_iter_next(const godot_variant *p_self, godot_variant *r_iter, bool *r_valid) {
	const Variant *self = (const Variant *)p_self;
	Variant *iter = (Variant *)r_iter;

	return self->iter_next(*iter, *r_valid);
}

godot_variant GDAPI godot_variant_iter_get(const godot_variant *p_self, godot_variant *r_iter, bool *r_valid) {
	const Variant *self = (const Variant *)p_self;
	Variant *iter = (Variant *)r_iter;

	Variant result = self->iter_next(*iter, *r_valid);
	godot_variant ret;
	memnew_placement_custom(&ret, Variant, Variant(result));
	return ret;
}

/// Variant functions.
godot_bool GDAPI godot_variant_hash_compare(const godot_variant *p_self, const godot_variant *p_other) {
	const Variant *self = (const Variant *)p_self;
	const Variant *other = (const Variant *)p_other;
	return self->hash_compare(*other);
}

godot_bool GDAPI godot_variant_booleanize(const godot_variant *p_self) {
	const Variant *self = (const Variant *)p_self;
	return self->booleanize();
}

void GDAPI godot_variant_blend(const godot_variant *p_a, const godot_variant *p_b, float p_c, godot_variant *r_dst) {
	const Variant *a = (const Variant *)p_a;
	const Variant *b = (const Variant *)p_b;
	Variant *dst = (Variant *)r_dst;
	Variant::blend(*a, *b, p_c, *dst);
}

void GDAPI godot_variant_interpolate(const godot_variant *p_a, const godot_variant *p_b, float p_c, godot_variant *r_dst) {
	const Variant *a = (const Variant *)p_a;
	const Variant *b = (const Variant *)p_b;
	Variant *dst = (Variant *)r_dst;
	Variant::interpolate(*a, *b, p_c, *dst);
}

godot_variant GDAPI godot_variant_duplicate(const godot_variant *p_self, godot_bool p_deep) {
	const Variant *self = (const Variant *)p_self;
	Variant result = self->duplicate(p_deep);
	godot_variant ret;
	memnew_placement_custom(&ret, Variant, Variant(result));
	return ret;
}

godot_string GDAPI godot_variant_stringify(const godot_variant *p_self) {
	const Variant *self = (const Variant *)p_self;
	String result = *self;
	godot_string ret;
	memnew_placement_custom(&ret, String, String(result));
	return ret;
}

// Discovery API

/// Operators
godot_validated_operator_evaluator GDAPI godot_variant_get_validated_operator_evaluator(godot_variant_operator p_operator, godot_variant_type p_type_a, godot_variant_type p_type_b) {
	return (godot_validated_operator_evaluator)Variant::get_validated_operator_evaluator((Variant::Operator)p_operator, (Variant::Type)p_type_a, (Variant::Type)p_type_b);
}

godot_ptr_operator_evaluator GDAPI godot_variant_get_ptr_operator_evaluator(godot_variant_operator p_operator, godot_variant_type p_type_a, godot_variant_type p_type_b) {
	return (godot_ptr_operator_evaluator)Variant::get_ptr_operator_evaluator((Variant::Operator)p_operator, (Variant::Type)p_type_a, (Variant::Type)p_type_b);
}

godot_variant_type GDAPI godot_variant_get_operator_return_type(godot_variant_operator p_operator, godot_variant_type p_type_a, godot_variant_type p_type_b) {
	return (godot_variant_type)Variant::get_operator_return_type((Variant::Operator)p_operator, (Variant::Type)p_type_a, (Variant::Type)p_type_b);
}

godot_string GDAPI godot_variant_get_operator_name(godot_variant_operator p_operator) {
	String op_name = Variant::get_operator_name((Variant::Operator)p_operator);
	godot_string ret;
	memnew_placement_custom(&ret, String, String(op_name));
	return ret;
}

/// Built-in Methods

bool GDAPI godot_variant_has_builtin_method(godot_variant_type p_type, const godot_string_name *p_method) {
	return Variant::has_builtin_method((Variant::Type)p_type, *((const StringName *)p_method));
}

bool GDAPI godot_variant_has_builtin_method_with_cstring(godot_variant_type p_type, const char *p_method) {
	return Variant::has_builtin_method((Variant::Type)p_type, StringName(p_method));
}

godot_validated_builtin_method GDAPI godot_variant_get_validated_builtin_method(godot_variant_type p_type, const godot_string_name *p_method) {
	return (godot_validated_builtin_method)Variant::get_validated_builtin_method((Variant::Type)p_type, *((const StringName *)p_method));
}

godot_validated_builtin_method GDAPI godot_variant_get_validated_builtin_method_with_cstring(godot_variant_type p_type, const char *p_method) {
	return (godot_validated_builtin_method)Variant::get_validated_builtin_method((Variant::Type)p_type, StringName(p_method));
}

godot_ptr_builtin_method GDAPI godot_variant_get_ptr_builtin_method(godot_variant_type p_type, const godot_string_name *p_method) {
	return (godot_ptr_builtin_method)Variant::get_ptr_builtin_method((Variant::Type)p_type, *((const StringName *)p_method));
}

godot_ptr_builtin_method GDAPI godot_variant_get_ptr_builtin_method_with_cstring(godot_variant_type p_type, const char *p_method) {
	return (godot_ptr_builtin_method)Variant::get_ptr_builtin_method((Variant::Type)p_type, StringName(p_method));
}

int GDAPI godot_variant_get_builtin_method_argument_count(godot_variant_type p_type, const godot_string_name *p_method) {
	return Variant::get_builtin_method_argument_count((Variant::Type)p_type, *((const StringName *)p_method));
}

int GDAPI godot_variant_get_builtin_method_argument_count_with_cstring(godot_variant_type p_type, const char *p_method) {
	return Variant::get_builtin_method_argument_count((Variant::Type)p_type, StringName(p_method));
}

godot_variant_type GDAPI godot_variant_get_builtin_method_argument_type(godot_variant_type p_type, const godot_string_name *p_method, int p_argument) {
	return (godot_variant_type)Variant::get_builtin_method_argument_type((Variant::Type)p_type, *((const StringName *)p_method), p_argument);
}

godot_variant_type GDAPI godot_variant_get_builtin_method_argument_type_with_cstring(godot_variant_type p_type, const char *p_method, int p_argument) {
	return (godot_variant_type)Variant::get_builtin_method_argument_type((Variant::Type)p_type, StringName(p_method), p_argument);
}

godot_string GDAPI godot_variant_get_builtin_method_argument_name(godot_variant_type p_type, const godot_string_name *p_method, int p_argument) {
	String name = Variant::get_builtin_method_argument_name((Variant::Type)p_type, *((const StringName *)p_method), p_argument);
	return *(godot_string *)&name;
}

godot_string GDAPI godot_variant_get_builtin_method_argument_name_with_cstring(godot_variant_type p_type, const char *p_method, int p_argument) {
	String name = Variant::get_builtin_method_argument_name((Variant::Type)p_type, StringName(p_method), p_argument);
	return *(godot_string *)&name;
}

bool GDAPI godot_variant_has_builtin_method_return_value(godot_variant_type p_type, const godot_string_name *p_method) {
	return Variant::has_builtin_method_return_value((Variant::Type)p_type, *((const StringName *)p_method));
}

bool GDAPI godot_variant_has_builtin_method_return_value_with_cstring(godot_variant_type p_type, const char *p_method) {
	return Variant::has_builtin_method_return_value((Variant::Type)p_type, StringName(p_method));
}

godot_variant_type GDAPI godot_variant_get_builtin_method_return_type(godot_variant_type p_type, const godot_string_name *p_method) {
	return (godot_variant_type)Variant::get_builtin_method_return_type((Variant::Type)p_type, *((const StringName *)p_method));
}

godot_variant_type GDAPI godot_variant_get_builtin_method_return_type_with_cstring(godot_variant_type p_type, const char *p_method) {
	return (godot_variant_type)Variant::get_builtin_method_return_type((Variant::Type)p_type, StringName(p_method));
}

bool GDAPI godot_variant_is_builtin_method_const(godot_variant_type p_type, const godot_string_name *p_method) {
	return Variant::is_builtin_method_const((Variant::Type)p_type, *((const StringName *)p_method));
}

bool GDAPI godot_variant_is_builtin_method_const_with_cstring(godot_variant_type p_type, const char *p_method) {
	return Variant::is_builtin_method_const((Variant::Type)p_type, StringName(p_method));
}

bool GDAPI godot_variant_is_builtin_method_static(godot_variant_type p_type, const godot_string_name *p_method) {
	return Variant::is_builtin_method_static((Variant::Type)p_type, *((const StringName *)p_method));
}

bool GDAPI godot_variant_is_builtin_method_static_with_cstring(godot_variant_type p_type, const char *p_method) {
	return Variant::is_builtin_method_static((Variant::Type)p_type, StringName(p_method));
}

bool GDAPI godot_variant_is_builtin_method_vararg(godot_variant_type p_type, const godot_string_name *p_method) {
	return Variant::is_builtin_method_vararg((Variant::Type)p_type, *((const StringName *)p_method));
}

bool GDAPI godot_variant_is_builtin_method_vararg_with_cstring(godot_variant_type p_type, const char *p_method) {
	return Variant::is_builtin_method_vararg((Variant::Type)p_type, StringName(p_method));
}

int GDAPI godot_variant_get_builtin_method_count(godot_variant_type p_type) {
	return Variant::get_builtin_method_count((Variant::Type)p_type);
}

void GDAPI godot_variant_get_builtin_method_list(godot_variant_type p_type, godot_string_name *r_list) {
	List<StringName> list;
	Variant::get_builtin_method_list((Variant::Type)p_type, &list);
	int i = 0;
	for (const List<StringName>::Element *E = list.front(); E; E = E->next()) {
		memnew_placement_custom(&r_list[i], StringName, StringName(E->get()));
	}
}

/// Constructors

int GDAPI godot_variant_get_constructor_count(godot_variant_type p_type) {
	return Variant::get_constructor_count((Variant::Type)p_type);
}

godot_validated_constructor GDAPI godot_variant_get_validated_constructor(godot_variant_type p_type, int p_constructor) {
	return (godot_validated_constructor)Variant::get_validated_constructor((Variant::Type)p_type, p_constructor);
}

godot_ptr_constructor GDAPI godot_variant_get_ptr_constructor(godot_variant_type p_type, int p_constructor) {
	return (godot_ptr_constructor)Variant::get_ptr_constructor((Variant::Type)p_type, p_constructor);
}

int GDAPI godot_variant_get_constructor_argument_count(godot_variant_type p_type, int p_constructor) {
	return Variant::get_constructor_argument_count((Variant::Type)p_type, p_constructor);
}

godot_variant_type GDAPI godot_variant_get_constructor_argument_type(godot_variant_type p_type, int p_constructor, int p_argument) {
	return (godot_variant_type)Variant::get_constructor_argument_type((Variant::Type)p_type, p_constructor, p_argument);
}

godot_string GDAPI godot_variant_get_constructor_argument_name(godot_variant_type p_type, int p_constructor, int p_argument) {
	String name = Variant::get_constructor_argument_name((Variant::Type)p_type, p_constructor, p_argument);
	godot_string ret;
	memnew_placement(&ret, String(name));
	return ret;
}

void GDAPI godot_variant_construct(godot_variant_type p_type, godot_variant *p_base, const godot_variant **p_args, int p_argcount, godot_variant_call_error *r_error) {
	Variant::construct((Variant::Type)p_type, *((Variant *)p_base), (const Variant **)p_args, p_argcount, *((Callable::CallError *)r_error));
}

/// Properties.
godot_variant_type GDAPI godot_variant_get_member_type(godot_variant_type p_type, const godot_string_name *p_member) {
	return (godot_variant_type)Variant::get_member_type((Variant::Type)p_type, *((const StringName *)p_member));
}

godot_variant_type GDAPI godot_variant_get_member_type_with_cstring(godot_variant_type p_type, const char *p_member) {
	return (godot_variant_type)Variant::get_member_type((Variant::Type)p_type, StringName(p_member));
}

int GDAPI godot_variant_get_member_count(godot_variant_type p_type) {
	return Variant::get_member_count((Variant::Type)p_type);
}

void GDAPI godot_variant_get_member_list(godot_variant_type p_type, godot_string_name *r_list) {
	List<StringName> members;
	Variant::get_member_list((Variant::Type)p_type, &members);
	int i = 0;
	for (const List<StringName>::Element *E = members.front(); E; E = E->next()) {
		memnew_placement_custom(&r_list[i++], StringName, StringName(E->get()));
	}
}

godot_validated_setter GDAPI godot_variant_get_validated_setter(godot_variant_type p_type, const godot_string_name *p_member) {
	return (godot_validated_setter)Variant::get_member_validated_setter((Variant::Type)p_type, *((const StringName *)p_member));
}

godot_validated_setter GDAPI godot_variant_get_validated_setter_with_cstring(godot_variant_type p_type, const char *p_member) {
	return (godot_validated_setter)Variant::get_member_validated_setter((Variant::Type)p_type, StringName(p_member));
}

godot_validated_getter GDAPI godot_variant_get_validated_getter(godot_variant_type p_type, const godot_string_name *p_member) {
	return (godot_validated_getter)Variant::get_member_validated_getter((Variant::Type)p_type, *((const StringName *)p_member));
}

godot_validated_getter GDAPI godot_variant_get_validated_getter_with_cstring(godot_variant_type p_type, const char *p_member) {
	return (godot_validated_getter)Variant::get_member_validated_getter((Variant::Type)p_type, StringName(p_member));
}

godot_ptr_setter GDAPI godot_variant_get_ptr_setter(godot_variant_type p_type, const godot_string_name *p_member) {
	return (godot_ptr_setter)Variant::get_member_ptr_setter((Variant::Type)p_type, *((const StringName *)p_member));
}

godot_ptr_setter GDAPI godot_variant_get_ptr_setter_with_cstring(godot_variant_type p_type, const char *p_member) {
	return (godot_ptr_setter)Variant::get_member_ptr_setter((Variant::Type)p_type, StringName(p_member));
}

godot_ptr_getter GDAPI godot_variant_get_ptr_getter(godot_variant_type p_type, const godot_string_name *p_member) {
	return (godot_ptr_getter)Variant::get_member_ptr_getter((Variant::Type)p_type, *((const StringName *)p_member));
}

godot_ptr_getter GDAPI godot_variant_get_ptr_getter_with_cstring(godot_variant_type p_type, const char *p_member) {
	return (godot_ptr_getter)Variant::get_member_ptr_getter((Variant::Type)p_type, StringName(p_member));
}

/// Indexing.
bool GDAPI godot_variant_has_indexing(godot_variant_type p_type) {
	return Variant::has_indexing((Variant::Type)p_type);
}

godot_variant_type GDAPI godot_variant_get_indexed_element_type(godot_variant_type p_type) {
	return (godot_variant_type)Variant::get_indexed_element_type((Variant::Type)p_type);
}

godot_validated_indexed_setter GDAPI godot_variant_get_validated_indexed_setter(godot_variant_type p_type) {
	return (godot_validated_indexed_setter)Variant::get_member_validated_indexed_setter((Variant::Type)p_type);
}

godot_validated_indexed_getter GDAPI godot_variant_get_validated_indexed_getter(godot_variant_type p_type) {
	return (godot_validated_indexed_getter)Variant::get_member_validated_indexed_getter((Variant::Type)p_type);
}

godot_ptr_indexed_setter GDAPI godot_variant_get_ptr_indexed_setter(godot_variant_type p_type) {
	return (godot_ptr_indexed_setter)Variant::get_member_ptr_indexed_setter((Variant::Type)p_type);
}

godot_ptr_indexed_getter GDAPI godot_variant_get_ptr_indexed_getter(godot_variant_type p_type) {
	return (godot_ptr_indexed_getter)Variant::get_member_ptr_indexed_getter((Variant::Type)p_type);
}

uint64_t GDAPI godot_variant_get_indexed_size(const godot_variant *p_self) {
	const Variant *self = (const Variant *)p_self;
	return self->get_indexed_size();
}

/// Keying.
bool GDAPI godot_variant_is_keyed(godot_variant_type p_type) {
	return Variant::is_keyed((Variant::Type)p_type);
}

godot_validated_keyed_setter GDAPI godot_variant_get_validated_keyed_setter(godot_variant_type p_type) {
	return (godot_validated_keyed_setter)Variant::get_member_validated_keyed_setter((Variant::Type)p_type);
}

godot_validated_keyed_getter GDAPI godot_variant_get_validated_keyed_getter(godot_variant_type p_type) {
	return (godot_validated_keyed_getter)Variant::get_member_validated_keyed_getter((Variant::Type)p_type);
}

godot_validated_keyed_checker GDAPI godot_variant_get_validated_keyed_checker(godot_variant_type p_type) {
	return (godot_validated_keyed_checker)Variant::get_member_validated_keyed_checker((Variant::Type)p_type);
}

godot_ptr_keyed_setter GDAPI godot_variant_get_ptr_keyed_setter(godot_variant_type p_type) {
	return (godot_ptr_keyed_setter)Variant::get_member_ptr_keyed_setter((Variant::Type)p_type);
}

godot_ptr_keyed_getter GDAPI godot_variant_get_ptr_keyed_getter(godot_variant_type p_type) {
	return (godot_ptr_keyed_getter)Variant::get_member_ptr_keyed_getter((Variant::Type)p_type);
}

godot_ptr_keyed_checker GDAPI godot_variant_get_ptr_keyed_checker(godot_variant_type p_type) {
	return (godot_ptr_keyed_checker)Variant::get_member_ptr_keyed_checker((Variant::Type)p_type);
}

/// Constants.
int GDAPI godot_variant_get_constants_count(godot_variant_type p_type) {
	return Variant::get_constants_count_for_type((Variant::Type)p_type);
}

void GDAPI godot_variant_get_constants_list(godot_variant_type p_type, godot_string_name *r_list) {
	List<StringName> constants;
	int i = 0;
	Variant::get_constants_for_type((Variant::Type)p_type, &constants);
	for (const List<StringName>::Element *E = constants.front(); E; E = E->next()) {
		memnew_placement_custom(&r_list[i++], StringName, StringName(E->get()));
	}
}

bool GDAPI godot_variant_has_constant(godot_variant_type p_type, const godot_string_name *p_constant) {
	return Variant::has_constant((Variant::Type)p_type, *((const StringName *)p_constant));
}

bool GDAPI godot_variant_has_constant_with_cstring(godot_variant_type p_type, const char *p_constant) {
	return Variant::has_constant((Variant::Type)p_type, StringName(p_constant));
}

godot_variant GDAPI godot_variant_get_constant_value(godot_variant_type p_type, const godot_string_name *p_constant) {
	Variant constant = Variant::get_constant_value((Variant::Type)p_type, *((const StringName *)p_constant));
	godot_variant ret;
	memnew_placement_custom(&ret, Variant, Variant(constant));
	return ret;
}

godot_variant GDAPI godot_variant_get_constant_value_with_cstring(godot_variant_type p_type, const char *p_constant) {
	Variant constant = Variant::get_constant_value((Variant::Type)p_type, StringName(p_constant));
	godot_variant ret;
	memnew_placement_custom(&ret, Variant, Variant(constant));
	return ret;
}

/// Utilities.
bool GDAPI godot_variant_has_utility_function(const godot_string_name *p_function) {
	return Variant::has_utility_function(*((const StringName *)p_function));
}

bool GDAPI godot_variant_has_utility_function_with_cstring(const char *p_function) {
	return Variant::has_utility_function(StringName(p_function));
}

void GDAPI godot_variant_call_utility_function(const godot_string_name *p_function, godot_variant *r_ret, const godot_variant **p_args, int p_argument_count, godot_variant_call_error *r_error) {
	const StringName *function = (const StringName *)p_function;
	Variant *ret = (Variant *)r_ret;
	const Variant **args = (const Variant **)p_args;
	Callable::CallError error;

	Variant::call_utility_function(*function, ret, args, p_argument_count, error);

	if (r_error) {
		r_error->error = (godot_variant_call_error_error)error.error;
		r_error->argument = error.argument;
		r_error->expected = (godot_variant_type)error.expected;
	}
}

void GDAPI godot_variant_call_utility_function_with_cstring(const char *p_function, godot_variant *r_ret, const godot_variant **p_args, int p_argument_count, godot_variant_call_error *r_error) {
	Variant *ret = (Variant *)r_ret;
	const Variant **args = (const Variant **)p_args;
	Callable::CallError error;

	Variant::call_utility_function(StringName(p_function), ret, args, p_argument_count, error);

	if (r_error) {
		r_error->error = (godot_variant_call_error_error)error.error;
		r_error->argument = error.argument;
		r_error->expected = (godot_variant_type)error.expected;
	}
}

godot_ptr_utility_function GDAPI godot_variant_get_ptr_utility_function(const godot_string_name *p_function) {
	return (godot_ptr_utility_function)Variant::get_ptr_utility_function(*((const StringName *)p_function));
}

godot_ptr_utility_function GDAPI godot_variant_get_ptr_utility_function_with_cstring(const char *p_function) {
	return (godot_ptr_utility_function)Variant::get_ptr_utility_function(StringName(p_function));
}

godot_validated_utility_function GDAPI godot_variant_get_validated_utility_function(const godot_string_name *p_function) {
	return (godot_validated_utility_function)Variant::get_validated_utility_function(*((const StringName *)p_function));
}

godot_validated_utility_function GDAPI godot_variant_get_validated_utility_function_with_cstring(const char *p_function) {
	return (godot_validated_utility_function)Variant::get_validated_utility_function(StringName(p_function));
}

godot_variant_utility_function_type GDAPI godot_variant_get_utility_function_type(const godot_string_name *p_function) {
	return (godot_variant_utility_function_type)Variant::get_utility_function_type(*((const StringName *)p_function));
}

godot_variant_utility_function_type GDAPI godot_variant_get_utility_function_type_with_cstring(const char *p_function) {
	return (godot_variant_utility_function_type)Variant::get_utility_function_type(StringName(p_function));
}

int GDAPI godot_variant_get_utility_function_argument_count(const godot_string_name *p_function) {
	return Variant::get_utility_function_argument_count(*((const StringName *)p_function));
}

int GDAPI godot_variant_get_utility_function_argument_count_with_cstring(const char *p_function) {
	return Variant::get_utility_function_argument_count(StringName(p_function));
}

godot_variant_type GDAPI godot_variant_get_utility_function_argument_type(const godot_string_name *p_function, int p_argument) {
	return (godot_variant_type)Variant::get_utility_function_argument_type(*((const StringName *)p_function), p_argument);
}

godot_variant_type GDAPI godot_variant_get_utility_function_argument_type_with_cstring(const char *p_function, int p_argument) {
	return (godot_variant_type)Variant::get_utility_function_argument_type(StringName(p_function), p_argument);
}

godot_string GDAPI godot_variant_get_utility_function_argument_name(const godot_string_name *p_function, int p_argument) {
	String argument_name = Variant::get_utility_function_argument_name(*((const StringName *)p_function), p_argument);
	godot_string ret;
	memnew_placement_custom(&ret, String, String(argument_name));
	return ret;
}

godot_string GDAPI godot_variant_get_utility_function_argument_name_with_cstring(const char *p_function, int p_argument) {
	String argument_name = Variant::get_utility_function_argument_name(StringName(p_function), p_argument);
	godot_string ret;
	memnew_placement_custom(&ret, String, String(argument_name));
	return ret;
}

bool GDAPI godot_variant_has_utility_function_return_value(const godot_string_name *p_function) {
	return Variant::has_utility_function_return_value(*((const StringName *)p_function));
}

bool GDAPI godot_variant_has_utility_function_return_value_with_cstring(const char *p_function) {
	return Variant::has_utility_function_return_value(StringName(p_function));
}

godot_variant_type GDAPI godot_variant_get_utility_function_return_type(const godot_string_name *p_function) {
	return (godot_variant_type)Variant::get_utility_function_return_type(*((const StringName *)p_function));
}

godot_variant_type GDAPI godot_variant_get_utility_function_return_type_with_cstring(const char *p_function) {
	return (godot_variant_type)Variant::get_utility_function_return_type(StringName(p_function));
}

bool GDAPI godot_variant_is_utility_function_vararg(const godot_string_name *p_function) {
	return Variant::is_utility_function_vararg(*((const StringName *)p_function));
}

bool GDAPI godot_variant_is_utility_function_vararg_with_cstring(const char *p_function) {
	return Variant::is_utility_function_vararg(StringName(p_function));
}

int GDAPI godot_variant_get_utility_function_count() {
	return Variant::get_utility_function_count();
}

void GDAPI godot_variant_get_utility_function_list(godot_string_name *r_functions) {
	List<StringName> functions;
	godot_string_name *func = r_functions;
	Variant::get_utility_function_list(&functions);

	for (const List<StringName>::Element *E = functions.front(); E; E = E->next()) {
		memnew_placement_custom(func++, StringName, StringName(E->get()));
	}
}

// Introspection.

godot_variant_type GDAPI godot_variant_get_type(const godot_variant *p_self) {
	const Variant *self = (const Variant *)p_self;
	return (godot_variant_type)self->get_type();
}

bool GDAPI godot_variant_has_method(const godot_variant *p_self, const godot_string_name *p_method) {
	const Variant *self = (const Variant *)p_self;
	const StringName *method = (const StringName *)p_method;
	return self->has_method(*method);
}

bool GDAPI godot_variant_has_member(godot_variant_type p_type, const godot_string_name *p_member) {
	return Variant::has_member((Variant::Type)p_type, *((const StringName *)p_member));
}

bool GDAPI godot_variant_has_key(const godot_variant *p_self, const godot_variant *p_key, bool *r_valid) {
	const Variant *self = (const Variant *)p_self;
	const Variant *key = (const Variant *)p_key;
	return self->has_key(*key, *r_valid);
}

godot_string GDAPI godot_variant_get_type_name(godot_variant_type p_type) {
	String name = Variant::get_type_name((Variant::Type)p_type);
	godot_string ret;
	memnew_placement_custom(&ret, String, String(name));
	return ret;
}

bool GDAPI godot_variant_can_convert(godot_variant_type p_from, godot_variant_type p_to) {
	return Variant::can_convert((Variant::Type)p_from, (Variant::Type)p_to);
}

bool GDAPI godot_variant_can_convert_strict(godot_variant_type p_from, godot_variant_type p_to) {
	return Variant::can_convert_strict((Variant::Type)p_from, (Variant::Type)p_to);
}

#ifdef __cplusplus
}
#endif
