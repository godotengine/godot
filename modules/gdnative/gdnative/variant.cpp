/*************************************************************************/
/*  variant.cpp                                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "core/reference.h"
#include "core/variant.h"

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

// Constructors

godot_variant_type GDAPI godot_variant_get_type(const godot_variant *p_self) {
	const Variant *self = (const Variant *)p_self;
	return (godot_variant_type)self->get_type();
}

void GDAPI godot_variant_new_copy(godot_variant *p_dest, const godot_variant *p_src) {
	Variant *dest = (Variant *)p_dest;
	Variant *src = (Variant *)p_src;
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

void GDAPI godot_variant_new_uint(godot_variant *r_dest, const uint64_t p_i) {
	Variant *dest = (Variant *)r_dest;
	memnew_placement_custom(dest, Variant, Variant(p_i));
}

void GDAPI godot_variant_new_int(godot_variant *r_dest, const int64_t p_i) {
	Variant *dest = (Variant *)r_dest;
	memnew_placement_custom(dest, Variant, Variant(p_i));
}

void GDAPI godot_variant_new_real(godot_variant *r_dest, const double p_r) {
	Variant *dest = (Variant *)r_dest;
	memnew_placement_custom(dest, Variant, Variant(p_r));
}

void GDAPI godot_variant_new_string(godot_variant *r_dest, const godot_string *p_s) {
	Variant *dest = (Variant *)r_dest;
	String *s = (String *)p_s;
	memnew_placement_custom(dest, Variant, Variant(*s));
}

void GDAPI godot_variant_new_string_name(godot_variant *r_dest, const godot_string_name *p_s) {
	Variant *dest = (Variant *)r_dest;
	StringName *s = (StringName *)p_s;
	memnew_placement_custom(dest, Variant, Variant(*s));
}

void GDAPI godot_variant_new_vector2(godot_variant *r_dest, const godot_vector2 *p_v2) {
	Variant *dest = (Variant *)r_dest;
	Vector2 *v2 = (Vector2 *)p_v2;
	memnew_placement_custom(dest, Variant, Variant(*v2));
}

void GDAPI godot_variant_new_vector2i(godot_variant *r_dest, const godot_vector2i *p_v2) {
	Variant *dest = (Variant *)r_dest;
	Vector2i *v2 = (Vector2i *)p_v2;
	memnew_placement_custom(dest, Variant, Variant(*v2));
}

void GDAPI godot_variant_new_rect2(godot_variant *r_dest, const godot_rect2 *p_rect2) {
	Variant *dest = (Variant *)r_dest;
	Rect2 *rect2 = (Rect2 *)p_rect2;
	memnew_placement_custom(dest, Variant, Variant(*rect2));
}

void GDAPI godot_variant_new_rect2i(godot_variant *r_dest, const godot_rect2i *p_rect2) {
	Variant *dest = (Variant *)r_dest;
	Rect2i *rect2 = (Rect2i *)p_rect2;
	memnew_placement_custom(dest, Variant, Variant(*rect2));
}

void GDAPI godot_variant_new_vector3(godot_variant *r_dest, const godot_vector3 *p_v3) {
	Variant *dest = (Variant *)r_dest;
	Vector3 *v3 = (Vector3 *)p_v3;
	memnew_placement_custom(dest, Variant, Variant(*v3));
}

void GDAPI godot_variant_new_vector3i(godot_variant *r_dest, const godot_vector3i *p_v3) {
	Variant *dest = (Variant *)r_dest;
	Vector3i *v3 = (Vector3i *)p_v3;
	memnew_placement_custom(dest, Variant, Variant(*v3));
}

void GDAPI godot_variant_new_transform2d(godot_variant *r_dest, const godot_transform2d *p_t2d) {
	Variant *dest = (Variant *)r_dest;
	Transform2D *t2d = (Transform2D *)p_t2d;
	memnew_placement_custom(dest, Variant, Variant(*t2d));
}

void GDAPI godot_variant_new_plane(godot_variant *r_dest, const godot_plane *p_plane) {
	Variant *dest = (Variant *)r_dest;
	Plane *plane = (Plane *)p_plane;
	memnew_placement_custom(dest, Variant, Variant(*plane));
}

void GDAPI godot_variant_new_quat(godot_variant *r_dest, const godot_quat *p_quat) {
	Variant *dest = (Variant *)r_dest;
	Quat *quat = (Quat *)p_quat;
	memnew_placement_custom(dest, Variant, Variant(*quat));
}

void GDAPI godot_variant_new_aabb(godot_variant *r_dest, const godot_aabb *p_aabb) {
	Variant *dest = (Variant *)r_dest;
	AABB *aabb = (AABB *)p_aabb;
	memnew_placement_custom(dest, Variant, Variant(*aabb));
}

void GDAPI godot_variant_new_basis(godot_variant *r_dest, const godot_basis *p_basis) {
	Variant *dest = (Variant *)r_dest;
	Basis *basis = (Basis *)p_basis;
	memnew_placement_custom(dest, Variant, Variant(*basis));
}

void GDAPI godot_variant_new_transform(godot_variant *r_dest, const godot_transform *p_trans) {
	Variant *dest = (Variant *)r_dest;
	Transform *trans = (Transform *)p_trans;
	memnew_placement_custom(dest, Variant, Variant(*trans));
}

void GDAPI godot_variant_new_color(godot_variant *r_dest, const godot_color *p_color) {
	Variant *dest = (Variant *)r_dest;
	Color *color = (Color *)p_color;
	memnew_placement_custom(dest, Variant, Variant(*color));
}

void GDAPI godot_variant_new_node_path(godot_variant *r_dest, const godot_node_path *p_np) {
	Variant *dest = (Variant *)r_dest;
	NodePath *np = (NodePath *)p_np;
	memnew_placement_custom(dest, Variant, Variant(*np));
}

void GDAPI godot_variant_new_rid(godot_variant *r_dest, const godot_rid *p_rid) {
	Variant *dest = (Variant *)r_dest;
	RID *rid = (RID *)p_rid;
	memnew_placement_custom(dest, Variant, Variant(*rid));
}

void GDAPI godot_variant_new_callable(godot_variant *r_dest, const godot_callable *p_cb) {
	Variant *dest = (Variant *)r_dest;
	Callable *cb = (Callable *)p_cb;
	memnew_placement_custom(dest, Variant, Variant(*cb));
}

void GDAPI godot_variant_new_signal(godot_variant *r_dest, const godot_signal *p_signal) {
	Variant *dest = (Variant *)r_dest;
	Signal *signal = (Signal *)p_signal;
	memnew_placement_custom(dest, Variant, Variant(*signal));
}

void GDAPI godot_variant_new_object(godot_variant *r_dest, const godot_object *p_obj) {
	Variant *dest = (Variant *)r_dest;
	Object *obj = (Object *)p_obj;
	Reference *reference = Object::cast_to<Reference>(obj);
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
	Dictionary *dict = (Dictionary *)p_dict;
	memnew_placement_custom(dest, Variant, Variant(*dict));
}

void GDAPI godot_variant_new_array(godot_variant *r_dest, const godot_array *p_arr) {
	Variant *dest = (Variant *)r_dest;
	Array *arr = (Array *)p_arr;
	memnew_placement_custom(dest, Variant, Variant(*arr));
}

void GDAPI godot_variant_new_packed_byte_array(godot_variant *r_dest, const godot_packed_byte_array *p_pba) {
	Variant *dest = (Variant *)r_dest;
	PackedByteArray *pba = (PackedByteArray *)p_pba;
	memnew_placement_custom(dest, Variant, Variant(*pba));
}

void GDAPI godot_variant_new_packed_int32_array(godot_variant *r_dest, const godot_packed_int32_array *p_pia) {
	Variant *dest = (Variant *)r_dest;
	PackedInt32Array *pia = (PackedInt32Array *)p_pia;
	memnew_placement_custom(dest, Variant, Variant(*pia));
}

void GDAPI godot_variant_new_packed_int64_array(godot_variant *r_dest, const godot_packed_int64_array *p_pia) {
	Variant *dest = (Variant *)r_dest;
	PackedInt64Array *pia = (PackedInt64Array *)p_pia;
	memnew_placement_custom(dest, Variant, Variant(*pia));
}

void GDAPI godot_variant_new_packed_float32_array(godot_variant *r_dest, const godot_packed_float32_array *p_pra) {
	Variant *dest = (Variant *)r_dest;
	PackedFloat32Array *pra = (PackedFloat32Array *)p_pra;
	memnew_placement_custom(dest, Variant, Variant(*pra));
}

void GDAPI godot_variant_new_packed_float64_array(godot_variant *r_dest, const godot_packed_float64_array *p_pra) {
	Variant *dest = (Variant *)r_dest;
	PackedFloat64Array *pra = (PackedFloat64Array *)p_pra;
	memnew_placement_custom(dest, Variant, Variant(*pra));
}

void GDAPI godot_variant_new_packed_string_array(godot_variant *r_dest, const godot_packed_string_array *p_psa) {
	Variant *dest = (Variant *)r_dest;
	PackedStringArray *psa = (PackedStringArray *)p_psa;
	memnew_placement_custom(dest, Variant, Variant(*psa));
}

void GDAPI godot_variant_new_packed_vector2_array(godot_variant *r_dest, const godot_packed_vector2_array *p_pv2a) {
	Variant *dest = (Variant *)r_dest;
	PackedVector2Array *pv2a = (PackedVector2Array *)p_pv2a;
	memnew_placement_custom(dest, Variant, Variant(*pv2a));
}

void GDAPI godot_variant_new_packed_vector3_array(godot_variant *r_dest, const godot_packed_vector3_array *p_pv3a) {
	Variant *dest = (Variant *)r_dest;
	PackedVector3Array *pv3a = (PackedVector3Array *)p_pv3a;
	memnew_placement_custom(dest, Variant, Variant(*pv3a));
}

void GDAPI godot_variant_new_packed_color_array(godot_variant *r_dest, const godot_packed_color_array *p_pca) {
	Variant *dest = (Variant *)r_dest;
	PackedColorArray *pca = (PackedColorArray *)p_pca;
	memnew_placement_custom(dest, Variant, Variant(*pca));
}

godot_bool GDAPI godot_variant_as_bool(const godot_variant *p_self) {
	const Variant *self = (const Variant *)p_self;
	return self->operator bool();
}

uint64_t GDAPI godot_variant_as_uint(const godot_variant *p_self) {
	const Variant *self = (const Variant *)p_self;
	return self->operator uint64_t();
}

int64_t GDAPI godot_variant_as_int(const godot_variant *p_self) {
	const Variant *self = (const Variant *)p_self;
	return self->operator int64_t();
}

double GDAPI godot_variant_as_real(const godot_variant *p_self) {
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

godot_variant GDAPI godot_variant_call(godot_variant *p_self, const godot_string *p_method, const godot_variant **p_args, const godot_int p_argcount, godot_variant_call_error *r_error) {
	Variant *self = (Variant *)p_self;
	String *method = (String *)p_method;
	const Variant **args = (const Variant **)p_args;
	godot_variant raw_dest;
	Variant *dest = (Variant *)&raw_dest;
	Callable::CallError error;
	memnew_placement_custom(dest, Variant, Variant(self->call(*method, args, p_argcount, error)));
	if (r_error) {
		r_error->error = (godot_variant_call_error_error)error.error;
		r_error->argument = error.argument;
		r_error->expected = (godot_variant_type)error.expected;
	}
	return raw_dest;
}

godot_bool GDAPI godot_variant_has_method(const godot_variant *p_self, const godot_string *p_method) {
	const Variant *self = (const Variant *)p_self;
	const String *method = (const String *)p_method;
	return self->has_method(*method);
}

godot_bool GDAPI godot_variant_operator_equal(const godot_variant *p_self, const godot_variant *p_other) {
	const Variant *self = (const Variant *)p_self;
	const Variant *other = (const Variant *)p_other;
	return self->operator==(*other);
}

godot_bool GDAPI godot_variant_operator_less(const godot_variant *p_self, const godot_variant *p_other) {
	const Variant *self = (const Variant *)p_self;
	const Variant *other = (const Variant *)p_other;
	return self->operator<(*other);
}

uint32_t GDAPI godot_variant_hash(const godot_variant *p_self) {
	const Variant *self = (const Variant *)p_self;
	return self->hash();
}

godot_bool GDAPI godot_variant_hash_compare(const godot_variant *p_self, const godot_variant *p_other) {
	const Variant *self = (const Variant *)p_self;
	const Variant *other = (const Variant *)p_other;
	return self->hash_compare(*other);
}

godot_bool GDAPI godot_variant_booleanize(const godot_variant *p_self) {
	const Variant *self = (const Variant *)p_self;
	return self->booleanize();
}

void GDAPI godot_variant_destroy(godot_variant *p_self) {
	Variant *self = (Variant *)p_self;
	self->~Variant();
}

// GDNative core 1.1

godot_string GDAPI godot_variant_get_operator_name(godot_variant_operator p_op) {
	Variant::Operator op = (Variant::Operator)p_op;
	godot_string raw_dest;
	String *dest = (String *)&raw_dest;
	memnew_placement(dest, String(Variant::get_operator_name(op))); // operator = is overloaded by String
	return raw_dest;
}

void GDAPI godot_variant_evaluate(godot_variant_operator p_op, const godot_variant *p_a, const godot_variant *p_b, godot_variant *r_ret, godot_bool *r_valid) {
	Variant::Operator op = (Variant::Operator)p_op;
	const Variant *a = (const Variant *)p_a;
	const Variant *b = (const Variant *)p_b;
	Variant *ret = (Variant *)r_ret;
	Variant::evaluate(op, *a, *b, *ret, *r_valid);
}

#ifdef __cplusplus
}
#endif
