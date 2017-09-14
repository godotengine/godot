/*************************************************************************/
/*  variant.cpp                                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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

void _variant_api_anchor() {}

#define memnew_placement_custom(m_placement, m_class, m_constr) _post_initialize(new (m_placement, sizeof(m_class), "") m_constr)

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

void GDAPI godot_variant_new_vector2(godot_variant *r_dest, const godot_vector2 *p_v2) {
	Variant *dest = (Variant *)r_dest;
	Vector2 *v2 = (Vector2 *)p_v2;
	memnew_placement_custom(dest, Variant, Variant(*v2));
}

void GDAPI godot_variant_new_rect2(godot_variant *r_dest, const godot_rect2 *p_rect2) {
	Variant *dest = (Variant *)r_dest;
	Rect2 *rect2 = (Rect2 *)p_rect2;
	memnew_placement_custom(dest, Variant, Variant(*rect2));
}

void GDAPI godot_variant_new_vector3(godot_variant *r_dest, const godot_vector3 *p_v3) {
	Variant *dest = (Variant *)r_dest;
	Vector3 *v3 = (Vector3 *)p_v3;
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

void GDAPI godot_variant_new_rect3(godot_variant *r_dest, const godot_rect3 *p_rect3) {
	Variant *dest = (Variant *)r_dest;
	Rect3 *rect3 = (Rect3 *)p_rect3;
	memnew_placement_custom(dest, Variant, Variant(*rect3));
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

void GDAPI godot_variant_new_object(godot_variant *r_dest, const godot_object *p_obj) {
	Variant *dest = (Variant *)r_dest;
	Object *obj = (Object *)p_obj;
	Reference *reference = Object::cast_to<Reference>(obj);
	REF ref;
	if (reference) {
		ref = REF(reference);
	}
	if (!ref.is_null()) {
		memnew_placement_custom(dest, Variant, Variant(ref.get_ref_ptr()));
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

void GDAPI godot_variant_new_pool_byte_array(godot_variant *r_dest, const godot_pool_byte_array *p_pba) {
	Variant *dest = (Variant *)r_dest;
	PoolByteArray *pba = (PoolByteArray *)p_pba;
	memnew_placement_custom(dest, Variant, Variant(*pba));
}

void GDAPI godot_variant_new_pool_int_array(godot_variant *r_dest, const godot_pool_int_array *p_pia) {
	Variant *dest = (Variant *)r_dest;
	PoolIntArray *pia = (PoolIntArray *)p_pia;
	memnew_placement_custom(dest, Variant, Variant(*pia));
}

void GDAPI godot_variant_new_pool_real_array(godot_variant *r_dest, const godot_pool_real_array *p_pra) {
	Variant *dest = (Variant *)r_dest;
	PoolRealArray *pra = (PoolRealArray *)p_pra;
	memnew_placement_custom(dest, Variant, Variant(*pra));
}

void GDAPI godot_variant_new_pool_string_array(godot_variant *r_dest, const godot_pool_string_array *p_psa) {
	Variant *dest = (Variant *)r_dest;
	PoolStringArray *psa = (PoolStringArray *)p_psa;
	memnew_placement_custom(dest, Variant, Variant(*psa));
}

void GDAPI godot_variant_new_pool_vector2_array(godot_variant *r_dest, const godot_pool_vector2_array *p_pv2a) {
	Variant *dest = (Variant *)r_dest;
	PoolVector2Array *pv2a = (PoolVector2Array *)p_pv2a;
	memnew_placement_custom(dest, Variant, Variant(*pv2a));
}

void GDAPI godot_variant_new_pool_vector3_array(godot_variant *r_dest, const godot_pool_vector3_array *p_pv3a) {
	Variant *dest = (Variant *)r_dest;
	PoolVector3Array *pv3a = (PoolVector3Array *)p_pv3a;
	memnew_placement_custom(dest, Variant, Variant(*pv3a));
}

void GDAPI godot_variant_new_pool_color_array(godot_variant *r_dest, const godot_pool_color_array *p_pca) {
	Variant *dest = (Variant *)r_dest;
	PoolColorArray *pca = (PoolColorArray *)p_pca;
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

godot_vector2 GDAPI godot_variant_as_vector2(const godot_variant *p_self) {
	godot_vector2 raw_dest;
	const Variant *self = (const Variant *)p_self;
	Vector2 *dest = (Vector2 *)&raw_dest;
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

godot_vector3 GDAPI godot_variant_as_vector3(const godot_variant *p_self) {
	godot_vector3 raw_dest;
	const Variant *self = (const Variant *)p_self;
	Vector3 *dest = (Vector3 *)&raw_dest;
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

godot_rect3 GDAPI godot_variant_as_rect3(const godot_variant *p_self) {
	godot_rect3 raw_dest;
	const Variant *self = (const Variant *)p_self;
	Rect3 *dest = (Rect3 *)&raw_dest;
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

godot_pool_byte_array GDAPI godot_variant_as_pool_byte_array(const godot_variant *p_self) {
	godot_pool_byte_array raw_dest;
	const Variant *self = (const Variant *)p_self;
	PoolByteArray *dest = (PoolByteArray *)&raw_dest;
	memnew_placement(dest, PoolByteArray(self->operator PoolByteArray())); // operator = is overloaded by PoolByteArray
	*dest = *self;
	return raw_dest;
}

godot_pool_int_array GDAPI godot_variant_as_pool_int_array(const godot_variant *p_self) {
	godot_pool_int_array raw_dest;
	const Variant *self = (const Variant *)p_self;
	PoolIntArray *dest = (PoolIntArray *)&raw_dest;
	memnew_placement(dest, PoolIntArray(self->operator PoolIntArray())); // operator = is overloaded by PoolIntArray
	*dest = *self;
	return raw_dest;
}

godot_pool_real_array GDAPI godot_variant_as_pool_real_array(const godot_variant *p_self) {
	godot_pool_real_array raw_dest;
	const Variant *self = (const Variant *)p_self;
	PoolRealArray *dest = (PoolRealArray *)&raw_dest;
	memnew_placement(dest, PoolRealArray(self->operator PoolRealArray())); // operator = is overloaded by PoolRealArray
	*dest = *self;
	return raw_dest;
}

godot_pool_string_array GDAPI godot_variant_as_pool_string_array(const godot_variant *p_self) {
	godot_pool_string_array raw_dest;
	const Variant *self = (const Variant *)p_self;
	PoolStringArray *dest = (PoolStringArray *)&raw_dest;
	memnew_placement(dest, PoolStringArray(self->operator PoolStringArray())); // operator = is overloaded by PoolStringArray
	*dest = *self;
	return raw_dest;
}

godot_pool_vector2_array GDAPI godot_variant_as_pool_vector2_array(const godot_variant *p_self) {
	godot_pool_vector2_array raw_dest;
	const Variant *self = (const Variant *)p_self;
	PoolVector2Array *dest = (PoolVector2Array *)&raw_dest;
	memnew_placement(dest, PoolVector2Array(self->operator PoolVector2Array())); // operator = is overloaded by PoolVector2Array
	*dest = *self;
	return raw_dest;
}

godot_pool_vector3_array GDAPI godot_variant_as_pool_vector3_array(const godot_variant *p_self) {
	godot_pool_vector3_array raw_dest;
	const Variant *self = (const Variant *)p_self;
	PoolVector3Array *dest = (PoolVector3Array *)&raw_dest;
	memnew_placement(dest, PoolVector3Array(self->operator PoolVector3Array())); // operator = is overloaded by PoolVector3Array
	*dest = *self;
	return raw_dest;
}

godot_pool_color_array GDAPI godot_variant_as_pool_color_array(const godot_variant *p_self) {
	godot_pool_color_array raw_dest;
	const Variant *self = (const Variant *)p_self;
	PoolColorArray *dest = (PoolColorArray *)&raw_dest;
	memnew_placement(dest, PoolColorArray(self->operator PoolColorArray())); // operator = is overloaded by PoolColorArray
	*dest = *self;
	return raw_dest;
}

godot_variant GDAPI godot_variant_call(godot_variant *p_self, const godot_string *p_method, const godot_variant **p_args, const godot_int p_argcount, godot_variant_call_error *r_error) {
	Variant *self = (Variant *)p_self;
	String *method = (String *)p_method;
	const Variant **args = (const Variant **)p_args;
	godot_variant raw_dest;
	Variant *dest = (Variant *)&raw_dest;
	Variant::CallError error;
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

godot_bool GDAPI godot_variant_hash_compare(const godot_variant *p_self, const godot_variant *p_other) {
	const Variant *self = (const Variant *)p_self;
	const Variant *other = (const Variant *)p_other;
	return self->hash_compare(*other);
}

godot_bool GDAPI godot_variant_booleanize(const godot_variant *p_self, godot_bool *r_valid) {
	const Variant *self = (const Variant *)p_self;
	bool &valid = *r_valid;
	return self->booleanize(valid);
}

void GDAPI godot_variant_destroy(godot_variant *p_self) {
	Variant *self = (Variant *)p_self;
	self->~Variant();
}

#ifdef __cplusplus
}
#endif
