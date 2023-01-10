/**************************************************************************/
/*  quat.cpp                                                              */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "gdnative/quat.h"

#include "core/math/quat.h"
#include "core/variant.h"

#ifdef __cplusplus
extern "C" {
#endif

static_assert(sizeof(godot_quat) == sizeof(Quat), "Quat size mismatch");

void GDAPI godot_quat_new(godot_quat *r_dest, const godot_real p_x, const godot_real p_y, const godot_real p_z, const godot_real p_w) {
	Quat *dest = (Quat *)r_dest;
	*dest = Quat(p_x, p_y, p_z, p_w);
}

void GDAPI godot_quat_new_with_axis_angle(godot_quat *r_dest, const godot_vector3 *p_axis, const godot_real p_angle) {
	const Vector3 *axis = (const Vector3 *)p_axis;
	Quat *dest = (Quat *)r_dest;
	*dest = Quat(*axis, p_angle);
}

void GDAPI godot_quat_new_with_basis(godot_quat *r_dest, const godot_basis *p_basis) {
	const Basis *basis = (const Basis *)p_basis;
	Quat *dest = (Quat *)r_dest;
	*dest = Quat(*basis);
}

void GDAPI godot_quat_new_with_euler(godot_quat *r_dest, const godot_vector3 *p_euler) {
	const Vector3 *euler = (const Vector3 *)p_euler;
	Quat *dest = (Quat *)r_dest;
	*dest = Quat(*euler);
}

godot_real GDAPI godot_quat_get_x(const godot_quat *p_self) {
	const Quat *self = (const Quat *)p_self;
	return self->x;
}

void GDAPI godot_quat_set_x(godot_quat *p_self, const godot_real val) {
	Quat *self = (Quat *)p_self;
	self->x = val;
}

godot_real GDAPI godot_quat_get_y(const godot_quat *p_self) {
	const Quat *self = (const Quat *)p_self;
	return self->y;
}

void GDAPI godot_quat_set_y(godot_quat *p_self, const godot_real val) {
	Quat *self = (Quat *)p_self;
	self->y = val;
}

godot_real GDAPI godot_quat_get_z(const godot_quat *p_self) {
	const Quat *self = (const Quat *)p_self;
	return self->z;
}

void GDAPI godot_quat_set_z(godot_quat *p_self, const godot_real val) {
	Quat *self = (Quat *)p_self;
	self->z = val;
}

godot_real GDAPI godot_quat_get_w(const godot_quat *p_self) {
	const Quat *self = (const Quat *)p_self;
	return self->w;
}

void GDAPI godot_quat_set_w(godot_quat *p_self, const godot_real val) {
	Quat *self = (Quat *)p_self;
	self->w = val;
}

godot_string GDAPI godot_quat_as_string(const godot_quat *p_self) {
	godot_string ret;
	const Quat *self = (const Quat *)p_self;
	memnew_placement(&ret, String(*self));
	return ret;
}

godot_real GDAPI godot_quat_length(const godot_quat *p_self) {
	const Quat *self = (const Quat *)p_self;
	return self->length();
}

godot_real GDAPI godot_quat_length_squared(const godot_quat *p_self) {
	const Quat *self = (const Quat *)p_self;
	return self->length_squared();
}

godot_quat GDAPI godot_quat_normalized(const godot_quat *p_self) {
	godot_quat dest;
	const Quat *self = (const Quat *)p_self;
	*((Quat *)&dest) = self->normalized();
	return dest;
}

godot_bool GDAPI godot_quat_is_normalized(const godot_quat *p_self) {
	const Quat *self = (const Quat *)p_self;
	return self->is_normalized();
}

godot_quat GDAPI godot_quat_inverse(const godot_quat *p_self) {
	godot_quat dest;
	const Quat *self = (const Quat *)p_self;
	*((Quat *)&dest) = self->inverse();
	return dest;
}

godot_real GDAPI godot_quat_dot(const godot_quat *p_self, const godot_quat *p_b) {
	const Quat *self = (const Quat *)p_self;
	const Quat *b = (const Quat *)p_b;
	return self->dot(*b);
}

godot_vector3 GDAPI godot_quat_xform(const godot_quat *p_self, const godot_vector3 *p_v) {
	godot_vector3 dest;
	const Quat *self = (const Quat *)p_self;
	const Vector3 *v = (const Vector3 *)p_v;
	*((Vector3 *)&dest) = self->xform(*v);
	return dest;
}

godot_quat GDAPI godot_quat_slerp(const godot_quat *p_self, const godot_quat *p_b, const godot_real p_t) {
	godot_quat dest;
	const Quat *self = (const Quat *)p_self;
	const Quat *b = (const Quat *)p_b;
	*((Quat *)&dest) = self->slerp(*b, p_t);
	return dest;
}

godot_quat GDAPI godot_quat_slerpni(const godot_quat *p_self, const godot_quat *p_b, const godot_real p_t) {
	godot_quat dest;
	const Quat *self = (const Quat *)p_self;
	const Quat *b = (const Quat *)p_b;
	*((Quat *)&dest) = self->slerpni(*b, p_t);
	return dest;
}

godot_quat GDAPI godot_quat_cubic_slerp(const godot_quat *p_self, const godot_quat *p_b, const godot_quat *p_pre_a, const godot_quat *p_post_b, const godot_real p_t) {
	godot_quat dest;
	const Quat *self = (const Quat *)p_self;
	const Quat *b = (const Quat *)p_b;
	const Quat *pre_a = (const Quat *)p_pre_a;
	const Quat *post_b = (const Quat *)p_post_b;
	*((Quat *)&dest) = self->cubic_slerp(*b, *pre_a, *post_b, p_t);
	return dest;
}

godot_quat GDAPI godot_quat_operator_multiply(const godot_quat *p_self, const godot_real p_b) {
	godot_quat raw_dest;
	Quat *dest = (Quat *)&raw_dest;
	const Quat *self = (const Quat *)p_self;
	*dest = *self * p_b;
	return raw_dest;
}

godot_quat GDAPI godot_quat_operator_add(const godot_quat *p_self, const godot_quat *p_b) {
	godot_quat raw_dest;
	Quat *dest = (Quat *)&raw_dest;
	const Quat *self = (const Quat *)p_self;
	const Quat *b = (const Quat *)p_b;
	*dest = *self + *b;
	return raw_dest;
}

godot_quat GDAPI godot_quat_operator_subtract(const godot_quat *p_self, const godot_quat *p_b) {
	godot_quat raw_dest;
	Quat *dest = (Quat *)&raw_dest;
	const Quat *self = (const Quat *)p_self;
	const Quat *b = (const Quat *)p_b;
	*dest = *self - *b;
	return raw_dest;
}

godot_quat GDAPI godot_quat_operator_divide(const godot_quat *p_self, const godot_real p_b) {
	godot_quat raw_dest;
	Quat *dest = (Quat *)&raw_dest;
	const Quat *self = (const Quat *)p_self;
	*dest = *self / p_b;
	return raw_dest;
}

godot_bool GDAPI godot_quat_operator_equal(const godot_quat *p_self, const godot_quat *p_b) {
	const Quat *self = (const Quat *)p_self;
	const Quat *b = (const Quat *)p_b;
	return *self == *b;
}

godot_quat GDAPI godot_quat_operator_neg(const godot_quat *p_self) {
	godot_quat raw_dest;
	Quat *dest = (Quat *)&raw_dest;
	const Quat *self = (const Quat *)p_self;
	*dest = -(*self);
	return raw_dest;
}

void GDAPI godot_quat_set_axis_angle(godot_quat *p_self, const godot_vector3 *p_axis, const godot_real p_angle) {
	Quat *self = (Quat *)p_self;
	const Vector3 *axis = (const Vector3 *)p_axis;
	self->set_axis_angle(*axis, p_angle);
}

#ifdef __cplusplus
}
#endif
