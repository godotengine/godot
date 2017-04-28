/*************************************************************************/
/*  godot_basis.cpp                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
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
#include "godot_basis.h"

#include "math/matrix3.h"

#ifdef __cplusplus
extern "C" {
#endif

void _basis_api_anchor() {
}

void GDAPI godot_basis_new(godot_basis *p_v) {
	Basis *v = (Basis *)p_v;
	*v = Basis();
}

void GDAPI godot_basis_new_with_euler_quat(godot_basis *p_v, const godot_quat *p_euler) {
	Basis *v = (Basis *)p_v;
	Quat *euler = (Quat *)p_euler;
	*v = Basis(*euler);
}

void GDAPI godot_basis_new_with_euler(godot_basis *p_v, const godot_vector3 p_euler) {
	Basis *v = (Basis *)p_v;
	Vector3 *euler = (Vector3 *)&p_euler;
	*v = Basis(*euler);
}

void GDAPI godot_basis_new_with_axis_and_angle(godot_basis *p_v, const godot_vector3 p_axis, const godot_real p_phi) {
	Basis *v = (Basis *)p_v;
	const Vector3 *axis = (Vector3 *)&p_axis;
	*v = Basis(*axis, p_phi);
}

void GDAPI godot_basis_new_with_rows(godot_basis *p_v, const godot_vector3 p_row0, const godot_vector3 p_row1, const godot_vector3 p_row2) {
	Basis *v = (Basis *)p_v;
	const Vector3 *row0 = (Vector3 *)&p_row0;
	const Vector3 *row1 = (Vector3 *)&p_row1;
	const Vector3 *row2 = (Vector3 *)&p_row2;
	*v = Basis(*row0, *row1, *row2);
}

godot_quat GDAPI godot_basis_as_quat(const godot_basis *p_v) {
	const Basis *v = (const Basis *)p_v;
	godot_quat quat;
	Quat *p_quat = (Quat *)&quat;
	*p_quat = v->operator Quat();
	return quat;
}

/*
 * p_elements is a pointer to an array of 3 (!!) vector3
 */
void GDAPI godot_basis_get_elements(godot_basis *p_v, godot_vector3 *p_elements) {
	Basis *v = (Basis *)p_v;
	Vector3 *elements = (Vector3 *)p_elements;
	elements[0] = v->elements[0];
	elements[1] = v->elements[1];
	elements[2] = v->elements[2];
}

godot_vector3 GDAPI godot_basis_get_axis(const godot_basis *p_v, const godot_int p_axis) {
	godot_vector3 dest;
	Vector3 *d = (Vector3 *)&dest;
	const Basis *v = (Basis *)p_v;
	*d = v->get_axis(p_axis);
	return dest;
}

void GDAPI godot_basis_set_axis(godot_basis *p_v, const godot_int p_axis, const godot_vector3 p_value) {
	Basis *v = (Basis *)p_v;
	const Vector3 *value = (Vector3 *)&p_value;
	v->set_axis(p_axis, *value);
}

godot_vector3 GDAPI godot_basis_get_row(const godot_basis *p_v, const godot_int p_row) {
	godot_vector3 dest;
	Vector3 *d = (Vector3 *)&dest;
	const Basis *v = (Basis *)p_v;
	*d = v->get_row(p_row);
	return dest;
}

void GDAPI godot_basis_set_row(godot_basis *p_v, const godot_int p_row, const godot_vector3 p_value) {
	Basis *v = (Basis *)p_v;
	const Vector3 *value = (Vector3 *)&p_value;
	v->set_row(p_row, *value);
}

godot_real godot_basis_determinant(const godot_basis *p_v) {
	Basis *v = (Basis *)p_v;
	return v->determinant();
}

godot_vector3 godot_basis_get_euler(const godot_basis *p_v) {
	godot_vector3 dest;
	Vector3 *d = (Vector3 *)&dest;
	const Basis *v = (Basis *)p_v;
	*d = v->get_euler();
	return dest;
}

godot_int godot_basis_get_orthogonal_index(const godot_basis *p_v) {
	const Basis *v = (Basis *)p_v;
	return v->get_orthogonal_index();
}

godot_vector3 godot_basis_get_scale(const godot_basis *p_v) {
	godot_vector3 dest;
	Vector3 *d = (Vector3 *)&dest;
	const Basis *v = (Basis *)p_v;
	*d = v->get_scale();
	return dest;
}

void godot_basis_inverse(godot_basis *p_dest, const godot_basis *p_v) {
	Basis *d = (Basis *)p_dest;
	const Basis *v = (Basis *)p_v;
	*d = v->inverse();
}

void godot_basis_orthonormalized(godot_basis *p_dest, const godot_basis *p_v) {
	Basis *d = (Basis *)p_dest;
	const Basis *v = (Basis *)p_v;
	*d = v->orthonormalized();
}

void godot_basis_rotated(godot_basis *p_dest, const godot_basis *p_v, const godot_vector3 p_axis, const godot_real p_phi) {
	Basis *d = (Basis *)p_dest;
	const Basis *v = (Basis *)p_v;
	const Vector3 *axis = (Vector3 *)&p_axis;
	*d = v->rotated(*axis, p_phi);
}

void godot_basis_scaled(godot_basis *p_dest, const godot_basis *p_v, const godot_vector3 p_scale) {
	Basis *d = (Basis *)p_dest;
	const Basis *v = (Basis *)p_v;
	const Vector3 *scale = (Vector3 *)&p_scale;
	*d = v->scaled(*scale);
}

godot_real godot_basis_tdotx(const godot_basis *p_v, const godot_vector3 p_with) {
	const Basis *v = (Basis *)p_v;
	const Vector3 *with = (Vector3 *)&p_with;
	return v->tdotx(*with);
}

godot_real godot_basis_tdoty(const godot_basis *p_v, const godot_vector3 p_with) {
	const Basis *v = (Basis *)p_v;
	const Vector3 *with = (Vector3 *)&p_with;
	return v->tdoty(*with);
}

godot_real godot_basis_tdotz(const godot_basis *p_v, const godot_vector3 p_with) {
	const Basis *v = (Basis *)p_v;
	const Vector3 *with = (Vector3 *)&p_with;
	return v->tdotz(*with);
}

void godot_basis_transposed(godot_basis *p_dest, const godot_basis *p_v) {
	Basis *d = (Basis *)p_dest;
	const Basis *v = (Basis *)p_v;
	*d = v->transposed();
}

godot_vector3 godot_basis_xform(const godot_basis *p_v, const godot_vector3 p_vect) {
	godot_vector3 dest;
	Vector3 *d = (Vector3 *)&dest;
	const Basis *v = (Basis *)p_v;
	const Vector3 *vect = (Vector3 *)&p_vect;
	*d = v->xform(*vect);
	return dest;
}

godot_vector3 godot_basis_xform_inv(const godot_basis *p_v, const godot_vector3 p_vect) {
	godot_vector3 dest;
	Vector3 *d = (Vector3 *)&dest;
	const Basis *v = (Basis *)p_v;
	const Vector3 *vect = (Vector3 *)&p_vect;
	*d = v->xform_inv(*vect);
	return dest;
}

#ifdef __cplusplus
}
#endif
