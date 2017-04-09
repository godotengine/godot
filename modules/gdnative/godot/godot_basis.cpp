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

void GDAPI godot_basis_new(godot_basis *p_basis) {
	Basis *basis = (Basis *)p_basis;
	*basis = Basis();
}

void GDAPI godot_basis_new_with_euler_quat(godot_basis *p_basis, const godot_quat *p_euler) {
	Basis *basis = (Basis *)p_basis;
	Quat *euler = (Quat *)p_euler;
	*basis = Basis(*euler);
}

void GDAPI godot_basis_new_with_euler(godot_basis *p_basis, const godot_vector3 *p_euler) {
	Basis *basis = (Basis *)p_basis;
	Vector3 *euler = (Vector3 *)p_euler;
	*basis = Basis(*euler);
}

godot_quat GDAPI godot_basis_as_quat(const godot_basis *p_basis) {
	const Basis *basis = (const Basis *)p_basis;
	godot_quat quat;
	Quat *p_quat = (Quat *)&quat;
	*p_quat = basis->operator Quat();
	return quat;
}

godot_vector3 GDAPI godot_basis_get_euler(const godot_basis *p_basis) {
	const Basis *basis = (const Basis *)p_basis;
	godot_vector3 euler;
	Vector3 *p_euler = (Vector3 *)&euler;
	*p_euler = basis->get_euler();
	return euler;
}

/*
 * p_elements is a pointer to an array of 3 (!!) vector3
 */
void GDAPI godot_basis_get_elements(godot_basis *p_basis, godot_vector3 *p_elements) {
	Basis *basis = (Basis *)p_basis;
	Vector3 *elements = (Vector3 *)p_elements;
	elements[0] = basis->elements[0];
	elements[1] = basis->elements[1];
	elements[2] = basis->elements[2];
}

#ifdef __cplusplus
}
#endif
