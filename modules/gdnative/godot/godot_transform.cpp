/*************************************************************************/
/*  godot_transform.cpp                                                  */
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
#include "godot_transform.h"

#include "math/transform.h"

#ifdef __cplusplus
extern "C" {
#endif

void _transform_api_anchor() {
}

void GDAPI godot_transform_new(godot_transform *p_trans) {
	Transform *trans = (Transform *)p_trans;
	*trans = Transform();
}

void GDAPI godot_transform_new_with_basis(godot_transform *p_trans, const godot_basis *p_basis) {
	Transform *trans = (Transform *)p_trans;
	const Basis *basis = (const Basis *)p_basis;
	*trans = Transform(*basis);
}

void GDAPI godot_transform_new_with_basis_origin(godot_transform *p_trans, const godot_basis *p_basis, const godot_vector3 *p_origin) {
	Transform *trans = (Transform *)p_trans;
	const Basis *basis = (const Basis *)p_basis;
	const Vector3 *origin = (const Vector3 *)p_origin;
	*trans = Transform(*basis, *origin);
}

godot_basis GDAPI *godot_transform_get_basis(godot_transform *p_trans) {
	Transform *trans = (Transform *)p_trans;
	return (godot_basis *)&trans->basis;
}

godot_vector3 GDAPI *godot_transform_get_origin(godot_transform *p_trans) {
	Transform *trans = (Transform *)p_trans;
	return (godot_vector3 *)&trans->origin;
}

#ifdef __cplusplus
}
#endif
