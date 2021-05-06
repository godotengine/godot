/*************************************************************************/
/*  quat.cpp                                                             */
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

#include "gdnative/quat.h"

#include "core/math/quat.h"

static_assert(sizeof(godot_quat) == sizeof(Quat), "Quat size mismatch");

#ifdef __cplusplus
extern "C" {
#endif

void GDAPI godot_quat_new(godot_quat *p_self) {
	memnew_placement(p_self, Quat);
}

void GDAPI godot_quat_new_copy(godot_quat *r_dest, const godot_quat *p_src) {
	memnew_placement(r_dest, Quat(*(Quat *)p_src));
}

godot_real_t GDAPI *godot_quat_operator_index(godot_quat *p_self, godot_int p_index) {
	Quat *self = (Quat *)p_self;
	return (godot_real_t *)&self->operator[](p_index);
}

const godot_real_t GDAPI *godot_quat_operator_index_const(const godot_quat *p_self, godot_int p_index) {
	const Quat *self = (const Quat *)p_self;
	return (const godot_real_t *)&self->operator[](p_index);
}

#ifdef __cplusplus
}
#endif
