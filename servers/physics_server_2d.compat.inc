/**************************************************************************/
/*  physics_server_2d.compat.inc                                          */
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

#ifndef DISABLE_DEPRECATED

void PhysicsServer2D::_area_add_shape_bind_compat_99017(RID p_area, RID p_shape, const Transform2D &p_transform, bool p_disabled) {
	area_add_shape(p_area, p_shape, p_transform, p_disabled, false);
}

void PhysicsServer2D::_area_set_shape_transform_compat_99017(RID p_area, int p_shape_idx, const Transform2D &p_transform) {
	area_set_shape_transform(p_area, p_shape_idx, p_transform, false);
}

void PhysicsServer2D::_bind_compatibility_methods() {
	ClassDB::bind_compatibility_method(D_METHOD("area_add_shape", "area", "shape", "transform", "disabled"), &PhysicsServer2D::_area_add_shape_bind_compat_99017, DEFVAL(Transform2D()), DEFVAL(false));
	ClassDB::bind_compatibility_method(D_METHOD("area_set_shape_transform", "area", "shape_idx", "transform"), &PhysicsServer2D::_area_set_shape_transform_compat_99017);
}

#endif
