/**************************************************************************/
/*  collision_object_2d.compat.inc                                        */
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

void CollisionObject2D::_shape_owner_add_shape_compat_99017(uint32_t p_owner, const Ref<Shape2D> &p_shape) {
	shape_owner_add_shape(p_owner, p_shape, false);
}

void CollisionObject2D::_shape_owner_set_transform_compat_99017(uint32_t p_owner, const Transform2D &p_transform) {
	shape_owner_set_transform(p_owner, p_transform, false);
}

void CollisionObject2D::_bind_compatibility_methods() {
	ClassDB::bind_compatibility_method(D_METHOD("shape_owner_add_shape", "owner_id", "shape"), &CollisionObject2D::_shape_owner_add_shape_compat_99017);
	ClassDB::bind_compatibility_method(D_METHOD("shape_owner_set_transform", "owner_id", "transform"), &CollisionObject2D::_shape_owner_set_transform_compat_99017);
}

#endif
