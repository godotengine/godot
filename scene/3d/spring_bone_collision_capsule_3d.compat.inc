/**************************************************************************/
/*  spring_bone_collision_sphere_3d.cpp                                   */
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

#include "spring_bone_collision_capsule_3d.h"

#include "core/object/class_db.h"

void SpringBoneCollisionCapsule3D::_bind_compatibility_methods() {
	ClassDB::bind_method(D_METHOD("set_inside", "inside"), &SpringBoneCollisionCapsule3D::_set_inside_bind_compat_122560);
	ClassDB::bind_method(D_METHOD("get_inside"), &SpringBoneCollisionCapsule3D::_is_inside_bind_compat_122560);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "inside"), "set_inside", "is_inside");
}
void SpringBoneCollisionCapsule3D::_set_inside_bind_compat_122560(bool p_enabled) {
	return (set_collide_mode(p_enabled ? SpringBoneCollision3D::COLLIDE_MODE_INSIDE : SpringBoneCollision3D::COLLIDE_MODE_JOINT));
}
bool SpringBoneCollisionCapsule3D::_is_inside_bind_compat_122560() const {
	return (get_collide_mode() == SpringBoneCollision3D::COLLIDE_MODE_INSIDE);
}

#endif // DISABLE_DEPRECATED
