/**************************************************************************/
/*  physics_server_3d.compat.inc                                          */
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

#include "physics_server_3d.h"

void PhysicsServer3D::_bind_compatibility_methods() {
	ClassDB::bind_compatibility_method(D_METHOD("generic_6dof_joint_set_param", "joint", "axis", "param", "value"), &PhysicsServer3D::_generic_6dof_joint_set_param_bind_compat_99341);
	ClassDB::bind_compatibility_method(D_METHOD("generic_6dof_joint_get_param", "joint", "axis", "param"), &PhysicsServer3D::_generic_6dof_joint_get_param_bind_compat_99341);
	ClassDB::bind_compatibility_method(D_METHOD("generic_6dof_joint_set_flag", "joint", "axis", "flag", "enable"), &PhysicsServer3D::_generic_6dof_joint_set_flag_bind_compat_99341);
	ClassDB::bind_compatibility_method(D_METHOD("generic_6dof_joint_get_flag", "joint", "axis", "flag"), &PhysicsServer3D::_generic_6dof_joint_get_flag_bind_compat_99341);
}

#endif // DISABLE_DEPRECATED
