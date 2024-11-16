/**************************************************************************/
/*  godot_physics_server_3d.compat.inc                                    */
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
#include "godot_physics_server_3d.h"

void GodotPhysicsServer3D::_generic_6dof_joint_set_param_bind_compat_99341(RID p_joint, Vector3::Axis p_axis, G6DOFJointAxisParam p_param, real_t p_value) {
	GodotPhysicsServer3D::generic_6dof_joint_set_param(p_joint, (int)p_axis, p_param, p_value);
}

real_t GodotPhysicsServer3D::_generic_6dof_joint_get_param_bind_compat_99341(RID p_joint, Vector3::Axis p_axis, G6DOFJointAxisParam p_param) const {
	return GodotPhysicsServer3D::generic_6dof_joint_get_param(p_joint, (int)p_axis, p_param);
}

void GodotPhysicsServer3D::_generic_6dof_joint_set_flag_bind_compat_99341(RID p_joint, Vector3::Axis p_axis, G6DOFJointAxisFlag p_flag, bool p_enable) {
	GodotPhysicsServer3D::generic_6dof_joint_set_flag(p_joint, (int)p_axis, p_flag, p_enable);
}

bool GodotPhysicsServer3D::_generic_6dof_joint_get_flag_bind_compat_99341(RID p_joint, Vector3::Axis p_axis, G6DOFJointAxisFlag p_flag) const {
	return GodotPhysicsServer3D::generic_6dof_joint_get_flag(p_joint, (int)p_axis, p_flag);
}

#endif //DISABLE_DEPRECATED
