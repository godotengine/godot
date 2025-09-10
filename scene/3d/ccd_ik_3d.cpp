/**************************************************************************/
/*  ccd_ik_3d.cpp                                                         */
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

#include "ccd_ik_3d.h"

void CCDIK3D::_solve_iteration(double p_delta, Skeleton3D *p_skeleton, IterateIK3DSetting *p_setting, const Vector3 &p_destination) {
	int joint_size = (int)p_setting->joints.size();
	int chain_size = (int)p_setting->chain.size();

	// Backwards.
	for (int ancestor = joint_size - 1; ancestor >= 0; ancestor--) {
		// Forwards.
		for (int i = ancestor; i < joint_size; i++) {
			IKModifier3DSolverInfo *solver_info = p_setting->solver_info_list[i];
			if (!solver_info || Math::is_zero_approx(solver_info->length)) {
				continue;
			}

			int HEAD = i;
			int TAIL = i + 1;

			Vector3 current_head = p_setting->chain[HEAD];
			Vector3 current_effector = p_setting->chain[chain_size - 1];
			Vector3 head_to_effector = current_effector - current_head;
			Vector3 head_to_destination = p_destination - current_head;

			if (Math::is_zero_approx(head_to_destination.length_squared() * head_to_effector.length_squared())) {
				continue;
			}

			Quaternion to_rot = Quaternion(head_to_effector.normalized(), head_to_destination.normalized());
			Vector3 to_tail = p_setting->chain[TAIL] - current_head;

			p_setting->update_chain_coordinate_fw(p_skeleton, TAIL, current_head + to_rot.xform(to_tail));

			if (p_setting->joint_settings[HEAD]->rotation_axis != ROTATION_AXIS_ALL) {
				p_setting->update_chain_coordinate_fw(p_skeleton, TAIL, p_setting->chain[HEAD] + p_setting->joint_settings[HEAD]->get_projected_rotation(solver_info->current_grest, p_setting->chain[TAIL] - p_setting->chain[HEAD]));
			}
			if (p_setting->joint_settings[HEAD]->limitation.is_valid()) {
				p_setting->update_chain_coordinate_fw(p_skeleton, TAIL, p_setting->chain[HEAD] + p_setting->joint_settings[HEAD]->get_limited_rotation(solver_info->current_grest, p_setting->chain[TAIL] - p_setting->chain[HEAD], solver_info->forward_vector));
			}
		}
	}
}
