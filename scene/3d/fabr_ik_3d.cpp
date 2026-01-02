/**************************************************************************/
/*  fabr_ik_3d.cpp                                                        */
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

#include "fabr_ik_3d.h"

void FABRIK3D::_solve_iteration(double p_delta, Skeleton3D *p_skeleton, IterateIK3DSetting *p_setting, const Vector3 &p_destination) {
	int joint_size = (int)p_setting->joints.size();

	// Backwards.
	bool first = true;
	for (int i = joint_size - 1; i >= 0; i--) {
		IKModifier3DSolverInfo *solver_info = p_setting->solver_info_list[i];
		if (!solver_info || Math::is_zero_approx(solver_info->length)) {
			continue;
		}

		int HEAD = i;
		int TAIL = i + 1;

		if (first) {
			p_setting->update_chain_coordinate_bw(p_skeleton, TAIL, p_destination);
			first = false;
		}

		p_setting->update_chain_coordinate_bw(p_skeleton, HEAD, limit_length(p_setting->chain[TAIL], p_setting->chain[HEAD], solver_info->length));

		if (p_setting->joint_settings[HEAD]->rotation_axis != ROTATION_AXIS_ALL) {
			p_setting->update_chain_coordinate_bw(p_skeleton, HEAD, p_setting->chain[TAIL] + p_setting->joint_settings[HEAD]->get_projected_rotation(solver_info->current_grest, p_setting->chain[HEAD] - p_setting->chain[TAIL]));
		}
		if (p_setting->joint_settings[HEAD]->limitation.is_valid()) {
			p_setting->update_chain_coordinate_bw(p_skeleton, HEAD, p_setting->chain[TAIL] + p_setting->joint_settings[HEAD]->get_limited_rotation(solver_info->current_grest, p_setting->chain[HEAD] - p_setting->chain[TAIL], solver_info->forward_vector));
		}
	}

	// Forwards.
	first = true;
	for (int i = 0; i < joint_size; i++) {
		IKModifier3DSolverInfo *solver_info = p_setting->solver_info_list[i];
		if (!solver_info || Math::is_zero_approx(solver_info->length)) {
			continue;
		}

		int HEAD = i;
		int TAIL = i + 1;

		if (first) {
			p_setting->update_chain_coordinate_fw(p_skeleton, HEAD, p_skeleton->get_bone_global_pose(p_setting->joints[HEAD].bone).origin);
			first = false;
		}

		p_setting->update_chain_coordinate_fw(p_skeleton, TAIL, limit_length(p_setting->chain[HEAD], p_setting->chain[TAIL], solver_info->length));

		if (p_setting->joint_settings[HEAD]->rotation_axis != ROTATION_AXIS_ALL) {
			p_setting->update_chain_coordinate_fw(p_skeleton, TAIL, p_setting->chain[HEAD] + p_setting->joint_settings[HEAD]->get_projected_rotation(solver_info->current_grest, p_setting->chain[TAIL] - p_setting->chain[HEAD]));
		}
		if (p_setting->joint_settings[HEAD]->limitation.is_valid()) {
			p_setting->update_chain_coordinate_fw(p_skeleton, TAIL, p_setting->chain[HEAD] + p_setting->joint_settings[HEAD]->get_limited_rotation(solver_info->current_grest, p_setting->chain[TAIL] - p_setting->chain[HEAD], solver_info->forward_vector));
		}
	}
}
