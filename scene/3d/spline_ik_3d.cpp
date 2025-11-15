/**************************************************************************/
/*  spline_ik_3d.cpp                                                      */
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

#include "spline_ik_3d.h"

bool SplineIK3D::_set(const StringName &p_path, const Variant &p_value) {
	String path = p_path;

	if (path.begins_with("settings/")) {
		int which = path.get_slicec('/', 1).to_int();
		String what = path.get_slicec('/', 2);
		ERR_FAIL_INDEX_V(which, (int)settings.size(), false);

		if (what == "path_3d") {
			set_path_3d(which, p_value);
		} else if (what == "tilt_enabled") {
			set_tilt_enabled(which, p_value);
		} else if (what == "tilt_fade_in") {
			set_tilt_fade_in(which, p_value);
		} else if (what == "tilt_fade_out") {
			set_tilt_fade_out(which, p_value);
		} else {
			return false;
		}
	}
	return true;
}

bool SplineIK3D::_get(const StringName &p_path, Variant &r_ret) const {
	String path = p_path;

	if (path.begins_with("settings/")) {
		int which = path.get_slicec('/', 1).to_int();
		String what = path.get_slicec('/', 2);
		ERR_FAIL_INDEX_V(which, (int)settings.size(), false);

		if (what == "path_3d") {
			r_ret = get_path_3d(which);
		} else if (what == "tilt_enabled") {
			r_ret = is_tilt_enabled(which);
		} else if (what == "tilt_fade_in") {
			r_ret = get_tilt_fade_in(which);
		} else if (what == "tilt_fade_out") {
			r_ret = get_tilt_fade_out(which);
		} else {
			return false;
		}
	}
	return true;
}

void SplineIK3D::_get_property_list(List<PropertyInfo> *p_list) const {
	LocalVector<PropertyInfo> props;
	for (uint32_t i = 0; i < settings.size(); i++) {
		String path = "settings/" + itos(i) + "/";
		props.push_back(PropertyInfo(Variant::NODE_PATH, path + "path_3d", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Path3D"));
		props.push_back(PropertyInfo(Variant::BOOL, path + "tilt_enabled"));
		props.push_back(PropertyInfo(Variant::INT, path + "tilt_fade_in", PROPERTY_HINT_RANGE, "-1,100,1,or_greater"));
		props.push_back(PropertyInfo(Variant::INT, path + "tilt_fade_out", PROPERTY_HINT_RANGE, "-1,100,1,or_greater"));
	}

	for (PropertyInfo &p : props) {
		_validate_dynamic_prop(p);
		p_list->push_back(p);
	}

	ChainIK3D::get_property_list(p_list);
}

void SplineIK3D::_validate_dynamic_prop(PropertyInfo &p_property) const {
	PackedStringArray split = p_property.name.split("/");
	if (split.size() > 2 && split[0] == "settings") {
		int which = split[1].to_int();
		if (split[2].begins_with("tilt_") && get_path_3d(which).is_empty()) {
			p_property.usage = PROPERTY_USAGE_NONE;
		} else if (split[2].begins_with("tilt_fade_") && !is_tilt_enabled(which)) {
			p_property.usage = PROPERTY_USAGE_NONE;
		}
	}
}

PackedStringArray SplineIK3D::get_configuration_warnings() const {
	PackedStringArray warnings = SkeletonModifier3D::get_configuration_warnings();
	for (uint32_t i = 0; i < sp_settings.size(); i++) {
		if (sp_settings[i]->path_3d.is_empty()) {
			warnings.push_back(RTR("Detecting settings with no Path3D set! SplineIK3D must have a Path3D to work."));
			break;
		}
	}
	return warnings;
}

// Setting.

void SplineIK3D::set_path_3d(int p_index, const NodePath &p_path_3d) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	sp_settings[p_index]->path_3d = p_path_3d;
	notify_property_list_changed();
	update_configuration_warnings();
}

NodePath SplineIK3D::get_path_3d(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), NodePath());
	return sp_settings[p_index]->path_3d;
}

void SplineIK3D::set_tilt_enabled(int p_index, bool p_enabled) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	sp_settings[p_index]->tilt_enabled = p_enabled;
	notify_property_list_changed();
}

bool SplineIK3D::is_tilt_enabled(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), false);
	return sp_settings[p_index]->tilt_enabled;
}

void SplineIK3D::set_tilt_fade_in(int p_index, int p_size) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	sp_settings[p_index]->tilt_fade_in = p_size;
}

int SplineIK3D::get_tilt_fade_in(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), -1);
	return sp_settings[p_index]->tilt_fade_in;
}

void SplineIK3D::set_tilt_fade_out(int p_index, int p_size) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	sp_settings[p_index]->tilt_fade_out = p_size;
}

int SplineIK3D::get_tilt_fade_out(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), -1);
	return sp_settings[p_index]->tilt_fade_out;
}

// Individual joints.

void SplineIK3D::_set_joint_count(int p_index, int p_count) {
	LocalVector<double> &twists = sp_settings[p_index]->twists;
	twists.resize(p_count);
	LocalVector<double> &accum = sp_settings[p_index]->chain_length_accum;
	accum.resize(p_count);
}

void SplineIK3D::_bind_methods() {
	// Setting.
	ClassDB::bind_method(D_METHOD("set_path_3d", "index", "path_3d"), &SplineIK3D::set_path_3d);
	ClassDB::bind_method(D_METHOD("get_path_3d", "index"), &SplineIK3D::get_path_3d);
	ClassDB::bind_method(D_METHOD("set_tilt_enabled", "index", "enabled"), &SplineIK3D::set_tilt_enabled);
	ClassDB::bind_method(D_METHOD("is_tilt_enabled", "index"), &SplineIK3D::is_tilt_enabled);
	ClassDB::bind_method(D_METHOD("set_tilt_fade_in", "index", "size"), &SplineIK3D::set_tilt_fade_in);
	ClassDB::bind_method(D_METHOD("get_tilt_fade_in", "index"), &SplineIK3D::get_tilt_fade_in);
	ClassDB::bind_method(D_METHOD("set_tilt_fade_out", "index", "size"), &SplineIK3D::set_tilt_fade_out);
	ClassDB::bind_method(D_METHOD("get_tilt_fade_out", "index"), &SplineIK3D::get_tilt_fade_out);

	ADD_ARRAY_COUNT("Settings", "setting_count", "set_setting_count", "get_setting_count", "settings/");
}

void SplineIK3D::_init_joints(Skeleton3D *p_skeleton, int p_index) {
	SplineIK3DSetting *setting = sp_settings[p_index];
	cached_space = p_skeleton->get_global_transform_interpolated();
	if (!setting->simulation_dirty) {
		if (mutable_bone_axes) {
			_update_bone_axis(p_skeleton, p_index);
		}
		return;
	}
	for (uint32_t i = 0; i < setting->solver_info_list.size(); i++) {
		if (setting->solver_info_list[i]) {
			memdelete(setting->solver_info_list[i]);
		}
	}
	setting->solver_info_list.clear();
	setting->solver_info_list.resize_initialized(setting->joints.size());
	setting->chain.clear();
	bool extend_end_bone = setting->extend_end_bone && setting->end_bone_length > 0;
	double accum = 0.0;
	for (uint32_t i = 0; i < setting->joints.size(); i++) {
		setting->chain.push_back(p_skeleton->get_bone_global_pose(setting->joints[i].bone).origin);
		bool last = i == setting->joints.size() - 1;
		if (last && extend_end_bone && setting->end_bone_length > 0) {
			Vector3 axis = IKModifier3D::get_bone_axis(p_skeleton, setting->end_bone.bone, setting->end_bone_direction, mutable_bone_axes);
			if (axis.is_zero_approx()) {
				setting->chain_length_accum[i] = accum;
				continue;
			}
			setting->solver_info_list[i] = memnew(IKModifier3DSolverInfo);
			setting->solver_info_list[i]->forward_vector = axis.normalized();
			setting->solver_info_list[i]->length = setting->end_bone_length;
			setting->chain.push_back(p_skeleton->get_bone_global_pose(setting->joints[i].bone).xform(axis * setting->end_bone_length));
		} else if (!last) {
			Vector3 axis = p_skeleton->get_bone_rest(setting->joints[i + 1].bone).origin;
			if (axis.is_zero_approx()) {
				setting->chain_length_accum[i] = accum;
				continue; // Means always we need to check solver info, but `!solver_info` means that the bone is zero length, so IK should skip it in the all process.
			}
			setting->solver_info_list[i] = memnew(IKModifier3DSolverInfo);
			setting->solver_info_list[i]->forward_vector = axis.normalized();
			setting->solver_info_list[i]->length = axis.length();
		}
		if (setting->solver_info_list[i]) {
			accum += setting->solver_info_list[i]->length;
		}
		setting->chain_length_accum[i] = accum;
	}

	if (mutable_bone_axes) {
		_update_bone_axis(p_skeleton, p_index);
#ifdef TOOLS_ENABLED
	} else {
		_make_gizmo_dirty();
#endif // TOOLS_ENABLED
	}

	setting->init_current_joint_rotations(p_skeleton);

	setting->simulation_dirty = false;
}

void SplineIK3D::_make_simulation_dirty(int p_index) {
	SplineIK3DSetting *setting = sp_settings[p_index];
	if (!setting) {
		return;
	}
	setting->simulation_dirty = true;
}

void SplineIK3D::_update_bone_axis(Skeleton3D *p_skeleton, int p_index) {
#ifdef TOOLS_ENABLED
	bool changed = false;
#endif // TOOLS_ENABLED
	SplineIK3DSetting *setting = sp_settings[p_index];
	const LocalVector<BoneJoint> &joints = setting->joints;
	const LocalVector<IKModifier3DSolverInfo *> &solver_info_list = setting->solver_info_list;
	int len = (int)solver_info_list.size() - 1;
	for (int j = 0; j < len; j++) {
		if (!solver_info_list[j]) {
			continue;
		}
		Vector3 axis = p_skeleton->get_bone_pose(joints[j + 1].bone).origin;
		if (axis.is_zero_approx()) {
			continue;
		}
		// Less computing.
#ifdef TOOLS_ENABLED
		if (!changed) {
			Vector3 old_v = solver_info_list[j]->forward_vector;
			solver_info_list[j]->forward_vector = axis.normalized();
			changed = changed || !old_v.is_equal_approx(solver_info_list[j]->forward_vector);
			float old_l = solver_info_list[j]->length;
			solver_info_list[j]->length = axis.length();
			changed = changed || !Math::is_equal_approx(old_l, solver_info_list[j]->length);
		} else {
			solver_info_list[j]->forward_vector = axis.normalized();
			solver_info_list[j]->length = axis.length();
		}
#else
		solver_info_list[j]->forward_vector = axis.normalized();
		solver_info_list[j]->length = axis.length();
#endif // TOOLS_ENABLED
	}
	if (setting->extend_end_bone && len >= 0) {
		if (solver_info_list[len]) {
			Vector3 axis = IKModifier3D::get_bone_axis(p_skeleton, setting->end_bone.bone, setting->end_bone_direction, mutable_bone_axes);
			if (!axis.is_zero_approx()) {
				solver_info_list[len]->forward_vector = axis.normalized();
				solver_info_list[len]->length = setting->end_bone_length;
			}
		}
	}
#ifdef TOOLS_ENABLED
	if (changed) {
		_make_gizmo_dirty();
	}
#endif // TOOLS_ENABLED
}

void SplineIK3D::_process_ik(Skeleton3D *p_skeleton, double p_delta) {
	for (uint32_t i = 0; i < settings.size(); i++) {
		_init_joints(p_skeleton, i);
		if (sp_settings[i]->joints.is_empty()) {
			continue; // Abort.
		}
		Path3D *path_3d = Object::cast_to<Path3D>(get_node_or_null(sp_settings[i]->path_3d));
		if (!path_3d) {
			continue; // Abort.
		}
		Ref<Curve3D> curve = path_3d->get_curve();
		if (curve.is_null() || curve->get_point_count() == 0) {
			continue; // Abort.
		}
		sp_settings[i]->cache_current_joint_rotations(p_skeleton); // Iterate over first to detect parent (outside of the chain) bone pose changes.
		_process_joints(p_delta, p_skeleton, sp_settings[i], curve, cached_space.affine_inverse() * path_3d->get_global_transform_interpolated());
	}
}

void SplineIK3D::_process_joints(double p_delta, Skeleton3D *p_skeleton, SplineIK3DSetting *p_setting, Ref<Curve3D> p_curve, const Transform3D &p_curve_space) {
	if (p_setting->solver_info_list.is_empty()) {
		return;
	}
	uint32_t joint_count = p_setting->joints.size();
	uint32_t joint_last = joint_count - 1;

	double path_length = p_curve->get_baked_length();
	PackedVector3Array points = p_curve->get_baked_points();
	Vector<real_t> tilts = p_curve->get_baked_tilts();
	Vector<real_t> dists = p_curve->get_baked_dist_cache();
	uint32_t point_count = points.size();
	uint32_t point_last = point_count - 1;

	// Make straight segment from root joint to start point.
	Vector3 start_point = p_curve_space.xform(points[0]);
	Vector3 start_vector = start_point - p_skeleton->get_bone_global_pose(p_setting->joints[0].bone).origin;
	double start_dist = start_vector.length();

	// Find first joint on the path.
	uint32_t chain_path_start = 0;
	while (chain_path_start < joint_count) {
		if (p_setting->chain_length_accum[chain_path_start] >= start_dist) {
			break;
		}
		chain_path_start++;
	}
	chain_path_start = (uint32_t)CLAMP((int)chain_path_start, 0, (int)joint_last);

	// For tilt fade-in, get bones length not on the path as denominator.
	double fade_in_denom = 0.0;
	int denom_start = p_setting->tilt_fade_in > 0 ? CLAMP(p_setting->tilt_fade_in - 1, (int)chain_path_start, (int)joint_count) : -1;
	int denom_start_to = denom_start - p_setting->tilt_fade_in;
	if (denom_start >= 0) {
		for (int i = denom_start; i > denom_start_to; i--) {
			if (i < 0) {
				break;
			}
			IKModifier3DSolverInfo *solver_info = p_setting->solver_info_list[i];
			if (!solver_info || Math::is_zero_approx(solver_info->length)) {
				continue;
			}
			fade_in_denom += solver_info->length;
		}
	}

	// Prepare for fade-out.
	uint32_t ended = 0;
	Vector3 end_point = p_curve_space.xform(points[point_last]);
	Vector3 end_vector;
	double end_to_end_length = 0.0;
	double fade_out_denom = 0.0;

	uint32_t last_nearest = 0;
	uint32_t last_nearest_next = 0;
	double last_interpolate = 0.0;

	for (uint32_t i = 0; i < p_setting->solver_info_list.size(); i++) {
		IKModifier3DSolverInfo *solver_info = p_setting->solver_info_list[i];
		if (!solver_info || Math::is_zero_approx(solver_info->length)) {
			continue;
		}
		uint32_t HEAD = i;
		uint32_t TAIL = i + 1;

		bool is_fitting_first = HEAD == chain_path_start;

		// Special case for out of path joints.
		if (point_count == 1 || HEAD <= chain_path_start) {
			// Set twist only for first fitting joint.
			if (!is_fitting_first) {
				p_setting->update_chain_coordinate(p_skeleton, TAIL, limit_length(p_setting->chain[HEAD], p_setting->chain[HEAD] + start_vector, solver_info->length));
			}
			if (p_setting->tilt_enabled) {
				if (p_setting->tilt_fade_in < 0) {
					p_setting->twists[HEAD] = 0.0;
				} else if (p_setting->tilt_fade_in == 0) {
					p_setting->twists[HEAD] = tilts[0];
				} else {
					// Decreases monotonically in a straight line, fetch the distance.
					double fade_in_dumping = CLAMP((double)(p_setting->chain[HEAD].distance_to(start_point) / fade_in_denom), 0.0, 1.0);
					p_setting->twists[HEAD] = Math::lerp((double)tilts[0], 0.0, fade_in_dumping);
				}
			}
			if (!is_fitting_first) {
				continue;
			}
		} else if (ended > 0) {
			p_setting->update_chain_coordinate(p_skeleton, TAIL, limit_length(p_setting->chain[HEAD], p_setting->chain[HEAD] + end_vector, solver_info->length));
			if (p_setting->tilt_enabled) {
				if (p_setting->tilt_fade_out < 0) {
					p_setting->twists[HEAD] = 0.0;
				} else if (p_setting->tilt_fade_out == 0) {
					p_setting->twists[HEAD] = tilts[point_last];
				} else {
					// Increases monotonically in a bended line, accumulate the distances.
					if (ended == 1) {
						end_to_end_length = p_setting->chain[TAIL].distance_to(end_point);
					} else {
						end_to_end_length += solver_info->length;
					}
					double fade_out_dumping = CLAMP(end_to_end_length / fade_out_denom, 0.0, 1.0);
					p_setting->twists[HEAD] = Math::lerp(ended == 1 ? Math::lerp((double)tilts[last_nearest], (double)tilts[last_nearest_next], last_interpolate) : (double)tilts[point_last], 0.0, fade_out_dumping);
					ended = 2;
				}
			}
			continue;
		}

		// General case.
		double lsq = solver_info->length * solver_info->length;
		Vector3 head_in_chain_space = p_curve_space.xform_inv(p_setting->chain[HEAD]);
		double interpolate = 0.0;
		uint32_t nearest = p_setting->find_nearest_point(head_in_chain_space, lsq, points, p_curve->is_closed(), last_nearest, &interpolate);
		if (nearest >= point_count) {
			if (HEAD == 0) {
				nearest = point_count - 2;
				interpolate = 1.0;
			} else {
				Vector3 chain_end = (p_setting->chain[HEAD] - p_setting->chain[HEAD - 1]).normalized();
				Vector3 path_end = (p_curve_space.xform(points[point_last]) - p_setting->chain[HEAD]).normalized();
				double rest_path_length = path_length - Math::lerp((double)dists[last_nearest], (double)dists[last_nearest_next], last_interpolate);
				interpolate = CLAMP(rest_path_length / solver_info->length, 0.0, 1.0); // End vector should be defined only one end bone to make neat interpolating.
				end_vector = chain_end.lerp(path_end, interpolate);

				int denom_end = p_setting->tilt_fade_out > 0 ? CLAMP((int)joint_last - p_setting->tilt_fade_out, 0, (int)last_nearest) : -1;
				int denom_end_to = denom_end + p_setting->tilt_fade_out;
				if (denom_end >= 0) {
					for (int e = denom_end; e < denom_end_to; e++) {
						if (e >= (int)joint_count) {
							break;
						}
						IKModifier3DSolverInfo *end_solver_info = p_setting->solver_info_list[e];
						if (!end_solver_info || Math::is_zero_approx(end_solver_info->length)) {
							continue;
						}
						fade_out_denom += end_solver_info->length;
					}
				}

				ended = 1;
				i--; // Will be processed above special case.
				continue;
			}
		}
		uint32_t nearest_next = p_curve->is_closed() ? Math::posmod(nearest + 1, point_count) : CLAMP(nearest, (uint32_t)0, point_last);
		p_setting->update_chain_coordinate(p_skeleton, TAIL, limit_length(p_setting->chain[HEAD], p_curve_space.xform(points[nearest].lerp(points[nearest_next], interpolate)), solver_info->length));
		if (!is_fitting_first) {
			p_setting->twists[HEAD] = Math::lerp((double)tilts[last_nearest], (double)tilts[last_nearest_next], last_interpolate);
		}
		last_nearest = nearest;
		last_nearest_next = nearest_next;
		last_interpolate = interpolate;
	}

	// Update virtual bone rest/poses.
	p_setting->cache_current_joint_rotations(p_skeleton, p_setting->tilt_enabled); // Pass p_setting->tilt_enabled to skip unneeded rotate process.

	// Apply the virtual bone rest/poses to the actual bones.
	for (uint32_t i = 0; i < p_setting->solver_info_list.size(); i++) {
		IKModifier3DSolverInfo *solver_info = p_setting->solver_info_list[i];
		if (!solver_info || Math::is_zero_approx(solver_info->length)) {
			continue;
		}
		p_skeleton->set_bone_pose_rotation(p_setting->joints[i].bone, solver_info->current_lpose);
	}
}

#ifdef TOOLS_ENABLED
Vector3 SplineIK3D::get_bone_vector(int p_index, int p_joint) const {
	Skeleton3D *skeleton = get_skeleton();
	if (!skeleton) {
		return Vector3();
	}
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), Vector3());
	SplineIK3DSetting *setting = sp_settings[p_index];
	if (!setting) {
		return Vector3();
	}
	const LocalVector<BoneJoint> &joints = setting->joints;
	ERR_FAIL_INDEX_V(p_joint, (int)joints.size(), Vector3());
	const LocalVector<IKModifier3DSolverInfo *> &solver_info_list = setting->solver_info_list;
	if (p_joint >= (int)solver_info_list.size() || !solver_info_list[p_joint]) {
		if (p_joint == (int)joints.size() - 1) {
			return IKModifier3D::get_bone_axis(skeleton, setting->end_bone.bone, setting->end_bone_direction, mutable_bone_axes) * setting->end_bone_length;
		}
		return mutable_bone_axes ? skeleton->get_bone_pose(joints[p_joint + 1].bone).origin : skeleton->get_bone_rest(joints[p_joint + 1].bone).origin;
	}
	return solver_info_list[p_joint]->forward_vector * solver_info_list[p_joint]->length;
}
#endif // TOOLS_ENABLED

SplineIK3D::~SplineIK3D() {
	clear_settings();
}
