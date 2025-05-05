/**************************************************************************/
/*  path_3d.cpp                                                           */
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

#include "path_3d.h"

#include "core/config/project_settings.h"

bool PathDebug3D::debug_enabled = false;
Color PathDebug3D::debug_paths_color = Color(0.1, 1.0, 0.7, 0.4);
float PathDebug3D::debug_paths_sample_interval = 0.1;
bool PathDebug3D::debug_paths_fish_bones_enabled = true;
int PathDebug3D::debug_paths_fish_bones_interval = 4;
Ref<Material> PathDebug3D::debug_default_material;

Mutex PathDebug3D::update_callbacks_mutex;
HashMap<Path3D *, Callable> PathDebug3D::update_callbacks;

void PathDebug3D::add_update_callback(Path3D *p_path, Callable p_callback) {
	MutexLock lock(update_callbacks_mutex);
	update_callbacks.insert(p_path, p_callback);
}

void PathDebug3D::remove_update_callback(Path3D *p_path) {
	MutexLock lock(update_callbacks_mutex);
	update_callbacks.erase(p_path);
}

void PathDebug3D::emit_update_callbacks() {
	MutexLock lock(update_callbacks_mutex);
	for (KeyValue<Path3D *, Callable> &kv : update_callbacks) {
		ERR_CONTINUE(!kv.value.is_valid());
		kv.value.call();
	}
}

void PathDebug3D::init_settings() {
#ifndef DISABLE_DEPRECATED
	if (!ProjectSettings::get_singleton()->has_setting("debug/shapes/paths/3d/geometry_color") && ProjectSettings::get_singleton()->has_setting("debug/shapes/paths/geometry_color")) {
		Color legacy_geometry_color = GLOBAL_GET("debug/shapes/paths/geometry_color");

		ProjectSettings::get_singleton()->set_setting("debug/shapes/paths/3d/geometry_color", legacy_geometry_color);
		ProjectSettings::get_singleton()->clear("debug/shapes/paths/geometry_color");
	}
#endif // DISABLE_DEPRECATED

	debug_paths_color = GLOBAL_DEF("debug/shapes/paths/3d/geometry_color", Color(0.1, 1.0, 0.7, 0.4));
	debug_paths_sample_interval = GLOBAL_DEF(PropertyInfo(Variant::FLOAT, "debug/shapes/paths/3d/sample_interval", PROPERTY_HINT_RANGE, "0.1,10,0.001,or_greater"), 0.1);
	debug_paths_fish_bones_enabled = GLOBAL_DEF(PropertyInfo(Variant::BOOL, "debug/shapes/paths/3d/fish_bones_enabled"), true);
	debug_paths_fish_bones_interval = GLOBAL_DEF(PropertyInfo(Variant::INT, "debug/shapes/paths/3d/fish_bones_interval", PROPERTY_HINT_RANGE, "1,10,1,or_greater"), 4);

	if (!debug_enabled && Engine::get_singleton()->is_editor_hint()) {
		debug_enabled = true;
	}
}

void PathDebug3D::init_materials() {
	if (debug_default_material.is_valid()) {
		return;
	}
	Ref<StandardMaterial3D> debug_material;
	debug_material.instantiate();
	debug_material->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
	debug_material->set_transparency(StandardMaterial3D::TRANSPARENCY_ALPHA);
	debug_material->set_flag(StandardMaterial3D::FLAG_SRGB_VERTEX_COLOR, true);
	debug_material->set_flag(StandardMaterial3D::FLAG_ALBEDO_FROM_VERTEX_COLOR, true);
	debug_material->set_flag(StandardMaterial3D::FLAG_DISABLE_FOG, true);
	debug_material->set_albedo(debug_paths_color);

	debug_default_material = debug_material;
}

void PathDebug3D::finish_materials() {
	if (debug_default_material.is_valid()) {
		debug_default_material.unref();
	};
}

void PathDebug3D::set_debug_enabled(bool p_enabled) {
	if (debug_enabled == p_enabled) {
		return;
	}
	debug_enabled = p_enabled;
	emit_update_callbacks();
}

bool PathDebug3D::is_debug_enabled() {
	return debug_enabled;
}

void PathDebug3D::set_debug_paths_color(const Color &p_color) {
	if (debug_paths_color == p_color) {
		return;
	}
	debug_paths_color = p_color;
	if (debug_default_material.is_valid()) {
		static_cast<Ref<StandardMaterial3D>>(debug_default_material)->set_albedo(debug_paths_color);
	}
	emit_update_callbacks();
}

Color PathDebug3D::get_debug_paths_color() {
	return debug_paths_color;
}

void PathDebug3D::set_debug_paths_sample_interval(float p_interval) {
	float _interval = MAX(0.1, p_interval);
	if (debug_paths_sample_interval == _interval) {
		return;
	}
	debug_paths_sample_interval = _interval;
	emit_update_callbacks();
}

float PathDebug3D::get_debug_paths_sample_interval() {
	return debug_paths_sample_interval;
}

void PathDebug3D::set_debug_paths_fish_bones_enabled(bool p_enabled) {
	if (debug_paths_fish_bones_enabled == p_enabled) {
		return;
	}
	debug_paths_fish_bones_enabled = p_enabled;
	emit_update_callbacks();
}

bool PathDebug3D::get_debug_paths_fish_bones_enabled() {
	return debug_paths_fish_bones_enabled;
}

void PathDebug3D::set_debug_paths_fish_bones_interval(int p_interval) {
	int _interval = MAX(1, p_interval);
	if (debug_paths_fish_bones_interval == _interval) {
		return;
	}
	debug_paths_fish_bones_interval = _interval;
	emit_update_callbacks();
}

int PathDebug3D::get_debug_paths_fish_bones_interval() {
	return debug_paths_fish_bones_interval;
}

Ref<Material> PathDebug3D::get_debug_material() {
	if (debug_default_material.is_null()) {
		init_materials();
	}
	return debug_default_material;
}

Path3D::Path3D() {
	set_notify_transform(true);
}

Path3D::~Path3D() {
	_debug_free();
}

void Path3D::set_update_callback(Callable p_callback) {
	update_callback = p_callback;
}

void Path3D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			PathDebug3D::add_update_callback(this, callable_mp(this, &Path3D::_on_debug_global_changed));
			_debug_create();
			_debug_update();
		} break;

		case NOTIFICATION_EXIT_TREE: {
			PathDebug3D::remove_update_callback(this);
			_debug_free();
		} break;

		case NOTIFICATION_VISIBILITY_CHANGED: {
			if (is_inside_tree() && debug_instance.is_valid()) {
				RS::get_singleton()->instance_set_visible(debug_instance, is_visible_in_tree());
			}
		} break;

		case NOTIFICATION_TRANSFORM_CHANGED: {
			if (is_inside_tree()) {
				if (debug_instance.is_valid()) {
					RS::get_singleton()->instance_set_transform(debug_instance, get_global_transform());
				}

				update_callback.call();
			}
		} break;
	}
}

void Path3D::_on_debug_global_changed() {
	if (_emitting_debug_changed) {
		return;
	}
	_emitting_debug_changed = true;
	callable_mp(this, &Path3D::_emit_debug_changed_deferred).call_deferred();
}

void Path3D::_emit_debug_changed_deferred() {
	_emitting_debug_changed = false;

	if (is_inside_tree()) {
		_debug_update();
		update_gizmos();
	}
}

void Path3D::_debug_create() {
	ERR_FAIL_NULL(RS::get_singleton());

	if (debug_mesh_rid.is_null()) {
		debug_mesh_rid = RS::get_singleton()->mesh_create();
	}

	if (debug_instance.is_null()) {
		debug_instance = RS::get_singleton()->instance_create();
	}

	RS::get_singleton()->instance_set_base(debug_instance, debug_mesh_rid);
	RS::get_singleton()->instance_geometry_set_cast_shadows_setting(debug_instance, RS::SHADOW_CASTING_SETTING_OFF);
}

void Path3D::_debug_free() {
	ERR_FAIL_NULL(RS::get_singleton());

	if (debug_instance.is_valid()) {
		RS::get_singleton()->free(debug_instance);
		debug_instance = RID();
	}
	if (debug_mesh_rid.is_valid()) {
		RS::get_singleton()->free(debug_mesh_rid);
		debug_mesh_rid = RID();
	}
}

void Path3D::_debug_update() {
	if (!is_inside_tree()) {
		return;
	}
	ERR_FAIL_NULL(RS::get_singleton());

	RenderingServer *rs = RS::get_singleton();

	ERR_FAIL_NULL(SceneTree::get_singleton());
	ERR_FAIL_NULL(RenderingServer::get_singleton());

	const bool path_debug_enabled = Engine::get_singleton()->is_editor_hint() || (PathDebug3D::is_debug_enabled() && debug_enabled);

	if (!path_debug_enabled) {
		_debug_free();
		return;
	}

	if (debug_mesh_rid.is_null() || debug_instance.is_null()) {
		_debug_create();
	}

	rs->mesh_clear(debug_mesh_rid);

	if (curve.is_null()) {
		return;
	}
	if (curve->get_point_count() < 2) {
		return;
	}

	const real_t baked_length = curve->get_baked_length();

	if (baked_length <= CMP_EPSILON) {
		return;
	}

	bool debug_paths_show_fish_bones = PathDebug3D::get_debug_paths_fish_bones_enabled();

	real_t sample_interval = PathDebug3D::get_debug_paths_sample_interval();

	int sample_count = int(baked_length / sample_interval) + 2;
	sample_interval = baked_length / (sample_count - 1);

	Vector<Transform3D> samples;
	samples.resize(sample_count);
	Transform3D *samples_ptrw = samples.ptrw();

	for (int i = 0; i < sample_count; i++) {
		samples_ptrw[i] = curve->sample_baked_with_rotation(i * sample_interval, true, true);
	}

	const Transform3D *samples_ptr = samples.ptr();

	// Render path lines.
	{
		Vector<Vector3> ribbon;
		ribbon.resize(sample_count);
		Vector3 *ribbon_ptrw = ribbon.ptrw();

		for (int i = 0; i < sample_count; i++) {
			ribbon_ptrw[i] = samples_ptr[i].origin;
		}

		Array ribbon_array;
		ribbon_array.resize(Mesh::ARRAY_MAX);
		ribbon_array[Mesh::ARRAY_VERTEX] = ribbon;

		rs->mesh_add_surface_from_arrays(debug_mesh_rid, RS::PRIMITIVE_LINE_STRIP, ribbon_array);
	}

	// Render path fish bones.
	if (debug_paths_show_fish_bones) {
		int fish_bones_interval = PathDebug3D::get_debug_paths_fish_bones_interval();

		const int vertex_per_bone = 4;
		Vector<Vector3> bones;
		bones.resize(sample_count * vertex_per_bone);
		Vector3 *bones_ptrw = bones.ptrw();

		for (int i = 0; i < sample_count / fish_bones_interval; i++) {
			const Transform3D &sample_transform = samples_ptr[i * fish_bones_interval];

			const Vector3 point = sample_transform.origin;
			const Vector3 &side = sample_transform.basis.get_column(0);
			const Vector3 &up = sample_transform.basis.get_column(1);
			const Vector3 &forward = sample_transform.basis.get_column(2);

			const Vector3 point_left = point + (side + forward - up * 0.3) * 0.06;
			const Vector3 point_right = point + (-side + forward - up * 0.3) * 0.06;

			const int bone_idx = i * vertex_per_bone;

			bones_ptrw[bone_idx] = point;
			bones_ptrw[bone_idx + 1] = point_left;
			bones_ptrw[bone_idx + 2] = point;
			bones_ptrw[bone_idx + 3] = point_right;
		}

		Array bone_array;
		bone_array.resize(Mesh::ARRAY_MAX);
		bone_array[Mesh::ARRAY_VERTEX] = bones;

		rs->mesh_add_surface_from_arrays(debug_mesh_rid, RS::PRIMITIVE_LINES, bone_array);
	}

	rs->instance_set_base(debug_instance, debug_mesh_rid);
	if (is_inside_tree()) {
		rs->instance_set_scenario(debug_instance, get_world_3d()->get_scenario());
		rs->instance_set_transform(debug_instance, get_global_transform());
		rs->instance_set_visible(debug_instance, is_visible_in_tree());
	}

	rs->instance_geometry_set_material_override(debug_instance, get_debug_material()->get_rid());
}

Ref<StandardMaterial3D> Path3D::get_debug_material() {
	if (debug_custom_enabled) {
		if (debug_custom_material.is_null()) {
			debug_custom_material.instantiate();
			debug_custom_material->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
			debug_custom_material->set_transparency(StandardMaterial3D::TRANSPARENCY_ALPHA);
			debug_custom_material->set_flag(StandardMaterial3D::FLAG_SRGB_VERTEX_COLOR, true);
			debug_custom_material->set_flag(StandardMaterial3D::FLAG_ALBEDO_FROM_VERTEX_COLOR, true);
			debug_custom_material->set_flag(StandardMaterial3D::FLAG_DISABLE_FOG, true);
		}
		debug_custom_material->set_albedo(debug_custom_color);
		return debug_custom_material;
	}
	return PathDebug3D::get_debug_material();
}

void Path3D::set_debug_enabled(bool p_enabled) {
	if (debug_enabled == p_enabled) {
		return;
	}

	debug_enabled = p_enabled;

	_debug_update();
	update_gizmos();
}

bool Path3D::get_debug_enabled() const {
	return debug_enabled;
}

void Path3D::set_debug_custom_color(const Color &p_color) {
	if (debug_custom_color == p_color) {
		return;
	}

	debug_custom_color = p_color;

	if (debug_custom_material.is_valid()) {
		debug_custom_material->set_albedo(debug_custom_color);
	}

	emit_signal(SNAME("debug_color_changed"));
}

const Color &Path3D::get_debug_custom_color() const {
	return debug_custom_color;
}

void Path3D::set_debug_custom_enabled(bool p_enabled) {
	if (debug_custom_enabled == p_enabled) {
		return;
	}

	debug_custom_enabled = p_enabled;

	_debug_update();
	update_gizmos();
}

bool Path3D::get_debug_custom_enabled() const {
	return debug_custom_enabled;
}

void Path3D::_curve_changed() {
	if (is_inside_tree() && Engine::get_singleton()->is_editor_hint()) {
		update_gizmos();
	}
	if (is_inside_tree()) {
		emit_signal(SNAME("curve_changed"));
	}

	// Update the configuration warnings of all children of type PathFollow
	// previously used for PathFollowOriented (now enforced orientation is done in PathFollow). Also trigger transform update on PathFollow3Ds in deferred mode.
	if (is_inside_tree()) {
		for (int i = 0; i < get_child_count(); i++) {
			PathFollow3D *child = Object::cast_to<PathFollow3D>(get_child(i));
			if (child) {
				child->update_configuration_warnings();
				child->update_transform();
			}
		}
	}

	_debug_update();
}

void Path3D::set_curve(const Ref<Curve3D> &p_curve) {
	if (curve.is_valid()) {
		curve->disconnect_changed(callable_mp(this, &Path3D::_curve_changed));
	}

	curve = p_curve;

	if (curve.is_valid()) {
		curve->connect_changed(callable_mp(this, &Path3D::_curve_changed));
	}
	_curve_changed();
}

Ref<Curve3D> Path3D::get_curve() const {
	return curve;
}

void Path3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_curve", "curve"), &Path3D::set_curve);
	ClassDB::bind_method(D_METHOD("get_curve"), &Path3D::get_curve);

	ClassDB::bind_method(D_METHOD("set_debug_enabled", "enabled"), &Path3D::set_debug_enabled);
	ClassDB::bind_method(D_METHOD("get_debug_enabled"), &Path3D::get_debug_enabled);

	ClassDB::bind_method(D_METHOD("set_debug_custom_enabled", "enabled"), &Path3D::set_debug_custom_enabled);
	ClassDB::bind_method(D_METHOD("get_debug_custom_enabled"), &Path3D::get_debug_custom_enabled);

	ClassDB::bind_method(D_METHOD("set_debug_custom_color", "debug_custom_color"), &Path3D::set_debug_custom_color);
	ClassDB::bind_method(D_METHOD("get_debug_custom_color"), &Path3D::get_debug_custom_color);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "curve", PROPERTY_HINT_RESOURCE_TYPE, "Curve3D", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_EDITOR_INSTANTIATE_OBJECT), "set_curve", "get_curve");

	ADD_GROUP("Debug", "debug_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "debug_enabled"), "set_debug_enabled", "get_debug_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "debug_custom_enabled"), "set_debug_custom_enabled", "get_debug_custom_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "debug_custom_color"), "set_debug_custom_color", "get_debug_custom_color");

	ADD_SIGNAL(MethodInfo("curve_changed"));
	ADD_SIGNAL(MethodInfo("debug_color_changed"));
}

void PathFollow3D::update_transform() {
	if (!path) {
		return;
	}

	Ref<Curve3D> c = path->get_curve();
	if (c.is_null()) {
		return;
	}

	real_t bl = c->get_baked_length();
	if (bl == 0.0) {
		return;
	}

	Transform3D t;

	if (rotation_mode == ROTATION_NONE) {
		Vector3 pos = c->sample_baked(progress, cubic);
		t.origin = pos;
	} else {
		t = c->sample_baked_with_rotation(progress, cubic, false);
		Vector3 tangent = -t.basis.get_column(2); // Retain tangent for applying tilt.
		t = PathFollow3D::correct_posture(t, rotation_mode);

		// Switch Z+ and Z- if necessary.
		if (use_model_front) {
			t.basis *= Basis::from_scale(Vector3(-1.0, 1.0, -1.0));
		}

		// Apply tilt *after* correct_posture().
		if (tilt_enabled) {
			const real_t tilt = c->sample_baked_tilt(progress);

			const Basis twist(tangent, tilt);
			t.basis = twist * t.basis;
		}
	}

	// Apply offset and scale.
	Vector3 scale = get_transform().basis.get_scale();
	t.translate_local(Vector3(h_offset, v_offset, 0));
	t.basis.scale_local(scale);

	set_transform(t);
}

void PathFollow3D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			Node *parent = get_parent();
			if (parent) {
				path = Object::cast_to<Path3D>(parent);
				update_transform();
			}
		} break;

		case NOTIFICATION_EXIT_TREE: {
			path = nullptr;
		} break;
	}
}

void PathFollow3D::set_cubic_interpolation_enabled(bool p_enabled) {
	cubic = p_enabled;
}

bool PathFollow3D::is_cubic_interpolation_enabled() const {
	return cubic;
}

void PathFollow3D::_validate_property(PropertyInfo &p_property) const {
	if (p_property.name == "offset") {
		real_t max = 10000;
		if (path && path->get_curve().is_valid()) {
			max = path->get_curve()->get_baked_length();
		}

		p_property.hint_string = "0," + rtos(max) + ",0.01,or_less,or_greater";
	}
}

PackedStringArray PathFollow3D::get_configuration_warnings() const {
	PackedStringArray warnings = Node3D::get_configuration_warnings();

	if (is_visible_in_tree() && is_inside_tree()) {
		if (!Object::cast_to<Path3D>(get_parent())) {
			warnings.push_back(RTR("PathFollow3D only works when set as a child of a Path3D node."));
		} else {
			Path3D *p = Object::cast_to<Path3D>(get_parent());
			if (p->get_curve().is_valid() && !p->get_curve()->is_up_vector_enabled() && rotation_mode == ROTATION_ORIENTED) {
				warnings.push_back(RTR("PathFollow3D's ROTATION_ORIENTED requires \"Up Vector\" to be enabled in its parent Path3D's Curve resource."));
			}
		}
	}

	return warnings;
}

Transform3D PathFollow3D::correct_posture(Transform3D p_transform, PathFollow3D::RotationMode p_rotation_mode) {
	Transform3D t = p_transform;

	// Modify frame according to rotation mode.
	if (p_rotation_mode == PathFollow3D::ROTATION_NONE) {
		// Clear rotation.
		t.basis = Basis();
	} else if (p_rotation_mode == PathFollow3D::ROTATION_ORIENTED) {
		Vector3 tangent = -t.basis.get_column(2);

		// Y-axis points up by default.
		t.basis = Basis::looking_at(tangent);
	} else {
		// Lock some euler axes.
		Vector3 euler = t.basis.get_euler_normalized(EulerOrder::YXZ);
		if (p_rotation_mode == PathFollow3D::ROTATION_Y) {
			// Only Y-axis allowed.
			euler[0] = 0;
			euler[2] = 0;
		} else if (p_rotation_mode == PathFollow3D::ROTATION_XY) {
			// XY allowed.
			euler[2] = 0;
		}

		Basis locked = Basis::from_euler(euler, EulerOrder::YXZ);
		t.basis = locked;
	}

	return t;
}

void PathFollow3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_progress", "progress"), &PathFollow3D::set_progress);
	ClassDB::bind_method(D_METHOD("get_progress"), &PathFollow3D::get_progress);

	ClassDB::bind_method(D_METHOD("set_h_offset", "h_offset"), &PathFollow3D::set_h_offset);
	ClassDB::bind_method(D_METHOD("get_h_offset"), &PathFollow3D::get_h_offset);

	ClassDB::bind_method(D_METHOD("set_v_offset", "v_offset"), &PathFollow3D::set_v_offset);
	ClassDB::bind_method(D_METHOD("get_v_offset"), &PathFollow3D::get_v_offset);

	ClassDB::bind_method(D_METHOD("set_progress_ratio", "ratio"), &PathFollow3D::set_progress_ratio);
	ClassDB::bind_method(D_METHOD("get_progress_ratio"), &PathFollow3D::get_progress_ratio);

	ClassDB::bind_method(D_METHOD("set_rotation_mode", "rotation_mode"), &PathFollow3D::set_rotation_mode);
	ClassDB::bind_method(D_METHOD("get_rotation_mode"), &PathFollow3D::get_rotation_mode);

	ClassDB::bind_method(D_METHOD("set_cubic_interpolation", "enabled"), &PathFollow3D::set_cubic_interpolation_enabled);
	ClassDB::bind_method(D_METHOD("get_cubic_interpolation"), &PathFollow3D::is_cubic_interpolation_enabled);

	ClassDB::bind_method(D_METHOD("set_use_model_front", "enabled"), &PathFollow3D::set_use_model_front);
	ClassDB::bind_method(D_METHOD("is_using_model_front"), &PathFollow3D::is_using_model_front);

	ClassDB::bind_method(D_METHOD("set_loop", "loop"), &PathFollow3D::set_loop);
	ClassDB::bind_method(D_METHOD("has_loop"), &PathFollow3D::has_loop);

	ClassDB::bind_method(D_METHOD("set_tilt_enabled", "enabled"), &PathFollow3D::set_tilt_enabled);
	ClassDB::bind_method(D_METHOD("is_tilt_enabled"), &PathFollow3D::is_tilt_enabled);

	ClassDB::bind_static_method("PathFollow3D", D_METHOD("correct_posture", "transform", "rotation_mode"), &PathFollow3D::correct_posture);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "progress", PROPERTY_HINT_RANGE, "0,10000,0.01,or_less,or_greater,suffix:m"), "set_progress", "get_progress");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "progress_ratio", PROPERTY_HINT_RANGE, "0,1,0.0001,or_less,or_greater", PROPERTY_USAGE_EDITOR), "set_progress_ratio", "get_progress_ratio");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "h_offset", PROPERTY_HINT_NONE, "suffix:m"), "set_h_offset", "get_h_offset");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "v_offset", PROPERTY_HINT_NONE, "suffix:m"), "set_v_offset", "get_v_offset");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "rotation_mode", PROPERTY_HINT_ENUM, "None,Y,XY,XYZ,Oriented"), "set_rotation_mode", "get_rotation_mode");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "use_model_front"), "set_use_model_front", "is_using_model_front");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "cubic_interp"), "set_cubic_interpolation", "get_cubic_interpolation");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "loop"), "set_loop", "has_loop");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "tilt_enabled"), "set_tilt_enabled", "is_tilt_enabled");

	BIND_ENUM_CONSTANT(ROTATION_NONE);
	BIND_ENUM_CONSTANT(ROTATION_Y);
	BIND_ENUM_CONSTANT(ROTATION_XY);
	BIND_ENUM_CONSTANT(ROTATION_XYZ);
	BIND_ENUM_CONSTANT(ROTATION_ORIENTED);
}

void PathFollow3D::set_progress(real_t p_progress) {
	ERR_FAIL_COND(!std::isfinite(p_progress));
	if (progress == p_progress) {
		return;
	}
	progress = p_progress;

	if (path) {
		if (path->get_curve().is_valid()) {
			real_t path_length = path->get_curve()->get_baked_length();

			if (loop && path_length) {
				progress = Math::fposmod(progress, path_length);
				if (!Math::is_zero_approx(p_progress) && Math::is_zero_approx(progress)) {
					progress = path_length;
				}
			} else {
				progress = CLAMP(progress, 0, path_length);
			}
		}

		update_transform();
	}
}

void PathFollow3D::set_h_offset(real_t p_h_offset) {
	if (h_offset == p_h_offset) {
		return;
	}
	h_offset = p_h_offset;
	update_transform();
}

real_t PathFollow3D::get_h_offset() const {
	return h_offset;
}

void PathFollow3D::set_v_offset(real_t p_v_offset) {
	if (v_offset == p_v_offset) {
		return;
	}
	v_offset = p_v_offset;
	update_transform();
}

real_t PathFollow3D::get_v_offset() const {
	return v_offset;
}

real_t PathFollow3D::get_progress() const {
	return progress;
}

void PathFollow3D::set_progress_ratio(real_t p_ratio) {
	ERR_FAIL_NULL_MSG(path, "Can only set progress ratio on a PathFollow3D that is the child of a Path3D which is itself part of the scene tree.");
	ERR_FAIL_COND_MSG(path->get_curve().is_null(), "Can't set progress ratio on a PathFollow3D that does not have a Curve.");
	ERR_FAIL_COND_MSG(!path->get_curve()->get_baked_length(), "Can't set progress ratio on a PathFollow3D that has a 0 length curve.");
	set_progress(p_ratio * path->get_curve()->get_baked_length());
}

real_t PathFollow3D::get_progress_ratio() const {
	if (path && path->get_curve().is_valid() && path->get_curve()->get_baked_length()) {
		return get_progress() / path->get_curve()->get_baked_length();
	} else {
		return 0;
	}
}

void PathFollow3D::set_rotation_mode(RotationMode p_rotation_mode) {
	if (rotation_mode == p_rotation_mode) {
		return;
	}
	rotation_mode = p_rotation_mode;

	update_configuration_warnings();
	update_transform();
}

PathFollow3D::RotationMode PathFollow3D::get_rotation_mode() const {
	return rotation_mode;
}

void PathFollow3D::set_use_model_front(bool p_use_model_front) {
	if (use_model_front == p_use_model_front) {
		return;
	}
	use_model_front = p_use_model_front;
	update_transform();
}

bool PathFollow3D::is_using_model_front() const {
	return use_model_front;
}

void PathFollow3D::set_loop(bool p_loop) {
	if (loop == p_loop) {
		return;
	}
	loop = p_loop;
	update_transform();
}

bool PathFollow3D::has_loop() const {
	return loop;
}

void PathFollow3D::set_tilt_enabled(bool p_enabled) {
	if (tilt_enabled == p_enabled) {
		return;
	}
	tilt_enabled = p_enabled;
	update_transform();
}

bool PathFollow3D::is_tilt_enabled() const {
	return tilt_enabled;
}
