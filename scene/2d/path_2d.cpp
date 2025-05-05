/**************************************************************************/
/*  path_2d.cpp                                                           */
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

#include "path_2d.h"

#include "core/config/project_settings.h"
#include "core/math/geometry_2d.h"
#include "scene/main/timer.h"

#ifdef TOOLS_ENABLED
#include "editor/themes/editor_scale.h"
#endif

bool PathDebug2D::debug_enabled = false;
Color PathDebug2D::debug_paths_color = Color(0.1, 1.0, 0.7, 0.4);
float PathDebug2D::debug_paths_width = 2.0;
float PathDebug2D::debug_paths_sample_interval = 5.0;
bool PathDebug2D::debug_paths_fish_bones_enabled = true;
int PathDebug2D::debug_paths_fish_bones_interval = 4;

Mutex PathDebug2D::update_callbacks_mutex;
HashMap<Path2D *, Callable> PathDebug2D::update_callbacks;

void PathDebug2D::add_update_callback(Path2D *p_path, Callable p_callback) {
	MutexLock lock(update_callbacks_mutex);
	update_callbacks.insert(p_path, p_callback);
}

void PathDebug2D::remove_update_callback(Path2D *p_path) {
	MutexLock lock(update_callbacks_mutex);
	update_callbacks.erase(p_path);
}

void PathDebug2D::emit_update_callbacks() {
	MutexLock lock(update_callbacks_mutex);
	for (KeyValue<Path2D *, Callable> &kv : update_callbacks) {
		ERR_CONTINUE(!kv.value.is_valid());
		kv.value.call();
	}
}

void PathDebug2D::init_settings() {
#ifndef DISABLE_DEPRECATED
	if (!ProjectSettings::get_singleton()->has_setting("debug/shapes/paths/2d/geometry_color") && ProjectSettings::get_singleton()->has_setting("debug/shapes/paths/geometry_color")) {
		Color legacy_geometry_color = GLOBAL_GET("debug/shapes/paths/geometry_color");

		ProjectSettings::get_singleton()->set_setting("debug/shapes/paths/2d/geometry_color", legacy_geometry_color);
		ProjectSettings::get_singleton()->clear("debug/shapes/paths/geometry_color");
	}

	if (ProjectSettings::get_singleton()->has_setting("debug/shapes/paths/geometry_width")) {
		debug_paths_width = GLOBAL_GET("debug/shapes/paths/geometry_width");
	}
#endif // DISABLE_DEPRECATED

	debug_paths_color = GLOBAL_DEF("debug/shapes/paths/2d/geometry_color", Color(0.1, 1.0, 0.7, 0.4));
	debug_paths_sample_interval = GLOBAL_DEF(PropertyInfo(Variant::FLOAT, "debug/shapes/paths/2d/sample_interval", PROPERTY_HINT_RANGE, "0.1,10,0.001,or_greater"), 5.0);
	debug_paths_fish_bones_enabled = GLOBAL_DEF(PropertyInfo(Variant::BOOL, "debug/shapes/paths/2d/fish_bones_enabled"), true);
	debug_paths_fish_bones_interval = GLOBAL_DEF(PropertyInfo(Variant::INT, "debug/shapes/paths/2d/fish_bones_interval", PROPERTY_HINT_RANGE, "1,10,1,or_greater"), 4);

	if (!debug_enabled && Engine::get_singleton()->is_editor_hint()) {
		debug_enabled = true;
	}
}

void PathDebug2D::set_debug_enabled(bool p_enabled) {
	if (debug_enabled == p_enabled) {
		return;
	}
	debug_enabled = p_enabled;
	emit_update_callbacks();
}

bool PathDebug2D::is_debug_enabled() {
	return debug_enabled;
}

void PathDebug2D::set_debug_paths_color(const Color &p_color) {
	if (debug_paths_color == p_color) {
		return;
	}
	debug_paths_color = p_color;
	emit_update_callbacks();
}

Color PathDebug2D::get_debug_paths_color() {
	return debug_paths_color;
}

void PathDebug2D::set_debug_paths_width(float p_width) {
	float _width = MAX(0.1, p_width);
	if (debug_paths_width == _width) {
		return;
	}
	debug_paths_width = _width;
	emit_update_callbacks();
}

float PathDebug2D::get_debug_paths_width() {
	return debug_paths_width;
}

void PathDebug2D::set_debug_paths_sample_interval(float p_interval) {
	float _interval = MAX(0.1, p_interval);
	if (debug_paths_sample_interval == _interval) {
		return;
	}
	debug_paths_sample_interval = _interval;
	emit_update_callbacks();
}

float PathDebug2D::get_debug_paths_sample_interval() {
	return debug_paths_sample_interval;
}

void PathDebug2D::set_debug_paths_fish_bones_enabled(bool p_enabled) {
	if (debug_paths_fish_bones_enabled == p_enabled) {
		return;
	}
	debug_paths_fish_bones_enabled = p_enabled;
	emit_update_callbacks();
}

bool PathDebug2D::get_debug_paths_fish_bones_enabled() {
	return debug_paths_fish_bones_enabled;
}

void PathDebug2D::set_debug_paths_fish_bones_interval(int p_interval) {
	int _interval = MAX(1, p_interval);
	if (debug_paths_fish_bones_interval == _interval) {
		return;
	}
	debug_paths_fish_bones_interval = _interval;
	emit_update_callbacks();
}

int PathDebug2D::get_debug_paths_fish_bones_interval() {
	return debug_paths_fish_bones_interval;
}

#ifdef DEBUG_ENABLED
Rect2 Path2D::_edit_get_rect() const {
	if (curve.is_null() || curve->get_point_count() == 0) {
		return Rect2(0, 0, 0, 0);
	}

	Rect2 aabb = Rect2(curve->get_point_position(0), Vector2(0, 0));

	for (int i = 0; i < curve->get_point_count(); i++) {
		for (int j = 0; j <= 8; j++) {
			real_t frac = j / 8.0;
			Vector2 p = curve->sample(i, frac);
			aabb.expand_to(p);
		}
	}

	return aabb;
}

bool Path2D::_edit_use_rect() const {
	return curve.is_valid() && curve->get_point_count() != 0;
}

bool Path2D::_edit_is_selected_on_click(const Point2 &p_point, double p_tolerance) const {
	if (curve.is_null()) {
		return false;
	}

	for (int i = 0; i < curve->get_point_count(); i++) {
		Vector2 segment_a = curve->get_point_position(i);

		for (int j = 1; j <= 8; j++) {
			real_t frac = j / 8.0;
			const Vector2 segment_b = curve->sample(i, frac);

			Vector2 p = Geometry2D::get_closest_point_to_segment(p_point, segment_a, segment_b);
			if (p.distance_to(p_point) <= p_tolerance) {
				return true;
			}

			segment_a = segment_b;
		}
	}

	return false;
}
#endif

void Path2D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			PathDebug2D::add_update_callback(this, callable_mp(this, &Path2D::_on_debug_global_changed));
			_debug_create();
			queue_redraw();
		} break;

		case NOTIFICATION_EXIT_TREE: {
			PathDebug2D::remove_update_callback(this);
			_debug_free();
		} break;

		case NOTIFICATION_DRAW: {
			_debug_update();
		} break;
	}
}

void Path2D::_on_debug_global_changed() {
	if (_emitting_debug_changed) {
		return;
	}
	_emitting_debug_changed = true;
	callable_mp(this, &Path2D::_emit_debug_changed_deferred).call_deferred();
}

void Path2D::_emit_debug_changed_deferred() {
	_emitting_debug_changed = false;

	if (is_inside_tree()) {
		queue_redraw();
	}
}

void Path2D::_debug_create() {
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

void Path2D::_debug_free() {
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

void Path2D::_debug_update() {
	if (!is_inside_tree()) {
		return;
	}

	ERR_FAIL_NULL(RS::get_singleton());

	RenderingServer *rs = RS::get_singleton();

	ERR_FAIL_NULL(SceneTree::get_singleton());
	ERR_FAIL_NULL(RenderingServer::get_singleton());

	const bool path_debug_enabled = Engine::get_singleton()->is_editor_hint() || (PathDebug2D::is_debug_enabled() && debug_enabled);

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

	const Color debug_default_color = PathDebug2D::get_debug_paths_color();

	Color debug_color = debug_default_color;
	if (debug_custom_enabled) {
		debug_color = debug_custom_color;
	}

	bool debug_paths_show_fish_bones = PathDebug2D::get_debug_paths_fish_bones_enabled();

	real_t sample_interval = PathDebug2D::get_debug_paths_sample_interval();

	const int sample_count = int(baked_length / sample_interval) + 2;
	sample_interval = baked_length / (sample_count - 1); // Recalculate real interval length.

	Vector<Transform2D> samples;
	samples.resize(sample_count);
	Transform2D *samples_ptrw = samples.ptrw();

	for (int i = 0; i < sample_count; i++) {
		samples_ptrw[i] = curve->sample_baked_with_rotation(i * sample_interval, false);
	}

	const Transform2D *samples_ptr = samples.ptr();

	// Draw curve segments
	{
		Vector<Vector2> ribbon;
		ribbon.resize(sample_count);
		Vector2 *ribbon_ptrw = ribbon.ptrw();

		for (int i = 0; i < sample_count; i++) {
			ribbon_ptrw[i] = samples_ptr[i].get_origin();
		}

		Array ribbon_array;
		ribbon_array.resize(Mesh::ARRAY_MAX);
		ribbon_array[Mesh::ARRAY_VERTEX] = ribbon;
		Vector<Color> ribbon_color;
		ribbon_color.resize(ribbon.size());
		ribbon_color.fill(debug_color);
		ribbon_array[Mesh::ARRAY_COLOR] = ribbon_color;

		rs->mesh_add_surface_from_arrays(debug_mesh_rid, RS::PRIMITIVE_LINE_STRIP, ribbon_array, Array(), Dictionary(), RS::ARRAY_FLAG_USE_2D_VERTICES);
	}

	// Render path fish bones.
	if (debug_paths_show_fish_bones) {
		int fish_bones_interval = PathDebug2D::get_debug_paths_fish_bones_interval();

		const int vertex_per_bone = 4;
		Vector<Vector2> bones;
		bones.resize(sample_count * vertex_per_bone);
		Vector2 *bones_ptrw = bones.ptrw();

		for (int i = 0; i < sample_count / fish_bones_interval; i++) {
			const Transform2D &sample_transform = samples_ptr[i * fish_bones_interval];

			const Vector2 point = sample_transform.get_origin();
			const Vector2 &side = sample_transform.columns[1];
			const Vector2 &forward = sample_transform.columns[0];

			const int bone_idx = i * vertex_per_bone;

			bones_ptrw[bone_idx] = point;
			bones_ptrw[bone_idx + 1] = point + (side - forward) * 5;
			bones_ptrw[bone_idx + 2] = point;
			bones_ptrw[bone_idx + 3] = point + (-side - forward) * 5;
		}

		Array bone_array;
		bone_array.resize(Mesh::ARRAY_MAX);
		bone_array[Mesh::ARRAY_VERTEX] = bones;
		Vector<Color> bones_color;
		bones_color.resize(bones.size());
		bones_color.fill(debug_color);
		bone_array[Mesh::ARRAY_COLOR] = bones_color;

		rs->mesh_add_surface_from_arrays(debug_mesh_rid, RS::PRIMITIVE_LINES, bone_array, Array(), Dictionary(), RS::ARRAY_FLAG_USE_2D_VERTICES);
	}

	rs->canvas_item_clear(get_canvas_item());
	rs->canvas_item_add_mesh(get_canvas_item(), debug_mesh_rid, Transform2D());
}

void Path2D::set_debug_enabled(bool p_enabled) {
	if (debug_enabled == p_enabled) {
		return;
	}

	debug_enabled = p_enabled;

	queue_redraw();
}

bool Path2D::get_debug_enabled() const {
	return debug_enabled;
}

void Path2D::set_debug_custom_color(const Color &p_color) {
	if (debug_custom_color == p_color) {
		return;
	}

	debug_custom_color = p_color;

	queue_redraw();
}

const Color &Path2D::get_debug_custom_color() const {
	return debug_custom_color;
}

void Path2D::set_debug_custom_enabled(bool p_enabled) {
	if (debug_custom_enabled == p_enabled) {
		return;
	}

	debug_custom_enabled = p_enabled;

	queue_redraw();
}

bool Path2D::get_debug_custom_enabled() const {
	return debug_custom_enabled;
}

void Path2D::_curve_changed() {
	if (!is_inside_tree()) {
		return;
	}

	for (int i = 0; i < get_child_count(); i++) {
		PathFollow2D *follow = Object::cast_to<PathFollow2D>(get_child(i));
		if (follow) {
			follow->path_changed();
		}
	}

	if (Engine::get_singleton()->is_editor_hint() || get_tree()->is_debugging_paths_hint()) {
		queue_redraw();
	}
}

void Path2D::set_curve(const Ref<Curve2D> &p_curve) {
	if (curve.is_valid()) {
		curve->disconnect_changed(callable_mp(this, &Path2D::_curve_changed));
	}

	curve = p_curve;

	if (curve.is_valid()) {
		curve->connect_changed(callable_mp(this, &Path2D::_curve_changed));
	}

	_curve_changed();
}

Ref<Curve2D> Path2D::get_curve() const {
	return curve;
}

void Path2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_curve", "curve"), &Path2D::set_curve);
	ClassDB::bind_method(D_METHOD("get_curve"), &Path2D::get_curve);

	ClassDB::bind_method(D_METHOD("set_debug_enabled", "enabled"), &Path2D::set_debug_enabled);
	ClassDB::bind_method(D_METHOD("get_debug_enabled"), &Path2D::get_debug_enabled);

	ClassDB::bind_method(D_METHOD("set_debug_custom_enabled", "enabled"), &Path2D::set_debug_custom_enabled);
	ClassDB::bind_method(D_METHOD("get_debug_custom_enabled"), &Path2D::get_debug_custom_enabled);

	ClassDB::bind_method(D_METHOD("set_debug_custom_color", "debug_custom_color"), &Path2D::set_debug_custom_color);
	ClassDB::bind_method(D_METHOD("get_debug_custom_color"), &Path2D::get_debug_custom_color);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "curve", PROPERTY_HINT_RESOURCE_TYPE, "Curve2D", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_EDITOR_INSTANTIATE_OBJECT), "set_curve", "get_curve");

	ADD_GROUP("Debug", "debug_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "debug_enabled"), "set_debug_enabled", "get_debug_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "debug_custom_enabled"), "set_debug_custom_enabled", "get_debug_custom_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "debug_custom_color"), "set_debug_custom_color", "get_debug_custom_color");
}

/////////////////////////////////////////////////////////////////////////////////

void PathFollow2D::path_changed() {
	if (update_timer && !update_timer->is_stopped()) {
		update_timer->start();
	} else {
		_update_transform();
	}
}

void PathFollow2D::_update_transform() {
	if (!path) {
		return;
	}

	Ref<Curve2D> c = path->get_curve();
	if (c.is_null()) {
		return;
	}

	real_t path_length = c->get_baked_length();
	if (path_length == 0) {
		return;
	}

	if (rotates) {
		Transform2D xform = c->sample_baked_with_rotation(progress, cubic);
		xform.translate_local(h_offset, v_offset);
		set_rotation(xform[0].angle());
		set_position(xform[2]);
	} else {
		Vector2 pos = c->sample_baked(progress, cubic);
		pos.x += h_offset;
		pos.y += v_offset;
		set_position(pos);
	}
}

void PathFollow2D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			if (Engine::get_singleton()->is_editor_hint()) {
				update_timer = memnew(Timer);
				update_timer->set_wait_time(0.2);
				update_timer->set_one_shot(true);
				update_timer->connect("timeout", callable_mp(this, &PathFollow2D::_update_transform));
				add_child(update_timer, false, Node::INTERNAL_MODE_BACK);
			}
		} break;

		case NOTIFICATION_ENTER_TREE: {
			path = Object::cast_to<Path2D>(get_parent());
			if (path) {
				_update_transform();
			}
		} break;

		case NOTIFICATION_EXIT_TREE: {
			path = nullptr;
		} break;
	}
}

void PathFollow2D::set_cubic_interpolation_enabled(bool p_enabled) {
	cubic = p_enabled;
}

bool PathFollow2D::is_cubic_interpolation_enabled() const {
	return cubic;
}

void PathFollow2D::_validate_property(PropertyInfo &p_property) const {
	if (p_property.name == "offset") {
		real_t max = 10000.0;
		if (path && path->get_curve().is_valid()) {
			max = path->get_curve()->get_baked_length();
		}

		p_property.hint_string = "0," + rtos(max) + ",0.01,or_less,or_greater";
	}
}

PackedStringArray PathFollow2D::get_configuration_warnings() const {
	PackedStringArray warnings = Node2D::get_configuration_warnings();

	if (is_visible_in_tree() && is_inside_tree()) {
		if (!Object::cast_to<Path2D>(get_parent())) {
			warnings.push_back(RTR("PathFollow2D only works when set as a child of a Path2D node."));
		}
	}

	return warnings;
}

void PathFollow2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_progress", "progress"), &PathFollow2D::set_progress);
	ClassDB::bind_method(D_METHOD("get_progress"), &PathFollow2D::get_progress);

	ClassDB::bind_method(D_METHOD("set_h_offset", "h_offset"), &PathFollow2D::set_h_offset);
	ClassDB::bind_method(D_METHOD("get_h_offset"), &PathFollow2D::get_h_offset);

	ClassDB::bind_method(D_METHOD("set_v_offset", "v_offset"), &PathFollow2D::set_v_offset);
	ClassDB::bind_method(D_METHOD("get_v_offset"), &PathFollow2D::get_v_offset);

	ClassDB::bind_method(D_METHOD("set_progress_ratio", "ratio"), &PathFollow2D::set_progress_ratio);
	ClassDB::bind_method(D_METHOD("get_progress_ratio"), &PathFollow2D::get_progress_ratio);

	ClassDB::bind_method(D_METHOD("set_rotates", "enabled"), &PathFollow2D::set_rotation_enabled);
	ClassDB::bind_method(D_METHOD("is_rotating"), &PathFollow2D::is_rotation_enabled);

	ClassDB::bind_method(D_METHOD("set_cubic_interpolation", "enabled"), &PathFollow2D::set_cubic_interpolation_enabled);
	ClassDB::bind_method(D_METHOD("get_cubic_interpolation"), &PathFollow2D::is_cubic_interpolation_enabled);

	ClassDB::bind_method(D_METHOD("set_loop", "loop"), &PathFollow2D::set_loop);
	ClassDB::bind_method(D_METHOD("has_loop"), &PathFollow2D::has_loop);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "progress", PROPERTY_HINT_RANGE, "0,10000,0.01,or_less,or_greater,suffix:px"), "set_progress", "get_progress");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "progress_ratio", PROPERTY_HINT_RANGE, "0,1,0.0001,or_less,or_greater", PROPERTY_USAGE_EDITOR), "set_progress_ratio", "get_progress_ratio");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "h_offset"), "set_h_offset", "get_h_offset");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "v_offset"), "set_v_offset", "get_v_offset");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "rotates"), "set_rotates", "is_rotating");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "cubic_interp"), "set_cubic_interpolation", "get_cubic_interpolation");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "loop"), "set_loop", "has_loop");
}

void PathFollow2D::set_progress(real_t p_progress) {
	ERR_FAIL_COND(!std::isfinite(p_progress));
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

		_update_transform();
	}
}

void PathFollow2D::set_h_offset(real_t p_h_offset) {
	h_offset = p_h_offset;
	if (path) {
		_update_transform();
	}
}

real_t PathFollow2D::get_h_offset() const {
	return h_offset;
}

void PathFollow2D::set_v_offset(real_t p_v_offset) {
	v_offset = p_v_offset;
	if (path) {
		_update_transform();
	}
}

real_t PathFollow2D::get_v_offset() const {
	return v_offset;
}

real_t PathFollow2D::get_progress() const {
	return progress;
}

void PathFollow2D::set_progress_ratio(real_t p_ratio) {
	ERR_FAIL_NULL_MSG(path, "Can only set progress ratio on a PathFollow2D that is the child of a Path2D which is itself part of the scene tree.");
	ERR_FAIL_COND_MSG(path->get_curve().is_null(), "Can't set progress ratio on a PathFollow2D that does not have a Curve.");
	ERR_FAIL_COND_MSG(!path->get_curve()->get_baked_length(), "Can't set progress ratio on a PathFollow2D that has a 0 length curve.");
	set_progress(p_ratio * path->get_curve()->get_baked_length());
}

real_t PathFollow2D::get_progress_ratio() const {
	if (path && path->get_curve().is_valid() && path->get_curve()->get_baked_length()) {
		return get_progress() / path->get_curve()->get_baked_length();
	} else {
		return 0;
	}
}

void PathFollow2D::set_rotation_enabled(bool p_enabled) {
	rotates = p_enabled;
	_update_transform();
}

bool PathFollow2D::is_rotation_enabled() const {
	return rotates;
}

void PathFollow2D::set_loop(bool p_loop) {
	loop = p_loop;
}

bool PathFollow2D::has_loop() const {
	return loop;
}
