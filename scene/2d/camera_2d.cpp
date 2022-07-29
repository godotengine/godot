/*************************************************************************/
/*  camera_2d.cpp                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "camera_2d.h"

#include "core/engine.h"
#include "core/math/math_funcs.h"
#include "scene/scene_string_names.h"
#include "servers/visual_server.h"

void Camera2D::_update_process_mode() {
	if (process_mode == CAMERA2D_PROCESS_IDLE) {
		set_process_internal(smoothing_active);
		set_physics_process_internal(false);
	} else {
		set_process_internal(false);
		set_physics_process_internal(smoothing_active);
	}
}

void Camera2D::_setup_viewport() {
	// Disconnect signal on previous viewport if there's one.
	if (viewport && viewport->is_connected("size_changed", this, "_update_size")) {
		viewport->disconnect("size_changed", this, "_update_size");
	}

	if (custom_viewport && ObjectDB::get_instance(custom_viewport_id)) {
		viewport = custom_viewport;
	} else {
		viewport = get_viewport();
	}

	RID vp = viewport->get_viewport_rid();
	group_name = "__cameras_" + itos(vp.get_id());
	canvas_group_name = "__cameras_c" + itos(canvas.get_id());
	add_to_group(group_name);
	add_to_group(canvas_group_name);

	viewport->connect("size_changed", this, "_update_size");
	_update_size();
}

Transform2D Camera2D::_get_camera_transform() {
	Transform2D xform;
	xform.scale_basis(zoom);
	Vector2 rotation_adjust;
	if (rotating) {
		real_t rotation = get_global_rotation();
		if (anchor_mode == ANCHOR_MODE_DRAG_CENTER) {
			rotation_adjust = screen_size * 0.5;
			rotation_adjust = rotation_adjust.rotated(rotation);
			rotation_adjust = screen_size * 0.5 - rotation_adjust;
		}
		xform.set_rotation(rotation);
	}
	xform.set_origin(current_position + rotation_adjust + offset);
	return xform;
}

void Camera2D::_update_viewport() {
	if (!is_inside_tree() || !viewport || !current) {
		return;
	}

	if (Engine::get_singleton()->is_editor_hint()) {
		update(); // Will just be drawn
		return;
	}

	if (!initialized) {
		_update_position();
	}

	ERR_FAIL_COND_MSG(custom_viewport && !ObjectDB::get_instance(custom_viewport_id),
			"Custom viewport does not match custom viewport id");
	Transform2D xform = _get_camera_transform().affine_inverse();
	viewport->set_canvas_transform(xform);
	Point2 screen_offset = (anchor_mode == ANCHOR_MODE_DRAG_CENTER ? (screen_size * 0.5) : Point2());
	get_tree()->call_group_flags(SceneTree::GROUP_CALL_REALTIME, group_name, "_camera_moved", xform, screen_offset);
}

void Camera2D::_update_size() {
	if (!is_inside_tree() || !viewport) {
		return;
	}

	ERR_FAIL_COND_MSG(custom_viewport && !ObjectDB::get_instance(custom_viewport_id),
			"Custom viewport does not match custom viewport id");

	if (anchor_mode == ANCHOR_MODE_DRAG_CENTER) {
		current_position += screen_size * 0.5;
		target_position += screen_size * 0.5;
	}
	screen_size = viewport->get_visible_rect().size * zoom;
	if (anchor_mode == ANCHOR_MODE_DRAG_CENTER) {
		current_position -= screen_size * 0.5;
		target_position -= screen_size * 0.5;
	}
	_update_position();
}

void Camera2D::_update_position() {
	if (!is_inside_tree()) {
		return;
	}

	target_position = get_global_position();
	if (!initialized) {
		current_position = target_position;
		if (anchor_mode == ANCHOR_MODE_DRAG_CENTER) {
			current_position -= screen_size * 0.5;
		}
		initialized = true;
	}

	if (anchor_mode == ANCHOR_MODE_DRAG_CENTER) {
		Vector2 half_screen_size = screen_size * 0.5;
		Point2 current_center = current_position + half_screen_size;
		if (h_offset_changed) {
			if (h_ofs > 0) {
				target_position.x += h_ofs * drag_margin[MARGIN_RIGHT] * half_screen_size.x;
			} else { // h_ofs <= 0
				target_position.x += h_ofs * drag_margin[MARGIN_LEFT] * half_screen_size.x;
			}
			h_offset_changed = false;
		} else if (h_drag_enabled && !Engine::get_singleton()->is_editor_hint()) {
			if (target_position.x > current_center.x + drag_margin[MARGIN_RIGHT] * half_screen_size.x) {
				target_position.x += drag_margin[MARGIN_RIGHT] * half_screen_size.x;
			} else if (target_position.x < current_center.x - drag_margin[MARGIN_LEFT] * half_screen_size.x) {
				target_position.x -= drag_margin[MARGIN_LEFT] * half_screen_size.x;
			} else {
				target_position.x = current_center.x;
			}
		}
		if (v_offset_changed) {
			if (v_ofs > 0) {
				target_position.y += v_ofs * drag_margin[MARGIN_BOTTOM] * half_screen_size.y;
			} else { // v_ofs <= 0
				target_position.y += v_ofs * drag_margin[MARGIN_TOP] * half_screen_size.y;
			}
			v_offset_changed = false;
		} else if (v_drag_enabled && !Engine::get_singleton()->is_editor_hint()) {
			if (target_position.y > current_center.y + drag_margin[MARGIN_BOTTOM] * half_screen_size.y) {
				target_position.y += drag_margin[MARGIN_BOTTOM] * half_screen_size.y;
			} else if (target_position.y < current_center.y - drag_margin[MARGIN_TOP] * half_screen_size.y) {
				target_position.y -= drag_margin[MARGIN_TOP] * half_screen_size.y;
			} else {
				target_position.y = current_center.y;
			}
		}
		target_position -= half_screen_size;
	}

	Vector2 limit_adjust;
	if (target_position.x < limit[MARGIN_LEFT]) {
		limit_adjust.x += limit[MARGIN_LEFT] - target_position.x;
	}
	if (target_position.x > limit[MARGIN_RIGHT] - screen_size.x) {
		limit_adjust.x += limit[MARGIN_RIGHT] - screen_size.x - target_position.x;
	}
	if (target_position.y < limit[MARGIN_TOP]) {
		limit_adjust.y += limit[MARGIN_TOP] - target_position.y;
	}
	if (target_position.y > limit[MARGIN_BOTTOM] - screen_size.y) {
		limit_adjust.y += limit[MARGIN_BOTTOM] - screen_size.y - target_position.y;
	}
	target_position += limit_adjust;

	if (!smoothing_active) {
		current_position = target_position;
		_update_viewport();
	}
}

// Must only be called by internal process notification!
void Camera2D::_update_scroll() {
	if (!initialized) {
		_update_position();
	}

	if (!smoothing_active) {
		current_position = target_position;
		_update_viewport();
		return;
	}

	if (!limit_smoothing_enabled) {
		if (current_position.x < limit[MARGIN_LEFT] && current_position.x < target_position.x) {
			if (target_position.x < limit[MARGIN_LEFT]) {
				current_position.x = target_position.x;
			} else {
				current_position.x = limit[MARGIN_LEFT];
			}
		}
		if (current_position.x > limit[MARGIN_RIGHT] - screen_size.x && current_position.x > target_position.x) {
			if (target_position.x > limit[MARGIN_RIGHT] - screen_size.x) {
				current_position.x = target_position.x;
			} else {
				current_position.x = limit[MARGIN_RIGHT] - screen_size.x;
			}
		}
		if (current_position.y < limit[MARGIN_TOP] && current_position.y < target_position.y) {
			if (target_position.y < limit[MARGIN_TOP]) {
				current_position.y = target_position.y;
			} else {
				current_position.y = limit[MARGIN_TOP];
			}
		}
		if (current_position.y > limit[MARGIN_BOTTOM] - screen_size.y && current_position.y > target_position.y) {
			if (target_position.y > limit[MARGIN_BOTTOM] - screen_size.y) {
				current_position.y = target_position.y;
			} else {
				current_position.y = limit[MARGIN_BOTTOM] - screen_size.y;
			}
		}
	}

	Vector2 difference = target_position - current_position;
	real_t distance = difference.length();
	Vector2 direction = difference.normalized();
	real_t time_step = process_mode == CAMERA2D_PROCESS_PHYSICS ? get_physics_process_delta_time() : get_process_delta_time();
	real_t step_size = distance * smoothing_speed * time_step;
	if (step_size > distance) {
		step_size = distance;
	}
	current_position += direction * step_size;
	_update_viewport();
}

void Camera2D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_INTERNAL_PROCESS:
		case NOTIFICATION_INTERNAL_PHYSICS_PROCESS: {
			_update_scroll();
		} break;

		case NOTIFICATION_TRANSFORM_CHANGED: {
			_update_position();
		} break;

		case NOTIFICATION_ENTER_TREE: {
			ERR_FAIL_COND(!is_inside_tree());
			initialized = false;
			canvas = get_canvas();
			_setup_viewport();
			_update_process_mode();
			_set_current(current);
		} break;

		case NOTIFICATION_EXIT_TREE: {
			const bool viewport_valid = !custom_viewport || ObjectDB::get_instance(custom_viewport_id);
			if (viewport && viewport_valid) {
				viewport->disconnect("size_changed", this, "_update_size");
				if (current) {
					viewport->set_canvas_transform(Transform2D());
				}
			}
			remove_from_group(group_name);
			remove_from_group(canvas_group_name);
			viewport = nullptr;
		} break;

#ifdef TOOLS_ENABLED
		case NOTIFICATION_DRAW: {
			if (!is_inside_tree() || !Engine::get_singleton()->is_editor_hint()) {
				break;
			}

			if (screen_drawing_enabled) {
				Color area_axis_color(1, 0.4, 1, 0.63);
				float area_axis_width = 1;
				if (current) {
					area_axis_width = 3;
				}

				Transform2D camera_transform = _get_camera_transform();
				Vector2 screen_endpoints[4] = {
					camera_transform.xform(Vector2(0, 0)),
					camera_transform.xform(Vector2(screen_size.width, 0)),
					camera_transform.xform(Vector2(screen_size.width, screen_size.height)),
					camera_transform.xform(Vector2(0, screen_size.height))
				};

				Transform2D inv_transform = get_global_transform().affine_inverse(); // undo global space

				for (int i = 0; i < 4; i++) {
					draw_line(inv_transform.xform(screen_endpoints[i]), inv_transform.xform(screen_endpoints[(i + 1) % 4]), area_axis_color, area_axis_width);
				}
			}

			if (limit_drawing_enabled) {
				Color limit_drawing_color(1, 1, 0.25, 0.63);
				float limit_drawing_width = 1;
				if (current) {
					limit_drawing_width = 3;
				}

				Vector2 camera_origin = get_global_transform().get_origin();
				Vector2 camera_scale = get_global_transform().get_scale().abs();
				Vector2 limit_points[4] = {
					(Vector2(limit[MARGIN_LEFT], limit[MARGIN_TOP]) - camera_origin) / camera_scale,
					(Vector2(limit[MARGIN_RIGHT], limit[MARGIN_TOP]) - camera_origin) / camera_scale,
					(Vector2(limit[MARGIN_RIGHT], limit[MARGIN_BOTTOM]) - camera_origin) / camera_scale,
					(Vector2(limit[MARGIN_LEFT], limit[MARGIN_BOTTOM]) - camera_origin) / camera_scale
				};

				for (int i = 0; i < 4; i++) {
					draw_line(limit_points[i], limit_points[(i + 1) % 4], limit_drawing_color, limit_drawing_width);
				}
			}

			if (margin_drawing_enabled) {
				Color margin_drawing_color(0.25, 1, 1, 0.63);
				float margin_drawing_width = 1;
				if (current) {
					margin_drawing_width = 3;
				}

				Transform2D camera_transform = _get_camera_transform();
				Vector2 margin_endpoints[4] = {
					camera_transform.xform(Vector2((screen_size.width / 2) - ((screen_size.width / 2) * drag_margin[MARGIN_LEFT]), (screen_size.height / 2) - ((screen_size.height / 2) * drag_margin[MARGIN_TOP]))),
					camera_transform.xform(Vector2((screen_size.width / 2) + ((screen_size.width / 2) * drag_margin[MARGIN_RIGHT]), (screen_size.height / 2) - ((screen_size.height / 2) * drag_margin[MARGIN_TOP]))),
					camera_transform.xform(Vector2((screen_size.width / 2) + ((screen_size.width / 2) * drag_margin[MARGIN_RIGHT]), (screen_size.height / 2) + ((screen_size.height / 2) * drag_margin[MARGIN_BOTTOM]))),
					camera_transform.xform(Vector2((screen_size.width / 2) - ((screen_size.width / 2) * drag_margin[MARGIN_LEFT]), (screen_size.height / 2) + ((screen_size.height / 2) * drag_margin[MARGIN_BOTTOM])))
				};

				Transform2D inv_transform = get_global_transform().affine_inverse(); // undo global space

				for (int i = 0; i < 4; i++) {
					draw_line(inv_transform.xform(margin_endpoints[i]), inv_transform.xform(margin_endpoints[(i + 1) % 4]), margin_drawing_color, margin_drawing_width);
				}
			}
		} break;
#endif
	}
}

void Camera2D::set_offset(const Vector2 &p_offset) {
	offset = p_offset;
	_update_viewport();
}

Vector2 Camera2D::get_offset() const {
	return offset;
}

void Camera2D::set_anchor_mode(AnchorMode p_anchor_mode) {
	anchor_mode = p_anchor_mode;
	_update_position();
}

Camera2D::AnchorMode Camera2D::get_anchor_mode() const {
	return anchor_mode;
}

void Camera2D::set_rotating(bool p_rotating) {
	rotating = p_rotating;
	_update_viewport();
}

bool Camera2D::is_rotating() const {
	return rotating;
}

void Camera2D::set_process_mode(Camera2DProcessMode p_mode) {
	if (process_mode == p_mode) {
		return;
	}
	process_mode = p_mode;
	_update_process_mode();
}

Camera2D::Camera2DProcessMode Camera2D::get_process_mode() const {
	return process_mode;
}

void Camera2D::_clear_current() {
	current = false;
	update();
}

void Camera2D::_set_current(bool p_current) {
	if (p_current && is_inside_tree()) {
		get_tree()->call_group_flags(SceneTree::GROUP_CALL_REALTIME, group_name, "_clear_current");
	}
	current = p_current;
	_update_position();
	update();
}

void Camera2D::make_current() {
	_set_current(true);
}

void Camera2D::clear_current() {
	_set_current(false);
}

bool Camera2D::is_current() const {
	return current;
}

void Camera2D::set_limit(Margin p_margin, int p_limit) {
	ERR_FAIL_INDEX((int)p_margin, 4);
	limit[p_margin] = p_limit;
	_update_position();
}

int Camera2D::get_limit(Margin p_margin) const {
	ERR_FAIL_INDEX_V((int)p_margin, 4, 0);
	return limit[p_margin];
}

void Camera2D::set_limit_smoothing_enabled(bool enable) {
	limit_smoothing_enabled = enable;
	_update_position();
}

bool Camera2D::is_limit_smoothing_enabled() const {
	return limit_smoothing_enabled;
}

void Camera2D::set_drag_margin(Margin p_margin, float p_drag_margin) {
	ERR_FAIL_INDEX((int)p_margin, 4);
	drag_margin[p_margin] = p_drag_margin;
	_update_position();
}

float Camera2D::get_drag_margin(Margin p_margin) const {
	ERR_FAIL_INDEX_V((int)p_margin, 4, 0);
	return drag_margin[p_margin];
}

Vector2 Camera2D::get_camera_position() const {
	if (anchor_mode == ANCHOR_MODE_DRAG_CENTER) {
		return target_position + screen_size * 0.5;
	}
	return target_position;
}

void Camera2D::force_update_scroll() {
	// No longer does anything. Kept for backwards compatibility.
}

void Camera2D::reset_smoothing() {
	_update_position();
	current_position = target_position;
}

void Camera2D::align() {
	h_offset_changed = true;
	v_offset_changed = true;
	reset_smoothing();
}

void Camera2D::set_smoothing_speed(float p_smoothing_speed) {
	smoothing_speed = p_smoothing_speed;
}

float Camera2D::get_smoothing_speed() const {
	return smoothing_speed;
}

Point2 Camera2D::get_camera_screen_center() const {
	return current_position + screen_size * 0.5;
}

void Camera2D::set_h_drag_enabled(bool p_enabled) {
	h_drag_enabled = p_enabled;
}

bool Camera2D::is_h_drag_enabled() const {
	return h_drag_enabled;
}

void Camera2D::set_v_drag_enabled(bool p_enabled) {
	v_drag_enabled = p_enabled;
}

bool Camera2D::is_v_drag_enabled() const {
	return v_drag_enabled;
}

void Camera2D::set_v_offset(float p_offset) {
	v_ofs = p_offset;
	v_offset_changed = true;
	_update_position();
}

float Camera2D::get_v_offset() const {
	return v_ofs;
}

void Camera2D::set_h_offset(float p_offset) {
	h_ofs = p_offset;
	h_offset_changed = true;
	_update_position();
}

float Camera2D::get_h_offset() const {
	return h_ofs;
}

void Camera2D::set_enable_follow_smoothing(bool p_enabled) {
	if (smoothing_enabled == p_enabled) {
		return;
	}

	// Separate the logic between enabled and active, because the smoothing
	// cannot be active in the editor. This can be done without a separate flag
	// but is bug prone so this approach is easier to follow.
	smoothing_enabled = p_enabled;
	smoothing_active = smoothing_enabled && !Engine::get_singleton()->is_editor_hint();
	// Keep the processing up to date after each change.
	_update_process_mode();
}

bool Camera2D::is_follow_smoothing_enabled() const {
	return smoothing_enabled;
}

void Camera2D::set_zoom(const Vector2 &p_zoom) {
	// Setting zoom to zero causes 'affine_invert' issues.
	ERR_FAIL_COND_MSG(Math::is_zero_approx(p_zoom.x) || Math::is_zero_approx(p_zoom.y),
			"Zoom level must be different from 0 (can be negative).");
	zoom = p_zoom;
	_update_size();
};

Vector2 Camera2D::get_zoom() const {
	return zoom;
};

void Camera2D::set_custom_viewport(Node *p_viewport) {
	ERR_FAIL_NULL(p_viewport);
	if (is_inside_tree()) {
		remove_from_group(group_name);
		remove_from_group(canvas_group_name);
	}

	if (custom_viewport && !ObjectDB::get_instance(custom_viewport_id)) {
		viewport = nullptr;
	}

	custom_viewport = Object::cast_to<Viewport>(p_viewport);

	if (custom_viewport) {
		custom_viewport_id = custom_viewport->get_instance_id();
	} else {
		custom_viewport_id = 0;
	}

	if (is_inside_tree()) {
		_setup_viewport();
	}
}

Node *Camera2D::get_custom_viewport() const {
	return custom_viewport;
}

void Camera2D::set_screen_drawing_enabled(bool enable) {
	screen_drawing_enabled = enable;
#ifdef TOOLS_ENABLED
	update();
#endif
}

bool Camera2D::is_screen_drawing_enabled() const {
	return screen_drawing_enabled;
}

void Camera2D::set_limit_drawing_enabled(bool enable) {
	limit_drawing_enabled = enable;
#ifdef TOOLS_ENABLED
	update();
#endif
}

bool Camera2D::is_limit_drawing_enabled() const {
	return limit_drawing_enabled;
}

void Camera2D::set_margin_drawing_enabled(bool enable) {
	margin_drawing_enabled = enable;
#ifdef TOOLS_ENABLED
	update();
#endif
}

bool Camera2D::is_margin_drawing_enabled() const {
	return margin_drawing_enabled;
}

void Camera2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_update_size"), &Camera2D::_update_size);

	ClassDB::bind_method(D_METHOD("set_offset", "offset"), &Camera2D::set_offset);
	ClassDB::bind_method(D_METHOD("get_offset"), &Camera2D::get_offset);

	ClassDB::bind_method(D_METHOD("set_anchor_mode", "anchor_mode"), &Camera2D::set_anchor_mode);
	ClassDB::bind_method(D_METHOD("get_anchor_mode"), &Camera2D::get_anchor_mode);

	ClassDB::bind_method(D_METHOD("set_rotating", "rotating"), &Camera2D::set_rotating);
	ClassDB::bind_method(D_METHOD("is_rotating"), &Camera2D::is_rotating);

	ClassDB::bind_method(D_METHOD("set_process_mode", "mode"), &Camera2D::set_process_mode);
	ClassDB::bind_method(D_METHOD("get_process_mode"), &Camera2D::get_process_mode);

	ClassDB::bind_method(D_METHOD("_clear_current"), &Camera2D::_clear_current);
	ClassDB::bind_method(D_METHOD("_set_current", "current"), &Camera2D::_set_current);
	ClassDB::bind_method(D_METHOD("make_current"), &Camera2D::make_current);
	ClassDB::bind_method(D_METHOD("clear_current"), &Camera2D::clear_current);
	ClassDB::bind_method(D_METHOD("is_current"), &Camera2D::is_current);

	ClassDB::bind_method(D_METHOD("set_limit", "margin", "limit"), &Camera2D::set_limit);
	ClassDB::bind_method(D_METHOD("get_limit", "margin"), &Camera2D::get_limit);

	ClassDB::bind_method(D_METHOD("set_limit_smoothing_enabled", "limit_smoothing_enabled"), &Camera2D::set_limit_smoothing_enabled);
	ClassDB::bind_method(D_METHOD("is_limit_smoothing_enabled"), &Camera2D::is_limit_smoothing_enabled);

	ClassDB::bind_method(D_METHOD("set_v_drag_enabled", "enabled"), &Camera2D::set_v_drag_enabled);
	ClassDB::bind_method(D_METHOD("is_v_drag_enabled"), &Camera2D::is_v_drag_enabled);

	ClassDB::bind_method(D_METHOD("set_h_drag_enabled", "enabled"), &Camera2D::set_h_drag_enabled);
	ClassDB::bind_method(D_METHOD("is_h_drag_enabled"), &Camera2D::is_h_drag_enabled);

	ClassDB::bind_method(D_METHOD("set_v_offset", "ofs"), &Camera2D::set_v_offset);
	ClassDB::bind_method(D_METHOD("get_v_offset"), &Camera2D::get_v_offset);

	ClassDB::bind_method(D_METHOD("set_h_offset", "ofs"), &Camera2D::set_h_offset);
	ClassDB::bind_method(D_METHOD("get_h_offset"), &Camera2D::get_h_offset);

	ClassDB::bind_method(D_METHOD("set_drag_margin", "margin", "drag_margin"), &Camera2D::set_drag_margin);
	ClassDB::bind_method(D_METHOD("get_drag_margin", "margin"), &Camera2D::get_drag_margin);

	ClassDB::bind_method(D_METHOD("get_camera_position"), &Camera2D::get_camera_position);
	ClassDB::bind_method(D_METHOD("get_camera_screen_center"), &Camera2D::get_camera_screen_center);

	ClassDB::bind_method(D_METHOD("set_zoom", "zoom"), &Camera2D::set_zoom);
	ClassDB::bind_method(D_METHOD("get_zoom"), &Camera2D::get_zoom);

	ClassDB::bind_method(D_METHOD("set_custom_viewport", "viewport"), &Camera2D::set_custom_viewport);
	ClassDB::bind_method(D_METHOD("get_custom_viewport"), &Camera2D::get_custom_viewport);

	ClassDB::bind_method(D_METHOD("set_follow_smoothing", "smoothing_speed"), &Camera2D::set_smoothing_speed);
	ClassDB::bind_method(D_METHOD("get_follow_smoothing"), &Camera2D::get_smoothing_speed);

	ClassDB::bind_method(D_METHOD("set_enable_follow_smoothing", "follow_smoothing"), &Camera2D::set_enable_follow_smoothing);
	ClassDB::bind_method(D_METHOD("is_follow_smoothing_enabled"), &Camera2D::is_follow_smoothing_enabled);

	ClassDB::bind_method(D_METHOD("force_update_scroll"), &Camera2D::force_update_scroll);
	ClassDB::bind_method(D_METHOD("reset_smoothing"), &Camera2D::reset_smoothing);
	ClassDB::bind_method(D_METHOD("align"), &Camera2D::align);

	ClassDB::bind_method(D_METHOD("set_screen_drawing_enabled", "screen_drawing_enabled"), &Camera2D::set_screen_drawing_enabled);
	ClassDB::bind_method(D_METHOD("is_screen_drawing_enabled"), &Camera2D::is_screen_drawing_enabled);

	ClassDB::bind_method(D_METHOD("set_limit_drawing_enabled", "limit_drawing_enabled"), &Camera2D::set_limit_drawing_enabled);
	ClassDB::bind_method(D_METHOD("is_limit_drawing_enabled"), &Camera2D::is_limit_drawing_enabled);

	ClassDB::bind_method(D_METHOD("set_margin_drawing_enabled", "margin_drawing_enabled"), &Camera2D::set_margin_drawing_enabled);
	ClassDB::bind_method(D_METHOD("is_margin_drawing_enabled"), &Camera2D::is_margin_drawing_enabled);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "offset"), "set_offset", "get_offset");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "anchor_mode", PROPERTY_HINT_ENUM, "Fixed TopLeft,Drag Center"), "set_anchor_mode", "get_anchor_mode");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "rotating"), "set_rotating", "is_rotating");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "current"), "_set_current", "is_current");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "zoom"), "set_zoom", "get_zoom");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "custom_viewport", PROPERTY_HINT_RESOURCE_TYPE, "Viewport", 0), "set_custom_viewport", "get_custom_viewport");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "process_mode", PROPERTY_HINT_ENUM, "Physics,Idle"), "set_process_mode", "get_process_mode");

	ADD_GROUP("Limit", "limit_");
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "limit_left"), "set_limit", "get_limit", MARGIN_LEFT);
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "limit_top"), "set_limit", "get_limit", MARGIN_TOP);
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "limit_right"), "set_limit", "get_limit", MARGIN_RIGHT);
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "limit_bottom"), "set_limit", "get_limit", MARGIN_BOTTOM);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "limit_smoothed"), "set_limit_smoothing_enabled", "is_limit_smoothing_enabled");

	ADD_GROUP("Draw Margin", "draw_margin_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "drag_margin_h_enabled"), "set_h_drag_enabled", "is_h_drag_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "drag_margin_v_enabled"), "set_v_drag_enabled", "is_v_drag_enabled");

	ADD_GROUP("Smoothing", "smoothing_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "smoothing_enabled"), "set_enable_follow_smoothing", "is_follow_smoothing_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "smoothing_speed", PROPERTY_HINT_RANGE, "0,60,1"), "set_follow_smoothing", "get_follow_smoothing");

	ADD_GROUP("Offset", "offset_");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "offset_h", PROPERTY_HINT_RANGE, "-1,1,0.01"), "set_h_offset", "get_h_offset");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "offset_v", PROPERTY_HINT_RANGE, "-1,1,0.01"), "set_v_offset", "get_v_offset");

	ADD_GROUP("Drag Margin", "drag_margin_");
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "drag_margin_left", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_drag_margin", "get_drag_margin", MARGIN_LEFT);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "drag_margin_top", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_drag_margin", "get_drag_margin", MARGIN_TOP);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "drag_margin_right", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_drag_margin", "get_drag_margin", MARGIN_RIGHT);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "drag_margin_bottom", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_drag_margin", "get_drag_margin", MARGIN_BOTTOM);

	ADD_GROUP("Editor", "editor_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "editor_draw_screen"), "set_screen_drawing_enabled", "is_screen_drawing_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "editor_draw_limits"), "set_limit_drawing_enabled", "is_limit_drawing_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "editor_draw_drag_margin"), "set_margin_drawing_enabled", "is_margin_drawing_enabled");

	BIND_ENUM_CONSTANT(ANCHOR_MODE_FIXED_TOP_LEFT);
	BIND_ENUM_CONSTANT(ANCHOR_MODE_DRAG_CENTER);
	BIND_ENUM_CONSTANT(CAMERA2D_PROCESS_PHYSICS);
	BIND_ENUM_CONSTANT(CAMERA2D_PROCESS_IDLE);
}

Camera2D::Camera2D() {
	anchor_mode = ANCHOR_MODE_DRAG_CENTER;
	rotating = false;
	current = false;
	limit[MARGIN_LEFT] = -10000000;
	limit[MARGIN_TOP] = -10000000;
	limit[MARGIN_RIGHT] = 10000000;
	limit[MARGIN_BOTTOM] = 10000000;

	drag_margin[MARGIN_LEFT] = 0.2;
	drag_margin[MARGIN_TOP] = 0.2;
	drag_margin[MARGIN_RIGHT] = 0.2;
	drag_margin[MARGIN_BOTTOM] = 0.2;
	smoothing_enabled = false;
	smoothing_active = false;
	limit_smoothing_enabled = false;
	initialized = false;

	viewport = nullptr;
	custom_viewport = nullptr;
	custom_viewport_id = 0;
	process_mode = CAMERA2D_PROCESS_IDLE;

	smoothing_speed = 5.0;
	zoom = Vector2(1, 1);

	screen_drawing_enabled = true;
	limit_drawing_enabled = false;
	margin_drawing_enabled = false;

	h_drag_enabled = false;
	v_drag_enabled = false;
	h_ofs = 0;
	v_ofs = 0;
	h_offset_changed = false;
	v_offset_changed = false;

	set_notify_transform(true);
}
