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

#include "core/config/project_settings.h"
#include "scene/main/window.h"

void Camera2D::_update_process_callback() {
	if (process_callback == CAMERA2D_PROCESS_IDLE) {
		set_process_internal(smoothing_active);
		set_physics_process_internal(false);
	} else {
		set_process_internal(false);
		set_physics_process_internal(smoothing_active);
	}
}

void Camera2D::_setup_viewport() {
	// Disconnect signal on previous viewport if there's one.
	Callable update_size = callable_mp(this, &Camera2D::_update_size);
	if (viewport && viewport->is_connected("size_changed", update_size)) {
		viewport->disconnect("size_changed", update_size);
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

	viewport->connect("size_changed", update_size);
	_update_size();
}

Transform2D Camera2D::_get_camera_transform() {
	Transform2D xform;
	xform.scale_basis(zoom_scale);
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
		queue_redraw(); // Will just be drawn.
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
	get_tree()->call_group(group_name, "_camera_moved", xform, screen_offset);
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
	screen_size = viewport->get_visible_rect().size * zoom_scale;
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
		if (drag_horizontal_offset_changed) {
			if (drag_horizontal_offset > 0) {
				target_position.x += drag_horizontal_offset * drag_margin[SIDE_RIGHT] * half_screen_size.x;
			} else { // drag_horizontal_offset <= 0
				target_position.x += drag_horizontal_offset * drag_margin[SIDE_LEFT] * half_screen_size.x;
			}
			drag_horizontal_offset_changed = false;
		} else if (drag_horizontal_enabled && !Engine::get_singleton()->is_editor_hint()) {
			if (target_position.x > current_center.x + drag_margin[SIDE_RIGHT] * half_screen_size.x) {
				target_position.x += drag_margin[SIDE_RIGHT] * half_screen_size.x;
			} else if (target_position.x < current_center.x - drag_margin[SIDE_LEFT] * half_screen_size.x) {
				target_position.x -= drag_margin[SIDE_LEFT] * half_screen_size.x;
			} else {
				target_position.x = current_center.x;
			}
		}
		if (drag_vertical_offset_changed) {
			if (drag_vertical_offset > 0) {
				target_position.y += drag_vertical_offset * drag_margin[SIDE_BOTTOM] * half_screen_size.y;
			} else { // drag_vertical_offset <= 0
				target_position.y += drag_vertical_offset * drag_margin[SIDE_TOP] * half_screen_size.y;
			}
			drag_vertical_offset_changed = false;
		} else if (drag_vertical_enabled && !Engine::get_singleton()->is_editor_hint()) {
			if (target_position.y > current_center.y + drag_margin[SIDE_BOTTOM] * half_screen_size.y) {
				target_position.y += drag_margin[SIDE_BOTTOM] * half_screen_size.y;
			} else if (target_position.y < current_center.y - drag_margin[SIDE_TOP] * half_screen_size.y) {
				target_position.y -= drag_margin[SIDE_TOP] * half_screen_size.y;
			} else {
				target_position.y = current_center.y;
			}
		}
		target_position -= half_screen_size;
	}

	Vector2 limit_adjust;
	if (target_position.x < limit[SIDE_LEFT]) {
		limit_adjust.x += limit[SIDE_LEFT] - target_position.x;
	}
	if (target_position.x > limit[SIDE_RIGHT] - screen_size.x) {
		limit_adjust.x += limit[SIDE_RIGHT] - screen_size.x - target_position.x;
	}
	if (target_position.y < limit[SIDE_TOP]) {
		limit_adjust.y += limit[SIDE_TOP] - target_position.y;
	}
	if (target_position.y > limit[SIDE_BOTTOM] - screen_size.y) {
		limit_adjust.y += limit[SIDE_BOTTOM] - screen_size.y - target_position.y;
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
		if (current_position.x < limit[SIDE_LEFT] && current_position.x < target_position.x) {
			if (target_position.x < limit[SIDE_LEFT]) {
				current_position.x = target_position.x;
			} else {
				current_position.x = limit[SIDE_LEFT];
			}
		}
		if (current_position.x > limit[SIDE_RIGHT] - screen_size.x && current_position.x > target_position.x) {
			if (target_position.x > limit[SIDE_RIGHT] - screen_size.x) {
				current_position.x = target_position.x;
			} else {
				current_position.x = limit[SIDE_RIGHT] - screen_size.x;
			}
		}
		if (current_position.y < limit[SIDE_TOP] && current_position.y < target_position.y) {
			if (target_position.y < limit[SIDE_TOP]) {
				current_position.y = target_position.y;
			} else {
				current_position.y = limit[SIDE_TOP];
			}
		}
		if (current_position.y > limit[SIDE_BOTTOM] - screen_size.y && current_position.y > target_position.y) {
			if (target_position.y > limit[SIDE_BOTTOM] - screen_size.y) {
				current_position.y = target_position.y;
			} else {
				current_position.y = limit[SIDE_BOTTOM] - screen_size.y;
			}
		}
	}

	Vector2 difference = target_position - current_position;
	real_t distance = difference.length();
	Vector2 direction = difference.normalized();
	real_t time_step = process_callback == CAMERA2D_PROCESS_PHYSICS ? get_physics_process_delta_time() : get_process_delta_time();
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
			_update_process_callback();
			set_current(current);
		} break;

		case NOTIFICATION_EXIT_TREE: {
			const bool viewport_valid = !custom_viewport || ObjectDB::get_instance(custom_viewport_id);
			if (viewport && viewport_valid) {
				viewport->disconnect("size_changed", callable_mp(this, &Camera2D::_update_size));
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
				real_t area_axis_width = 1;
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
				real_t limit_drawing_width = 1;
				if (current) {
					limit_drawing_width = 3;
				}

				Vector2 camera_origin = get_global_position();
				Vector2 camera_scale = get_global_scale().abs();
				Vector2 limit_points[4] = {
					(Vector2(limit[SIDE_LEFT], limit[SIDE_TOP]) - camera_origin) / camera_scale,
					(Vector2(limit[SIDE_RIGHT], limit[SIDE_TOP]) - camera_origin) / camera_scale,
					(Vector2(limit[SIDE_RIGHT], limit[SIDE_BOTTOM]) - camera_origin) / camera_scale,
					(Vector2(limit[SIDE_LEFT], limit[SIDE_BOTTOM]) - camera_origin) / camera_scale
				};

				for (int i = 0; i < 4; i++) {
					draw_line(limit_points[i], limit_points[(i + 1) % 4], limit_drawing_color, limit_drawing_width);
				}
			}

			if (margin_drawing_enabled) {
				Color margin_drawing_color(0.25, 1, 1, 0.63);
				real_t margin_drawing_width = 1;
				if (current) {
					margin_drawing_width = 3;
				}

				Transform2D camera_transform = _get_camera_transform();
				Vector2 margin_endpoints[4] = {
					camera_transform.xform(Vector2((screen_size.width / 2) - ((screen_size.width / 2) * drag_margin[SIDE_LEFT]), (screen_size.height / 2) - ((screen_size.height / 2) * drag_margin[SIDE_TOP]))),
					camera_transform.xform(Vector2((screen_size.width / 2) + ((screen_size.width / 2) * drag_margin[SIDE_RIGHT]), (screen_size.height / 2) - ((screen_size.height / 2) * drag_margin[SIDE_TOP]))),
					camera_transform.xform(Vector2((screen_size.width / 2) + ((screen_size.width / 2) * drag_margin[SIDE_RIGHT]), (screen_size.height / 2) + ((screen_size.height / 2) * drag_margin[SIDE_BOTTOM]))),
					camera_transform.xform(Vector2((screen_size.width / 2) - ((screen_size.width / 2) * drag_margin[SIDE_LEFT]), (screen_size.height / 2) + ((screen_size.height / 2) * drag_margin[SIDE_BOTTOM])))
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

void Camera2D::set_process_callback(Camera2DProcessCallback p_mode) {
	if (process_callback == p_mode) {
		return;
	}
	process_callback = p_mode;
	_update_process_callback();
}

Camera2D::Camera2DProcessCallback Camera2D::get_process_callback() const {
	return process_callback;
}

void Camera2D::_clear_current() {
	current = false;
	queue_redraw();
}

void Camera2D::set_current(bool p_current) {
	if (is_inside_tree()) {
		if (p_current) {
			get_tree()->call_group(group_name, "_clear_current");
			viewport->_camera_2d_set(this);
		} else {
			viewport->_camera_2d_set(nullptr);
		}
	}
	current = p_current;
	queue_redraw();
}

bool Camera2D::is_current() const {
	return current;
}

void Camera2D::set_limit(Side p_side, int p_limit) {
	ERR_FAIL_INDEX((int)p_side, 4);
	limit[p_side] = p_limit;
	_update_position();
}

int Camera2D::get_limit(Side p_side) const {
	ERR_FAIL_INDEX_V((int)p_side, 4, 0);
	return limit[p_side];
}

void Camera2D::set_limit_smoothing_enabled(bool enable) {
	limit_smoothing_enabled = enable;
	_update_position();
}

bool Camera2D::is_limit_smoothing_enabled() const {
	return limit_smoothing_enabled;
}

void Camera2D::set_drag_margin(Side p_side, real_t p_drag_margin) {
	ERR_FAIL_INDEX((int)p_side, 4);
	drag_margin[p_side] = p_drag_margin;
	_update_position();
}

real_t Camera2D::get_drag_margin(Side p_side) const {
	ERR_FAIL_INDEX_V((int)p_side, 4, 0);
	return drag_margin[p_side];
}

Point2 Camera2D::get_camera_position() const {
	if (anchor_mode == ANCHOR_MODE_DRAG_CENTER) {
		return target_position + screen_size * 0.5;
	}
	return target_position;
}

void Camera2D::reset_smoothing() {
	_update_position();
	current_position = target_position;
}

void Camera2D::align() {
	drag_horizontal_offset_changed = true;
	drag_vertical_offset_changed = true;
	reset_smoothing();
}

void Camera2D::set_smoothing_speed(real_t p_smoothing_speed) {
	smoothing_speed = p_smoothing_speed;
}

real_t Camera2D::get_smoothing_speed() const {
	return smoothing_speed;
}

Point2 Camera2D::get_camera_screen_center() const {
	return current_position + screen_size * 0.5;
}

void Camera2D::set_drag_horizontal_enabled(bool p_enabled) {
	drag_horizontal_enabled = p_enabled;
}

bool Camera2D::is_drag_horizontal_enabled() const {
	return drag_horizontal_enabled;
}

void Camera2D::set_drag_vertical_enabled(bool p_enabled) {
	drag_vertical_enabled = p_enabled;
}

bool Camera2D::is_drag_vertical_enabled() const {
	return drag_vertical_enabled;
}

void Camera2D::set_drag_vertical_offset(real_t p_offset) {
	drag_vertical_offset = p_offset;
	drag_vertical_offset_changed = true;
	_update_position();
}

real_t Camera2D::get_drag_vertical_offset() const {
	return drag_vertical_offset;
}

void Camera2D::set_drag_horizontal_offset(real_t p_offset) {
	drag_horizontal_offset = p_offset;
	drag_horizontal_offset_changed = true;
	_update_position();
}

real_t Camera2D::get_drag_horizontal_offset() const {
	return drag_horizontal_offset;
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
	_update_process_callback();

	notify_property_list_changed();
}

bool Camera2D::is_follow_smoothing_enabled() const {
	return smoothing_enabled;
}

void Camera2D::set_zoom(const Vector2 &p_zoom) {
	ERR_FAIL_COND_MSG(Math::is_zero_approx(p_zoom.x) || Math::is_zero_approx(p_zoom.y),
			"Zoom level must be different from 0 (can be negative).");

	zoom = p_zoom;
	zoom_scale = Vector2(1, 1) / zoom;
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

	custom_viewport = Object::cast_to<Viewport>(p_viewport);

	if (custom_viewport) {
		custom_viewport_id = custom_viewport->get_instance_id();
	} else {
		custom_viewport_id = ObjectID();
	}

	if (is_inside_tree()) {
		if (custom_viewport) {
			viewport = custom_viewport;
		} else {
			viewport = get_viewport();
		}

		RID vp = viewport->get_viewport_rid();
		group_name = "__cameras_" + itos(vp.get_id());
		canvas_group_name = "__cameras_c" + itos(canvas.get_id());
		add_to_group(group_name);
		add_to_group(canvas_group_name);
	}
}

Node *Camera2D::get_custom_viewport() const {
	return custom_viewport;
}

void Camera2D::set_screen_drawing_enabled(bool enable) {
	screen_drawing_enabled = enable;
#ifdef TOOLS_ENABLED
	queue_redraw();
#endif
}

bool Camera2D::is_screen_drawing_enabled() const {
	return screen_drawing_enabled;
}

void Camera2D::set_limit_drawing_enabled(bool enable) {
	limit_drawing_enabled = enable;
#ifdef TOOLS_ENABLED
	queue_redraw();
#endif
}

bool Camera2D::is_limit_drawing_enabled() const {
	return limit_drawing_enabled;
}

void Camera2D::set_margin_drawing_enabled(bool enable) {
	margin_drawing_enabled = enable;
#ifdef TOOLS_ENABLED
	queue_redraw();
#endif
}

bool Camera2D::is_margin_drawing_enabled() const {
	return margin_drawing_enabled;
}

void Camera2D::_validate_property(PropertyInfo &p_property) const {
	if (!smoothing_enabled && p_property.name == "smoothing_speed") {
		p_property.usage = PROPERTY_USAGE_NO_EDITOR;
	}
}

void Camera2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_offset", "offset"), &Camera2D::set_offset);
	ClassDB::bind_method(D_METHOD("get_offset"), &Camera2D::get_offset);

	ClassDB::bind_method(D_METHOD("set_anchor_mode", "anchor_mode"), &Camera2D::set_anchor_mode);
	ClassDB::bind_method(D_METHOD("get_anchor_mode"), &Camera2D::get_anchor_mode);

	ClassDB::bind_method(D_METHOD("set_rotating", "rotating"), &Camera2D::set_rotating);
	ClassDB::bind_method(D_METHOD("is_rotating"), &Camera2D::is_rotating);

	ClassDB::bind_method(D_METHOD("set_process_callback", "mode"), &Camera2D::set_process_callback);
	ClassDB::bind_method(D_METHOD("get_process_callback"), &Camera2D::get_process_callback);

	ClassDB::bind_method(D_METHOD("_clear_current"), &Camera2D::_clear_current);
	ClassDB::bind_method(D_METHOD("set_current", "current"), &Camera2D::set_current);
	ClassDB::bind_method(D_METHOD("is_current"), &Camera2D::is_current);

	ClassDB::bind_method(D_METHOD("set_limit", "margin", "limit"), &Camera2D::set_limit);
	ClassDB::bind_method(D_METHOD("get_limit", "margin"), &Camera2D::get_limit);

	ClassDB::bind_method(D_METHOD("set_limit_smoothing_enabled", "limit_smoothing_enabled"), &Camera2D::set_limit_smoothing_enabled);
	ClassDB::bind_method(D_METHOD("is_limit_smoothing_enabled"), &Camera2D::is_limit_smoothing_enabled);

	ClassDB::bind_method(D_METHOD("set_drag_vertical_enabled", "enabled"), &Camera2D::set_drag_vertical_enabled);
	ClassDB::bind_method(D_METHOD("is_drag_vertical_enabled"), &Camera2D::is_drag_vertical_enabled);

	ClassDB::bind_method(D_METHOD("set_drag_horizontal_enabled", "enabled"), &Camera2D::set_drag_horizontal_enabled);
	ClassDB::bind_method(D_METHOD("is_drag_horizontal_enabled"), &Camera2D::is_drag_horizontal_enabled);

	ClassDB::bind_method(D_METHOD("set_drag_vertical_offset", "offset"), &Camera2D::set_drag_vertical_offset);
	ClassDB::bind_method(D_METHOD("get_drag_vertical_offset"), &Camera2D::get_drag_vertical_offset);

	ClassDB::bind_method(D_METHOD("set_drag_horizontal_offset", "offset"), &Camera2D::set_drag_horizontal_offset);
	ClassDB::bind_method(D_METHOD("get_drag_horizontal_offset"), &Camera2D::get_drag_horizontal_offset);

	ClassDB::bind_method(D_METHOD("set_drag_margin", "margin", "drag_margin"), &Camera2D::set_drag_margin);
	ClassDB::bind_method(D_METHOD("get_drag_margin", "margin"), &Camera2D::get_drag_margin);

	ClassDB::bind_method(D_METHOD("get_target_position"), &Camera2D::get_camera_position);
	ClassDB::bind_method(D_METHOD("get_screen_center_position"), &Camera2D::get_camera_screen_center);

	ClassDB::bind_method(D_METHOD("set_zoom", "zoom"), &Camera2D::set_zoom);
	ClassDB::bind_method(D_METHOD("get_zoom"), &Camera2D::get_zoom);

	ClassDB::bind_method(D_METHOD("set_custom_viewport", "viewport"), &Camera2D::set_custom_viewport);
	ClassDB::bind_method(D_METHOD("get_custom_viewport"), &Camera2D::get_custom_viewport);

	ClassDB::bind_method(D_METHOD("set_smoothing_speed", "smoothing_speed"), &Camera2D::set_smoothing_speed);
	ClassDB::bind_method(D_METHOD("get_smoothing_speed"), &Camera2D::get_smoothing_speed);

	ClassDB::bind_method(D_METHOD("set_enable_follow_smoothing", "follow_smoothing"), &Camera2D::set_enable_follow_smoothing);
	ClassDB::bind_method(D_METHOD("is_follow_smoothing_enabled"), &Camera2D::is_follow_smoothing_enabled);

	ClassDB::bind_method(D_METHOD("reset_smoothing"), &Camera2D::reset_smoothing);
	ClassDB::bind_method(D_METHOD("align"), &Camera2D::align);

	ClassDB::bind_method(D_METHOD("set_screen_drawing_enabled", "screen_drawing_enabled"), &Camera2D::set_screen_drawing_enabled);
	ClassDB::bind_method(D_METHOD("is_screen_drawing_enabled"), &Camera2D::is_screen_drawing_enabled);

	ClassDB::bind_method(D_METHOD("set_limit_drawing_enabled", "limit_drawing_enabled"), &Camera2D::set_limit_drawing_enabled);
	ClassDB::bind_method(D_METHOD("is_limit_drawing_enabled"), &Camera2D::is_limit_drawing_enabled);

	ClassDB::bind_method(D_METHOD("set_margin_drawing_enabled", "margin_drawing_enabled"), &Camera2D::set_margin_drawing_enabled);
	ClassDB::bind_method(D_METHOD("is_margin_drawing_enabled"), &Camera2D::is_margin_drawing_enabled);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "offset", PROPERTY_HINT_NONE, "suffix:px"), "set_offset", "get_offset");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "anchor_mode", PROPERTY_HINT_ENUM, "Fixed TopLeft,Drag Center"), "set_anchor_mode", "get_anchor_mode");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "rotating"), "set_rotating", "is_rotating");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "current"), "set_current", "is_current");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "zoom", PROPERTY_HINT_LINK), "set_zoom", "get_zoom");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "custom_viewport", PROPERTY_HINT_RESOURCE_TYPE, "Viewport", PROPERTY_USAGE_NONE), "set_custom_viewport", "get_custom_viewport");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "process_callback", PROPERTY_HINT_ENUM, "Physics,Idle"), "set_process_callback", "get_process_callback");

	ADD_GROUP("Limit", "limit_");
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "limit_left", PROPERTY_HINT_NONE, "suffix:px"), "set_limit", "get_limit", SIDE_LEFT);
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "limit_top", PROPERTY_HINT_NONE, "suffix:px"), "set_limit", "get_limit", SIDE_TOP);
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "limit_right", PROPERTY_HINT_NONE, "suffix:px"), "set_limit", "get_limit", SIDE_RIGHT);
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "limit_bottom", PROPERTY_HINT_NONE, "suffix:px"), "set_limit", "get_limit", SIDE_BOTTOM);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "limit_smoothed"), "set_limit_smoothing_enabled", "is_limit_smoothing_enabled");

	ADD_GROUP("Smoothing", "smoothing_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "smoothing_enabled"), "set_enable_follow_smoothing", "is_follow_smoothing_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "smoothing_speed", PROPERTY_HINT_RANGE, "0,60,1"), "set_smoothing_speed", "get_smoothing_speed");

	ADD_GROUP("Drag", "drag_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "drag_horizontal_enabled"), "set_drag_horizontal_enabled", "is_drag_horizontal_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "drag_vertical_enabled"), "set_drag_vertical_enabled", "is_drag_vertical_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "drag_horizontal_offset", PROPERTY_HINT_RANGE, "-1,1,0.01"), "set_drag_horizontal_offset", "get_drag_horizontal_offset");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "drag_vertical_offset", PROPERTY_HINT_RANGE, "-1,1,0.01"), "set_drag_vertical_offset", "get_drag_vertical_offset");
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "drag_left_margin", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_drag_margin", "get_drag_margin", SIDE_LEFT);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "drag_top_margin", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_drag_margin", "get_drag_margin", SIDE_TOP);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "drag_right_margin", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_drag_margin", "get_drag_margin", SIDE_RIGHT);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "drag_bottom_margin", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_drag_margin", "get_drag_margin", SIDE_BOTTOM);

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
	limit[SIDE_LEFT] = -10000000;
	limit[SIDE_TOP] = -10000000;
	limit[SIDE_RIGHT] = 10000000;
	limit[SIDE_BOTTOM] = 10000000;

	drag_margin[SIDE_LEFT] = 0.2;
	drag_margin[SIDE_TOP] = 0.2;
	drag_margin[SIDE_RIGHT] = 0.2;
	drag_margin[SIDE_BOTTOM] = 0.2;

	set_notify_transform(true);
}
