/*************************************************************************/
/*  camera_2d.cpp                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "scene/main/window.h"

void Camera2D::_update_scroll() {
	if (!is_inside_tree()) {
		return;
	}

	if (Engine::get_singleton()->is_editor_hint()) {
		update(); //will just be drawn
		return;
	}

	if (!viewport) {
		return;
	}

	if (current) {
		ERR_FAIL_COND(custom_viewport && !ObjectDB::get_instance(custom_viewport_id));

		Transform2D xform = get_camera_transform();

		viewport->set_canvas_transform(xform);

		Size2 screen_size = _get_camera_screen_size();
		Point2 screen_offset = (anchor_mode == ANCHOR_MODE_DRAG_CENTER ? (screen_size * 0.5) : Point2());

		get_tree()->call_group_flags(SceneTree::GROUP_CALL_REALTIME, group_name, "_camera_moved", xform, screen_offset);
	};
}

void Camera2D::_update_process_callback() {
	if (Engine::get_singleton()->is_editor_hint()) {
		set_process_internal(false);
		set_physics_process_internal(false);
	} else if (process_callback == CAMERA2D_PROCESS_IDLE) {
		set_process_internal(true);
		set_physics_process_internal(false);
	} else {
		set_process_internal(false);
		set_physics_process_internal(true);
	}
}

void Camera2D::set_zoom(const Vector2 &p_zoom) {
	// Setting zoom to zero causes 'affine_invert' issues
	ERR_FAIL_COND_MSG(Math::is_zero_approx(p_zoom.x) || Math::is_zero_approx(p_zoom.y), "Zoom level must be different from 0 (can be negative).");

	zoom = p_zoom;
	Point2 old_smoothed_camera_pos = smoothed_camera_pos;
	_update_scroll();
	smoothed_camera_pos = old_smoothed_camera_pos;
};

Vector2 Camera2D::get_zoom() const {
	return zoom;
};

Transform2D Camera2D::get_camera_transform() {
	if (!get_tree()) {
		return Transform2D();
	}

	ERR_FAIL_COND_V(custom_viewport && !ObjectDB::get_instance(custom_viewport_id), Transform2D());

	Size2 screen_size = _get_camera_screen_size();

	Point2 new_camera_pos = get_global_position();
	Point2 ret_camera_pos;

	if (!first) {
		if (anchor_mode == ANCHOR_MODE_DRAG_CENTER) {
			if (drag_horizontal_enabled && !Engine::get_singleton()->is_editor_hint() && !drag_horizontal_offset_changed) {
				camera_pos.x = MIN(camera_pos.x, (new_camera_pos.x + screen_size.x * 0.5 * zoom.x * drag_margin[SIDE_LEFT]));
				camera_pos.x = MAX(camera_pos.x, (new_camera_pos.x - screen_size.x * 0.5 * zoom.x * drag_margin[SIDE_RIGHT]));
			} else {
				if (drag_horizontal_offset < 0) {
					camera_pos.x = new_camera_pos.x + screen_size.x * 0.5 * drag_margin[SIDE_RIGHT] * drag_horizontal_offset;
				} else {
					camera_pos.x = new_camera_pos.x + screen_size.x * 0.5 * drag_margin[SIDE_LEFT] * drag_horizontal_offset;
				}

				drag_horizontal_offset_changed = false;
			}

			if (drag_vertical_enabled && !Engine::get_singleton()->is_editor_hint() && !drag_vertical_offset_changed) {
				camera_pos.y = MIN(camera_pos.y, (new_camera_pos.y + screen_size.y * 0.5 * zoom.y * drag_margin[SIDE_TOP]));
				camera_pos.y = MAX(camera_pos.y, (new_camera_pos.y - screen_size.y * 0.5 * zoom.y * drag_margin[SIDE_BOTTOM]));

			} else {
				if (drag_vertical_offset < 0) {
					camera_pos.y = new_camera_pos.y + screen_size.y * 0.5 * drag_margin[SIDE_BOTTOM] * drag_vertical_offset;
				} else {
					camera_pos.y = new_camera_pos.y + screen_size.y * 0.5 * drag_margin[SIDE_TOP] * drag_vertical_offset;
				}

				drag_vertical_offset_changed = false;
			}

		} else if (anchor_mode == ANCHOR_MODE_FIXED_TOP_LEFT) {
			camera_pos = new_camera_pos;
		}

		Point2 screen_offset = (anchor_mode == ANCHOR_MODE_DRAG_CENTER ? (screen_size * 0.5 * zoom) : Point2());
		Rect2 screen_rect(-screen_offset + camera_pos, screen_size * zoom);

		if (limit_smoothing_enabled) {
			if (screen_rect.position.x < limit[SIDE_LEFT]) {
				camera_pos.x -= screen_rect.position.x - limit[SIDE_LEFT];
			}

			if (screen_rect.position.x + screen_rect.size.x > limit[SIDE_RIGHT]) {
				camera_pos.x -= screen_rect.position.x + screen_rect.size.x - limit[SIDE_RIGHT];
			}

			if (screen_rect.position.y + screen_rect.size.y > limit[SIDE_BOTTOM]) {
				camera_pos.y -= screen_rect.position.y + screen_rect.size.y - limit[SIDE_BOTTOM];
			}

			if (screen_rect.position.y < limit[SIDE_TOP]) {
				camera_pos.y -= screen_rect.position.y - limit[SIDE_TOP];
			}
		}

		if (smoothing_enabled && !Engine::get_singleton()->is_editor_hint()) {
			real_t c = smoothing * (process_callback == CAMERA2D_PROCESS_PHYSICS ? get_physics_process_delta_time() : get_process_delta_time());
			smoothed_camera_pos = ((camera_pos - smoothed_camera_pos) * c) + smoothed_camera_pos;
			ret_camera_pos = smoothed_camera_pos;
			//camera_pos=camera_pos*(1.0-smoothing)+new_camera_pos*smoothing;
		} else {
			ret_camera_pos = smoothed_camera_pos = camera_pos;
		}

	} else {
		ret_camera_pos = smoothed_camera_pos = camera_pos = new_camera_pos;
		first = false;
	}

	Point2 screen_offset = (anchor_mode == ANCHOR_MODE_DRAG_CENTER ? (screen_size * 0.5 * zoom) : Point2());

	real_t angle = get_global_rotation();
	if (rotating) {
		screen_offset = screen_offset.rotated(angle);
	}

	Rect2 screen_rect(-screen_offset + ret_camera_pos, screen_size * zoom);

	if (!limit_smoothing_enabled) {
		if (screen_rect.position.x < limit[SIDE_LEFT]) {
			screen_rect.position.x = limit[SIDE_LEFT];
		}

		if (screen_rect.position.x + screen_rect.size.x > limit[SIDE_RIGHT]) {
			screen_rect.position.x = limit[SIDE_RIGHT] - screen_rect.size.x;
		}

		if (screen_rect.position.y + screen_rect.size.y > limit[SIDE_BOTTOM]) {
			screen_rect.position.y = limit[SIDE_BOTTOM] - screen_rect.size.y;
		}

		if (screen_rect.position.y < limit[SIDE_TOP]) {
			screen_rect.position.y = limit[SIDE_TOP];
		}
	}

	if (offset != Vector2()) {
		screen_rect.position += offset;
	}

	camera_screen_center = screen_rect.get_center();

	Transform2D xform;
	xform.scale_basis(zoom);
	if (rotating) {
		xform.set_rotation(angle);
	}
	xform.set_origin(screen_rect.position /*.floor()*/);

	/*
	if (0) {
		xform = get_global_transform() * xform;
	} else {
		xform.elements[2]+=get_global_transform().get_origin();
	}
*/

	return (xform).affine_inverse();
}

void Camera2D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_INTERNAL_PROCESS:
		case NOTIFICATION_INTERNAL_PHYSICS_PROCESS: {
			_update_scroll();

		} break;
		case NOTIFICATION_TRANSFORM_CHANGED: {
			if (!is_processing_internal() && !is_physics_processing_internal()) {
				_update_scroll();
			}

		} break;
		case NOTIFICATION_ENTER_TREE: {
			ERR_FAIL_COND(!is_inside_tree());
			if (custom_viewport && ObjectDB::get_instance(custom_viewport_id)) {
				viewport = custom_viewport;
			} else {
				viewport = get_viewport();
			}

			if (is_current()) {
				viewport->_camera_2d_set(this);
			}

			canvas = get_canvas();

			RID vp = viewport->get_viewport_rid();

			group_name = "__cameras_" + itos(vp.get_id());
			canvas_group_name = "__cameras_c" + itos(canvas.get_id());
			add_to_group(group_name);
			add_to_group(canvas_group_name);

			_update_process_callback();
			_update_scroll();
			first = true;

		} break;
		case NOTIFICATION_EXIT_TREE: {
			if (is_current()) {
				if (viewport && !(custom_viewport && !ObjectDB::get_instance(custom_viewport_id))) {
					viewport->set_canvas_transform(Transform2D());
					clear_current();
					current = true;
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
				if (is_current()) {
					area_axis_width = 3;
				}

				Transform2D inv_camera_transform = get_camera_transform().affine_inverse();
				Size2 screen_size = _get_camera_screen_size();

				Vector2 screen_endpoints[4] = {
					inv_camera_transform.xform(Vector2(0, 0)),
					inv_camera_transform.xform(Vector2(screen_size.width, 0)),
					inv_camera_transform.xform(Vector2(screen_size.width, screen_size.height)),
					inv_camera_transform.xform(Vector2(0, screen_size.height))
				};

				Transform2D inv_transform = get_global_transform().affine_inverse(); // undo global space

				for (int i = 0; i < 4; i++) {
					draw_line(inv_transform.xform(screen_endpoints[i]), inv_transform.xform(screen_endpoints[(i + 1) % 4]), area_axis_color, area_axis_width);
				}
			}

			if (limit_drawing_enabled) {
				Color limit_drawing_color(1, 1, 0.25, 0.63);
				real_t limit_drawing_width = 1;
				if (is_current()) {
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
				if (is_current()) {
					margin_drawing_width = 3;
				}

				Transform2D inv_camera_transform = get_camera_transform().affine_inverse();
				Size2 screen_size = _get_camera_screen_size();

				Vector2 margin_endpoints[4] = {
					inv_camera_transform.xform(Vector2((screen_size.width / 2) - ((screen_size.width / 2) * drag_margin[SIDE_LEFT]), (screen_size.height / 2) - ((screen_size.height / 2) * drag_margin[SIDE_TOP]))),
					inv_camera_transform.xform(Vector2((screen_size.width / 2) + ((screen_size.width / 2) * drag_margin[SIDE_RIGHT]), (screen_size.height / 2) - ((screen_size.height / 2) * drag_margin[SIDE_TOP]))),
					inv_camera_transform.xform(Vector2((screen_size.width / 2) + ((screen_size.width / 2) * drag_margin[SIDE_RIGHT]), (screen_size.height / 2) + ((screen_size.height / 2) * drag_margin[SIDE_BOTTOM]))),
					inv_camera_transform.xform(Vector2((screen_size.width / 2) - ((screen_size.width / 2) * drag_margin[SIDE_LEFT]), (screen_size.height / 2) + ((screen_size.height / 2) * drag_margin[SIDE_BOTTOM])))
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
	Point2 old_smoothed_camera_pos = smoothed_camera_pos;
	_update_scroll();
	smoothed_camera_pos = old_smoothed_camera_pos;
}

Vector2 Camera2D::get_offset() const {
	return offset;
}

void Camera2D::set_anchor_mode(AnchorMode p_anchor_mode) {
	anchor_mode = p_anchor_mode;
	_update_scroll();
}

Camera2D::AnchorMode Camera2D::get_anchor_mode() const {
	return anchor_mode;
}

void Camera2D::set_rotating(bool p_rotating) {
	rotating = p_rotating;
	Point2 old_smoothed_camera_pos = smoothed_camera_pos;
	_update_scroll();
	smoothed_camera_pos = old_smoothed_camera_pos;
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

void Camera2D::_make_current(Object *p_which) {
	if (p_which == this) {
		current = true;
		if (is_inside_tree()) {
			get_viewport()->_camera_2d_set(this);
			update();
		}
	} else {
		current = false;
		if (is_inside_tree()) {
			if (get_viewport()->get_camera_2d() == this) {
				get_viewport()->_camera_2d_set(nullptr);
			}
			update();
		}
	}
}

void Camera2D::set_current(bool p_current) {
	if (p_current) {
		make_current();
	} else {
		if (current) {
			clear_current();
		}
	}
}

bool Camera2D::is_current() const {
	return current;
}

void Camera2D::make_current() {
	if (is_inside_tree()) {
		get_tree()->call_group_flags(SceneTree::GROUP_CALL_REALTIME, group_name, "_make_current", this);
	} else {
		current = true;
	}
	_update_scroll();
}

void Camera2D::clear_current() {
	if (is_inside_tree()) {
		get_tree()->call_group_flags(SceneTree::GROUP_CALL_REALTIME, group_name, "_make_current", (Object *)nullptr);
	} else {
		current = false;
	}
}

void Camera2D::set_limit(Side p_side, int p_limit) {
	ERR_FAIL_INDEX((int)p_side, 4);
	limit[p_side] = p_limit;
	update();
}

int Camera2D::get_limit(Side p_side) const {
	ERR_FAIL_INDEX_V((int)p_side, 4, 0);
	return limit[p_side];
}

void Camera2D::set_limit_smoothing_enabled(bool enable) {
	limit_smoothing_enabled = enable;
	_update_scroll();
}

bool Camera2D::is_limit_smoothing_enabled() const {
	return limit_smoothing_enabled;
}

void Camera2D::set_drag_margin(Side p_side, real_t p_drag_margin) {
	ERR_FAIL_INDEX((int)p_side, 4);
	drag_margin[p_side] = p_drag_margin;
	update();
}

real_t Camera2D::get_drag_margin(Side p_side) const {
	ERR_FAIL_INDEX_V((int)p_side, 4, 0);
	return drag_margin[p_side];
}

Vector2 Camera2D::get_camera_position() const {
	return camera_pos;
}

void Camera2D::force_update_scroll() {
	_update_scroll();
}

void Camera2D::reset_smoothing() {
	_update_scroll();
	smoothed_camera_pos = camera_pos;
}

void Camera2D::align() {
	ERR_FAIL_COND(custom_viewport && !ObjectDB::get_instance(custom_viewport_id));

	Size2 screen_size = _get_camera_screen_size();

	Point2 current_camera_pos = get_global_position();
	if (anchor_mode == ANCHOR_MODE_DRAG_CENTER) {
		if (drag_horizontal_offset < 0) {
			camera_pos.x = current_camera_pos.x + screen_size.x * 0.5 * drag_margin[SIDE_RIGHT] * drag_horizontal_offset;
		} else {
			camera_pos.x = current_camera_pos.x + screen_size.x * 0.5 * drag_margin[SIDE_LEFT] * drag_horizontal_offset;
		}
		if (drag_vertical_offset < 0) {
			camera_pos.y = current_camera_pos.y + screen_size.y * 0.5 * drag_margin[SIDE_TOP] * drag_vertical_offset;
		} else {
			camera_pos.y = current_camera_pos.y + screen_size.y * 0.5 * drag_margin[SIDE_BOTTOM] * drag_vertical_offset;
		}
	} else if (anchor_mode == ANCHOR_MODE_FIXED_TOP_LEFT) {
		camera_pos = current_camera_pos;
	}

	_update_scroll();
}

void Camera2D::set_follow_smoothing(real_t p_speed) {
	smoothing = p_speed;
	if (smoothing > 0 && !(is_inside_tree() && Engine::get_singleton()->is_editor_hint())) {
		set_process_internal(true);
	} else {
		set_process_internal(false);
	}
}

real_t Camera2D::get_follow_smoothing() const {
	return smoothing;
}

Point2 Camera2D::get_camera_screen_center() const {
	return camera_screen_center;
}

Size2 Camera2D::_get_camera_screen_size() const {
	// special case if the camera2D is in the root viewport
	if (Engine::get_singleton()->is_editor_hint() && get_viewport()->get_parent_viewport() == get_tree()->get_root()) {
		return Size2(ProjectSettings::get_singleton()->get("display/window/size/width"), ProjectSettings::get_singleton()->get("display/window/size/height"));
	}
	return get_viewport_rect().size;
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
	Point2 old_smoothed_camera_pos = smoothed_camera_pos;
	_update_scroll();
	smoothed_camera_pos = old_smoothed_camera_pos;
}

real_t Camera2D::get_drag_vertical_offset() const {
	return drag_vertical_offset;
}

void Camera2D::set_drag_horizontal_offset(real_t p_offset) {
	drag_horizontal_offset = p_offset;
	drag_horizontal_offset_changed = true;
	Point2 old_smoothed_camera_pos = smoothed_camera_pos;
	_update_scroll();
	smoothed_camera_pos = old_smoothed_camera_pos;
}

real_t Camera2D::get_drag_horizontal_offset() const {
	return drag_horizontal_offset;
}

void Camera2D::_set_old_smoothing(real_t p_enable) {
	//compatibility
	if (p_enable > 0) {
		smoothing_enabled = true;
		set_follow_smoothing(p_enable);
	}
}

void Camera2D::set_enable_follow_smoothing(bool p_enabled) {
	smoothing_enabled = p_enabled;
	notify_property_list_changed();
}

bool Camera2D::is_follow_smoothing_enabled() const {
	return smoothing_enabled;
}

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

void Camera2D::_validate_property(PropertyInfo &property) const {
	if (!smoothing_enabled && property.name == "smoothing_speed") {
		property.usage = PROPERTY_USAGE_NOEDITOR;
	}
}

void Camera2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_offset", "offset"), &Camera2D::set_offset);
	ClassDB::bind_method(D_METHOD("get_offset"), &Camera2D::get_offset);

	ClassDB::bind_method(D_METHOD("set_anchor_mode", "anchor_mode"), &Camera2D::set_anchor_mode);
	ClassDB::bind_method(D_METHOD("get_anchor_mode"), &Camera2D::get_anchor_mode);

	ClassDB::bind_method(D_METHOD("set_rotating", "rotating"), &Camera2D::set_rotating);
	ClassDB::bind_method(D_METHOD("is_rotating"), &Camera2D::is_rotating);

	ClassDB::bind_method(D_METHOD("_update_scroll"), &Camera2D::_update_scroll);

	ClassDB::bind_method(D_METHOD("set_process_callback", "mode"), &Camera2D::set_process_callback);
	ClassDB::bind_method(D_METHOD("get_process_callback"), &Camera2D::get_process_callback);

	ClassDB::bind_method(D_METHOD("set_current", "current"), &Camera2D::set_current);
	ClassDB::bind_method(D_METHOD("is_current"), &Camera2D::is_current);
	ClassDB::bind_method(D_METHOD("_make_current"), &Camera2D::_make_current);

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

	ClassDB::bind_method(D_METHOD("get_camera_position"), &Camera2D::get_camera_position);
	ClassDB::bind_method(D_METHOD("get_camera_screen_center"), &Camera2D::get_camera_screen_center);

	ClassDB::bind_method(D_METHOD("set_zoom", "zoom"), &Camera2D::set_zoom);
	ClassDB::bind_method(D_METHOD("get_zoom"), &Camera2D::get_zoom);

	ClassDB::bind_method(D_METHOD("set_custom_viewport", "viewport"), &Camera2D::set_custom_viewport);
	ClassDB::bind_method(D_METHOD("get_custom_viewport"), &Camera2D::get_custom_viewport);

	ClassDB::bind_method(D_METHOD("set_follow_smoothing", "follow_smoothing"), &Camera2D::set_follow_smoothing);
	ClassDB::bind_method(D_METHOD("get_follow_smoothing"), &Camera2D::get_follow_smoothing);

	ClassDB::bind_method(D_METHOD("set_enable_follow_smoothing", "follow_smoothing"), &Camera2D::set_enable_follow_smoothing);
	ClassDB::bind_method(D_METHOD("is_follow_smoothing_enabled"), &Camera2D::is_follow_smoothing_enabled);

	ClassDB::bind_method(D_METHOD("force_update_scroll"), &Camera2D::force_update_scroll);
	ClassDB::bind_method(D_METHOD("reset_smoothing"), &Camera2D::reset_smoothing);
	ClassDB::bind_method(D_METHOD("align"), &Camera2D::align);

	ClassDB::bind_method(D_METHOD("_set_old_smoothing", "follow_smoothing"), &Camera2D::_set_old_smoothing);

	ClassDB::bind_method(D_METHOD("set_screen_drawing_enabled", "screen_drawing_enabled"), &Camera2D::set_screen_drawing_enabled);
	ClassDB::bind_method(D_METHOD("is_screen_drawing_enabled"), &Camera2D::is_screen_drawing_enabled);

	ClassDB::bind_method(D_METHOD("set_limit_drawing_enabled", "limit_drawing_enabled"), &Camera2D::set_limit_drawing_enabled);
	ClassDB::bind_method(D_METHOD("is_limit_drawing_enabled"), &Camera2D::is_limit_drawing_enabled);

	ClassDB::bind_method(D_METHOD("set_margin_drawing_enabled", "margin_drawing_enabled"), &Camera2D::set_margin_drawing_enabled);
	ClassDB::bind_method(D_METHOD("is_margin_drawing_enabled"), &Camera2D::is_margin_drawing_enabled);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "offset"), "set_offset", "get_offset");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "anchor_mode", PROPERTY_HINT_ENUM, "Fixed TopLeft,Drag Center"), "set_anchor_mode", "get_anchor_mode");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "rotating"), "set_rotating", "is_rotating");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "current"), "set_current", "is_current");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "zoom"), "set_zoom", "get_zoom");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "custom_viewport", PROPERTY_HINT_RESOURCE_TYPE, "Viewport", PROPERTY_USAGE_NONE), "set_custom_viewport", "get_custom_viewport");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "process_callback", PROPERTY_HINT_ENUM, "Physics,Idle"), "set_process_callback", "get_process_callback");

	ADD_GROUP("Limit", "limit_");
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "limit_left"), "set_limit", "get_limit", SIDE_LEFT);
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "limit_top"), "set_limit", "get_limit", SIDE_TOP);
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "limit_right"), "set_limit", "get_limit", SIDE_RIGHT);
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "limit_bottom"), "set_limit", "get_limit", SIDE_BOTTOM);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "limit_smoothed"), "set_limit_smoothing_enabled", "is_limit_smoothing_enabled");

	ADD_GROUP("Smoothing", "smoothing_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "smoothing_enabled"), "set_enable_follow_smoothing", "is_follow_smoothing_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "smoothing_speed"), "set_follow_smoothing", "get_follow_smoothing");

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
