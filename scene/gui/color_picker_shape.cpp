/**************************************************************************/
/*  color_picker_shape.cpp                                                */
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

#include "color_picker_shape.h"

#include "scene/gui/margin_container.h"

void ColorPickerShape::_emit_color_changed() {
	color_picker->emit_signal(SNAME("color_changed"), color_picker->color);
}

bool ColorPickerShape::can_handle(const Ref<InputEvent> &p_event, Vector2 &r_position, bool *r_is_click) {
	Ref<InputEventMouseButton> mb = p_event;
	if (mb.is_valid()) {
		if (mb->get_button_index() != MouseButton::LEFT) {
			return false;
		}

		if (r_is_click) {
			*r_is_click = true;
		}

		if (mb->is_pressed()) {
			is_dragging = true;
			r_position = mb->get_position();
			return true;
		} else {
			_emit_color_changed();
			color_picker->add_recent_preset(color_picker->color);
			is_dragging = false;
			return false;
		}
	}

	Ref<InputEventMouseMotion> mm = p_event;
	if (is_dragging && mm.is_valid()) {
		r_position = mm->get_position();
		return true;
	}
	return false;
}

void ColorPickerShape::apply_color() {
	color_picker->_copy_hsv_to_color();
	color_picker->last_color = color_picker->color;
	color_picker->set_pick_color(color_picker->color);

	if (!color_picker->deferred_mode_enabled) {
		_emit_color_changed();
	}
}

void ColorPickerShape::cancel_event() {
	is_dragging = false;
}

void ColorPickerShape::draw_focus_rect(Control *p_control, const Rect2 &p_rect) {
	if (!p_control->has_focus()) {
		return;
	}

	Rect2 focus_rect;
	if (p_rect.has_area()) {
		focus_rect = p_rect;
	} else {
		focus_rect = Rect2(Vector2(), p_control->get_size());
	}

	const RID ci = p_control->get_canvas_item();
	if (!cursor_editing) {
		RenderingServer::get_singleton()->canvas_item_add_rect(ci, focus_rect, color_picker->theme_cache.focused_not_editing_cursor_color);
	}
	color_picker->theme_cache.picker_focus_rectangle->draw(ci, focus_rect);
}

void ColorPickerShape::draw_focus_circle(Control *p_control) {
	if (!p_control->has_focus()) {
		return;
	}

	const Rect2 focus_rect(Vector2(), p_control->get_size());
	const RID ci = p_control->get_canvas_item();
	if (!cursor_editing) {
		RenderingServer::get_singleton()->canvas_item_add_circle(ci, focus_rect.get_center(), focus_rect.get_size().y * 0.5, color_picker->theme_cache.focused_not_editing_cursor_color);
	}
	color_picker->theme_cache.picker_focus_circle->draw(ci, focus_rect);
}

void ColorPickerShape::draw_sv_square(Control *p_control, const Rect2 &p_square, bool p_draw_focus) {
	const Vector2 end = p_square.get_end();
	PackedVector2Array points = {
		p_square.position,
		Vector2(end.x, p_square.position.y),
		end,
		Vector2(p_square.position.x, end.y),
	};

	Color color1 = color_picker->color;
	color1.set_hsv(color_picker->h, 1, 1);
	Color color2 = color1;
	color2.set_hsv(color_picker->h, 1, 0);

	PackedColorArray colors = {
		Color(1, 1, 1, 1),
		Color(1, 1, 1, 1),
		Color(0, 0, 0, 1),
		Color(0, 0, 0, 1)
	};
	p_control->draw_polygon(points, colors);

	colors = {
		Color(color1, 0),
		Color(color1, 1),
		Color(color2, 1),
		Color(color2, 0)
	};
	p_control->draw_polygon(points, colors);

	Vector2 cursor_pos;
	cursor_pos.x = CLAMP(p_square.position.x + p_square.size.x * color_picker->s, p_square.position.x, end.x);
	cursor_pos.y = CLAMP(p_square.position.y + p_square.size.y * (1.0 - color_picker->v), p_square.position.y, end.y);

	if (p_draw_focus) {
		draw_focus_rect(p_control, p_square);
	}
	draw_cursor(p_control, cursor_pos);
}

void ColorPickerShape::draw_cursor(Control *p_control, const Vector2 &p_center, bool p_draw_bg) {
	const Vector2 position = p_center - color_picker->theme_cache.picker_cursor->get_size() * 0.5;
	if (p_draw_bg) {
		p_control->draw_texture(color_picker->theme_cache.picker_cursor_bg, position, Color(color_picker->color, 1.0));
	}
	p_control->draw_texture(color_picker->theme_cache.picker_cursor, position);
}

void ColorPickerShape::draw_circle_cursor(Control *p_control, float p_hue) {
	const Vector2 center = p_control->get_size() * 0.5;
	const Vector2 cursor_pos(
			center.x + (center.x * Math::cos(p_hue * Math_TAU) * color_picker->s),
			center.y + (center.y * Math::sin(p_hue * Math_TAU) * color_picker->s));

	draw_cursor(p_control, cursor_pos);
}

void ColorPickerShape::connect_shape_focus(Control *p_shape) {
	p_shape->set_focus_mode(Control::FOCUS_ALL);
	p_shape->connect(SceneStringName(focus_entered), callable_mp(this, &ColorPickerShape::shape_focus_entered));
	p_shape->connect(SceneStringName(focus_exited), callable_mp(this, &ColorPickerShape::shape_focus_exited));
}

void ColorPickerShape::shape_focus_entered() {
	Input *input = Input::get_singleton();
	if (!(input->is_action_pressed("ui_up") || input->is_action_pressed("ui_down") || input->is_action_pressed("ui_left") || input->is_action_pressed("ui_right"))) {
		cursor_editing = true;
	}
}

void ColorPickerShape::shape_focus_exited() {
	cursor_editing = false;
}

void ColorPickerShape::handle_cursor_editing(const Ref<InputEvent> &p_event, Control *p_control) {
	if (p_event->is_action_pressed("ui_accept", false, true)) {
		cursor_editing = !cursor_editing;
		p_control->queue_redraw();
		color_picker->accept_event();
	}

	if (cursor_editing && p_event->is_action_pressed("ui_cancel", false, true)) {
		cursor_editing = false;
		p_control->queue_redraw();
		color_picker->accept_event();
	}

	if (!cursor_editing) {
		return;
	}

	Input *input = Input::get_singleton();
	bool is_joypad_event = Object::cast_to<InputEventJoypadMotion>(p_event.ptr()) || Object::cast_to<InputEventJoypadButton>(p_event.ptr());

	if (p_event->is_action_pressed("ui_left", true) || p_event->is_action_pressed("ui_right", true) || p_event->is_action_pressed("ui_up", true) || p_event->is_action_pressed("ui_down", true)) {
		if (is_joypad_event) {
			if (color_picker->is_processing_internal()) {
				color_picker->accept_event();
				return;
			}
			color_picker->set_process_internal(true);
		}

		Vector2 color_change_vector = Vector2(
				input->is_action_pressed("ui_right") - input->is_action_pressed("ui_left"),
				input->is_action_pressed("ui_down") - input->is_action_pressed("ui_up"));
		update_cursor(color_change_vector, p_event->is_echo());
		color_picker->accept_event();
	}
}

int ColorPickerShape::get_edge_h_change(const Vector2 &p_color_change_vector) {
	int h_change = 0;

	if (color_picker->h > 0 && color_picker->h < 0.5) {
		h_change -= p_color_change_vector.x;
	} else if (color_picker->h > 0.5 && color_picker->h < 1) {
		h_change += p_color_change_vector.x;
	}

	if (color_picker->h > 0.25 && color_picker->h < 0.75) {
		h_change -= p_color_change_vector.y;
	} else if (color_picker->h < 0.25 || color_picker->h > 0.75) {
		h_change += p_color_change_vector.y;
	}
	return h_change;
}

float ColorPickerShape::get_h_on_circle_edge(const Vector2 &p_color_change_vector) {
	int h_change = get_edge_h_change(p_color_change_vector);

	float target_h = Math::wrapf(color_picker->h + h_change / 360.0, 0, 1);
	int current_quarter = color_picker->h * 4;
	int future_quarter = target_h * 4;
	if (p_color_change_vector.y > 0 && ((future_quarter == 0 && current_quarter == 1) || (future_quarter == 1 && current_quarter == 0))) {
		target_h = 0.25f;
	} else if (p_color_change_vector.y < 0 && ((future_quarter == 2 && current_quarter == 3) || (future_quarter == 3 && current_quarter == 2))) {
		target_h = 0.75f;
	} else if (p_color_change_vector.x < 0 && ((future_quarter == 1 && current_quarter == 2) || (future_quarter == 2 && current_quarter == 1))) {
		target_h = 0.5f;
	} else if (p_color_change_vector.x > 0 && ((future_quarter == 3 && current_quarter == 0) || (future_quarter == 0 && current_quarter == 3))) {
		target_h = 0;
	}
	return target_h;
}

void ColorPickerShape::initialize_controls() {
	_initialize_controls();
	update_theme();
	is_initialized = true;
}

void ColorPickerShape::update_cursor(const Vector2 &p_color_change_vector, bool p_is_echo) {
	if (p_color_change_vector.is_zero_approx()) {
		echo_multiplier = 1.0;
	} else {
		echo_multiplier = p_is_echo ? CLAMP(echo_multiplier * 1.1, 1, 25) : 1;
		_update_cursor(p_color_change_vector * echo_multiplier, p_is_echo);
		apply_color();
	}
}

ColorPickerShape::ColorPickerShape(ColorPicker *p_color_picker) {
	color_picker = p_color_picker;
}

void ColorPickerShapeRectangle::_sv_square_input(const Ref<InputEvent> &p_event) {
	handle_cursor_editing(p_event, sv_square);

	Vector2 event_position;
	if (!can_handle(p_event, event_position)) {
		return;
	}
	event_position = (event_position / sv_square->get_size()).clampf(0.0, 1.0);

	color_picker->s = event_position.x;
	color_picker->v = 1.0 - event_position.y;

	apply_color();
}

void ColorPickerShapeRectangle::_hue_slider_input(const Ref<InputEvent> &p_event) {
	handle_cursor_editing(p_event, hue_slider);

	Vector2 event_position;
	if (!can_handle(p_event, event_position)) {
		return;
	}
	color_picker->h = CLAMP(event_position.y / hue_slider->get_size().y, 0.0, 1.0);
	apply_color();
}

void ColorPickerShapeRectangle::_sv_square_draw() {
	draw_sv_square(sv_square, Rect2(Vector2(), sv_square->get_size()));
}

void ColorPickerShapeRectangle::_hue_slider_draw() {
	const Vector2 size = hue_slider->get_size();
	hue_slider->draw_texture_rect(color_picker->theme_cache.color_hue, Rect2(0, 0, -size.y, size.x), false, Color(1, 1, 1), true);

	draw_focus_rect(hue_slider);

	int y = size.y * color_picker->h;
	const Color color = Color::from_hsv(color_picker->h, 1, 1);
	hue_slider->draw_line(Vector2(0, y), Vector2(size.x, y), color.inverted());
}

void ColorPickerShapeRectangle::_initialize_controls() {
	sv_square = memnew(Control);
	color_picker->shape_container->add_child(sv_square);
	sv_square->connect(SceneStringName(gui_input), callable_mp(this, &ColorPickerShapeRectangle::_sv_square_input));
	sv_square->connect(SceneStringName(draw), callable_mp(this, &ColorPickerShapeRectangle::_sv_square_draw));
	connect_shape_focus(sv_square);

	hue_slider = memnew(Control);
	color_picker->shape_container->add_child(hue_slider);
	hue_slider->connect(SceneStringName(gui_input), callable_mp(this, &ColorPickerShapeRectangle::_hue_slider_input));
	hue_slider->connect(SceneStringName(draw), callable_mp(this, &ColorPickerShapeRectangle::_hue_slider_draw));
	connect_shape_focus(hue_slider);

	controls.append(sv_square);
	controls.append(hue_slider);
}

void ColorPickerShapeRectangle::_update_cursor(const Vector2 &p_color_change_vector, bool p_is_echo) {
	if (sv_square->has_focus()) {
		color_picker->s = CLAMP(color_picker->s + p_color_change_vector.x / 100.0, 0, 1);
		color_picker->v = CLAMP(color_picker->v - p_color_change_vector.y / 100.0, 0, 1);
	} else if (hue_slider->has_focus()) {
		color_picker->h = CLAMP(color_picker->h + p_color_change_vector.y * echo_multiplier / 360.0, 0, 1);
	}
}

void ColorPickerShapeRectangle::update_theme() {
	const ColorPicker::ThemeCache &theme_cache = color_picker->theme_cache;
	sv_square->set_custom_minimum_size(Size2(theme_cache.sv_width, theme_cache.sv_height));
	hue_slider->set_custom_minimum_size(Size2(theme_cache.h_width, 0));
}

void ColorPickerShapeRectangle::grab_focus() {
	hue_slider->grab_focus();
}

float ColorPickerShapeWheel::_get_h_on_wheel(const Vector2 &p_color_change_vector) {
	int h_change = get_edge_h_change(p_color_change_vector);

	float target_h = Math::wrapf(color_picker->h + h_change / 360.0, 0, 1);
	int current_quarter = color_picker->h * 4;
	int future_quarter = target_h * 4;

	if (p_color_change_vector.y > 0 && ((future_quarter == 0 && current_quarter == 1) || (future_quarter == 1 && current_quarter == 0))) {
		rotate_next_echo_event = !rotate_next_echo_event;
	} else if (p_color_change_vector.y < 0 && ((future_quarter == 2 && current_quarter == 3) || (future_quarter == 3 && current_quarter == 2))) {
		rotate_next_echo_event = !rotate_next_echo_event;
	} else if (p_color_change_vector.x < 0 && ((future_quarter == 1 && current_quarter == 2) || (future_quarter == 2 && current_quarter == 1))) {
		rotate_next_echo_event = !rotate_next_echo_event;
	} else if (p_color_change_vector.x > 0 && ((future_quarter == 3 && current_quarter == 0) || (future_quarter == 0 && current_quarter == 3))) {
		rotate_next_echo_event = !rotate_next_echo_event;
	}

	return target_h;
}

void ColorPickerShapeWheel::_reset_wheel_focus() {
	wheel_focused = true;
}

void ColorPickerShapeWheel::_wheel_input(const Ref<InputEvent> &p_event) {
	handle_cursor_editing(p_event, wheel_uv);
	if (!cursor_editing) {
		// Wheel and inner square are the same control, so focus has to be moved manually.
		if (!wheel_focused && p_event->is_action_pressed("ui_down", true)) {
			wheel_focused = true;
			wheel_uv->queue_redraw();
			color_picker->accept_event();
			return;
		} else if (wheel_focused && p_event->is_action_pressed("ui_up", true)) {
			wheel_focused = false;
			wheel_uv->queue_redraw();
			color_picker->accept_event();
			return;
		}
	}

	Vector2 event_position;
	bool is_click = false;
	if (!can_handle(p_event, event_position, &is_click)) {
		if (is_click) {
			// Released mouse button while dragging wheel.
			spinning = false;
		}
		return;
	}
	const Vector2 uv_size = wheel_uv->get_size();
	const Vector2 ring_radius = uv_size * Math_SQRT12 * WHEEL_RADIUS;
	const Vector2 center = uv_size * 0.5;

	if (is_click && !spinning) {
		real_t dist = center.distance_to(event_position);
		if (dist >= center.x * WHEEL_RADIUS * 2.0 && dist <= center.x) {
			spinning = true;
			if (!wheel_focused) {
				cursor_editing = true;
				wheel_focused = true;
			}
		} else if (dist > center.x) {
			// Clicked outside the wheel.
			cancel_event();
			return;
		}
	};

	if (spinning) {
		real_t rad = center.angle_to_point(event_position);
		color_picker->h = ((rad >= 0) ? rad : (Math_TAU + rad)) / Math_TAU;
		apply_color();
		return;
	}

	const Rect2 uv_rect(center - ring_radius, ring_radius * 2.0);
	event_position -= uv_rect.position;
	event_position /= uv_rect.size;

	if (is_click && (event_position.x < 0 || event_position.x > 1 || event_position.y < 0 || event_position.y > 1)) {
		// Clicked inside the wheel, but outside the square.
		cancel_event();
		return;
	}

	event_position = event_position.clampf(0.0, 1.0);

	color_picker->s = event_position.x;
	color_picker->v = 1.0 - event_position.y;
	if (wheel_focused) {
		cursor_editing = true;
		wheel_focused = false;
	}

	apply_color();
}

void ColorPickerShapeWheel::_wheel_draw() {
	wheel->draw_rect(Rect2(Point2(), wheel->get_size()), Color(1, 1, 1));
}

void ColorPickerShapeWheel::_wheel_uv_draw() {
	const Vector2 uv_size = wheel_uv->get_size();
	const Vector2 ring_radius = uv_size * Math_SQRT12 * WHEEL_RADIUS;
	const Vector2 center = uv_size * 0.5;

	const Rect2 uv_rect(center - ring_radius, ring_radius * 2.0);
	draw_sv_square(wheel_uv, uv_rect, !wheel_focused);
	if (wheel_focused) {
		draw_focus_circle(wheel_uv);
	}

	float radius = WHEEL_RADIUS * 2.0;
	radius += (1.0 - radius) * 0.5;
	const Vector2 cursor_pos = center +
			Vector2(center.x * Math::cos(color_picker->h * Math_TAU) * radius,
					center.y * Math::sin(color_picker->h * Math_TAU) * radius);
	draw_cursor(wheel_uv, cursor_pos, false);
}

void ColorPickerShapeWheel::_initialize_controls() {
	wheel_margin = memnew(MarginContainer);
	color_picker->shape_container->add_child(wheel_margin);

	Ref<ShaderMaterial> material;
	material.instantiate();
	material->set_shader(ColorPicker::wheel_shader);
	material->set_shader_parameter("wheel_radius", WHEEL_RADIUS);

	wheel = memnew(Control);
	wheel->set_material(material);
	wheel_margin->add_child(wheel);
	wheel->connect(SceneStringName(draw), callable_mp(this, &ColorPickerShapeWheel::_wheel_draw));

	wheel_uv = memnew(Control);
	wheel_margin->add_child(wheel_uv);
	wheel_uv->connect(SceneStringName(focus_entered), callable_mp(this, &ColorPickerShapeWheel::_reset_wheel_focus));
	wheel_uv->connect(SceneStringName(gui_input), callable_mp(this, &ColorPickerShapeWheel::_wheel_input));
	wheel_uv->connect(SceneStringName(draw), callable_mp(this, &ColorPickerShapeWheel::_wheel_uv_draw));
	connect_shape_focus(wheel_uv);

	controls.append(wheel_margin);
	controls.append(wheel);
	controls.append(wheel_uv);
}

void ColorPickerShapeWheel::_update_cursor(const Vector2 &p_color_change_vector, bool p_is_echo) {
	if (wheel_focused) {
		if (p_is_echo && rotate_next_echo_event) {
			color_picker->h = _get_h_on_wheel(-p_color_change_vector);
		} else {
			rotate_next_echo_event = false;
			color_picker->h = _get_h_on_wheel(p_color_change_vector);
		}
	} else {
		color_picker->s = CLAMP(color_picker->s + p_color_change_vector.x / 100.0, 0, 1);
		color_picker->v = CLAMP(color_picker->v - p_color_change_vector.y / 100.0, 0, 1);
	}
}

void ColorPickerShapeWheel::update_theme() {
	const ColorPicker::ThemeCache &theme_cache = color_picker->theme_cache;
	wheel_margin->set_custom_minimum_size(Size2(theme_cache.sv_width, theme_cache.sv_height));
	wheel_margin->add_theme_constant_override(SNAME("margin_bottom"), 8 * theme_cache.base_scale);
}

void ColorPickerShapeWheel::grab_focus() {
	wheel_uv->grab_focus();
}

void ColorPickerShapeCircle::update_circle_cursor(const Vector2 &p_color_change_vector, const Vector2 &p_center, const Vector2 &p_hue_offset) {
	if (circle_keyboard_joypad_picker_cursor_position == Vector2i()) {
		circle_keyboard_joypad_picker_cursor_position = p_center + p_hue_offset;
	}

	Vector2i potential_cursor_position = circle_keyboard_joypad_picker_cursor_position + p_color_change_vector;
	real_t potential_new_cursor_distance = p_center.distance_to(potential_cursor_position);
	real_t dist_pre = p_center.distance_to(circle_keyboard_joypad_picker_cursor_position);
	if (color_picker->s < 1 || potential_new_cursor_distance < dist_pre) {
		circle_keyboard_joypad_picker_cursor_position += p_color_change_vector;
		real_t dist = p_center.distance_to(circle_keyboard_joypad_picker_cursor_position);
		real_t rad = p_center.angle_to_point(circle_keyboard_joypad_picker_cursor_position);
		color_picker->h = ((rad >= 0) ? rad : (Math_TAU + rad)) / Math_TAU;
		color_picker->s = CLAMP(dist / p_center.x, 0, 1);
	} else {
		color_picker->h = get_h_on_circle_edge(p_color_change_vector);
		circle_keyboard_joypad_picker_cursor_position = Vector2i();
	}
}

void ColorPickerShapeCircle::_initialize_controls() {
	circle_margin = memnew(MarginContainer);
	color_picker->shape_container->add_child(circle_margin);

	Ref<ShaderMaterial> material;
	material.instantiate();
	material->set_shader(_get_shader());

	circle = memnew(Control);
	circle->set_material(material);
	circle_margin->add_child(circle);
	circle->connect(SceneStringName(draw), callable_mp(this, &ColorPickerShapeCircle::_circle_draw));

	circle_overlay = memnew(Control);
	circle_overlay->set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT);
	circle->add_child(circle_overlay);
	circle_overlay->connect(SceneStringName(gui_input), callable_mp(this, &ColorPickerShapeCircle::_circle_input));
	circle_overlay->connect(SceneStringName(draw), callable_mp(this, &ColorPickerShapeCircle::_circle_overlay_draw));
	connect_shape_focus(circle_overlay);

	value_slider = memnew(Control);
	color_picker->shape_container->add_child(value_slider);
	value_slider->connect(SceneStringName(gui_input), callable_mp(this, &ColorPickerShapeCircle::_value_slider_input));
	value_slider->connect(SceneStringName(draw), callable_mp(this, &ColorPickerShapeCircle::_value_slider_draw));
	connect_shape_focus(value_slider);

	controls.append(circle_margin);
	controls.append(circle);
	controls.append(circle_overlay);
	controls.append(value_slider);
}

void ColorPickerShapeCircle::update_theme() {
	const ColorPicker::ThemeCache &theme_cache = color_picker->theme_cache;
	circle_margin->set_custom_minimum_size(Size2(theme_cache.sv_width, theme_cache.sv_height));
	circle_margin->add_theme_constant_override(SNAME("margin_bottom"), 8 * theme_cache.base_scale);
	value_slider->set_custom_minimum_size(Size2(theme_cache.h_width, 0));
}

void ColorPickerShapeCircle::grab_focus() {
	circle_overlay->grab_focus();
}

void ColorPickerShapeVHSCircle::_circle_input(const Ref<InputEvent> &p_event) {
	handle_cursor_editing(p_event, circle_overlay);

	Vector2 event_position;
	bool is_click = false;
	if (!can_handle(p_event, event_position, &is_click)) {
		return;
	}

	Vector2 center = circle->get_size() * 0.5;
	real_t dist = center.distance_to(event_position);
	if (is_click && dist > center.x) {
		// Clicked outside the circle.
		cancel_event();
		return;
	}

	real_t rad = center.angle_to_point(event_position);
	color_picker->h = ((rad >= 0) ? rad : (Math_TAU + rad)) / Math_TAU;
	color_picker->s = CLAMP(dist / center.x, 0, 1);
	color_picker->ok_hsl_h = color_picker->h;
	color_picker->ok_hsl_s = color_picker->s;
	circle_keyboard_joypad_picker_cursor_position = Vector2i();

	apply_color();
}

void ColorPickerShapeVHSCircle::_value_slider_input(const Ref<InputEvent> &p_event) {
	handle_cursor_editing(p_event, value_slider);

	Vector2 event_position;
	if (!can_handle(p_event, event_position)) {
		return;
	}
	color_picker->v = 1.0 - CLAMP(event_position.y / value_slider->get_size().y, 0.0, 1.0);
	apply_color();
}

void ColorPickerShapeVHSCircle::_circle_draw() {
	Ref<ShaderMaterial> material = circle->get_material();
	material->set_shader_parameter(SNAME("v"), color_picker->v);
	circle->draw_rect(Rect2(Point2(), circle->get_size()), Color(1, 1, 1));
}

void ColorPickerShapeVHSCircle::_circle_overlay_draw() {
	draw_focus_circle(circle_overlay);
	draw_circle_cursor(circle_overlay, color_picker->h);
}

void ColorPickerShapeVHSCircle::_value_slider_draw() {
	const Vector2 size = value_slider->get_size();
	PackedVector2Array points{
		Vector2(),
		Vector2(size.x, 0),
		size,
		Vector2(0, size.y)
	};

	Color color = Color::from_hsv(color_picker->h, color_picker->s, 1);
	PackedColorArray colors = {
		color,
		color,
		Color(),
		Color()
	};

	value_slider->draw_polygon(points, colors);

	draw_focus_rect(value_slider);

	int y = size.y * (1 - CLAMP(color_picker->v, 0, 1));
	color.set_hsv(color_picker->h, 1, color_picker->v);
	value_slider->draw_line(Vector2(0, y), Vector2(size.x, y), color.inverted());
}

void ColorPickerShapeVHSCircle::_update_cursor(const Vector2 &p_color_change_vector, bool p_is_echo) {
	if (circle_overlay->has_focus()) {
		const Vector2 center = circle_overlay->get_size() / 2.0;
		const Vector2 hue_offset = center * Vector2(Math::cos(color_picker->h * Math_TAU), Math::sin(color_picker->h * Math_TAU)) * color_picker->s;
		update_circle_cursor(p_color_change_vector, center, hue_offset);
	} else if (value_slider->has_focus()) {
		color_picker->v = CLAMP(color_picker->v - p_color_change_vector.y * echo_multiplier / 100.0, 0, 1);
	}
}

void ColorPickerShapeOKHSLCircle::_circle_input(const Ref<InputEvent> &p_event) {
	handle_cursor_editing(p_event, circle_overlay);

	Vector2 event_position;
	bool is_click = false;
	if (!can_handle(p_event, event_position, &is_click)) {
		return;
	}

	const Vector2 center = circle->get_size() * 0.5;
	real_t dist = center.distance_to(event_position);
	if (is_click && dist > center.x) {
		// Clicked outside the circle.
		cancel_event();
		return;
	}

	real_t rad = center.angle_to_point(event_position);
	color_picker->h = ((rad >= 0) ? rad : (Math_TAU + rad)) / Math_TAU;
	color_picker->s = CLAMP(dist / center.x, 0, 1);
	color_picker->ok_hsl_h = color_picker->h;
	color_picker->ok_hsl_s = color_picker->s;
	circle_keyboard_joypad_picker_cursor_position = Vector2i();

	apply_color();
}

void ColorPickerShapeOKHSLCircle::_value_slider_input(const Ref<InputEvent> &p_event) {
	handle_cursor_editing(p_event, value_slider);

	Vector2 event_position;
	if (!can_handle(p_event, event_position)) {
		return;
	}
	color_picker->ok_hsl_l = 1.0 - CLAMP(event_position.y / value_slider->get_size().y, 0.0, 1.0);
	apply_color();
}

void ColorPickerShapeOKHSLCircle::_circle_draw() {
	Ref<ShaderMaterial> material = circle->get_material();
	material->set_shader_parameter(SNAME("ok_hsl_l"), color_picker->ok_hsl_l);
	circle->draw_rect(Rect2(Point2(), circle->get_size()), Color(1, 1, 1));
}

void ColorPickerShapeOKHSLCircle::_circle_overlay_draw() {
	draw_focus_circle(circle_overlay);
	draw_circle_cursor(circle_overlay, color_picker->ok_hsl_h);
}

void ColorPickerShapeOKHSLCircle::_value_slider_draw() {
	const float ok_hsl_h = color_picker->ok_hsl_h;
	const float ok_hsl_s = color_picker->ok_hsl_s;
	const float ok_hsl_l = color_picker->ok_hsl_l;

	const Vector2 size = value_slider->get_size();
	PackedVector2Array points{
		Vector2(size.x, 0),
		Vector2(size.x, size.y * 0.5),
		size,
		Vector2(0, size.y),
		Vector2(0, size.y * 0.5),
		Vector2()
	};

	Color color1 = Color::from_ok_hsl(ok_hsl_h, ok_hsl_s, 1);
	Color color2 = Color::from_ok_hsl(ok_hsl_h, ok_hsl_s, 0.5);
	Color color3 = Color::from_ok_hsl(ok_hsl_h, ok_hsl_s, 0);
	PackedColorArray colors = {
		color1,
		color2,
		color3,
		color3,
		color2,
		color1,
	};
	value_slider->draw_polygon(points, colors);

	draw_focus_rect(value_slider);

	int y = size.y * (1 - CLAMP(ok_hsl_l, 0, 1));
	value_slider->draw_line(Vector2(0, y), Vector2(size.x, y), Color::from_hsv(ok_hsl_h, 1, ok_hsl_l).inverted());
}

void ColorPickerShapeOKHSLCircle::_update_cursor(const Vector2 &p_color_change_vector, bool p_is_echo) {
	if (circle_overlay->has_focus()) {
		const Vector2 center = circle_overlay->get_size() / 2.0;
		const Vector2 hue_offset = center * Vector2(Math::cos(color_picker->ok_hsl_h * Math_TAU), Math::sin(color_picker->ok_hsl_h * Math_TAU)) * color_picker->ok_hsl_s;
		update_circle_cursor(p_color_change_vector, center, hue_offset);
		color_picker->ok_hsl_h = color_picker->h;
		color_picker->ok_hsl_s = color_picker->s;
	} else if (value_slider->has_focus()) {
		color_picker->ok_hsl_l = CLAMP(color_picker->ok_hsl_l - p_color_change_vector.y * echo_multiplier / 100.0, 0, 1);
	}
}
