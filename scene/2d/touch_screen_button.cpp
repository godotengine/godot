/**************************************************************************/
/*  touch_screen_button.cpp                                               */
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

#include "touch_screen_button.h"

#include "core/input_map.h"
#include "core/os/input.h"
#include "core/os/os.h"

void TouchScreenButton::set_texture(const Ref<Texture> &p_texture) {
	texture = p_texture;
	update();
}

Ref<Texture> TouchScreenButton::get_texture() const {
	return texture;
}

void TouchScreenButton::set_texture_pressed(const Ref<Texture> &p_texture_pressed) {
	texture_pressed = p_texture_pressed;
	update();
}

Ref<Texture> TouchScreenButton::get_texture_pressed() const {
	return texture_pressed;
}

void TouchScreenButton::set_bitmask(const Ref<BitMap> &p_bitmask) {
	bitmask = p_bitmask;
}

Ref<BitMap> TouchScreenButton::get_bitmask() const {
	return bitmask;
}

void TouchScreenButton::set_shape(const Ref<Shape2D> &p_shape) {
	if (shape.is_valid()) {
		shape->disconnect("changed", this, "update");
	}

	shape = p_shape;

	if (shape.is_valid()) {
		shape->connect("changed", this, "update");
	}

	update();
}

Ref<Shape2D> TouchScreenButton::get_shape() const {
	return shape;
}

void TouchScreenButton::set_shape_centered(bool p_shape_centered) {
	shape_centered = p_shape_centered;
	update();
}

bool TouchScreenButton::is_shape_visible() const {
	return shape_visible;
}

void TouchScreenButton::set_shape_visible(bool p_shape_visible) {
	shape_visible = p_shape_visible;
	update();
}

bool TouchScreenButton::is_shape_centered() const {
	return shape_centered;
}

void TouchScreenButton::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_DRAW: {
			if (!is_inside_tree()) {
				return;
			}
			if (!Engine::get_singleton()->is_editor_hint() && !OS::get_singleton()->has_touchscreen_ui_hint() && visibility == VISIBILITY_TOUCHSCREEN_ONLY) {
				return;
			}

			if (finger_pressed != -1) {
				if (texture_pressed.is_valid()) {
					draw_texture(texture_pressed, Point2());
				} else if (texture.is_valid()) {
					draw_texture(texture, Point2());
				}

			} else {
				if (texture.is_valid()) {
					draw_texture(texture, Point2());
				}
			}

			if (!shape_visible) {
				return;
			}
			if (!Engine::get_singleton()->is_editor_hint() && !get_tree()->is_debugging_collisions_hint()) {
				return;
			}
			if (shape.is_valid()) {
				Color draw_col = get_tree()->get_debug_collisions_color();

				Vector2 size = texture.is_null() ? shape->get_rect().size : texture->get_size();
				Vector2 pos = shape_centered ? size * 0.5f : Vector2();
				draw_set_transform_matrix(get_canvas_transform().translated(pos));
				shape->draw(get_canvas_item(), draw_col);
			}

		} break;
		case NOTIFICATION_ENTER_TREE: {
			if (!Engine::get_singleton()->is_editor_hint() && !OS::get_singleton()->has_touchscreen_ui_hint() && visibility == VISIBILITY_TOUCHSCREEN_ONLY) {
				return;
			}
			update();

			if (!Engine::get_singleton()->is_editor_hint()) {
				set_process_input(is_visible_in_tree());
			}

		} break;
		case NOTIFICATION_EXIT_TREE: {
			if (is_pressed()) {
				_release(true);
			}
		} break;
		case NOTIFICATION_VISIBILITY_CHANGED: {
			if (Engine::get_singleton()->is_editor_hint()) {
				break;
			}
			if (is_visible_in_tree()) {
				set_process_input(true);
			} else {
				set_process_input(false);
				if (is_pressed()) {
					_release();
				}
			}
		} break;
		case NOTIFICATION_PAUSED: {
			if (is_pressed()) {
				_release();
			}
		} break;
	}
}

bool TouchScreenButton::is_pressed() const {
	return finger_pressed != -1;
}

void TouchScreenButton::set_action(const String &p_action) {
	action = p_action;
}

String TouchScreenButton::get_action() const {
	return action;
}

void TouchScreenButton::_input(const Ref<InputEvent> &p_event) {
	ERR_FAIL_COND(p_event.is_null());

	if (!is_visible_in_tree()) {
		return;
	}

	if (p_event->get_device() != 0) {
		return;
	}

	const InputEventScreenTouch *st = Object::cast_to<InputEventScreenTouch>(*p_event);

	if (passby_press) {
		const InputEventScreenDrag *sd = Object::cast_to<InputEventScreenDrag>(*p_event);

		if (st && !st->is_pressed() && finger_pressed == st->get_index()) {
			_release();
		}

		if ((st && st->is_pressed()) || sd) {
			int index = st ? st->get_index() : sd->get_index();
			Point2 coord = st ? st->get_position() : sd->get_position();

			if (finger_pressed == -1 || index == finger_pressed) {
				if (_is_point_inside(coord)) {
					if (finger_pressed == -1) {
						_press(index);
					}
				} else {
					if (finger_pressed != -1) {
						_release();
					}
				}
			}
		}

	} else {
		if (st) {
			if (st->is_pressed()) {
				const bool can_press = finger_pressed == -1;
				if (!can_press) {
					return; //already fingering
				}

				if (_is_point_inside(st->get_position())) {
					_press(st->get_index());
				}
			} else {
				if (st->get_index() == finger_pressed) {
					_release();
				}
			}
		}
	}
}

bool TouchScreenButton::_is_point_inside(const Point2 &p_point) {
	Point2 coord = (get_global_transform_with_canvas()).affine_inverse().xform(p_point);

	bool touched = false;
	bool check_rect = true;

	if (shape.is_valid()) {
		check_rect = false;

		Vector2 size = texture.is_null() ? shape->get_rect().size : texture->get_size();
		Transform2D xform = shape_centered ? Transform2D().translated(size * 0.5f) : Transform2D();
		touched = shape->collide(xform, unit_rect, Transform2D(0, coord + Vector2(0.5, 0.5)));
	}

	if (bitmask.is_valid()) {
		check_rect = false;
		if (!touched && Rect2(Point2(), bitmask->get_size()).has_point(coord)) {
			if (bitmask->get_bit(coord)) {
				touched = true;
			}
		}
	}

	if (!touched && check_rect) {
		if (texture.is_valid()) {
			touched = Rect2(Size2(), texture->get_size()).has_point(coord);
		}
	}

	return touched;
}

void TouchScreenButton::_press(int p_finger_pressed) {
	finger_pressed = p_finger_pressed;

	if (action != StringName()) {
		Input::get_singleton()->action_press(action);
		Ref<InputEventAction> iea;
		iea.instance();
		iea->set_action(action);
		iea->set_pressed(true);
		get_tree()->input_event(iea);
	}

	emit_signal("pressed");
	update();
}

void TouchScreenButton::_release(bool p_exiting_tree) {
	finger_pressed = -1;

	if (action != StringName()) {
		Input::get_singleton()->action_release(action);
		if (!p_exiting_tree) {
			Ref<InputEventAction> iea;
			iea.instance();
			iea->set_action(action);
			iea->set_pressed(false);
			get_tree()->input_event(iea);
		}
	}

	if (!p_exiting_tree) {
		emit_signal("released");
		update();
	}
}

#ifdef TOOLS_ENABLED
Rect2 TouchScreenButton::_edit_get_rect() const {
	if (texture.is_null()) {
		return CanvasItem::_edit_get_rect();
	}

	return Rect2(Size2(), texture->get_size());
}

bool TouchScreenButton::_edit_use_rect() const {
	return !texture.is_null();
}
#endif

Rect2 TouchScreenButton::get_anchorable_rect() const {
	if (texture.is_null()) {
		return CanvasItem::get_anchorable_rect();
	}

	return Rect2(Size2(), texture->get_size());
}

void TouchScreenButton::set_visibility_mode(VisibilityMode p_mode) {
	visibility = p_mode;
	update();
}

TouchScreenButton::VisibilityMode TouchScreenButton::get_visibility_mode() const {
	return visibility;
}

void TouchScreenButton::set_passby_press(bool p_enable) {
	passby_press = p_enable;
}

bool TouchScreenButton::is_passby_press_enabled() const {
	return passby_press;
}

void TouchScreenButton::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_texture", "texture"), &TouchScreenButton::set_texture);
	ClassDB::bind_method(D_METHOD("get_texture"), &TouchScreenButton::get_texture);

	ClassDB::bind_method(D_METHOD("set_texture_pressed", "texture_pressed"), &TouchScreenButton::set_texture_pressed);
	ClassDB::bind_method(D_METHOD("get_texture_pressed"), &TouchScreenButton::get_texture_pressed);

	ClassDB::bind_method(D_METHOD("set_bitmask", "bitmask"), &TouchScreenButton::set_bitmask);
	ClassDB::bind_method(D_METHOD("get_bitmask"), &TouchScreenButton::get_bitmask);

	ClassDB::bind_method(D_METHOD("set_shape", "shape"), &TouchScreenButton::set_shape);
	ClassDB::bind_method(D_METHOD("get_shape"), &TouchScreenButton::get_shape);

	ClassDB::bind_method(D_METHOD("set_shape_centered", "bool"), &TouchScreenButton::set_shape_centered);
	ClassDB::bind_method(D_METHOD("is_shape_centered"), &TouchScreenButton::is_shape_centered);

	ClassDB::bind_method(D_METHOD("set_shape_visible", "bool"), &TouchScreenButton::set_shape_visible);
	ClassDB::bind_method(D_METHOD("is_shape_visible"), &TouchScreenButton::is_shape_visible);

	ClassDB::bind_method(D_METHOD("set_action", "action"), &TouchScreenButton::set_action);
	ClassDB::bind_method(D_METHOD("get_action"), &TouchScreenButton::get_action);

	ClassDB::bind_method(D_METHOD("set_visibility_mode", "mode"), &TouchScreenButton::set_visibility_mode);
	ClassDB::bind_method(D_METHOD("get_visibility_mode"), &TouchScreenButton::get_visibility_mode);

	ClassDB::bind_method(D_METHOD("set_passby_press", "enabled"), &TouchScreenButton::set_passby_press);
	ClassDB::bind_method(D_METHOD("is_passby_press_enabled"), &TouchScreenButton::is_passby_press_enabled);

	ClassDB::bind_method(D_METHOD("is_pressed"), &TouchScreenButton::is_pressed);

	ClassDB::bind_method(D_METHOD("_input"), &TouchScreenButton::_input);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "normal", PROPERTY_HINT_RESOURCE_TYPE, "Texture"), "set_texture", "get_texture");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "pressed", PROPERTY_HINT_RESOURCE_TYPE, "Texture"), "set_texture_pressed", "get_texture_pressed");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "bitmask", PROPERTY_HINT_RESOURCE_TYPE, "BitMap"), "set_bitmask", "get_bitmask");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "shape", PROPERTY_HINT_RESOURCE_TYPE, "Shape2D"), "set_shape", "get_shape");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "shape_centered"), "set_shape_centered", "is_shape_centered");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "shape_visible"), "set_shape_visible", "is_shape_visible");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "passby_press"), "set_passby_press", "is_passby_press_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "action"), "set_action", "get_action");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "visibility_mode", PROPERTY_HINT_ENUM, "Always,TouchScreen Only"), "set_visibility_mode", "get_visibility_mode");

	ADD_SIGNAL(MethodInfo("pressed"));
	ADD_SIGNAL(MethodInfo("released"));

	BIND_ENUM_CONSTANT(VISIBILITY_ALWAYS);
	BIND_ENUM_CONSTANT(VISIBILITY_TOUCHSCREEN_ONLY);
}

TouchScreenButton::TouchScreenButton() {
	finger_pressed = -1;
	passby_press = false;
	visibility = VISIBILITY_ALWAYS;
	shape_centered = true;
	shape_visible = true;
	unit_rect = Ref<RectangleShape2D>(memnew(RectangleShape2D));
	unit_rect->set_extents(Vector2(0.5, 0.5));
}
