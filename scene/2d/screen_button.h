/*************************************************************************/
/*  screen_button.h                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifndef SCREEN_BUTTON_H
#define SCREEN_BUTTON_H

#include "scene/2d/node_2d.h"
#include "scene/resources/bit_mask.h"
#include "scene/resources/rectangle_shape_2d.h"
#include "scene/resources/texture.h"

class TouchScreenButton : public Node2D {

	GDCLASS(TouchScreenButton, Node2D);

public:
	enum VisibilityMode {
		VISIBILITY_ALWAYS,
		VISIBILITY_TOUCHSCREEN_ONLY
	};

private:
	Ref<Texture> texture;
	Ref<Texture> texture_pressed;
	Ref<BitMap> bitmask;
	Ref<Shape2D> shape;
	bool shape_centered;
	bool shape_visible;

	Ref<RectangleShape2D> unit_rect;

	StringName action;
	bool passby_press;
	int finger_pressed;

	VisibilityMode visibility;

	void _input(const Ref<InputEvent> &p_event);

	bool _is_point_inside(const Point2 &p_point);

	void _press(int p_finger_pressed);
	void _release(bool p_exiting_tree = false);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_texture(const Ref<Texture> &p_texture);
	Ref<Texture> get_texture() const;

	void set_texture_pressed(const Ref<Texture> &p_texture_pressed);
	Ref<Texture> get_texture_pressed() const;

	void set_bitmask(const Ref<BitMap> &p_bitmask);
	Ref<BitMap> get_bitmask() const;

	void set_shape(const Ref<Shape2D> &p_shape);
	Ref<Shape2D> get_shape() const;

	void set_shape_centered(bool p_shape_centered);
	bool is_shape_centered() const;

	void set_shape_visible(bool p_shape_visible);
	bool is_shape_visible() const;

	void set_action(const String &p_action);
	String get_action() const;

	void set_passby_press(bool p_enable);
	bool is_passby_press_enabled() const;

	void set_visibility_mode(VisibilityMode p_mode);
	VisibilityMode get_visibility_mode() const;

	bool is_pressed() const;

	Rect2 _edit_get_rect() const;

	TouchScreenButton();
};

VARIANT_ENUM_CAST(TouchScreenButton::VisibilityMode);

#endif // SCREEN_BUTTON_H
