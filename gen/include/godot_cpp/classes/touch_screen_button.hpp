/**************************************************************************/
/*  touch_screen_button.hpp                                               */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#pragma once

#include <godot_cpp/classes/node2d.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/variant/string.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class BitMap;
class Shape2D;
class Texture2D;

class TouchScreenButton : public Node2D {
	GDEXTENSION_CLASS(TouchScreenButton, Node2D)

public:
	enum VisibilityMode {
		VISIBILITY_ALWAYS = 0,
		VISIBILITY_TOUCHSCREEN_ONLY = 1,
	};

	void set_texture_normal(const Ref<Texture2D> &p_texture);
	Ref<Texture2D> get_texture_normal() const;
	void set_texture_pressed(const Ref<Texture2D> &p_texture);
	Ref<Texture2D> get_texture_pressed() const;
	void set_bitmask(const Ref<BitMap> &p_bitmask);
	Ref<BitMap> get_bitmask() const;
	void set_shape(const Ref<Shape2D> &p_shape);
	Ref<Shape2D> get_shape() const;
	void set_shape_centered(bool p_bool);
	bool is_shape_centered() const;
	void set_shape_visible(bool p_bool);
	bool is_shape_visible() const;
	void set_action(const String &p_action);
	String get_action() const;
	void set_visibility_mode(TouchScreenButton::VisibilityMode p_mode);
	TouchScreenButton::VisibilityMode get_visibility_mode() const;
	void set_passby_press(bool p_enabled);
	bool is_passby_press_enabled() const;
	bool is_pressed() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Node2D::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(TouchScreenButton::VisibilityMode);

