/**************************************************************************/
/*  texture_rect.hpp                                                      */
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

#include <godot_cpp/classes/control.hpp>
#include <godot_cpp/classes/ref.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Texture2D;

class TextureRect : public Control {
	GDEXTENSION_CLASS(TextureRect, Control)

public:
	enum ExpandMode {
		EXPAND_KEEP_SIZE = 0,
		EXPAND_IGNORE_SIZE = 1,
		EXPAND_FIT_WIDTH = 2,
		EXPAND_FIT_WIDTH_PROPORTIONAL = 3,
		EXPAND_FIT_HEIGHT = 4,
		EXPAND_FIT_HEIGHT_PROPORTIONAL = 5,
	};

	enum StretchMode {
		STRETCH_SCALE = 0,
		STRETCH_TILE = 1,
		STRETCH_KEEP = 2,
		STRETCH_KEEP_CENTERED = 3,
		STRETCH_KEEP_ASPECT = 4,
		STRETCH_KEEP_ASPECT_CENTERED = 5,
		STRETCH_KEEP_ASPECT_COVERED = 6,
	};

	void set_texture(const Ref<Texture2D> &p_texture);
	Ref<Texture2D> get_texture() const;
	void set_expand_mode(TextureRect::ExpandMode p_expand_mode);
	TextureRect::ExpandMode get_expand_mode() const;
	void set_flip_h(bool p_enable);
	bool is_flipped_h() const;
	void set_flip_v(bool p_enable);
	bool is_flipped_v() const;
	void set_stretch_mode(TextureRect::StretchMode p_stretch_mode);
	TextureRect::StretchMode get_stretch_mode() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Control::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(TextureRect::ExpandMode);
VARIANT_ENUM_CAST(TextureRect::StretchMode);

