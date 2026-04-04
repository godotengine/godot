/**************************************************************************/
/*  texture_button.hpp                                                    */
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

#include <godot_cpp/classes/base_button.hpp>
#include <godot_cpp/classes/ref.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class BitMap;
class Texture2D;

class TextureButton : public BaseButton {
	GDEXTENSION_CLASS(TextureButton, BaseButton)

public:
	enum StretchMode {
		STRETCH_SCALE = 0,
		STRETCH_TILE = 1,
		STRETCH_KEEP = 2,
		STRETCH_KEEP_CENTERED = 3,
		STRETCH_KEEP_ASPECT = 4,
		STRETCH_KEEP_ASPECT_CENTERED = 5,
		STRETCH_KEEP_ASPECT_COVERED = 6,
	};

	void set_texture_normal(const Ref<Texture2D> &p_texture);
	void set_texture_pressed(const Ref<Texture2D> &p_texture);
	void set_texture_hover(const Ref<Texture2D> &p_texture);
	void set_texture_disabled(const Ref<Texture2D> &p_texture);
	void set_texture_focused(const Ref<Texture2D> &p_texture);
	void set_click_mask(const Ref<BitMap> &p_mask);
	void set_ignore_texture_size(bool p_ignore);
	void set_stretch_mode(TextureButton::StretchMode p_mode);
	void set_flip_h(bool p_enable);
	bool is_flipped_h() const;
	void set_flip_v(bool p_enable);
	bool is_flipped_v() const;
	Ref<Texture2D> get_texture_normal() const;
	Ref<Texture2D> get_texture_pressed() const;
	Ref<Texture2D> get_texture_hover() const;
	Ref<Texture2D> get_texture_disabled() const;
	Ref<Texture2D> get_texture_focused() const;
	Ref<BitMap> get_click_mask() const;
	bool get_ignore_texture_size() const;
	TextureButton::StretchMode get_stretch_mode() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		BaseButton::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(TextureButton::StretchMode);

