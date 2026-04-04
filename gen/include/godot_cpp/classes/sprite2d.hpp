/**************************************************************************/
/*  sprite2d.hpp                                                          */
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
#include <godot_cpp/variant/rect2.hpp>
#include <godot_cpp/variant/vector2.hpp>
#include <godot_cpp/variant/vector2i.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Texture2D;

class Sprite2D : public Node2D {
	GDEXTENSION_CLASS(Sprite2D, Node2D)

public:
	void set_texture(const Ref<Texture2D> &p_texture);
	Ref<Texture2D> get_texture() const;
	void set_centered(bool p_centered);
	bool is_centered() const;
	void set_offset(const Vector2 &p_offset);
	Vector2 get_offset() const;
	void set_flip_h(bool p_flip_h);
	bool is_flipped_h() const;
	void set_flip_v(bool p_flip_v);
	bool is_flipped_v() const;
	void set_region_enabled(bool p_enabled);
	bool is_region_enabled() const;
	bool is_pixel_opaque(const Vector2 &p_pos) const;
	void set_region_rect(const Rect2 &p_rect);
	Rect2 get_region_rect() const;
	void set_region_filter_clip_enabled(bool p_enabled);
	bool is_region_filter_clip_enabled() const;
	void set_frame(int32_t p_frame);
	int32_t get_frame() const;
	void set_frame_coords(const Vector2i &p_coords);
	Vector2i get_frame_coords() const;
	void set_vframes(int32_t p_vframes);
	int32_t get_vframes() const;
	void set_hframes(int32_t p_hframes);
	int32_t get_hframes() const;
	Rect2 get_rect() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Node2D::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

