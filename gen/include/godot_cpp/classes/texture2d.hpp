/**************************************************************************/
/*  texture2d.hpp                                                         */
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

#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/texture.hpp>
#include <godot_cpp/variant/color.hpp>
#include <godot_cpp/variant/vector2.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Image;
class RID;
struct Rect2;
class Resource;

class Texture2D : public Texture {
	GDEXTENSION_CLASS(Texture2D, Texture)

public:
	int32_t get_width() const;
	int32_t get_height() const;
	Vector2 get_size() const;
	bool has_alpha() const;
	void draw(const RID &p_canvas_item, const Vector2 &p_position, const Color &p_modulate = Color(1, 1, 1, 1), bool p_transpose = false) const;
	void draw_rect(const RID &p_canvas_item, const Rect2 &p_rect, bool p_tile, const Color &p_modulate = Color(1, 1, 1, 1), bool p_transpose = false) const;
	void draw_rect_region(const RID &p_canvas_item, const Rect2 &p_rect, const Rect2 &p_src_rect, const Color &p_modulate = Color(1, 1, 1, 1), bool p_transpose = false, bool p_clip_uv = true) const;
	Ref<Image> get_image() const;
	Ref<Resource> create_placeholder() const;
	virtual int32_t _get_width() const;
	virtual int32_t _get_height() const;
	virtual bool _is_pixel_opaque(int32_t p_x, int32_t p_y) const;
	virtual bool _has_alpha() const;
	virtual void _draw(const RID &p_to_canvas_item, const Vector2 &p_pos, const Color &p_modulate, bool p_transpose) const;
	virtual void _draw_rect(const RID &p_to_canvas_item, const Rect2 &p_rect, bool p_tile, const Color &p_modulate, bool p_transpose) const;
	virtual void _draw_rect_region(const RID &p_to_canvas_item, const Rect2 &p_rect, const Rect2 &p_src_rect, const Color &p_modulate, bool p_transpose, bool p_clip_uv) const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Texture::register_virtuals<T, B>();
		if constexpr (!std::is_same_v<decltype(&B::_get_width), decltype(&T::_get_width)>) {
			BIND_VIRTUAL_METHOD(T, _get_width, 3905245786);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_height), decltype(&T::_get_height)>) {
			BIND_VIRTUAL_METHOD(T, _get_height, 3905245786);
		}
		if constexpr (!std::is_same_v<decltype(&B::_is_pixel_opaque), decltype(&T::_is_pixel_opaque)>) {
			BIND_VIRTUAL_METHOD(T, _is_pixel_opaque, 2522259332);
		}
		if constexpr (!std::is_same_v<decltype(&B::_has_alpha), decltype(&T::_has_alpha)>) {
			BIND_VIRTUAL_METHOD(T, _has_alpha, 36873697);
		}
		if constexpr (!std::is_same_v<decltype(&B::_draw), decltype(&T::_draw)>) {
			BIND_VIRTUAL_METHOD(T, _draw, 1384643611);
		}
		if constexpr (!std::is_same_v<decltype(&B::_draw_rect), decltype(&T::_draw_rect)>) {
			BIND_VIRTUAL_METHOD(T, _draw_rect, 3819628907);
		}
		if constexpr (!std::is_same_v<decltype(&B::_draw_rect_region), decltype(&T::_draw_rect_region)>) {
			BIND_VIRTUAL_METHOD(T, _draw_rect_region, 4094143664);
		}
	}

public:
};

} // namespace godot

