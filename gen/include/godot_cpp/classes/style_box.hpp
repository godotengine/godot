/**************************************************************************/
/*  style_box.hpp                                                         */
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

#include <godot_cpp/classes/global_constants.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/resource.hpp>
#include <godot_cpp/variant/rect2.hpp>
#include <godot_cpp/variant/vector2.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class CanvasItem;
class RID;

class StyleBox : public Resource {
	GDEXTENSION_CLASS(StyleBox, Resource)

public:
	Vector2 get_minimum_size() const;
	void set_content_margin(Side p_margin, float p_offset);
	void set_content_margin_all(float p_offset);
	float get_content_margin(Side p_margin) const;
	float get_margin(Side p_margin) const;
	Vector2 get_offset() const;
	void draw(const RID &p_canvas_item, const Rect2 &p_rect) const;
	CanvasItem *get_current_item_drawn() const;
	bool test_mask(const Vector2 &p_point, const Rect2 &p_rect) const;
	virtual void _draw(const RID &p_to_canvas_item, const Rect2 &p_rect) const;
	virtual Rect2 _get_draw_rect(const Rect2 &p_rect) const;
	virtual Vector2 _get_minimum_size() const;
	virtual bool _test_mask(const Vector2 &p_point, const Rect2 &p_rect) const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Resource::register_virtuals<T, B>();
		if constexpr (!std::is_same_v<decltype(&B::_draw), decltype(&T::_draw)>) {
			BIND_VIRTUAL_METHOD(T, _draw, 2275962004);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_draw_rect), decltype(&T::_get_draw_rect)>) {
			BIND_VIRTUAL_METHOD(T, _get_draw_rect, 408950903);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_minimum_size), decltype(&T::_get_minimum_size)>) {
			BIND_VIRTUAL_METHOD(T, _get_minimum_size, 3341600327);
		}
		if constexpr (!std::is_same_v<decltype(&B::_test_mask), decltype(&T::_test_mask)>) {
			BIND_VIRTUAL_METHOD(T, _test_mask, 3735564539);
		}
	}

public:
};

} // namespace godot

