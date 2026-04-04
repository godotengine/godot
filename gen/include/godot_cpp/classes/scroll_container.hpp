/**************************************************************************/
/*  scroll_container.hpp                                                  */
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

#include <godot_cpp/classes/container.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Control;
class HScrollBar;
class VScrollBar;

class ScrollContainer : public Container {
	GDEXTENSION_CLASS(ScrollContainer, Container)

public:
	enum ScrollMode {
		SCROLL_MODE_DISABLED = 0,
		SCROLL_MODE_AUTO = 1,
		SCROLL_MODE_SHOW_ALWAYS = 2,
		SCROLL_MODE_SHOW_NEVER = 3,
		SCROLL_MODE_RESERVE = 4,
	};

	enum ScrollHintMode {
		SCROLL_HINT_MODE_DISABLED = 0,
		SCROLL_HINT_MODE_ALL = 1,
		SCROLL_HINT_MODE_TOP_AND_LEFT = 2,
		SCROLL_HINT_MODE_BOTTOM_AND_RIGHT = 3,
	};

	void set_h_scroll(int32_t p_value);
	int32_t get_h_scroll() const;
	void set_v_scroll(int32_t p_value);
	int32_t get_v_scroll() const;
	void set_horizontal_custom_step(float p_value);
	float get_horizontal_custom_step() const;
	void set_vertical_custom_step(float p_value);
	float get_vertical_custom_step() const;
	void set_horizontal_scroll_mode(ScrollContainer::ScrollMode p_enable);
	ScrollContainer::ScrollMode get_horizontal_scroll_mode() const;
	void set_vertical_scroll_mode(ScrollContainer::ScrollMode p_enable);
	ScrollContainer::ScrollMode get_vertical_scroll_mode() const;
	void set_deadzone(int32_t p_deadzone);
	int32_t get_deadzone() const;
	void set_scroll_hint_mode(ScrollContainer::ScrollHintMode p_scroll_hint_mode);
	ScrollContainer::ScrollHintMode get_scroll_hint_mode() const;
	void set_tile_scroll_hint(bool p_tile_scroll_hint);
	bool is_scroll_hint_tiled();
	void set_follow_focus(bool p_enabled);
	bool is_following_focus() const;
	HScrollBar *get_h_scroll_bar();
	VScrollBar *get_v_scroll_bar();
	void ensure_control_visible(Control *p_control);
	void set_draw_focus_border(bool p_draw);
	bool get_draw_focus_border();

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Container::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(ScrollContainer::ScrollMode);
VARIANT_ENUM_CAST(ScrollContainer::ScrollHintMode);

