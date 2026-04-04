/**************************************************************************/
/*  split_container.hpp                                                   */
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
#include <godot_cpp/variant/packed_int32_array.hpp>
#include <godot_cpp/variant/typed_array.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Control;

class SplitContainer : public Container {
	GDEXTENSION_CLASS(SplitContainer, Container)

public:
	enum DraggerVisibility {
		DRAGGER_VISIBLE = 0,
		DRAGGER_HIDDEN = 1,
		DRAGGER_HIDDEN_COLLAPSED = 2,
	};

	void set_split_offsets(const PackedInt32Array &p_offsets);
	PackedInt32Array get_split_offsets() const;
	void clamp_split_offset(int32_t p_priority_index = 0);
	void set_collapsed(bool p_collapsed);
	bool is_collapsed() const;
	void set_dragger_visibility(SplitContainer::DraggerVisibility p_mode);
	SplitContainer::DraggerVisibility get_dragger_visibility() const;
	void set_vertical(bool p_vertical);
	bool is_vertical() const;
	void set_dragging_enabled(bool p_dragging_enabled);
	bool is_dragging_enabled() const;
	void set_drag_area_margin_begin(int32_t p_margin);
	int32_t get_drag_area_margin_begin() const;
	void set_drag_area_margin_end(int32_t p_margin);
	int32_t get_drag_area_margin_end() const;
	void set_drag_area_offset(int32_t p_offset);
	int32_t get_drag_area_offset() const;
	void set_drag_area_highlight_in_editor(bool p_drag_area_highlight_in_editor);
	bool is_drag_area_highlight_in_editor_enabled() const;
	TypedArray<Control> get_drag_area_controls();
	void set_touch_dragger_enabled(bool p_enabled);
	bool is_touch_dragger_enabled() const;
	Control *get_drag_area_control();
	void set_split_offset(int32_t p_offset);
	int32_t get_split_offset() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Container::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(SplitContainer::DraggerVisibility);

