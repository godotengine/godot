/**************************************************************************/
/*  tab_container.hpp                                                     */
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
#include <godot_cpp/classes/control.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/tab_bar.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/variant.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Node;
class Popup;
class Texture2D;
struct Vector2;

class TabContainer : public Container {
	GDEXTENSION_CLASS(TabContainer, Container)

public:
	enum TabPosition {
		POSITION_TOP = 0,
		POSITION_BOTTOM = 1,
		POSITION_MAX = 2,
	};

	int32_t get_tab_count() const;
	void set_current_tab(int32_t p_tab_idx);
	int32_t get_current_tab() const;
	int32_t get_previous_tab() const;
	bool select_previous_available();
	bool select_next_available();
	Control *get_current_tab_control() const;
	TabBar *get_tab_bar() const;
	Control *get_tab_control(int32_t p_tab_idx) const;
	void set_tab_alignment(TabBar::AlignmentMode p_alignment);
	TabBar::AlignmentMode get_tab_alignment() const;
	void set_tabs_position(TabContainer::TabPosition p_tabs_position);
	TabContainer::TabPosition get_tabs_position() const;
	void set_clip_tabs(bool p_clip_tabs);
	bool get_clip_tabs() const;
	void set_tabs_visible(bool p_visible);
	bool are_tabs_visible() const;
	void set_all_tabs_in_front(bool p_is_front);
	bool is_all_tabs_in_front() const;
	void set_tab_title(int32_t p_tab_idx, const String &p_title);
	String get_tab_title(int32_t p_tab_idx) const;
	void set_tab_tooltip(int32_t p_tab_idx, const String &p_tooltip);
	String get_tab_tooltip(int32_t p_tab_idx) const;
	void set_tab_icon(int32_t p_tab_idx, const Ref<Texture2D> &p_icon);
	Ref<Texture2D> get_tab_icon(int32_t p_tab_idx) const;
	void set_tab_icon_max_width(int32_t p_tab_idx, int32_t p_width);
	int32_t get_tab_icon_max_width(int32_t p_tab_idx) const;
	void set_tab_disabled(int32_t p_tab_idx, bool p_disabled);
	bool is_tab_disabled(int32_t p_tab_idx) const;
	void set_tab_hidden(int32_t p_tab_idx, bool p_hidden);
	bool is_tab_hidden(int32_t p_tab_idx) const;
	void set_tab_metadata(int32_t p_tab_idx, const Variant &p_metadata);
	Variant get_tab_metadata(int32_t p_tab_idx) const;
	void set_tab_button_icon(int32_t p_tab_idx, const Ref<Texture2D> &p_icon);
	Ref<Texture2D> get_tab_button_icon(int32_t p_tab_idx) const;
	int32_t get_tab_idx_at_point(const Vector2 &p_point) const;
	int32_t get_tab_idx_from_control(Control *p_control) const;
	void set_popup(Node *p_popup);
	Popup *get_popup() const;
	void set_switch_on_drag_hover(bool p_enabled);
	bool get_switch_on_drag_hover() const;
	void set_drag_to_rearrange_enabled(bool p_enabled);
	bool get_drag_to_rearrange_enabled() const;
	void set_tabs_rearrange_group(int32_t p_group_id);
	int32_t get_tabs_rearrange_group() const;
	void set_use_hidden_tabs_for_min_size(bool p_enabled);
	bool get_use_hidden_tabs_for_min_size() const;
	void set_tab_focus_mode(Control::FocusMode p_focus_mode);
	Control::FocusMode get_tab_focus_mode() const;
	void set_deselect_enabled(bool p_enabled);
	bool get_deselect_enabled() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Container::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(TabContainer::TabPosition);

