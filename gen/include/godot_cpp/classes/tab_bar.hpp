/**************************************************************************/
/*  tab_bar.hpp                                                           */
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
#include <godot_cpp/classes/texture2d.hpp>
#include <godot_cpp/variant/rect2.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/variant.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

struct Vector2;

class TabBar : public Control {
	GDEXTENSION_CLASS(TabBar, Control)

public:
	enum AlignmentMode {
		ALIGNMENT_LEFT = 0,
		ALIGNMENT_CENTER = 1,
		ALIGNMENT_RIGHT = 2,
		ALIGNMENT_MAX = 3,
	};

	enum CloseButtonDisplayPolicy {
		CLOSE_BUTTON_SHOW_NEVER = 0,
		CLOSE_BUTTON_SHOW_ACTIVE_ONLY = 1,
		CLOSE_BUTTON_SHOW_ALWAYS = 2,
		CLOSE_BUTTON_MAX = 3,
	};

	void set_tab_count(int32_t p_count);
	int32_t get_tab_count() const;
	void set_current_tab(int32_t p_tab_idx);
	int32_t get_current_tab() const;
	int32_t get_previous_tab() const;
	bool select_previous_available();
	bool select_next_available();
	void set_tab_title(int32_t p_tab_idx, const String &p_title);
	String get_tab_title(int32_t p_tab_idx) const;
	void set_tab_tooltip(int32_t p_tab_idx, const String &p_tooltip);
	String get_tab_tooltip(int32_t p_tab_idx) const;
	void set_tab_text_direction(int32_t p_tab_idx, Control::TextDirection p_direction);
	Control::TextDirection get_tab_text_direction(int32_t p_tab_idx) const;
	void set_tab_language(int32_t p_tab_idx, const String &p_language);
	String get_tab_language(int32_t p_tab_idx) const;
	void set_tab_icon(int32_t p_tab_idx, const Ref<Texture2D> &p_icon);
	Ref<Texture2D> get_tab_icon(int32_t p_tab_idx) const;
	void set_tab_icon_max_width(int32_t p_tab_idx, int32_t p_width);
	int32_t get_tab_icon_max_width(int32_t p_tab_idx) const;
	void set_tab_button_icon(int32_t p_tab_idx, const Ref<Texture2D> &p_icon);
	Ref<Texture2D> get_tab_button_icon(int32_t p_tab_idx) const;
	void set_tab_disabled(int32_t p_tab_idx, bool p_disabled);
	bool is_tab_disabled(int32_t p_tab_idx) const;
	void set_tab_hidden(int32_t p_tab_idx, bool p_hidden);
	bool is_tab_hidden(int32_t p_tab_idx) const;
	void set_tab_metadata(int32_t p_tab_idx, const Variant &p_metadata);
	Variant get_tab_metadata(int32_t p_tab_idx) const;
	void remove_tab(int32_t p_tab_idx);
	void add_tab(const String &p_title = String(), const Ref<Texture2D> &p_icon = nullptr);
	int32_t get_tab_idx_at_point(const Vector2 &p_point) const;
	void set_tab_alignment(TabBar::AlignmentMode p_alignment);
	TabBar::AlignmentMode get_tab_alignment() const;
	void set_clip_tabs(bool p_clip_tabs);
	bool get_clip_tabs() const;
	int32_t get_tab_offset() const;
	bool get_offset_buttons_visible() const;
	void ensure_tab_visible(int32_t p_idx);
	Rect2 get_tab_rect(int32_t p_tab_idx) const;
	void move_tab(int32_t p_from, int32_t p_to);
	void set_close_with_middle_mouse(bool p_enabled);
	bool get_close_with_middle_mouse() const;
	void set_tab_close_display_policy(TabBar::CloseButtonDisplayPolicy p_policy);
	TabBar::CloseButtonDisplayPolicy get_tab_close_display_policy() const;
	void set_max_tab_width(int32_t p_width);
	int32_t get_max_tab_width() const;
	void set_scrolling_enabled(bool p_enabled);
	bool get_scrolling_enabled() const;
	void set_drag_to_rearrange_enabled(bool p_enabled);
	bool get_drag_to_rearrange_enabled() const;
	void set_switch_on_drag_hover(bool p_enabled);
	bool get_switch_on_drag_hover() const;
	void set_tabs_rearrange_group(int32_t p_group_id);
	int32_t get_tabs_rearrange_group() const;
	void set_scroll_to_selected(bool p_enabled);
	bool get_scroll_to_selected() const;
	void set_select_with_rmb(bool p_enabled);
	bool get_select_with_rmb() const;
	void set_deselect_enabled(bool p_enabled);
	bool get_deselect_enabled() const;
	void clear_tabs();

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Control::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(TabBar::AlignmentMode);
VARIANT_ENUM_CAST(TabBar::CloseButtonDisplayPolicy);

