/*************************************************************************/
/*  tabs.h                                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#ifndef TABS_H
#define TABS_H

#include "scene/gui/control.h"

class Tabs : public Control {
	GDCLASS(Tabs, Control);

public:
	enum TabAlign {

		ALIGN_LEFT,
		ALIGN_CENTER,
		ALIGN_RIGHT,
		ALIGN_MAX
	};

	enum CloseButtonDisplayPolicy {

		CLOSE_BUTTON_SHOW_NEVER,
		CLOSE_BUTTON_SHOW_ACTIVE_ONLY,
		CLOSE_BUTTON_SHOW_ALWAYS,
		CLOSE_BUTTON_MAX
	};

private:
	struct Tab {
		String text;
		String xl_text;
		Ref<Texture2D> icon;
		int ofs_cache;
		bool disabled;
		int size_cache;
		int size_text;
		int x_cache;
		int x_size_cache;

		Ref<Texture2D> right_button;
		Rect2 rb_rect;
		Rect2 cb_rect;
	};

	int offset;
	int max_drawn_tab;
	int highlight_arrow;
	bool buttons_visible;
	bool missing_right;
	Vector<Tab> tabs;
	int current;
	int previous;
	int _get_top_margin() const;
	TabAlign tab_align;
	int rb_hover;
	bool rb_pressing;

	bool select_with_rmb;

	int cb_hover;
	bool cb_pressing;
	CloseButtonDisplayPolicy cb_displaypolicy;

	int hover; // Hovered tab.
	int min_width;
	bool scrolling_enabled;
	bool drag_to_rearrange_enabled;
	int tabs_rearrange_group;

	int get_tab_width(int p_idx) const;
	void _ensure_no_over_offset();

	void _update_hover();
	void _update_cache();

	void _on_mouse_exited();

protected:
	void _gui_input(const Ref<InputEvent> &p_event);
	void _notification(int p_what);
	static void _bind_methods();

	Variant get_drag_data(const Point2 &p_point) override;
	bool can_drop_data(const Point2 &p_point, const Variant &p_data) const override;
	void drop_data(const Point2 &p_point, const Variant &p_data) override;
	int get_tab_idx_at_point(const Point2 &p_point) const;

public:
	void add_tab(const String &p_str = "", const Ref<Texture2D> &p_icon = Ref<Texture2D>());

	void set_tab_title(int p_tab, const String &p_title);
	String get_tab_title(int p_tab) const;

	void set_tab_icon(int p_tab, const Ref<Texture2D> &p_icon);
	Ref<Texture2D> get_tab_icon(int p_tab) const;

	void set_tab_disabled(int p_tab, bool p_disabled);
	bool get_tab_disabled(int p_tab) const;

	void set_tab_right_button(int p_tab, const Ref<Texture2D> &p_right_button);
	Ref<Texture2D> get_tab_right_button(int p_tab) const;

	void set_tab_align(TabAlign p_align);
	TabAlign get_tab_align() const;

	void move_tab(int from, int to);

	void set_tab_close_display_policy(CloseButtonDisplayPolicy p_policy);
	CloseButtonDisplayPolicy get_tab_close_display_policy() const;

	int get_tab_count() const;
	void set_current_tab(int p_current);
	int get_current_tab() const;
	int get_previous_tab() const;
	int get_hovered_tab() const;

	int get_tab_offset() const;
	bool get_offset_buttons_visible() const;

	void remove_tab(int p_idx);

	void clear_tabs();

	void set_scrolling_enabled(bool p_enabled);
	bool get_scrolling_enabled() const;

	void set_drag_to_rearrange_enabled(bool p_enabled);
	bool get_drag_to_rearrange_enabled() const;
	void set_tabs_rearrange_group(int p_group_id);
	int get_tabs_rearrange_group() const;

	void set_select_with_rmb(bool p_enabled);
	bool get_select_with_rmb() const;

	void ensure_tab_visible(int p_idx);
	void set_min_width(int p_width);

	Rect2 get_tab_rect(int p_tab) const;
	Size2 get_minimum_size() const override;

	Tabs();
};

VARIANT_ENUM_CAST(Tabs::TabAlign);
VARIANT_ENUM_CAST(Tabs::CloseButtonDisplayPolicy);

#endif // TABS_H
