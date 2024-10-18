/**************************************************************************/
/*  option_button.h                                                       */
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

#ifndef OPTION_BUTTON_H
#define OPTION_BUTTON_H

#include "scene/gui/button.h"
#include "scene/gui/popup_menu.h"
#include "scene/property_list_helper.h"

class OptionButton : public Button {
public:
	enum Type {
		DROPDOWN,
		HYBRID,
		CAROUSEL,
	};

private:
	GDCLASS(OptionButton, Button);

	bool disable_shortcuts = false;
	PopupMenu *popup = nullptr;
	int current = -1;
	bool fit_to_longest_item = true;
	Vector2 _cached_size;
	bool cache_refresh_pending = false;
	bool allow_reselect = false;
	bool initialized = false;
	int queued_current = -1;
	MouseButton current_mouse_button = MouseButton::NONE;
	bool carousel_wraparound = true;
	Type type = DROPDOWN;
	DrawMode left_arrow_mode = DRAW_NORMAL;
	DrawMode right_arrow_mode = DRAW_NORMAL;

	struct ThemeCache {
		Ref<StyleBox> normal;

		Color font_color;
		Color font_focus_color;
		Color font_pressed_color;
		Color font_hover_color;
		Color font_hover_pressed_color;
		Color font_disabled_color;

		int h_separation = 0;

		Ref<Texture2D> arrow_icon;
		Ref<Texture2D> left_arrow_normal_icon;
		Ref<Texture2D> left_arrow_hover_icon;
		Ref<Texture2D> left_arrow_hover_pressed_icon;
		Ref<Texture2D> left_arrow_pressed_icon;
		Ref<Texture2D> left_arrow_disabled_icon;
		Ref<Texture2D> right_arrow_normal_icon;
		Ref<Texture2D> right_arrow_hover_icon;
		Ref<Texture2D> right_arrow_hover_pressed_icon;
		Ref<Texture2D> right_arrow_pressed_icon;
		Ref<Texture2D> right_arrow_disabled_icon;
		int arrow_margin = 0;
		int modulate_arrow = 0;
	} theme_cache;

	static inline PropertyListHelper base_property_helper;
	PropertyListHelper property_helper;

	void _focused(int p_which);
	void _selected(int p_which);
	void _select(int p_which, bool p_emit = false);
	void _select_int(int p_which);
	void _refresh_size_cache();
	void _dummy_setter() {} // Stub for PropertyListHelper (_set() doesn't use it).

	bool _is_over_arrow(bool p_arrow, Vector2 p_pos);
	void _set_arrow_mode(bool p_arrow, DrawMode p_mode);

	bool _is_arrow_hovered(bool p_arrow);
	bool _is_arrow_pressed(bool p_arrow);
	bool _is_arrow_disabled(bool p_arrow);
	void _set_arrow_pressed(bool p_arrow, bool p_pressed);
	void _set_arrow_hovered(bool p_arrow, bool p_hovered);
	void _set_arrow_disabled(bool p_arrow, bool p_disabled);

	Ref<Texture2D> _get_arrow_icon(bool p_arrow);

	void _on_left_pressed();
	void _on_right_pressed();
	void _select_previous(bool p_emit);
	void _select_next(bool p_emit);
	bool _has_next_selectable_item();
	bool _has_previous_selectable_item();

	virtual void pressed() override;

protected:
	Size2 get_minimum_size() const override;
	virtual void _queue_update_size_cache() override;

	void _notification(int p_what);
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const { return property_helper.property_get_value(p_name, r_ret); }
	void _get_property_list(List<PropertyInfo> *p_list) const { property_helper.get_property_list(p_list); }
	bool _property_can_revert(const StringName &p_name) const { return property_helper.property_can_revert(p_name); }
	bool _property_get_revert(const StringName &p_name, Variant &r_property) const { return property_helper.property_get_revert(p_name, r_property); }
	void _validate_property(PropertyInfo &p_property) const;
	static void _bind_methods();

	virtual void shortcut_input(const Ref<InputEvent> &p_event) override;
	virtual void gui_input(const Ref<InputEvent> &p_event) override;

public:
	// ATTENTION: This is used by the POT generator's scene parser. If the number of properties returned by `_get_items()` ever changes,
	// this value should be updated to reflect the new size.
	static const int ITEM_PROPERTY_SIZE = 5;

	void add_icon_item(const Ref<Texture2D> &p_icon, const String &p_label, int p_id = -1);
	void add_item(const String &p_label, int p_id = -1);

	void set_item_text(int p_idx, const String &p_text);
	void set_item_icon(int p_idx, const Ref<Texture2D> &p_icon);
	void set_item_id(int p_idx, int p_id);
	void set_item_metadata(int p_idx, const Variant &p_metadata);
	void set_item_disabled(int p_idx, bool p_disabled);
	void set_item_tooltip(int p_idx, const String &p_tooltip);

	String get_item_text(int p_idx) const;
	Ref<Texture2D> get_item_icon(int p_idx) const;
	int get_item_id(int p_idx) const;
	int get_item_index(int p_id) const;
	Variant get_item_metadata(int p_idx) const;
	bool is_item_disabled(int p_idx) const;
	bool is_item_separator(int p_idx) const;
	String get_item_tooltip(int p_idx) const;

	bool has_selectable_items() const;
	int get_selectable_item(bool p_from_last = false) const;

	void set_item_count(int p_count);
	int get_item_count() const;
	void set_fit_to_longest_item(bool p_fit);
	bool is_fit_to_longest_item() const;

	void set_allow_reselect(bool p_allow);
	bool get_allow_reselect() const;

	void add_separator(const String &p_text = "");

	void clear();

	void select(int p_idx);
	int get_selected() const;
	int get_selected_id() const;
	Variant get_selected_metadata() const;

	void remove_item(int p_idx);

	PopupMenu *get_popup() const;
	void show_popup();

	void set_disable_shortcuts(bool p_disabled);

	void set_carousel_wraparound(bool p_carousel_wraparound);
	bool is_carousel_wraparound();

	void set_type(Type p_type);
	Type get_type();

	OptionButton(const String &p_text = String());
	~OptionButton();
};

VARIANT_ENUM_CAST(OptionButton::Type);

#endif // OPTION_BUTTON_H
