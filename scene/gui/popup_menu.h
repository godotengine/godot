/*************************************************************************/
/*  popup_menu.h                                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef POPUP_MENU_H
#define POPUP_MENU_H

#include "core/input/shortcut.h"
#include "scene/gui/margin_container.h"
#include "scene/gui/popup.h"
#include "scene/gui/scroll_container.h"
#include "scene/resources/text_line.h"

class PopupMenu : public Popup {
	GDCLASS(PopupMenu, Popup);

	struct Item {
		Ref<Texture2D> icon;
		String text;
		String xl_text;
		Ref<TextLine> text_buf;
		Ref<TextLine> accel_text_buf;

		Dictionary opentype_features;
		String language;
		Control::TextDirection text_direction = Control::TEXT_DIRECTION_AUTO;

		bool checked = false;
		enum {
			CHECKABLE_TYPE_NONE,
			CHECKABLE_TYPE_CHECK_BOX,
			CHECKABLE_TYPE_RADIO_BUTTON,
		} checkable_type;
		int max_states = 0;
		int state = 0;
		bool separator = false;
		bool disabled = false;
		bool dirty = true;
		int id = 0;
		Variant metadata;
		String submenu;
		String tooltip;
		Key accel = Key::NONE;
		int _ofs_cache = 0;
		int _height_cache = 0;
		int h_ofs = 0;
		Ref<Shortcut> shortcut;
		bool shortcut_is_global = false;
		bool shortcut_is_disabled = false;

		// Returns (0,0) if icon is null.
		Size2 get_icon_size() const {
			return icon.is_null() ? Size2() : icon->get_size();
		}

		Item() {
			text_buf.instantiate();
			accel_text_buf.instantiate();
			checkable_type = CHECKABLE_TYPE_NONE;
		}
	};

	bool close_allowed = false;

	Timer *minimum_lifetime_timer = nullptr;
	Timer *submenu_timer;
	List<Rect2> autohide_areas;
	Vector<Item> items;
	MouseButton initial_button_mask = MouseButton::NONE;
	bool during_grabbed_click = false;
	int mouse_over = -1;
	int submenu_over = -1;
	Rect2 parent_rect;
	String _get_accel_text(const Item &p_item) const;
	int _get_mouse_over(const Point2 &p_over) const;
	virtual Size2 _get_contents_minimum_size() const override;

	int _get_item_height(int p_item) const;
	int _get_items_total_height() const;
	void _scroll_to_item(int p_item);

	void _shape_item(int p_item);

	virtual void gui_input(const Ref<InputEvent> &p_event);
	void _activate_submenu(int p_over);
	void _submenu_timeout();

	uint64_t popup_time_msec = 0;
	bool hide_on_item_selection = true;
	bool hide_on_checkable_item_selection = true;
	bool hide_on_multistate_item_selection = false;
	Vector2 moved;

	Map<Ref<Shortcut>, int> shortcut_refcount;

	void _ref_shortcut(Ref<Shortcut> p_sc);
	void _unref_shortcut(Ref<Shortcut> p_sc);

	bool allow_search = true;
	uint64_t search_time_msec = 0;
	String search_string = "";

	MarginContainer *margin_container;
	ScrollContainer *scroll_container;
	Control *control;

	void _draw_items();
	void _draw_background();

	void _minimum_lifetime_timeout();
	void _close_pressed();

protected:
	void _notification(int p_what);
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;
	static void _bind_methods();

public:
	// ATTENTION: This is used by the POT generator's scene parser. If the number of properties returned by `_get_items()` ever changes,
	// this value should be updated to reflect the new size.
	static const int ITEM_PROPERTY_SIZE = 10;

	void add_item(const String &p_label, int p_id = -1, Key p_accel = Key::NONE);
	void add_icon_item(const Ref<Texture2D> &p_icon, const String &p_label, int p_id = -1, Key p_accel = Key::NONE);
	void add_check_item(const String &p_label, int p_id = -1, Key p_accel = Key::NONE);
	void add_icon_check_item(const Ref<Texture2D> &p_icon, const String &p_label, int p_id = -1, Key p_accel = Key::NONE);
	void add_radio_check_item(const String &p_label, int p_id = -1, Key p_accel = Key::NONE);
	void add_icon_radio_check_item(const Ref<Texture2D> &p_icon, const String &p_label, int p_id = -1, Key p_accel = Key::NONE);

	void add_multistate_item(const String &p_label, int p_max_states, int p_default_state = 0, int p_id = -1, Key p_accel = Key::NONE);

	void add_shortcut(const Ref<Shortcut> &p_shortcut, int p_id = -1, bool p_global = false);
	void add_icon_shortcut(const Ref<Texture2D> &p_icon, const Ref<Shortcut> &p_shortcut, int p_id = -1, bool p_global = false);
	void add_check_shortcut(const Ref<Shortcut> &p_shortcut, int p_id = -1, bool p_global = false);
	void add_icon_check_shortcut(const Ref<Texture2D> &p_icon, const Ref<Shortcut> &p_shortcut, int p_id = -1, bool p_global = false);
	void add_radio_check_shortcut(const Ref<Shortcut> &p_shortcut, int p_id = -1, bool p_global = false);
	void add_icon_radio_check_shortcut(const Ref<Texture2D> &p_icon, const Ref<Shortcut> &p_shortcut, int p_id = -1, bool p_global = false);

	void add_submenu_item(const String &p_label, const String &p_submenu, int p_id = -1);

	void set_item_text(int p_idx, const String &p_text);

	void set_item_text_direction(int p_idx, Control::TextDirection p_text_direction);
	void set_item_opentype_feature(int p_idx, const String &p_name, int p_value);
	void clear_item_opentype_features(int p_idx);
	void set_item_language(int p_idx, const String &p_language);
	void set_item_icon(int p_idx, const Ref<Texture2D> &p_icon);
	void set_item_checked(int p_idx, bool p_checked);
	void set_item_id(int p_idx, int p_id);
	void set_item_accelerator(int p_idx, Key p_accel);
	void set_item_metadata(int p_idx, const Variant &p_meta);
	void set_item_disabled(int p_idx, bool p_disabled);
	void set_item_submenu(int p_idx, const String &p_submenu);
	void set_item_as_separator(int p_idx, bool p_separator);
	void set_item_as_checkable(int p_idx, bool p_checkable);
	void set_item_as_radio_checkable(int p_idx, bool p_radio_checkable);
	void set_item_tooltip(int p_idx, const String &p_tooltip);
	void set_item_shortcut(int p_idx, const Ref<Shortcut> &p_shortcut, bool p_global = false);
	void set_item_h_offset(int p_idx, int p_offset);
	void set_item_multistate(int p_idx, int p_state);
	void toggle_item_multistate(int p_idx);
	void set_item_shortcut_disabled(int p_idx, bool p_disabled);

	void toggle_item_checked(int p_idx);

	String get_item_text(int p_idx) const;
	Control::TextDirection get_item_text_direction(int p_idx) const;
	int get_item_opentype_feature(int p_idx, const String &p_name) const;
	String get_item_language(int p_idx) const;
	int get_item_idx_from_text(const String &text) const;
	Ref<Texture2D> get_item_icon(int p_idx) const;
	bool is_item_checked(int p_idx) const;
	int get_item_id(int p_idx) const;
	int get_item_index(int p_id) const;
	Key get_item_accelerator(int p_idx) const;
	Variant get_item_metadata(int p_idx) const;
	bool is_item_disabled(int p_idx) const;
	String get_item_submenu(int p_idx) const;
	bool is_item_separator(int p_idx) const;
	bool is_item_checkable(int p_idx) const;
	bool is_item_radio_checkable(int p_idx) const;
	bool is_item_shortcut_disabled(int p_idx) const;
	String get_item_tooltip(int p_idx) const;
	Ref<Shortcut> get_item_shortcut(int p_idx) const;
	int get_item_state(int p_idx) const;

	int get_current_index() const;

	void set_item_count(int p_count);
	int get_item_count() const;

	bool activate_item_by_event(const Ref<InputEvent> &p_event, bool p_for_global_only = false);
	void activate_item(int p_item);

	void remove_item(int p_idx);

	void add_separator(const String &p_text = String(), int p_id = -1);

	void clear();

	void set_parent_rect(const Rect2 &p_rect);

	virtual String get_tooltip(const Point2 &p_pos) const;

	virtual void get_translatable_strings(List<String> *p_strings) const override;

	void add_autohide_area(const Rect2 &p_area);
	void clear_autohide_areas();

	void set_hide_on_item_selection(bool p_enabled);
	bool is_hide_on_item_selection() const;

	void set_hide_on_checkable_item_selection(bool p_enabled);
	bool is_hide_on_checkable_item_selection() const;

	void set_hide_on_multistate_item_selection(bool p_enabled);
	bool is_hide_on_multistate_item_selection() const;

	void set_submenu_popup_delay(float p_time);
	float get_submenu_popup_delay() const;

	void set_allow_search(bool p_allow);
	bool get_allow_search() const;

	virtual void popup(const Rect2 &p_bounds = Rect2());

	void take_mouse_focus();

	PopupMenu();
	~PopupMenu();
};

#endif
