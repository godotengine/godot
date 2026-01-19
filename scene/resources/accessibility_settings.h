/**************************************************************************/
/*  accessibility_settings.h                                              */
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

#pragma once

#include "core/io/resource.h"
#include "scene/property_list_helper.h"
#include "servers/display/accessibility_server.h"
#include "servers/text/text_server.h"

class InputEventKey;

class AccessibilitySettings : public Resource {
	GDCLASS(AccessibilitySettings, Resource);

public:
	enum ValueType {
		VALUE_STRING = (1 << 0),
		VALUE_NUMBER = (1 << 1),
		VALUE_COLOR = (1 << 2),
	};

private:
	struct ActionInfo {
		int64_t id = 0;
		String description;
	};

	ObjectID owner; // Node or parent AccessibilitySettings.
	RID subelement_rid;
	ObjectID focused_subelement;

	String name;
	String description;
	AccessibilityServerEnums::AccessibilityLiveMode accessibility_live = AccessibilityServerEnums::AccessibilityLiveMode::LIVE_OFF;

	AccessibilityServerEnums::AccessibilityRole role = AccessibilityServerEnums::ROLE_CONTAINER;
	String role_description;

	BitField<ValueType> value_type = 0;
	String value;
	real_t value_num = 0.0;
	Vector2 value_num_range = Vector2(-Math::INF, Math::INF);
	real_t value_num_step = 1.0;
	real_t value_num_jump = 1.0;
	Color value_color;

	String state_description;
	String tooltip;
	String extra_info;
	BitField<AccessibilityServerEnums::AccessibilityFlagsBit> flags = 0;
	bool state_checked = false;
	Ref<InputEventKey> shortcut;
	Vector<ActionInfo> custom_actions;
	BitField<AccessibilityServerEnums::AccessibilityActionBit> supported_actions = 0;
	BitField<AccessibilityServerEnums::AccessibilityActionBit> overridden_actions = 0;

	Vector2 scroll_x_range;
	Vector2 scroll_y_range;
	Vector2 scroll_pos;

	HorizontalAlignment text_align = HORIZONTAL_ALIGNMENT_LEFT;
	String placeholder;
	String url;
	String language;
	TextServer::Orientation orientation = TextServer::ORIENTATION_HORIZONTAL;
	Color bg_color = Color(0, 0, 0, 0);
	Color fg_color = Color(0, 0, 0, 1);

	struct {
		int items = 0;
		bool vertical = true;
	} list_data;
	struct {
		int index = -1;
		int level = 0;
		bool selected = false;
		bool expanded = false;
	} list_item_data;
	struct {
		int rows = 0;
		int cols = 0;
	} table_data;
	struct {
		Vector2i pos = Vector2i(-1, -1);
		Vector2i span = Vector2i(1, 1);
	} table_cell_data;

	struct SubElement {
		Ref<AccessibilitySettings> settings;
		Rect2 bounds;
	};
	Vector<SubElement> subelements;

	static inline PropertyListHelper custom_action_base_property_helper;
	PropertyListHelper custom_action_property_helper;

	static inline PropertyListHelper subelement_base_property_helper;
	PropertyListHelper subelement_property_helper;

	void _action_custom(const Variant &p_data);
	void _action_default(AccessibilityServerEnums::AccessibilityAction p_action, const Variant &p_data);

	Node *_get_owner_node() const;

	void _reset_focus();

	bool _is_role_scrollable() const { return (role == AccessibilityServerEnums::AccessibilityRole::ROLE_SCROLL_BAR || role == AccessibilityServerEnums::AccessibilityRole::ROLE_SCROLL_VIEW); }
	bool _is_role_list() const { return (role == AccessibilityServerEnums::AccessibilityRole::ROLE_TREE || role == AccessibilityServerEnums::AccessibilityRole::ROLE_LIST || role == AccessibilityServerEnums::AccessibilityRole::ROLE_LIST_BOX || role == AccessibilityServerEnums::AccessibilityRole::ROLE_TAB_BAR || role == AccessibilityServerEnums::AccessibilityRole::ROLE_MENU); }
	bool _is_role_list_item() const { return (role == AccessibilityServerEnums::AccessibilityRole::ROLE_TREE_ITEM || role == AccessibilityServerEnums::AccessibilityRole::ROLE_LIST_ITEM || role == AccessibilityServerEnums::AccessibilityRole::ROLE_LIST_BOX_OPTION || role == AccessibilityServerEnums::AccessibilityRole::ROLE_TAB || role == AccessibilityServerEnums::AccessibilityRole::ROLE_MENU_ITEM || role == AccessibilityServerEnums::AccessibilityRole::ROLE_MENU_ITEM_CHECK_BOX || role == AccessibilityServerEnums::AccessibilityRole::ROLE_MENU_ITEM_RADIO); }
	bool _is_role_table() const { return (role == AccessibilityServerEnums::AccessibilityRole::ROLE_TABLE); }
	bool _is_role_table_row() const { return (role == AccessibilityServerEnums::AccessibilityRole::ROLE_ROW || role == AccessibilityServerEnums::AccessibilityRole::ROLE_ROW_GROUP || role == AccessibilityServerEnums::AccessibilityRole::ROLE_ROW_HEADER); }
	bool _is_role_table_col() const { return (role == AccessibilityServerEnums::AccessibilityRole::ROLE_COLUMN_HEADER); }
	bool _is_role_table_cell() const { return (role == AccessibilityServerEnums::AccessibilityRole::ROLE_CELL); }

protected:
	static void _bind_methods();
	void _validate_property(PropertyInfo &p_property) const;

	bool _set(const StringName &p_name, const Variant &p_value) {
		return custom_action_property_helper.property_set_value(p_name, p_value) || subelement_property_helper.property_set_value(p_name, p_value);
	}
	bool _get(const StringName &p_name, Variant &r_ret) const {
		return custom_action_property_helper.property_get_value(p_name, r_ret) || subelement_property_helper.property_get_value(p_name, r_ret);
	}
	void _get_property_list(List<PropertyInfo> *p_list) const {
		custom_action_property_helper.get_property_list(p_list);
		subelement_property_helper.get_property_list(p_list);
	}
	bool _property_can_revert(const StringName &p_name) const {
		return custom_action_property_helper.property_can_revert(p_name) || subelement_property_helper.property_can_revert(p_name);
	}
	bool _property_get_revert(const StringName &p_name, Variant &r_property) const {
		return custom_action_property_helper.property_get_revert(p_name, r_property) || subelement_property_helper.property_get_revert(p_name, r_property);
	}

	void _queue_update();

	GDVIRTUAL0(_invalidate)
	GDVIRTUAL1(_update, RID)

public:
	// Node callbacks.
	ObjectID _get_owner() const { return owner; }
	void _set_owner(ObjectID p_owner);

	void _invalidate_elements();
	void _update_elements(const RID &p_rid);

	// Subelement.
	RID _get_subelement_rid() const;
	void _set_subelement_rid(const RID &p_rid);

	ObjectID get_focused_subelement() const;
	void grab_subelement_focus();

	// Name.
	String get_accessibility_name() const;
	void set_accessibility_name(const String &p_name);

	String get_accessibility_description() const;
	void set_accessibility_description(const String &p_description);

	AccessibilityServerEnums::AccessibilityLiveMode get_accessibility_live() const;
	void set_accessibility_live(AccessibilityServerEnums::AccessibilityLiveMode p_mode);

	// Role.
	AccessibilityServerEnums::AccessibilityRole get_role() const;
	void set_role(AccessibilityServerEnums::AccessibilityRole p_role);

	String get_role_description() const;
	void set_role_description(const String &p_description);

	// Value.
	BitField<ValueType> get_value_type() const;
	void set_value_type(BitField<ValueType> p_type);

	String get_value() const;
	void set_value(const String &p_value);

	real_t get_value_num() const;
	void set_value_num(real_t p_value);

	Vector2 get_value_num_range() const;
	void set_value_num_range(const Vector2 &p_range);

	real_t get_value_num_step() const;
	void set_value_num_step(real_t p_step);

	real_t get_value_num_jump() const;
	void set_value_num_jump(real_t p_jump);

	Color get_value_color() const;
	void set_value_color(const Color &p_value);

	// Common.
	String get_state_description() const;
	void set_state_description(const String &p_description);

	String get_tooltip() const;
	void set_tooltip(const String &p_tooltip);

	String get_extra_info() const;
	void set_extra_info(const String &p_extra_info);

	BitField<AccessibilityServerEnums::AccessibilityFlagsBit> get_flags() const;
	void set_flags(BitField<AccessibilityServerEnums::AccessibilityFlagsBit> p_flags);

	bool get_state_checked() const;
	void set_state_checked(bool p_checekd);

	Ref<InputEventKey> get_shortcut() const;
	void set_shortcut(const Ref<InputEventKey> &p_shortcut);

	// Action.
	void set_supported_actions(BitField<AccessibilityServerEnums::AccessibilityActionBit> p_actions);
	BitField<AccessibilityServerEnums::AccessibilityActionBit> get_supported_actions() const;

	void set_overridden_actions(BitField<AccessibilityServerEnums::AccessibilityActionBit> p_actions);
	BitField<AccessibilityServerEnums::AccessibilityActionBit> get_overridden_actions() const;

	bool add_custom_action(const String &p_description, int64_t p_id = -1);
	void remove_custom_action(int32_t p_idx);

	int32_t get_custom_action_count() const;
	void set_custom_action_count(int32_t p_count);
	int64_t get_custom_action_id(int32_t p_idx) const;
	void set_custom_action_id(int32_t p_idx, int64_t p_id);
	String get_custom_action_description(int32_t p_idx) const;
	void set_custom_action_description(int32_t p_idx, const String &p_description);

	// Scroll.
	// ROLE_SCROLL_BAR, ROLE_SCROLL_VIEW
	Vector2 get_scroll_x_range() const;
	void set_scroll_x_range(const Vector2 &p_range);

	Vector2 get_scroll_y_range() const;
	void set_scroll_y_range(const Vector2 &p_range);

	Vector2 get_scroll_position() const;
	void set_scroll_position(const Vector2 &p_position);

	// Text.
	HorizontalAlignment get_text_alignment() const;
	void set_text_alignment(HorizontalAlignment p_align);

	String get_placeholder() const;
	void set_placeholder(const String &p_placeholder);

	String get_url() const;
	void set_url(const String &p_url);

	String get_text_language() const;
	void set_text_language(const String &p_language);

	TextServer::Orientation get_text_orientation() const;
	void set_text_orientation(TextServer::Orientation p_orientation);

	Color get_background_color() const;
	void set_background_color(const Color &p_color);

	Color get_foreground_color() const;
	void set_foreground_color(const Color &p_color);

	// List.
	// ROLE_TREE, ROLE_LIST, ROLE_LIST_BOX
	int get_list_item_count() const;
	void set_list_item_count(int p_count);

	bool get_list_vertical() const;
	void set_list_vertical(bool p_vertical);

	// List Item.
	// ROLE_TREE_ITEM, ROLE_LIST_ITEM, ROLE_LIST_BOX_OPTION
	int get_list_item_index() const;
	void set_list_item_index(int p_index);

	int get_list_item_level() const;
	void set_list_item_level(int p_level);

	bool get_list_item_selected() const;
	void set_list_item_selected(bool p_selected);

	bool get_list_item_expanded() const;
	void set_list_item_expanded(bool p_expanded);

	// Table.
	// ROLE_TABLE
	int get_table_row_count() const;
	void set_table_row_count(int p_count);

	int get_table_column_count() const;
	void set_table_column_count(int p_count);

	// Table cells/headers.
	// ROLE_ROW, ROLE_ROW_GROUP, ROLE_ROW_HEADER, ROLE_COLUMN_HEADER, ROLE_CELL
	Vector2i get_table_cell_position() const;
	void set_table_cell_position(const Vector2i &p_position);

	Vector2i get_table_cell_span() const;
	void set_table_cell_span(const Vector2i &p_span);

	// Subelements.
	bool add_subelement(const Ref<AccessibilitySettings> &p_settings, const Rect2 &p_bounds);
	void remove_subelement(int32_t p_idx);

	int32_t get_subelement_count() const;
	void set_subelement_count(int32_t p_count);
	Ref<AccessibilitySettings> get_subelement(int32_t p_idx) const;
	void set_subelement(int32_t p_idx, const Ref<AccessibilitySettings> &p_settings);
	Rect2 get_subelement_bounds(int32_t p_idx) const;
	void set_subelement_bounds(int32_t p_idx, const Rect2 &p_bounds);

	AccessibilitySettings();
	~AccessibilitySettings();
};

VARIANT_BITFIELD_CAST(AccessibilitySettings::ValueType);
