/**************************************************************************/
/*  accessibility_settings.cpp                                            */
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

#include "accessibility_settings.h"

#include "core/input/shortcut.h"
#include "core/object/callable_mp.h"
#include "core/object/class_db.h"
#include "scene/main/node.h"

void AccessibilitySettings::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_accessibility_name", "name"), &AccessibilitySettings::set_accessibility_name);
	ClassDB::bind_method(D_METHOD("get_accessibility_name"), &AccessibilitySettings::get_accessibility_name);
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "accessibility_name", PROPERTY_HINT_NONE, ""), "set_accessibility_name", "get_accessibility_name");

	ClassDB::bind_method(D_METHOD("set_accessibility_description", "description"), &AccessibilitySettings::set_accessibility_description);
	ClassDB::bind_method(D_METHOD("get_accessibility_description"), &AccessibilitySettings::get_accessibility_description);
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "accessibility_description", PROPERTY_HINT_NONE, ""), "set_accessibility_description", "get_accessibility_description");

	ClassDB::bind_method(D_METHOD("set_accessibility_live", "mode"), &AccessibilitySettings::set_accessibility_live);
	ClassDB::bind_method(D_METHOD("get_accessibility_live"), &AccessibilitySettings::get_accessibility_live);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "accessibility_live", PROPERTY_HINT_ENUM, "Off,Polite,Assertive"), "set_accessibility_live", "get_accessibility_live");

	// Role.
	ClassDB::bind_method(D_METHOD("set_role", "role"), &AccessibilitySettings::set_role);
	ClassDB::bind_method(D_METHOD("get_role"), &AccessibilitySettings::get_role);
	ClassDB::bind_method(D_METHOD("set_role_description", "role_description"), &AccessibilitySettings::set_role_description);
	ClassDB::bind_method(D_METHOD("get_role_description"), &AccessibilitySettings::get_role_description);

	ADD_GROUP("Role", "");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "role", PROPERTY_HINT_ENUM, "Unknown,Default Button,Audio,Video,Static Text,Container,Panel,Button,Link,Check Box,Radio Button,Check Button,Scroll Bar,Scroll View,Splitter,Slider,Spin Button,Progress Indicator,Text Field,Multiline Text Field,Color Picker,Table,Cell,Row,Row Group,Row Header,Column Header,Tree,Tree Item,List,List Item,List Box,List Box Option,Tab Bar,Tab,Tab Panel,Menu Bar,Menu,Menu Item,Menu Item Check Box,Menu Item Radio,Image,Window,Title Bar,Dialog,Tooltip,Region"), "set_role", "get_role");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "role_description", PROPERTY_HINT_NONE, ""), "set_role_description", "get_role_description");

	// Value.
	ClassDB::bind_method(D_METHOD("set_value_type", "value_type"), &AccessibilitySettings::set_value_type);
	ClassDB::bind_method(D_METHOD("get_value_type"), &AccessibilitySettings::get_value_type);
	ClassDB::bind_method(D_METHOD("set_value", "value"), &AccessibilitySettings::set_value);
	ClassDB::bind_method(D_METHOD("get_value"), &AccessibilitySettings::get_value);
	ClassDB::bind_method(D_METHOD("set_value_num", "value_num"), &AccessibilitySettings::set_value_num);
	ClassDB::bind_method(D_METHOD("get_value_num"), &AccessibilitySettings::get_value_num);
	ClassDB::bind_method(D_METHOD("set_value_num_range", "value_num_range"), &AccessibilitySettings::set_value_num_range);
	ClassDB::bind_method(D_METHOD("get_value_num_range"), &AccessibilitySettings::get_value_num_range);
	ClassDB::bind_method(D_METHOD("set_value_num_step", "value_num_step"), &AccessibilitySettings::set_value_num_step);
	ClassDB::bind_method(D_METHOD("get_value_num_step"), &AccessibilitySettings::get_value_num_step);
	ClassDB::bind_method(D_METHOD("set_value_num_jump", "value_num_jump"), &AccessibilitySettings::set_value_num_jump);
	ClassDB::bind_method(D_METHOD("get_value_num_jump"), &AccessibilitySettings::get_value_num_jump);
	ClassDB::bind_method(D_METHOD("set_value_color", "value_color"), &AccessibilitySettings::set_value_color);
	ClassDB::bind_method(D_METHOD("get_value_color"), &AccessibilitySettings::get_value_color);

	ADD_GROUP("Value", "");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "value_type", PROPERTY_HINT_FLAGS, "String,Number,Color"), "set_value_type", "get_value_type");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "value", PROPERTY_HINT_NONE, ""), "set_value", "get_value");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "value_num", PROPERTY_HINT_NONE, ""), "set_value_num", "get_value_num");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "value_num_range", PROPERTY_HINT_NONE, ""), "set_value_num_range", "get_value_num_range");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "value_num_step", PROPERTY_HINT_NONE, ""), "set_value_num_step", "get_value_num_step");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "value_num_jump", PROPERTY_HINT_NONE, ""), "set_value_num_jump", "get_value_num_jump");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "value_color", PROPERTY_HINT_NONE, ""), "set_value_color", "get_value_color");

	BIND_BITFIELD_FLAG(VALUE_STRING);
	BIND_BITFIELD_FLAG(VALUE_NUMBER);
	BIND_BITFIELD_FLAG(VALUE_COLOR);

	// Common.
	ClassDB::bind_method(D_METHOD("set_state_description", "state_description"), &AccessibilitySettings::set_state_description);
	ClassDB::bind_method(D_METHOD("get_state_description"), &AccessibilitySettings::get_state_description);
	ClassDB::bind_method(D_METHOD("set_tooltip", "tooltip"), &AccessibilitySettings::set_tooltip);
	ClassDB::bind_method(D_METHOD("get_tooltip"), &AccessibilitySettings::get_tooltip);
	ClassDB::bind_method(D_METHOD("set_extra_info", "extra_info"), &AccessibilitySettings::set_extra_info);
	ClassDB::bind_method(D_METHOD("get_extra_info"), &AccessibilitySettings::get_extra_info);
	ClassDB::bind_method(D_METHOD("set_flags", "flags"), &AccessibilitySettings::set_flags);
	ClassDB::bind_method(D_METHOD("get_flags"), &AccessibilitySettings::get_flags);
	ClassDB::bind_method(D_METHOD("set_state_checked", "state_checked"), &AccessibilitySettings::set_state_checked);
	ClassDB::bind_method(D_METHOD("get_state_checked"), &AccessibilitySettings::get_state_checked);
	ClassDB::bind_method(D_METHOD("set_shortcut", "shortcut"), &AccessibilitySettings::set_shortcut);
	ClassDB::bind_method(D_METHOD("get_shortcut"), &AccessibilitySettings::get_shortcut);

	ADD_GROUP("Common", "");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "state_description", PROPERTY_HINT_NONE, ""), "set_state_description", "get_state_description");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "tooltip", PROPERTY_HINT_NONE, ""), "set_tooltip", "get_tooltip");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "extra_info", PROPERTY_HINT_NONE, ""), "set_extra_info", "get_extra_info");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "flags", PROPERTY_HINT_FLAGS, "Hidden,Multiselectable,Required,Visited,Busy,Modal,Touch Passthrough,Readonly,Disabled,Clips Children"), "set_flags", "get_flags");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "state_checked", PROPERTY_HINT_NONE, ""), "set_state_checked", "get_state_checked");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "shortcut", PROPERTY_HINT_RESOURCE_TYPE, Shortcut::get_class_static()), "set_shortcut", "get_shortcut");

	// Actions.
	ClassDB::bind_method(D_METHOD("set_supported_actions", "actions"), &AccessibilitySettings::set_supported_actions);
	ClassDB::bind_method(D_METHOD("get_supported_actions"), &AccessibilitySettings::get_supported_actions);

	ClassDB::bind_method(D_METHOD("set_overridden_actions", "actions"), &AccessibilitySettings::set_overridden_actions);
	ClassDB::bind_method(D_METHOD("get_overridden_actions"), &AccessibilitySettings::get_overridden_actions);

	ClassDB::bind_method(D_METHOD("add_custom_action", "description", "id"), &AccessibilitySettings::add_custom_action);
	ClassDB::bind_method(D_METHOD("remove_custom_action", "idx"), &AccessibilitySettings::remove_custom_action);
	ClassDB::bind_method(D_METHOD("get_custom_action_count"), &AccessibilitySettings::get_custom_action_count);
	ClassDB::bind_method(D_METHOD("set_custom_action_count", "count"), &AccessibilitySettings::set_custom_action_count);
	ClassDB::bind_method(D_METHOD("get_custom_action_id", "idx"), &AccessibilitySettings::get_custom_action_id);
	ClassDB::bind_method(D_METHOD("set_custom_action_id", "idx", "id"), &AccessibilitySettings::set_custom_action_id);
	ClassDB::bind_method(D_METHOD("get_custom_action_description", "idx"), &AccessibilitySettings::get_custom_action_description);
	ClassDB::bind_method(D_METHOD("set_custom_action_description", "idx", "description"), &AccessibilitySettings::set_custom_action_description);

	ADD_GROUP("Actions", "");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "supported_actions", PROPERTY_HINT_FLAGS, "Click,Focus,Blur,Collapse,Expand,Decrement,Increment,Hide Tooltip,Show Tooltip,Set Text Selection,Replace Text Selection,Scroll Backward,Scroll Down,Scroll Forward,Scroll Left,Scroll Right,Scroll Up,Scroll Into View,Scroll to Point,Set Scroll Offset,Set Value,Show Context Menu"), "set_supported_actions", "get_supported_actions");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "overridden_actions", PROPERTY_HINT_FLAGS, "Click,Focus,Blur,Collapse,Expand,Decrement,Increment,Hide Tooltip,Show Tooltip,Set Text Selection,Replace Text Selection,Scroll Backward,Scroll Down,Scroll Forward,Scroll Left,Scroll Right,Scroll Up,Scroll Into View,Scroll to Point,Set Scroll Offset,Set Value,Show Context Menu"), "set_overridden_actions", "get_overridden_actions");
	ADD_ARRAY_COUNT("Custom Actions", "custom_action_count", "set_custom_action_count", "get_custom_action_count", "custom_action_");

	ADD_GROUP("Custom Actions", "custom_action_");
	custom_action_base_property_helper.set_prefix("custom_action_");
	custom_action_base_property_helper.set_array_length_getter(&AccessibilitySettings::get_custom_action_count);
	custom_action_base_property_helper.register_property(PropertyInfo(Variant::INT, "id", PROPERTY_HINT_NONE, ""), -1, &AccessibilitySettings::set_custom_action_id, &AccessibilitySettings::get_custom_action_id);
	custom_action_base_property_helper.register_property(PropertyInfo(Variant::STRING, "description"), String(), &AccessibilitySettings::set_custom_action_description, &AccessibilitySettings::get_custom_action_description);
	PropertyListHelper::register_base_helper(&custom_action_base_property_helper);

	// Text.
	ClassDB::bind_method(D_METHOD("set_text_alignment", "text_alignment"), &AccessibilitySettings::set_text_alignment);
	ClassDB::bind_method(D_METHOD("get_text_alignment"), &AccessibilitySettings::get_text_alignment);
	ClassDB::bind_method(D_METHOD("set_placeholder", "placeholder"), &AccessibilitySettings::set_placeholder);
	ClassDB::bind_method(D_METHOD("get_placeholder"), &AccessibilitySettings::get_placeholder);
	ClassDB::bind_method(D_METHOD("set_url", "url"), &AccessibilitySettings::set_url);
	ClassDB::bind_method(D_METHOD("get_url"), &AccessibilitySettings::get_url);
	ClassDB::bind_method(D_METHOD("set_text_orientation", "text_orientation"), &AccessibilitySettings::set_text_orientation);
	ClassDB::bind_method(D_METHOD("get_text_orientation"), &AccessibilitySettings::get_text_orientation);
	ClassDB::bind_method(D_METHOD("set_background_color", "background_color"), &AccessibilitySettings::set_background_color);
	ClassDB::bind_method(D_METHOD("get_background_color"), &AccessibilitySettings::get_background_color);
	ClassDB::bind_method(D_METHOD("set_foreground_color", "foreground_color"), &AccessibilitySettings::set_foreground_color);
	ClassDB::bind_method(D_METHOD("get_foreground_color"), &AccessibilitySettings::get_foreground_color);

	ADD_GROUP("Text", "");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "text_alignment", PROPERTY_HINT_ENUM, "Left,Center,Right,Fill"), "set_text_alignment", "get_text_alignment");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "placeholder", PROPERTY_HINT_NONE, ""), "set_placeholder", "get_placeholder");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "url", PROPERTY_HINT_NONE, ""), "set_url", "get_url");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "text_orientation", PROPERTY_HINT_ENUM, "Horizontal,Vertical"), "set_text_orientation", "get_text_orientation");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "background_color", PROPERTY_HINT_NONE, ""), "set_background_color", "get_background_color");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "foreground_color", PROPERTY_HINT_NONE, ""), "set_foreground_color", "get_foreground_color");

	ADD_GROUP("Role Specific", "");

	// Scroll.
	ClassDB::bind_method(D_METHOD("set_scroll_x_range", "scroll_range"), &AccessibilitySettings::set_scroll_x_range);
	ClassDB::bind_method(D_METHOD("get_scroll_x_range"), &AccessibilitySettings::get_scroll_x_range);
	ClassDB::bind_method(D_METHOD("set_scroll_y_range", "scroll_range"), &AccessibilitySettings::set_scroll_y_range);
	ClassDB::bind_method(D_METHOD("get_scroll_y_range"), &AccessibilitySettings::get_scroll_y_range);
	ClassDB::bind_method(D_METHOD("set_scroll_position", "scroll_position"), &AccessibilitySettings::set_scroll_position);
	ClassDB::bind_method(D_METHOD("get_scroll_position"), &AccessibilitySettings::get_scroll_position);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "scroll_x_range", PROPERTY_HINT_NONE, ""), "set_scroll_x_range", "get_scroll_x_range");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "scroll_y_range", PROPERTY_HINT_NONE, ""), "set_scroll_y_range", "get_scroll_y_range");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "scroll_position", PROPERTY_HINT_NONE, ""), "set_scroll_position", "get_scroll_position");

	// List.
	ClassDB::bind_method(D_METHOD("set_list_item_count", "list_item_count"), &AccessibilitySettings::set_list_item_count);
	ClassDB::bind_method(D_METHOD("get_list_item_count"), &AccessibilitySettings::get_list_item_count);
	ClassDB::bind_method(D_METHOD("set_list_vertical", "list_vertical"), &AccessibilitySettings::set_list_vertical);
	ClassDB::bind_method(D_METHOD("get_list_vertical"), &AccessibilitySettings::get_list_vertical);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "list_item_count", PROPERTY_HINT_RANGE, "0,2048,1,or_greater"), "set_list_item_count", "get_list_item_count");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "list_vertical", PROPERTY_HINT_NONE, ""), "set_list_vertical", "get_list_vertical");

	// List Item.
	ClassDB::bind_method(D_METHOD("set_list_item_index", "list_item_index"), &AccessibilitySettings::set_list_item_index);
	ClassDB::bind_method(D_METHOD("get_list_item_index"), &AccessibilitySettings::get_list_item_index);
	ClassDB::bind_method(D_METHOD("set_list_item_level", "list_item_level"), &AccessibilitySettings::set_list_item_level);
	ClassDB::bind_method(D_METHOD("get_list_item_level"), &AccessibilitySettings::get_list_item_level);
	ClassDB::bind_method(D_METHOD("set_list_item_selected", "list_item_selected"), &AccessibilitySettings::set_list_item_selected);
	ClassDB::bind_method(D_METHOD("get_list_item_selected"), &AccessibilitySettings::get_list_item_selected);
	ClassDB::bind_method(D_METHOD("set_list_item_expanded", "list_item_expanded"), &AccessibilitySettings::set_list_item_expanded);
	ClassDB::bind_method(D_METHOD("get_list_item_expanded"), &AccessibilitySettings::get_list_item_expanded);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "list_item_index", PROPERTY_HINT_RANGE, "0,2048,1,or_greater"), "set_list_item_index", "get_list_item_index");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "list_item_level", PROPERTY_HINT_RANGE, "0,2048,1,or_greater"), "set_list_item_level", "get_list_item_level");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "list_item_selected", PROPERTY_HINT_NONE, ""), "set_list_item_selected", "get_list_item_selected");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "list_item_expanded", PROPERTY_HINT_NONE, ""), "set_list_item_expanded", "get_list_item_expanded");

	// Table.
	ClassDB::bind_method(D_METHOD("set_table_row_count", "table_row_count"), &AccessibilitySettings::set_table_row_count);
	ClassDB::bind_method(D_METHOD("get_table_row_count"), &AccessibilitySettings::get_table_row_count);
	ClassDB::bind_method(D_METHOD("set_table_column_count", "table_column_count"), &AccessibilitySettings::set_table_column_count);
	ClassDB::bind_method(D_METHOD("get_table_column_count"), &AccessibilitySettings::get_table_column_count);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "table_row_count", PROPERTY_HINT_RANGE, "0,2048,1,or_greater"), "set_table_row_count", "get_table_row_count");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "table_column_count", PROPERTY_HINT_RANGE, "0,2048,1,or_greater"), "set_table_column_count", "get_table_column_count");

	// Table cells/headers.
	ClassDB::bind_method(D_METHOD("set_table_cell_position", "table_cell_position"), &AccessibilitySettings::set_table_cell_position);
	ClassDB::bind_method(D_METHOD("get_table_cell_position"), &AccessibilitySettings::get_table_cell_position);
	ClassDB::bind_method(D_METHOD("set_table_cell_span", "table_cell_span"), &AccessibilitySettings::set_table_cell_span);
	ClassDB::bind_method(D_METHOD("get_table_cell_span"), &AccessibilitySettings::get_table_cell_span);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2I, "table_cell_position", PROPERTY_HINT_NONE, ""), "set_table_cell_position", "get_table_cell_position");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2I, "table_cell_span", PROPERTY_HINT_NONE, ""), "set_table_cell_span", "get_table_cell_span");

	// Subelements.
	ClassDB::bind_method(D_METHOD("add_subelement", "settings", "bounds"), &AccessibilitySettings::add_subelement);
	ClassDB::bind_method(D_METHOD("remove_subelement", "idx"), &AccessibilitySettings::remove_subelement);
	ClassDB::bind_method(D_METHOD("get_subelement_count"), &AccessibilitySettings::get_subelement_count);
	ClassDB::bind_method(D_METHOD("set_subelement_count", "count"), &AccessibilitySettings::set_subelement_count);
	ClassDB::bind_method(D_METHOD("get_subelement", "idx"), &AccessibilitySettings::get_subelement);
	ClassDB::bind_method(D_METHOD("set_subelement", "idx", "settings"), &AccessibilitySettings::set_subelement);
	ClassDB::bind_method(D_METHOD("get_subelement_bounds", "idx"), &AccessibilitySettings::get_subelement_bounds);
	ClassDB::bind_method(D_METHOD("set_subelement_bounds", "idx", "bounds"), &AccessibilitySettings::set_subelement_bounds);

	ClassDB::bind_method(D_METHOD("grab_subelement_focus"), &AccessibilitySettings::grab_subelement_focus);

	ADD_ARRAY_COUNT("Subelements", "subelement_count", "set_subelement_count", "get_subelement_count", "subelement_");

	ADD_GROUP("Subelements", "subelement_");
	subelement_base_property_helper.set_prefix("subelement_");
	subelement_base_property_helper.set_array_length_getter(&AccessibilitySettings::get_subelement_count);
	subelement_base_property_helper.register_property(PropertyInfo(Variant::OBJECT, "settings", PROPERTY_HINT_RESOURCE_TYPE, AccessibilitySettings::get_class_static()), Ref<AccessibilitySettings>(), &AccessibilitySettings::set_subelement, &AccessibilitySettings::get_subelement);
	subelement_base_property_helper.register_property(PropertyInfo(Variant::RECT2, "bounds"), Rect2(), &AccessibilitySettings::set_subelement_bounds, &AccessibilitySettings::get_subelement_bounds);
	PropertyListHelper::register_base_helper(&subelement_base_property_helper);

	// Actions.
	ADD_SIGNAL(MethodInfo("action_custom", PropertyInfo(Variant::INT, "id")));
	ADD_SIGNAL(MethodInfo("action_click"));
	ADD_SIGNAL(MethodInfo("action_focus"));
	ADD_SIGNAL(MethodInfo("action_blur"));
	ADD_SIGNAL(MethodInfo("action_collapse"));
	ADD_SIGNAL(MethodInfo("action_expand"));
	ADD_SIGNAL(MethodInfo("action_increment"));
	ADD_SIGNAL(MethodInfo("action_decrement"));
	ADD_SIGNAL(MethodInfo("action_scroll_down", PropertyInfo(Variant::FLOAT, "value")));
	ADD_SIGNAL(MethodInfo("action_scroll_up", PropertyInfo(Variant::FLOAT, "value")));
	ADD_SIGNAL(MethodInfo("action_scroll_left", PropertyInfo(Variant::FLOAT, "value")));
	ADD_SIGNAL(MethodInfo("action_scroll_right", PropertyInfo(Variant::FLOAT, "value")));
	ADD_SIGNAL(MethodInfo("action_scroll_to_point", PropertyInfo(Variant::VECTOR2, "pos")));
	ADD_SIGNAL(MethodInfo("action_scroll_set_offset", PropertyInfo(Variant::VECTOR2, "offset")));
	ADD_SIGNAL(MethodInfo("action_scroll_into_view"));
	ADD_SIGNAL(MethodInfo("action_set_value", PropertyInfo(Variant::NIL, "value")));
	ADD_SIGNAL(MethodInfo("show_context_menu"));

	// Virtual.
	GDVIRTUAL_BIND(_invalidate);
	GDVIRTUAL_BIND(_update, "rid");
}

void AccessibilitySettings::_validate_property(PropertyInfo &p_property) const {
	if (p_property.name.begins_with("value") && p_property.name != "value_type") {
		p_property.usage = PROPERTY_USAGE_NONE;
	}
	if (value_type.has_flag(VALUE_STRING) && p_property.name == "value") {
		p_property.usage = PROPERTY_USAGE_DEFAULT;
	}
	if (value_type.has_flag(VALUE_NUMBER) && p_property.name.begins_with("value_num")) {
		p_property.usage = PROPERTY_USAGE_DEFAULT;
	}
	if (value_type.has_flag(VALUE_COLOR) && p_property.name == "value_color") {
		p_property.usage = PROPERTY_USAGE_DEFAULT;
	}

	if ((p_property.name == "scroll_x_range" || p_property.name == "scroll_y_range" || p_property.name == "scroll_position") && !_is_role_scrollable()) {
		p_property.usage = PROPERTY_USAGE_NONE;
	}
	if ((p_property.name == "list_item_count" || p_property.name == "list_vertical") && !_is_role_list()) {
		p_property.usage = PROPERTY_USAGE_NONE;
	}
	if ((p_property.name == "list_item_index" || p_property.name == "list_item_level" || p_property.name == "list_item_selected" || p_property.name == "list_item_expanded") && !_is_role_list_item()) {
		p_property.usage = PROPERTY_USAGE_NONE;
	}
	if ((p_property.name == "table_row_count" || p_property.name == "table_column_count") && !_is_role_table()) {
		p_property.usage = PROPERTY_USAGE_NONE;
	}
	if (p_property.name == "table_cell_position" && !_is_role_table_cell() && !_is_role_table_col() && !_is_role_table_row()) {
		p_property.usage = PROPERTY_USAGE_NONE;
	}
	if (p_property.name == "table_cell_span" && !_is_role_table_cell()) {
		p_property.usage = PROPERTY_USAGE_NONE;
	}
}

void AccessibilitySettings::_action_custom(const Variant &p_data) {
	emit_signal("action_custom", p_data);
}

void AccessibilitySettings::_action_default(AccessibilityServerEnums::AccessibilityAction p_action, const Variant &p_data) {
	switch (p_action) {
		case AccessibilityServerEnums::AccessibilityAction::ACTION_CLICK: {
			emit_signal("action_click");
		} break;
		case AccessibilityServerEnums::AccessibilityAction::ACTION_FOCUS: {
			emit_signal("action_focus");
		} break;
		case AccessibilityServerEnums::AccessibilityAction::ACTION_BLUR: {
			emit_signal("action_blur");
		} break;
		case AccessibilityServerEnums::AccessibilityAction::ACTION_COLLAPSE: {
			emit_signal("action_collapse");
		} break;
		case AccessibilityServerEnums::AccessibilityAction::ACTION_EXPAND: {
			emit_signal("action_expand");
		} break;
		case AccessibilityServerEnums::AccessibilityAction::ACTION_DECREMENT: {
			emit_signal("action_increment");
		} break;
		case AccessibilityServerEnums::AccessibilityAction::ACTION_INCREMENT: {
			emit_signal("action_decrement");
		} break;
		case AccessibilityServerEnums::AccessibilityAction::ACTION_SCROLL_DOWN: {
			emit_signal("action_scroll_down", p_data);
		} break;
		case AccessibilityServerEnums::AccessibilityAction::ACTION_SCROLL_LEFT: {
			emit_signal("action_scroll_left", p_data);
		} break;
		case AccessibilityServerEnums::AccessibilityAction::ACTION_SCROLL_RIGHT: {
			emit_signal("action_scroll_right", p_data);
		} break;
		case AccessibilityServerEnums::AccessibilityAction::ACTION_SCROLL_UP: {
			emit_signal("action_scroll_up", p_data);
		} break;
		case AccessibilityServerEnums::AccessibilityAction::ACTION_SCROLL_INTO_VIEW: {
			emit_signal("action_scroll_into_view");
		} break;
		case AccessibilityServerEnums::AccessibilityAction::ACTION_SCROLL_TO_POINT: {
			emit_signal("action_scroll_to_point", p_data);
		} break;
		case AccessibilityServerEnums::AccessibilityAction::ACTION_SET_SCROLL_OFFSET: {
			emit_signal("action_scroll_set_offset", p_data);
		} break;
		case AccessibilityServerEnums::AccessibilityAction::ACTION_SET_VALUE: {
			emit_signal("action_set_value", p_data);
		} break;
		case AccessibilityServerEnums::AccessibilityAction::ACTION_SHOW_CONTEXT_MENU: {
			emit_signal("show_context_menu");
		} break;
		default: {
			WARN_PRINT(vformat("Unsupported action %d.", p_action));
		} break;
	}
}

void AccessibilitySettings::_set_owner(ObjectID p_owner) {
	if (owner == p_owner) {
		return;
	}
	_reset_focus();
	_invalidate_elements();
	owner = p_owner;
	_queue_update();
}

void AccessibilitySettings::_invalidate_elements() {
	if (subelement_rid.is_valid()) {
		AccessibilityServer::get_singleton()->free_element(subelement_rid);
		subelement_rid = RID();
	}
	for (const SubElement &se : subelements) {
		if (se.settings.is_valid()) {
			se.settings->_invalidate_elements();
		}
	}
	GDVIRTUAL_CALL(_invalidate);
}

void AccessibilitySettings::_update_elements(const RID &p_rid) {
	ERR_FAIL_COND(p_rid.is_null());

	AccessibilityServer *as = AccessibilityServer::get_singleton();
	ERR_FAIL_NULL(as);

	Node *node = _get_owner_node();
	bool translate = node && node->can_auto_translate();

	if (!name.is_empty()) {
		as->update_set_name(p_rid, translate ? tr(name) : name); // Do not unset if empty, it can be set by Node instead.
	}
	as->update_set_description(p_rid, translate ? tr(description) : description);
	as->update_set_live(p_rid, accessibility_live);

	as->update_set_role(p_rid, role);
	as->update_set_role_description(p_rid, translate ? tr(role_description) : role_description);
	switch (value_type) {
		case VALUE_COLOR: {
			as->update_set_color_value(p_rid, value_color);
		} break;
		case VALUE_NUMBER: {
			as->update_set_num_value(p_rid, value_num);
			as->update_set_num_jump(p_rid, value_num_jump);
			as->update_set_num_range(p_rid, value_num_range.x, value_num_range.y);
			as->update_set_num_step(p_rid, value_num_step);
		} break;
		case VALUE_STRING: {
			as->update_set_value(p_rid, value);
		} break;
	}
	as->update_set_state_description(p_rid, translate ? tr(state_description) : state_description);
	as->update_set_tooltip(p_rid, translate ? tr(tooltip) : tooltip);

	as->update_set_extra_info(p_rid, translate ? tr(extra_info) : extra_info);
	for (int i = 0; i < AccessibilityServerEnums::AccessibilityFlags::FLAG_MAX; i++) {
		as->update_set_flag(p_rid, AccessibilityServerEnums::AccessibilityFlags(i), flags.has_flag((AccessibilityServerEnums::AccessibilityFlagsBit)(1 << i)));
	}
	as->update_set_checked(p_rid, state_checked);
	if (shortcut.is_valid()) {
		as->update_set_shortcut(p_rid, shortcut->as_text()); // TODO: not implemented by AccessKit, string format might change.
	}
	if (!custom_actions.is_empty()) {
		as->update_add_action(p_rid, AccessibilityServerEnums::AccessibilityAction::ACTION_CUSTOM, callable_mp(this, &AccessibilitySettings::_action_custom));
		for (int i = 0; i < custom_actions.size(); i++) {
			as->update_add_custom_action(p_rid, custom_actions[i].id, translate ? tr(custom_actions[i].description) : custom_actions[i].description);
		}
	}

	for (int i = 0; i < AccessibilityServerEnums::AccessibilityAction::ACTION_CUSTOM; i++) {
		if (overridden_actions.has_flag(AccessibilityServerEnums::AccessibilityActionBit(1 << i))) {
			as->update_add_action(p_rid, AccessibilityServerEnums::AccessibilityAction(i), callable_mp(this, &AccessibilitySettings::_action_default).bind((AccessibilityServerEnums::AccessibilityAction)i));
		} else if (!supported_actions.has_flag(AccessibilityServerEnums::AccessibilityActionBit(1 << i))) {
			as->update_remove_action(p_rid, AccessibilityServerEnums::AccessibilityAction(i));
		}
	}

	if (_is_role_scrollable()) {
		if (scroll_x_range.x != scroll_x_range.y) {
			as->update_set_scroll_x_range(p_rid, scroll_x_range.x, scroll_x_range.y);
			as->update_set_scroll_x(p_rid, scroll_pos.x);
		}
		if (scroll_y_range.x != scroll_y_range.y) {
			as->update_set_scroll_y_range(p_rid, scroll_y_range.x, scroll_y_range.y);
			as->update_set_scroll_y(p_rid, scroll_pos.y);
		}
	}

	as->update_set_text_align(p_rid, text_align);
	as->update_set_placeholder(p_rid, translate ? tr(placeholder) : placeholder);
	as->update_set_url(p_rid, url);
	as->update_set_language(p_rid, language);
	as->update_set_text_orientation(p_rid, orientation != TextServer::ORIENTATION_HORIZONTAL);
	as->update_set_background_color(p_rid, bg_color);
	as->update_set_foreground_color(p_rid, fg_color);

	// List.
	if (_is_role_list()) {
		as->update_set_list_item_count(p_rid, list_data.items);
		as->update_set_list_orientation(p_rid, list_data.vertical);
	}

	// List Item.
	if (_is_role_list_item()) {
		as->update_set_list_item_index(p_rid, list_item_data.index);
		as->update_set_list_item_level(p_rid, list_item_data.level);
		as->update_set_list_item_selected(p_rid, list_item_data.selected);
		as->update_set_list_item_expanded(p_rid, list_item_data.expanded);
	}

	// Table.
	if (_is_role_table()) {
		as->update_set_table_row_count(p_rid, table_data.rows);
		as->update_set_table_column_count(p_rid, table_data.cols);
	}

	// Table item.
	if (_is_role_table_row()) {
		as->update_set_table_row_index(p_rid, table_cell_data.pos.y);
	}
	if (_is_role_table_col()) {
		as->update_set_table_column_index(p_rid, table_cell_data.pos.x);
	}
	if (_is_role_table_cell()) {
		as->update_set_table_cell_position(p_rid, table_cell_data.pos.y, table_cell_data.pos.x);
		as->update_set_table_cell_span(p_rid, table_cell_data.span.y, table_cell_data.span.x);
	}

	GDVIRTUAL_CALL(_update, p_rid);

	for (const SubElement &se : subelements) {
		if (se.settings.is_valid()) {
			RID sub_rid = se.settings->_get_subelement_rid();
			if (sub_rid.is_null()) {
				sub_rid = as->create_sub_element(p_rid, se.settings->get_role());
				se.settings->_set_subelement_rid(sub_rid);
			}
			se.settings->_update_elements(sub_rid);
			as->update_set_bounds(sub_rid, se.bounds);
		}
	}
}

RID AccessibilitySettings::_get_subelement_rid() const {
	return subelement_rid;
}

void AccessibilitySettings::_set_subelement_rid(const RID &p_rid) {
	subelement_rid = p_rid;
}

ObjectID AccessibilitySettings::get_focused_subelement() const {
	return focused_subelement;
}

void AccessibilitySettings::grab_subelement_focus() {
	AccessibilitySettings *settings = ObjectDB::get_instance<AccessibilitySettings>(owner);
	while (settings) {
		settings->focused_subelement = get_instance_id();
		settings = ObjectDB::get_instance<AccessibilitySettings>(settings->owner);
	}
}

void AccessibilitySettings::_reset_focus() {
	AccessibilitySettings *settings = ObjectDB::get_instance<AccessibilitySettings>(owner);
	if (settings) {
		if (settings->focused_subelement == get_instance_id()) {
			settings->focused_subelement = ObjectID();
		}
		settings->_reset_focus();
	}
}

Node *AccessibilitySettings::_get_owner_node() const {
	Node *node = ObjectDB::get_instance<Node>(owner);
	if (node) {
		return node;
	}
	AccessibilitySettings *settings = ObjectDB::get_instance<AccessibilitySettings>(owner);
	if (settings) {
		return settings->_get_owner_node();
	}
	return nullptr;
}

void AccessibilitySettings::_queue_update() {
	Node *node = ObjectDB::get_instance<Node>(owner);
	if (node) {
		node->queue_accessibility_update();
		return;
	}
	AccessibilitySettings *settings = ObjectDB::get_instance<AccessibilitySettings>(owner);
	if (settings) {
		settings->_queue_update();
		return;
	}
}

// Name.
String AccessibilitySettings::get_accessibility_name() const {
	return name;
}

void AccessibilitySettings::set_accessibility_name(const String &p_name) {
	if (name == p_name) {
		return;
	}
	name = p_name;
	_queue_update();
}

String AccessibilitySettings::get_accessibility_description() const {
	return description;
}

void AccessibilitySettings::set_accessibility_description(const String &p_description) {
	if (description == p_description) {
		return;
	}
	description = p_description;
	_queue_update();
}

AccessibilityServerEnums::AccessibilityLiveMode AccessibilitySettings::get_accessibility_live() const {
	return accessibility_live;
}

void AccessibilitySettings::set_accessibility_live(AccessibilityServerEnums::AccessibilityLiveMode p_mode) {
	if (accessibility_live == p_mode) {
		return;
	}
	accessibility_live = p_mode;
	_queue_update();
}

// Role.
AccessibilityServerEnums::AccessibilityRole AccessibilitySettings::get_role() const {
	return role;
}

void AccessibilitySettings::set_role(AccessibilityServerEnums::AccessibilityRole p_role) {
	if (role == p_role) {
		return;
	}
	role = p_role;
	notify_property_list_changed();
	_queue_update();
}

String AccessibilitySettings::get_role_description() const {
	return role_description;
}

void AccessibilitySettings::set_role_description(const String &p_description) {
	if (role_description == p_description) {
		return;
	}
	role_description = p_description;
	_queue_update();
}

// Value.
BitField<AccessibilitySettings::ValueType> AccessibilitySettings::get_value_type() const {
	return value_type;
}

void AccessibilitySettings::set_value_type(BitField<AccessibilitySettings::ValueType> p_type) {
	if (value_type == p_type) {
		return;
	}
	value_type = p_type;
	notify_property_list_changed();
	_queue_update();
}

String AccessibilitySettings::get_value() const {
	return value;
}

void AccessibilitySettings::set_value(const String &p_value) {
	if (value == p_value) {
		return;
	}
	value = p_value;
	_queue_update();
}

real_t AccessibilitySettings::get_value_num() const {
	return value_num;
}

void AccessibilitySettings::set_value_num(real_t p_value) {
	if (value_num == p_value) {
		return;
	}
	value_num = p_value;
	_queue_update();
}

Vector2 AccessibilitySettings::get_value_num_range() const {
	return value_num_range;
}

void AccessibilitySettings::set_value_num_range(const Vector2 &p_range) {
	if (value_num_range == p_range) {
		return;
	}
	value_num_range = p_range;
	_queue_update();
}

real_t AccessibilitySettings::get_value_num_step() const {
	return value_num_step;
}

void AccessibilitySettings::set_value_num_step(real_t p_step) {
	if (value_num_step == p_step) {
		return;
	}
	value_num_step = p_step;
	_queue_update();
}

real_t AccessibilitySettings::get_value_num_jump() const {
	return value_num_jump;
}

void AccessibilitySettings::set_value_num_jump(real_t p_jump) {
	if (value_num_jump == p_jump) {
		return;
	}
	value_num_jump = p_jump;
	_queue_update();
}

Color AccessibilitySettings::get_value_color() const {
	return value_color;
}

void AccessibilitySettings::set_value_color(const Color &p_value) {
	if (value_color == p_value) {
		return;
	}
	value_color = p_value;
	_queue_update();
}

// Common.
String AccessibilitySettings::get_state_description() const {
	return state_description;
}

void AccessibilitySettings::set_state_description(const String &p_description) {
	if (state_description == p_description) {
		return;
	}
	state_description = p_description;
	_queue_update();
}

String AccessibilitySettings::get_tooltip() const {
	return tooltip;
}

void AccessibilitySettings::set_tooltip(const String &p_tooltip) {
	if (tooltip == p_tooltip) {
		return;
	}
	tooltip = p_tooltip;
	_queue_update();
}

String AccessibilitySettings::get_extra_info() const {
	return extra_info;
}

void AccessibilitySettings::set_extra_info(const String &p_extra_info) {
	if (extra_info == p_extra_info) {
		return;
	}
	extra_info = p_extra_info;
	_queue_update();
}

BitField<AccessibilityServerEnums::AccessibilityFlagsBit> AccessibilitySettings::get_flags() const {
	return flags;
}

void AccessibilitySettings::set_flags(BitField<AccessibilityServerEnums::AccessibilityFlagsBit> p_flags) {
	if (flags == p_flags) {
		return;
	}
	flags = p_flags;
	_queue_update();
}

bool AccessibilitySettings::get_state_checked() const {
	return state_checked;
}

void AccessibilitySettings::set_state_checked(bool p_checked) {
	if (state_checked == p_checked) {
		return;
	}
	state_checked = p_checked;
	_queue_update();
}

Ref<InputEventKey> AccessibilitySettings::get_shortcut() const {
	return shortcut;
}

void AccessibilitySettings::set_shortcut(const Ref<InputEventKey> &p_shortcut) {
	if (shortcut == p_shortcut) {
		return;
	}
	shortcut = p_shortcut;
	_queue_update();
}

void AccessibilitySettings::set_supported_actions(BitField<AccessibilityServerEnums::AccessibilityActionBit> p_actions) {
	if (supported_actions == p_actions) {
		return;
	}
	supported_actions = p_actions;
	_queue_update();
}

BitField<AccessibilityServerEnums::AccessibilityActionBit> AccessibilitySettings::get_supported_actions() const {
	return supported_actions;
}

void AccessibilitySettings::set_overridden_actions(BitField<AccessibilityServerEnums::AccessibilityActionBit> p_actions) {
	if (overridden_actions == p_actions) {
		return;
	}
	overridden_actions = p_actions;
	_queue_update();
}

BitField<AccessibilityServerEnums::AccessibilityActionBit> AccessibilitySettings::get_overridden_actions() const {
	return overridden_actions;
}

bool AccessibilitySettings::add_custom_action(const String &p_description, int64_t p_id) {
	ERR_FAIL_COND_V(p_description.is_empty(), false);

	custom_actions.push_back({ (p_id < 0) ? custom_actions.size() : p_id, p_description });
	_queue_update();
	return true;
}

void AccessibilitySettings::remove_custom_action(int32_t p_idx) {
	ERR_FAIL_INDEX(p_idx, custom_actions.size());
	custom_actions.remove_at(p_idx);
	_queue_update();
}

int32_t AccessibilitySettings::get_custom_action_count() const {
	return custom_actions.size();
}

void AccessibilitySettings::set_custom_action_count(int32_t p_count) {
	ERR_FAIL_COND(p_count < 0);
	if (custom_actions.size() != p_count) {
		custom_actions.resize(p_count);
		notify_property_list_changed();
		_queue_update();
	}
}

int64_t AccessibilitySettings::get_custom_action_id(int32_t p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, custom_actions.size(), -1);

	return custom_actions[p_idx].id;
}

void AccessibilitySettings::set_custom_action_id(int32_t p_idx, int64_t p_id) {
	ERR_FAIL_INDEX(p_idx, custom_actions.size());
	if (custom_actions[p_idx].id == p_id) {
		return;
	}
	custom_actions.write[p_idx].id = p_id;
	_queue_update();
}

String AccessibilitySettings::get_custom_action_description(int32_t p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, custom_actions.size(), String());

	return custom_actions[p_idx].description;
}

void AccessibilitySettings::set_custom_action_description(int32_t p_idx, const String &p_description) {
	ERR_FAIL_INDEX(p_idx, custom_actions.size());
	if (custom_actions[p_idx].description == p_description) {
		return;
	}
	custom_actions.write[p_idx].description = p_description;
	_queue_update();
}

// Scroll.
Vector2 AccessibilitySettings::get_scroll_x_range() const {
	return scroll_x_range;
}

void AccessibilitySettings::set_scroll_x_range(const Vector2 &p_range) {
	if (scroll_x_range == p_range) {
		return;
	}
	scroll_x_range = p_range;
	_queue_update();
}

Vector2 AccessibilitySettings::get_scroll_y_range() const {
	return scroll_y_range;
}

void AccessibilitySettings::set_scroll_y_range(const Vector2 &p_range) {
	if (scroll_y_range == p_range) {
		return;
	}
	scroll_y_range = p_range;
	_queue_update();
}

Vector2 AccessibilitySettings::get_scroll_position() const {
	return scroll_pos;
}

void AccessibilitySettings::set_scroll_position(const Vector2 &p_position) {
	if (scroll_pos == p_position) {
		return;
	}
	scroll_pos = p_position;
	_queue_update();
}

// Text.
HorizontalAlignment AccessibilitySettings::get_text_alignment() const {
	return text_align;
}

void AccessibilitySettings::set_text_alignment(HorizontalAlignment p_align) {
	if (text_align == p_align) {
		return;
	}
	text_align = p_align;
	_queue_update();
}

String AccessibilitySettings::get_placeholder() const {
	return placeholder;
}

void AccessibilitySettings::set_placeholder(const String &p_placeholder) {
	if (placeholder == p_placeholder) {
		return;
	}
	placeholder = p_placeholder;
	_queue_update();
}

String AccessibilitySettings::get_url() const {
	return url;
}

void AccessibilitySettings::set_url(const String &p_url) {
	if (url == p_url) {
		return;
	}
	url = p_url;
	_queue_update();
}

String AccessibilitySettings::get_text_language() const {
	return language;
}

void AccessibilitySettings::set_text_language(const String &p_language) {
	if (language == p_language) {
		return;
	}
	language = p_language;
	_queue_update();
}

TextServer::Orientation AccessibilitySettings::get_text_orientation() const {
	return orientation;
}

void AccessibilitySettings::set_text_orientation(TextServer::Orientation p_orientation) {
	if (orientation == p_orientation) {
		return;
	}
	orientation = p_orientation;
	_queue_update();
}

Color AccessibilitySettings::get_background_color() const {
	return bg_color;
}

void AccessibilitySettings::set_background_color(const Color &p_color) {
	if (bg_color == p_color) {
		return;
	}
	bg_color = p_color;
	_queue_update();
}

Color AccessibilitySettings::get_foreground_color() const {
	return fg_color;
}

void AccessibilitySettings::set_foreground_color(const Color &p_color) {
	if (fg_color == p_color) {
		return;
	}
	fg_color = p_color;
	_queue_update();
}

// List.
int AccessibilitySettings::get_list_item_count() const {
	return list_data.items;
}

void AccessibilitySettings::set_list_item_count(int p_count) {
	if (list_data.items == p_count) {
		return;
	}
	list_data.items = p_count;
	_queue_update();
}

bool AccessibilitySettings::get_list_vertical() const {
	return list_data.vertical;
}

void AccessibilitySettings::set_list_vertical(bool p_vertical) {
	if (list_data.vertical == p_vertical) {
		return;
	}
	list_data.vertical = p_vertical;
	_queue_update();
}

// List Item.
int AccessibilitySettings::get_list_item_index() const {
	return list_item_data.index;
}

void AccessibilitySettings::set_list_item_index(int p_index) {
	if (list_item_data.index == p_index) {
		return;
	}
	list_item_data.index = p_index;
	_queue_update();
}

int AccessibilitySettings::get_list_item_level() const {
	return list_item_data.level;
}

void AccessibilitySettings::set_list_item_level(int p_level) {
	if (list_item_data.level == p_level) {
		return;
	}
	list_item_data.level = p_level;
	_queue_update();
}

bool AccessibilitySettings::get_list_item_selected() const {
	return list_item_data.selected;
}

void AccessibilitySettings::set_list_item_selected(bool p_selected) {
	if (list_item_data.selected == p_selected) {
		return;
	}
	list_item_data.selected = p_selected;
	_queue_update();
}

bool AccessibilitySettings::get_list_item_expanded() const {
	return list_item_data.expanded;
}

void AccessibilitySettings::set_list_item_expanded(bool p_expanded) {
	if (list_item_data.expanded == p_expanded) {
		return;
	}
	list_item_data.expanded = p_expanded;
	_queue_update();
}

// Table.
int AccessibilitySettings::get_table_row_count() const {
	return table_data.rows;
}

void AccessibilitySettings::set_table_row_count(int p_count) {
	if (table_data.rows == p_count) {
		return;
	}
	table_data.rows = p_count;
	_queue_update();
}

int AccessibilitySettings::get_table_column_count() const {
	return table_data.cols;
}

void AccessibilitySettings::set_table_column_count(int p_count) {
	if (table_data.cols == p_count) {
		return;
	}
	table_data.cols = p_count;
	_queue_update();
}

// Table cells/headers.
Vector2i AccessibilitySettings::get_table_cell_position() const {
	return table_cell_data.pos;
}

void AccessibilitySettings::set_table_cell_position(const Vector2i &p_position) {
	if (table_cell_data.pos == p_position) {
		return;
	}
	table_cell_data.pos = p_position;
	_queue_update();
}

Vector2i AccessibilitySettings::get_table_cell_span() const {
	return table_cell_data.span;
}

void AccessibilitySettings::set_table_cell_span(const Vector2i &p_span) {
	if (table_cell_data.span == p_span) {
		return;
	}
	table_cell_data.span = p_span;
	_queue_update();
}

// Subelement.

bool AccessibilitySettings::add_subelement(const Ref<AccessibilitySettings> &p_settings, const Rect2 &p_bounds) {
	ERR_FAIL_COND_V(p_settings.is_null(), false);
	ERR_FAIL_COND_V(p_settings->_get_owner().is_valid(), false);

	p_settings->_set_owner(get_instance_id());
	subelements.push_back({ p_settings, p_bounds });
	_queue_update();
	return true;
}

void AccessibilitySettings::remove_subelement(int32_t p_idx) {
	ERR_FAIL_INDEX(p_idx, subelements.size());
	if (subelements[p_idx].settings.is_valid()) {
		subelements[p_idx].settings->_set_owner(ObjectID());
	}
	subelements.remove_at(p_idx);
	_queue_update();
}

int32_t AccessibilitySettings::get_subelement_count() const {
	return subelements.size();
}

void AccessibilitySettings::set_subelement_count(int32_t p_count) {
	ERR_FAIL_COND(p_count < 0);
	if (subelements.size() != p_count) {
		subelements.resize(p_count);
		notify_property_list_changed();
		_queue_update();
	}
}

Ref<AccessibilitySettings> AccessibilitySettings::get_subelement(int32_t p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, subelements.size(), Ref<AccessibilitySettings>());

	return subelements[p_idx].settings;
}

void AccessibilitySettings::set_subelement(int32_t p_idx, const Ref<AccessibilitySettings> &p_settings) {
	ERR_FAIL_INDEX(p_idx, subelements.size());
	if (subelements[p_idx].settings == p_settings) {
		return;
	}
	ERR_FAIL_COND(p_settings->_get_owner().is_valid());

	if (subelements[p_idx].settings.is_valid()) {
		subelements[p_idx].settings->_set_owner(ObjectID());
	}
	subelements.write[p_idx].settings = p_settings;
	if (subelements[p_idx].settings.is_valid()) {
		subelements[p_idx].settings->_set_owner(get_instance_id());
	}
	_queue_update();
}

Rect2 AccessibilitySettings::get_subelement_bounds(int32_t p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, subelements.size(), Rect2());

	return subelements[p_idx].bounds;
}

void AccessibilitySettings::set_subelement_bounds(int32_t p_idx, const Rect2 &p_bounds) {
	ERR_FAIL_INDEX(p_idx, subelements.size());

	if (subelements[p_idx].bounds == p_bounds) {
		return;
	}
	subelements.write[p_idx].bounds = p_bounds;
	_queue_update();
}

AccessibilitySettings::AccessibilitySettings() {
	custom_action_property_helper.setup_for_instance(custom_action_base_property_helper, this);
	subelement_property_helper.setup_for_instance(subelement_base_property_helper, this);
}

AccessibilitySettings::~AccessibilitySettings() {
	_invalidate_elements();
}
