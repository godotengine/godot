/**************************************************************************/
/*  accessibility_server.cpp                                              */
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

#include "accessibility_server.h"

#include "core/object/class_db.h"
#include "servers/display/accessibility_server_dummy.h"

AccessibilityServer::AccessibilityServerCreate AccessibilityServer::server_create_functions[AccessibilityServer::MAX_SERVERS] = {
	{ "dummy", &AccessibilityServerDummy::create_func }
};

int AccessibilityServer::server_create_count = 1;

void AccessibilityServer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("is_supported"), &AccessibilityServer::is_supported);

	ClassDB::bind_method(D_METHOD("create_element", "window_id", "role"), &AccessibilityServer::create_element);
	ClassDB::bind_method(D_METHOD("create_sub_element", "parent_rid", "role", "insert_pos"), &AccessibilityServer::create_sub_element, DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("create_sub_text_edit_elements", "parent_rid", "shaped_text", "min_height", "insert_pos", "is_last_line"), &AccessibilityServer::create_sub_text_edit_elements, DEFVAL(-1), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("has_element", "id"), &AccessibilityServer::has_element);
	ClassDB::bind_method(D_METHOD("free_element", "id"), &AccessibilityServer::free_element);
	ClassDB::bind_method(D_METHOD("element_set_meta", "id", "meta"), &AccessibilityServer::element_set_meta);
	ClassDB::bind_method(D_METHOD("element_get_meta", "id"), &AccessibilityServer::element_get_meta);

	ClassDB::bind_method(D_METHOD("_update_if_active", "callback"), &AccessibilityServer::update_if_active);

	ClassDB::bind_method(D_METHOD("set_window_rect", "window_id", "rect_out", "rect_in"), &AccessibilityServer::set_window_rect);
	ClassDB::bind_method(D_METHOD("set_window_focused", "window_id", "focused"), &AccessibilityServer::set_window_focused);

	ClassDB::bind_method(D_METHOD("update_set_focus", "id"), &AccessibilityServer::update_set_focus);
	ClassDB::bind_method(D_METHOD("get_window_root", "window_id"), &AccessibilityServer::get_window_root);
	ClassDB::bind_method(D_METHOD("update_set_role", "id", "role"), &AccessibilityServer::update_set_role);
	ClassDB::bind_method(D_METHOD("update_set_name", "id", "name"), &AccessibilityServer::update_set_name);
	ClassDB::bind_method(D_METHOD("update_set_braille_label", "id", "name"), &AccessibilityServer::update_set_braille_label);
	ClassDB::bind_method(D_METHOD("update_set_braille_role_description", "id", "description"), &AccessibilityServer::update_set_braille_role_description);
	ClassDB::bind_method(D_METHOD("update_set_extra_info", "id", "name"), &AccessibilityServer::update_set_extra_info);
	ClassDB::bind_method(D_METHOD("update_set_description", "id", "description"), &AccessibilityServer::update_set_description);
	ClassDB::bind_method(D_METHOD("update_set_value", "id", "value"), &AccessibilityServer::update_set_value);
	ClassDB::bind_method(D_METHOD("update_set_tooltip", "id", "tooltip"), &AccessibilityServer::update_set_tooltip);
	ClassDB::bind_method(D_METHOD("update_set_bounds", "id", "rect"), &AccessibilityServer::update_set_bounds);
	ClassDB::bind_method(D_METHOD("update_set_transform", "id", "transform"), &AccessibilityServer::update_set_transform);
	ClassDB::bind_method(D_METHOD("update_add_child", "id", "child_id"), &AccessibilityServer::update_add_child);
	ClassDB::bind_method(D_METHOD("update_add_related_controls", "id", "related_id"), &AccessibilityServer::update_add_related_controls);
	ClassDB::bind_method(D_METHOD("update_add_related_details", "id", "related_id"), &AccessibilityServer::update_add_related_details);
	ClassDB::bind_method(D_METHOD("update_add_related_described_by", "id", "related_id"), &AccessibilityServer::update_add_related_described_by);
	ClassDB::bind_method(D_METHOD("update_add_related_flow_to", "id", "related_id"), &AccessibilityServer::update_add_related_flow_to);
	ClassDB::bind_method(D_METHOD("update_add_related_labeled_by", "id", "related_id"), &AccessibilityServer::update_add_related_labeled_by);
	ClassDB::bind_method(D_METHOD("update_add_related_radio_group", "id", "related_id"), &AccessibilityServer::update_add_related_radio_group);
	ClassDB::bind_method(D_METHOD("update_set_active_descendant", "id", "other_id"), &AccessibilityServer::update_set_active_descendant);
	ClassDB::bind_method(D_METHOD("update_set_next_on_line", "id", "other_id"), &AccessibilityServer::update_set_next_on_line);
	ClassDB::bind_method(D_METHOD("update_set_previous_on_line", "id", "other_id"), &AccessibilityServer::update_set_previous_on_line);
	ClassDB::bind_method(D_METHOD("update_set_member_of", "id", "group_id"), &AccessibilityServer::update_set_member_of);
	ClassDB::bind_method(D_METHOD("update_set_in_page_link_target", "id", "other_id"), &AccessibilityServer::update_set_in_page_link_target);
	ClassDB::bind_method(D_METHOD("update_set_error_message", "id", "other_id"), &AccessibilityServer::update_set_error_message);
	ClassDB::bind_method(D_METHOD("update_set_live", "id", "live"), &AccessibilityServer::update_set_live);
	ClassDB::bind_method(D_METHOD("update_add_action", "id", "action", "callable"), &AccessibilityServer::update_add_action);
	ClassDB::bind_method(D_METHOD("update_add_custom_action", "id", "action_id", "action_description"), &AccessibilityServer::update_add_custom_action);
	ClassDB::bind_method(D_METHOD("update_set_table_row_count", "id", "count"), &AccessibilityServer::update_set_table_row_count);
	ClassDB::bind_method(D_METHOD("update_set_table_column_count", "id", "count"), &AccessibilityServer::update_set_table_column_count);
	ClassDB::bind_method(D_METHOD("update_set_table_row_index", "id", "index"), &AccessibilityServer::update_set_table_row_index);
	ClassDB::bind_method(D_METHOD("update_set_table_column_index", "id", "index"), &AccessibilityServer::update_set_table_column_index);
	ClassDB::bind_method(D_METHOD("update_set_table_cell_position", "id", "row_index", "column_index"), &AccessibilityServer::update_set_table_cell_position);
	ClassDB::bind_method(D_METHOD("update_set_table_cell_span", "id", "row_span", "column_span"), &AccessibilityServer::update_set_table_cell_span);
	ClassDB::bind_method(D_METHOD("update_set_list_item_count", "id", "size"), &AccessibilityServer::update_set_list_item_count);
	ClassDB::bind_method(D_METHOD("update_set_list_item_index", "id", "index"), &AccessibilityServer::update_set_list_item_index);
	ClassDB::bind_method(D_METHOD("update_set_list_item_level", "id", "level"), &AccessibilityServer::update_set_list_item_level);
	ClassDB::bind_method(D_METHOD("update_set_list_item_selected", "id", "selected"), &AccessibilityServer::update_set_list_item_selected);
	ClassDB::bind_method(D_METHOD("update_set_list_item_expanded", "id", "expanded"), &AccessibilityServer::update_set_list_item_expanded);
	ClassDB::bind_method(D_METHOD("update_set_popup_type", "id", "popup"), &AccessibilityServer::update_set_popup_type);
	ClassDB::bind_method(D_METHOD("update_set_checked", "id", "checekd"), &AccessibilityServer::update_set_checked);
	ClassDB::bind_method(D_METHOD("update_set_num_value", "id", "position"), &AccessibilityServer::update_set_num_value);
	ClassDB::bind_method(D_METHOD("update_set_num_range", "id", "min", "max"), &AccessibilityServer::update_set_num_range);
	ClassDB::bind_method(D_METHOD("update_set_num_step", "id", "step"), &AccessibilityServer::update_set_num_step);
	ClassDB::bind_method(D_METHOD("update_set_num_jump", "id", "jump"), &AccessibilityServer::update_set_num_jump);
	ClassDB::bind_method(D_METHOD("update_set_scroll_x", "id", "position"), &AccessibilityServer::update_set_scroll_x);
	ClassDB::bind_method(D_METHOD("update_set_scroll_x_range", "id", "min", "max"), &AccessibilityServer::update_set_scroll_x_range);
	ClassDB::bind_method(D_METHOD("update_set_scroll_y", "id", "position"), &AccessibilityServer::update_set_scroll_y);
	ClassDB::bind_method(D_METHOD("update_set_scroll_y_range", "id", "min", "max"), &AccessibilityServer::update_set_scroll_y_range);
	ClassDB::bind_method(D_METHOD("update_set_text_decorations", "id", "underline", "strikethrough", "overline", "color"), &AccessibilityServer::update_set_text_decorations, DEFVAL(Color(0, 0, 0, 1)));
	ClassDB::bind_method(D_METHOD("update_set_text_align", "id", "align"), &AccessibilityServer::update_set_text_align);
	ClassDB::bind_method(D_METHOD("update_set_text_selection", "id", "text_start_id", "start_char", "text_end_id", "end_char"), &AccessibilityServer::update_set_text_selection);
	ClassDB::bind_method(D_METHOD("update_set_flag", "id", "flag", "value"), &AccessibilityServer::update_set_flag);
	ClassDB::bind_method(D_METHOD("update_set_classname", "id", "classname"), &AccessibilityServer::update_set_classname);
	ClassDB::bind_method(D_METHOD("update_set_placeholder", "id", "placeholder"), &AccessibilityServer::update_set_placeholder);
	ClassDB::bind_method(D_METHOD("update_set_language", "id", "language"), &AccessibilityServer::update_set_language);
	ClassDB::bind_method(D_METHOD("update_set_text_orientation", "id", "vertical"), &AccessibilityServer::update_set_text_orientation);
	ClassDB::bind_method(D_METHOD("update_set_list_orientation", "id", "vertical"), &AccessibilityServer::update_set_list_orientation);
	ClassDB::bind_method(D_METHOD("update_set_shortcut", "id", "shortcut"), &AccessibilityServer::update_set_shortcut);
	ClassDB::bind_method(D_METHOD("update_set_url", "id", "url"), &AccessibilityServer::update_set_url);
	ClassDB::bind_method(D_METHOD("update_set_role_description", "id", "description"), &AccessibilityServer::update_set_role_description);
	ClassDB::bind_method(D_METHOD("update_set_state_description", "id", "description"), &AccessibilityServer::update_set_state_description);
	ClassDB::bind_method(D_METHOD("update_set_color_value", "id", "color"), &AccessibilityServer::update_set_color_value);
	ClassDB::bind_method(D_METHOD("update_set_background_color", "id", "color"), &AccessibilityServer::update_set_background_color);
	ClassDB::bind_method(D_METHOD("update_set_foreground_color", "id", "color"), &AccessibilityServer::update_set_foreground_color);

	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_UNKNOWN);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_DEFAULT_BUTTON);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_AUDIO);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_VIDEO);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_STATIC_TEXT);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_CONTAINER);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_PANEL);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_BUTTON);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_LINK);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_CHECK_BOX);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_RADIO_BUTTON);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_CHECK_BUTTON);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_SCROLL_BAR);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_SCROLL_VIEW);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_SPLITTER);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_SLIDER);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_SPIN_BUTTON);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_PROGRESS_INDICATOR);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_TEXT_FIELD);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_MULTILINE_TEXT_FIELD);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_COLOR_PICKER);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_TABLE);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_CELL);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_ROW);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_ROW_GROUP);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_ROW_HEADER);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_COLUMN_HEADER);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_TREE);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_TREE_ITEM);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_LIST);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_LIST_ITEM);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_LIST_BOX);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_LIST_BOX_OPTION);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_TAB_BAR);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_TAB);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_TAB_PANEL);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_MENU_BAR);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_MENU);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_MENU_ITEM);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_MENU_ITEM_CHECK_BOX);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_MENU_ITEM_RADIO);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_IMAGE);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_WINDOW);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_TITLE_BAR);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_DIALOG);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_TOOLTIP);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_REGION);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_TEXT_RUN);

	BIND_ENUM_CONSTANT(AccessibilityServerEnums::POPUP_MENU);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::POPUP_LIST);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::POPUP_TREE);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::POPUP_DIALOG);

	BIND_ENUM_CONSTANT(AccessibilityServerEnums::FLAG_HIDDEN);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::FLAG_MULTISELECTABLE);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::FLAG_REQUIRED);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::FLAG_VISITED);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::FLAG_BUSY);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::FLAG_MODAL);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::FLAG_TOUCH_PASSTHROUGH);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::FLAG_READONLY);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::FLAG_DISABLED);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::FLAG_CLIPS_CHILDREN);

	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ACTION_CLICK);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ACTION_FOCUS);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ACTION_BLUR);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ACTION_COLLAPSE);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ACTION_EXPAND);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ACTION_DECREMENT);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ACTION_INCREMENT);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ACTION_HIDE_TOOLTIP);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ACTION_SHOW_TOOLTIP);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ACTION_SET_TEXT_SELECTION);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ACTION_REPLACE_SELECTED_TEXT);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ACTION_SCROLL_BACKWARD);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ACTION_SCROLL_DOWN);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ACTION_SCROLL_FORWARD);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ACTION_SCROLL_LEFT);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ACTION_SCROLL_RIGHT);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ACTION_SCROLL_UP);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ACTION_SCROLL_INTO_VIEW);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ACTION_SCROLL_TO_POINT);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ACTION_SET_SCROLL_OFFSET);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ACTION_SET_VALUE);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ACTION_SHOW_CONTEXT_MENU);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ACTION_CUSTOM);

	BIND_ENUM_CONSTANT(AccessibilityServerEnums::LIVE_OFF);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::LIVE_POLITE);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::LIVE_ASSERTIVE);

	BIND_ENUM_CONSTANT(AccessibilityServerEnums::SCROLL_UNIT_ITEM);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::SCROLL_UNIT_PAGE);

	BIND_ENUM_CONSTANT(AccessibilityServerEnums::SCROLL_HINT_TOP_LEFT);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::SCROLL_HINT_BOTTOM_RIGHT);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::SCROLL_HINT_TOP_EDGE);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::SCROLL_HINT_BOTTOM_EDGE);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::SCROLL_HINT_LEFT_EDGE);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::SCROLL_HINT_RIGHT_EDGE);
}

AccessibilityServer *AccessibilityServer::create(int p_index, Error &r_error) {
	ERR_FAIL_INDEX_V(p_index, server_create_count, nullptr);
	return server_create_functions[p_index].create_function(r_error);
}

void AccessibilityServer::register_create_function(const char *p_name, CreateFunction p_function) {
	ERR_FAIL_COND(server_create_count == MAX_SERVERS);
	// Dummy server is always last
	server_create_functions[server_create_count] = server_create_functions[server_create_count - 1];
	server_create_functions[server_create_count - 1].name = p_name;
	server_create_functions[server_create_count - 1].create_function = p_function;
	server_create_count++;
}

int AccessibilityServer::get_create_function_count() {
	return server_create_count;
}

const char *AccessibilityServer::get_create_function_name(int p_index) {
	ERR_FAIL_INDEX_V(p_index, server_create_count, nullptr);
	return server_create_functions[p_index].name;
}

AccessibilityServer::AccessibilityServer() {
	singleton = this;
}

AccessibilityServer::~AccessibilityServer() {
	singleton = nullptr;
}
