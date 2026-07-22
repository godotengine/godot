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
	ClassDB::bind_method(D_METHOD("element_set_parent", "id", "parent_id"), &AccessibilityServer::element_set_parent);
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
	ClassDB::bind_method(D_METHOD("update_clear_children", "id"), &AccessibilityServer::update_clear_children);
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
	ClassDB::bind_method(D_METHOD("update_set_author_id", "id", "author_id"), &AccessibilityServer::update_set_author_id);
	ClassDB::bind_method(D_METHOD("update_set_expanded", "id", "state"), &AccessibilityServer::update_set_expanded);
	ClassDB::bind_method(D_METHOD("update_set_checked_state", "id", "state"), &AccessibilityServer::update_set_checked_state);
	ClassDB::bind_method(D_METHOD("update_set_selected_state", "id", "state"), &AccessibilityServer::update_set_selected_state);
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
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_GROUP);
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
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_COMBO_BOX);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_EDITABLE_COMBO_BOX);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_MENU_LIST_OPTION);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_MENU_LIST_POPUP);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_SEARCH_INPUT);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_DATE_INPUT);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_DATE_TIME_INPUT);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_WEEK_INPUT);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_MONTH_INPUT);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_TIME_INPUT);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_EMAIL_INPUT);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_NUMBER_INPUT);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_PASSWORD_INPUT);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_PHONE_NUMBER_INPUT);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_URL_INPUT);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_SWITCH);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_PARAGRAPH);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_LABEL);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_ABBR);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_ALERT);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_ALERT_DIALOG);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_APPLICATION);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_ARTICLE);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_BANNER);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_BLOCKQUOTE);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_CANVAS);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_CAPTION);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_CARET);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_CODE);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_COMPLEMENTARY);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_COMMENT);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_CONTENT_DELETION);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_CONTENT_INSERTION);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_CONTENT_INFO);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_DEFINITION);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_DESCRIPTION_LIST);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_DETAILS);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_DISCLOSURE_TRIANGLE);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_DOCUMENT);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_EMBEDDED_OBJECT);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_EMPHASIS);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_FEED);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_FIGURE);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_FIGURE_CAPTION);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_FOOTER);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_FORM);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_GRID);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_GRID_CELL);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_HEADER);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_HEADING);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_IFRAME);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_IFRAME_PRESENTATIONAL);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_IME_CANDIDATE);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_KEYBOARD);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_LEGEND);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_LINE_BREAK);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_LIST_MARKER);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_LOG);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_MAIN);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_MARK);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_MARQUEE);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_MATH);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_METER);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_NAVIGATION);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_NOTE);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_PLUGIN_OBJECT);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_RADIO_GROUP);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_ROOT_WEB_AREA);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_RUBY);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_RUBY_ANNOTATION);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_SEARCH);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_SECTION);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_SECTION_HEADER);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_SECTION_FOOTER);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_STATUS);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_STRONG);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_SUGGESTION);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_SVG_ROOT);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_TERM);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_TIMER);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_TOOLBAR);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_TREE_GRID);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_WEB_VIEW);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_LIST_GRID);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_TERMINAL);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_GRAPHICS_DOCUMENT);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_GRAPHICS_OBJECT);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_GRAPHICS_SYMBOL);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_PDF_ROOT);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_PDF_ACTIONABLE_HIGHLIGHT);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_DOC_ABSTRACT);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_DOC_ACKNOWLEDGEMENTS);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_DOC_AFTERWORD);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_DOC_APPENDIX);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_DOC_BACK_LINK);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_DOC_BIBLIO_ENTRY);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_DOC_BIBLIOGRAPHY);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_DOC_BIBLIO_REF);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_DOC_CHAPTER);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_DOC_COLOPHON);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_DOC_CONCLUSION);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_DOC_COVER);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_DOC_CREDIT);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_DOC_CREDITS);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_DOC_DEDICATION);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_DOC_ENDNOTE);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_DOC_ENDNOTES);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_DOC_EPIGRAPH);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_DOC_EPILOGUE);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_DOC_ERRATA);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_DOC_EXAMPLE);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_DOC_FOOTNOTE);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_DOC_FOREWORD);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_DOC_GLOSSARY);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_DOC_GLOSS_REF);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_DOC_INDEX);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_DOC_INTRODUCTION);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_DOC_NOTE_REF);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_DOC_NOTICE);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_DOC_PAGE_BREAK);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_DOC_PAGE_FOOTER);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_DOC_PAGE_HEADER);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_DOC_PAGE_LIST);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_DOC_PART);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_DOC_PREFACE);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_DOC_PROLOGUE);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_DOC_PULLQUOTE);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_DOC_QNA);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_DOC_SUBTITLE);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_DOC_TIP);
	BIND_ENUM_CONSTANT(AccessibilityServerEnums::ROLE_DOC_TOC);

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
