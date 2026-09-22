/**************************************************************************/
/*  accessibility_server.h                                                */
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

#include "core/object/object.h"
#include "core/variant/type_info.h"
#include "servers/display/accessibility_server_enums.h"
#include "servers/display/display_server_enums.h"

class AccessibilityServer : public Object {
	GDCLASS(AccessibilityServer, Object)

public:
	typedef AccessibilityServer *(*CreateFunction)(Error &r_error);

protected:
	static inline AccessibilityServer *singleton = nullptr;
	static inline AccessibilityServerEnums::AccessibilityMode accessibility_mode = AccessibilityServerEnums::AccessibilityMode::ACCESSIBILITY_AUTO;

	static void _bind_methods();

	enum {
		MAX_SERVERS = 64
	};

	struct AccessibilityServerCreate {
		const char *name;
		CreateFunction create_function;
	};

	static AccessibilityServerCreate server_create_functions[MAX_SERVERS];
	static int server_create_count;

public:
	_FORCE_INLINE_ static AccessibilityServerEnums::AccessibilityMode get_mode() { return accessibility_mode; }
	static void set_mode(AccessibilityServerEnums::AccessibilityMode p_mode) { accessibility_mode = p_mode; }

	_FORCE_INLINE_ static AccessibilityServer *get_singleton() { return singleton; }

	virtual bool is_supported() const { return false; }

	virtual bool window_create(DisplayServerEnums::WindowID p_window_id, void *p_handle) = 0;
	virtual void window_destroy(DisplayServerEnums::WindowID p_window_id) = 0;

	virtual RID create_element(DisplayServerEnums::WindowID p_window_id, AccessibilityServerEnums::AccessibilityRole p_role) = 0;
	virtual RID create_sub_element(const RID &p_parent_rid, AccessibilityServerEnums::AccessibilityRole p_role, int p_insert_pos = -1) = 0;
	virtual RID create_sub_text_edit_elements(const RID &p_parent_rid, const RID &p_shaped_text, float p_min_height, int p_insert_pos = -1, bool p_is_last_line = false) = 0;
	virtual bool has_element(const RID &p_id) const = 0;
	virtual void free_element(const RID &p_id) = 0;

	virtual void element_set_meta(const RID &p_id, const Variant &p_meta) = 0;
	virtual Variant element_get_meta(const RID &p_id) const = 0;

	virtual void update_if_active(const Callable &p_callable) = 0;

	virtual RID get_window_root(DisplayServerEnums::WindowID p_window_id) const = 0;
	virtual void update_set_focus(const RID &p_id) = 0;

	virtual void set_window_rect(DisplayServerEnums::WindowID p_window_id, const Rect2 &p_rect_out, const Rect2 &p_rect_in) = 0;
	virtual void set_window_focused(DisplayServerEnums::WindowID p_window_id, bool p_focused) = 0;
	virtual void set_window_callbacks(DisplayServerEnums::WindowID p_window_id, const Callable &p_activate_callable, const Callable &p_deativate_callable) = 0;
	virtual void window_activation_completed(DisplayServerEnums::WindowID p_window_id) = 0;
	virtual void window_deactivation_completed(DisplayServerEnums::WindowID p_window_id) = 0;

	virtual void update_set_role(const RID &p_id, AccessibilityServerEnums::AccessibilityRole p_role) = 0;
	virtual void update_set_name(const RID &p_id, const String &p_name) = 0;
	virtual void update_set_braille_label(const RID &p_id, const String &p_name) = 0;
	virtual void update_set_braille_role_description(const RID &p_id, const String &p_description) = 0;
	virtual void update_set_extra_info(const RID &p_id, const String &p_name_extra_info) = 0;
	virtual void update_set_description(const RID &p_id, const String &p_description) = 0;
	virtual void update_set_value(const RID &p_id, const String &p_value) = 0;
	virtual void update_set_tooltip(const RID &p_id, const String &p_tooltip) = 0;
	virtual void update_set_bounds(const RID &p_id, const Rect2 &p_rect) = 0;
	virtual void update_set_transform(const RID &p_id, const Transform2D &p_transform) = 0;
	virtual void update_add_child(const RID &p_id, const RID &p_child_id) = 0;
	virtual void update_add_related_controls(const RID &p_id, const RID &p_related_id) = 0;
	virtual void update_add_related_details(const RID &p_id, const RID &p_related_id) = 0;
	virtual void update_add_related_described_by(const RID &p_id, const RID &p_related_id) = 0;
	virtual void update_add_related_flow_to(const RID &p_id, const RID &p_related_id) = 0;
	virtual void update_add_related_labeled_by(const RID &p_id, const RID &p_related_id) = 0;
	virtual void update_add_related_radio_group(const RID &p_id, const RID &p_related_id) = 0;
	virtual void update_set_active_descendant(const RID &p_id, const RID &p_other_id) = 0;
	virtual void update_set_next_on_line(const RID &p_id, const RID &p_other_id) = 0;
	virtual void update_set_previous_on_line(const RID &p_id, const RID &p_other_id) = 0;
	virtual void update_set_member_of(const RID &p_id, const RID &p_group_id) = 0;
	virtual void update_set_in_page_link_target(const RID &p_id, const RID &p_other_id) = 0;
	virtual void update_set_error_message(const RID &p_id, const RID &p_other_id) = 0;
	virtual void update_set_live(const RID &p_id, AccessibilityServerEnums::AccessibilityLiveMode p_live) = 0;
	virtual void update_add_action(const RID &p_id, AccessibilityServerEnums::AccessibilityAction p_action, const Callable &p_callable) = 0;
	virtual void update_add_custom_action(const RID &p_id, int p_action_id, const String &p_action_description) = 0;
	virtual void update_set_table_row_count(const RID &p_id, int p_count) = 0;
	virtual void update_set_table_column_count(const RID &p_id, int p_count) = 0;
	virtual void update_set_table_row_index(const RID &p_id, int p_index) = 0;
	virtual void update_set_table_column_index(const RID &p_id, int p_index) = 0;
	virtual void update_set_table_cell_position(const RID &p_id, int p_row_index, int p_column_index) = 0;
	virtual void update_set_table_cell_span(const RID &p_id, int p_row_span, int p_column_span) = 0;
	virtual void update_set_list_item_count(const RID &p_id, int p_size) = 0;
	virtual void update_set_list_item_index(const RID &p_id, int p_index) = 0;
	virtual void update_set_list_item_level(const RID &p_id, int p_level) = 0;
	virtual void update_set_list_item_selected(const RID &p_id, bool p_selected) = 0;
	virtual void update_set_list_item_expanded(const RID &p_id, bool p_expanded) = 0;
	virtual void update_set_popup_type(const RID &p_id, AccessibilityServerEnums::AccessibilityPopupType p_popup) = 0;
	virtual void update_set_checked(const RID &p_id, bool p_checekd) = 0;
	virtual void update_set_num_value(const RID &p_id, double p_position) = 0;
	virtual void update_set_num_range(const RID &p_id, double p_min, double p_max) = 0;
	virtual void update_set_num_step(const RID &p_id, double p_step) = 0;
	virtual void update_set_num_jump(const RID &p_id, double p_jump) = 0;
	virtual void update_set_scroll_x(const RID &p_id, double p_position) = 0;
	virtual void update_set_scroll_x_range(const RID &p_id, double p_min, double p_max) = 0;
	virtual void update_set_scroll_y(const RID &p_id, double p_position) = 0;
	virtual void update_set_scroll_y_range(const RID &p_id, double p_min, double p_max) = 0;
	virtual void update_set_text_decorations(const RID &p_id, bool p_underline, bool p_strikethrough, bool p_overline, const Color &p_color) = 0;
	virtual void update_set_text_align(const RID &p_id, HorizontalAlignment p_align) = 0;
	virtual void update_set_text_selection(const RID &p_id, const RID &p_text_start_id, int p_start_char, const RID &p_text_end_id, int p_end_char) = 0;
	virtual void update_set_flag(const RID &p_id, AccessibilityServerEnums::AccessibilityFlags p_flag, bool p_value) = 0;
	virtual void update_set_classname(const RID &p_id, const String &p_classname) = 0;
	virtual void update_set_placeholder(const RID &p_id, const String &p_placeholder) = 0;
	virtual void update_set_language(const RID &p_id, const String &p_language) = 0;
	virtual void update_set_text_orientation(const RID &p_id, bool p_vertical) = 0;
	virtual void update_set_list_orientation(const RID &p_id, bool p_vertical) = 0;
	virtual void update_set_shortcut(const RID &p_id, const String &p_shortcut) = 0;
	virtual void update_set_url(const RID &p_id, const String &p_url) = 0;
	virtual void update_set_role_description(const RID &p_id, const String &p_description) = 0;
	virtual void update_set_state_description(const RID &p_id, const String &p_description) = 0;
	virtual void update_set_color_value(const RID &p_id, const Color &p_color) = 0;
	virtual void update_set_background_color(const RID &p_id, const Color &p_color) = 0;
	virtual void update_set_foreground_color(const RID &p_id, const Color &p_color) = 0;

	static void register_create_function(const char *p_name, CreateFunction p_function);
	static int get_create_function_count();
	static const char *get_create_function_name(int p_index);
	static AccessibilityServer *create(int p_index, Error &r_error);

	AccessibilityServer();
	virtual ~AccessibilityServer();
};

VARIANT_ENUM_CAST_EXT(AccessibilityServerEnums::AccessibilityAction, AccessibilityServer::AccessibilityAction)
VARIANT_ENUM_CAST_EXT(AccessibilityServerEnums::AccessibilityFlags, AccessibilityServer::AccessibilityFlags)
VARIANT_ENUM_CAST_EXT(AccessibilityServerEnums::AccessibilityLiveMode, AccessibilityServer::AccessibilityLiveMode)
VARIANT_ENUM_CAST_EXT(AccessibilityServerEnums::AccessibilityPopupType, AccessibilityServer::AccessibilityPopupType)
VARIANT_ENUM_CAST_EXT(AccessibilityServerEnums::AccessibilityRole, AccessibilityServer::AccessibilityRole)
VARIANT_ENUM_CAST_EXT(AccessibilityServerEnums::AccessibilityScrollUnit, AccessibilityServer::AccessibilityScrollUnit)
VARIANT_ENUM_CAST_EXT(AccessibilityServerEnums::AccessibilityScrollHint, AccessibilityServer::AccessibilityScrollHint)
