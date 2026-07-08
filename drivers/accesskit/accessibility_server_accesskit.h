/**************************************************************************/
/*  accessibility_server_accesskit.h                                      */
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

#ifdef ACCESSKIT_ENABLED

#include "core/templates/rid_owner.h"
#include "servers/display/accessibility_server.h"

#include <accesskit.h>

class AccessibilityServerAccessKit : public AccessibilityServer {
	GDSOFTCLASS(AccessibilityServerAccessKit, AccessibilityServer);

	static AccessibilityServer *create_func(Error &r_error);

	struct AccessibilityElement {
		HashMap<accesskit_action, Callable> actions;

		DisplayServerEnums::WindowID window_id = DisplayServerEnums::INVALID_WINDOW_ID;
		RID parent;
		LocalVector<RID> children;
		Vector3i run;
		Variant meta;
		String name;
		String name_extra_info;
		Variant value;
		int64_t flags = 0;

		accesskit_role role = ACCESSKIT_ROLE_UNKNOWN;
		accesskit_node *node = nullptr;
	};
	mutable RID_PtrOwner<AccessibilityElement> rid_owner{ 65536, 1048576 };

	struct WindowData {
		// Adapter.
#ifdef WINDOWS_ENABLED
		accesskit_windows_subclassing_adapter *adapter = nullptr;
#endif
#ifdef MACOS_ENABLED
		accesskit_macos_subclassing_adapter *adapter = nullptr;
#endif
#ifdef LINUXBSD_ENABLED
		accesskit_unix_adapter *adapter = nullptr;
#endif

		RID root_id;
		bool initial_update_completed = false;
		HashSet<RID> update;
		Callable activate;
		Callable deactivate;
		bool activated = false;
	};

	RID focus;

	HashMap<DisplayServerEnums::WindowID, WindowData> windows;

	HashMap<AccessibilityServerEnums::AccessibilityRole, accesskit_role> role_map;
	HashMap<AccessibilityServerEnums::AccessibilityAction, accesskit_action> action_map;

	_FORCE_INLINE_ accesskit_role _accessibility_role(AccessibilityServerEnums::AccessibilityRole p_role) const;
	_FORCE_INLINE_ accesskit_action _accessibility_action(AccessibilityServerEnums::AccessibilityAction p_action) const;

	void _free_recursive(WindowData *p_wd, const RID &p_id);
	_FORCE_INLINE_ void _ensure_node(const RID &p_id, AccessibilityElement *p_ae);

	static void _accessibility_action_callback(struct accesskit_action_request *p_request, void *p_user_data);
	static accesskit_tree_update *_accessibility_initial_tree_update_callback(void *p_user_data);
	static void _accessibility_deactivation_callback(void *p_user_data);
	static accesskit_tree_update *_accessibility_build_tree_update(void *p_user_data);

	bool in_accessibility_update = false;
	Callable update_cb;

public:
	bool is_supported() const override { return true; }

	bool window_create(DisplayServerEnums::WindowID p_window_id, void *p_handle) override;
	void window_destroy(DisplayServerEnums::WindowID p_window_id) override;

	RID create_element(DisplayServerEnums::WindowID p_window_id, AccessibilityServerEnums::AccessibilityRole p_role) override;
	RID create_sub_element(const RID &p_parent_rid, AccessibilityServerEnums::AccessibilityRole p_role, int p_insert_pos = -1) override;
	virtual RID create_sub_text_edit_elements(const RID &p_parent_rid, const RID &p_shaped_text, float p_min_height, int p_insert_pos = -1, bool p_is_last_line = false) override;
	bool has_element(const RID &p_id) const override;
	void free_element(const RID &p_id) override;

	void element_set_meta(const RID &p_id, const Variant &p_meta) override;
	Variant element_get_meta(const RID &p_id) const override;

	void update_if_active(const Callable &p_callable) override;

	void update_set_focus(const RID &p_id) override;
	RID get_window_root(DisplayServerEnums::WindowID p_window_id) const override;

	void set_window_rect(DisplayServerEnums::WindowID p_window_id, const Rect2 &p_rect_out, const Rect2 &p_rect_in) override;
	void set_window_focused(DisplayServerEnums::WindowID p_window_id, bool p_focused) override;
	void set_window_callbacks(DisplayServerEnums::WindowID p_window_id, const Callable &p_activate_callable, const Callable &p_deativate_callable) override;
	void window_activation_completed(DisplayServerEnums::WindowID p_window_id) override;
	void window_deactivation_completed(DisplayServerEnums::WindowID p_window_id) override;

	void update_set_role(const RID &p_id, AccessibilityServerEnums::AccessibilityRole p_role) override;
	void update_set_name(const RID &p_id, const String &p_name) override;
	void update_set_braille_label(const RID &p_id, const String &p_name) override;
	void update_set_braille_role_description(const RID &p_id, const String &p_description) override;
	void update_set_extra_info(const RID &p_id, const String &p_name_extra_info) override;
	void update_set_description(const RID &p_id, const String &p_description) override;
	void update_set_value(const RID &p_id, const String &p_value) override;
	void update_set_tooltip(const RID &p_id, const String &p_tooltip) override;
	void update_set_bounds(const RID &p_id, const Rect2 &p_rect) override;
	void update_set_transform(const RID &p_id, const Transform2D &p_transform) override;
	void update_add_child(const RID &p_id, const RID &p_child_id) override;
	void update_add_related_controls(const RID &p_id, const RID &p_related_id) override;
	void update_add_related_details(const RID &p_id, const RID &p_related_id) override;
	void update_add_related_described_by(const RID &p_id, const RID &p_related_id) override;
	void update_add_related_flow_to(const RID &p_id, const RID &p_related_id) override;
	void update_add_related_labeled_by(const RID &p_id, const RID &p_related_id) override;
	void update_add_related_radio_group(const RID &p_id, const RID &p_related_id) override;
	void update_set_active_descendant(const RID &p_id, const RID &p_other_id) override;
	void update_set_next_on_line(const RID &p_id, const RID &p_other_id) override;
	void update_set_previous_on_line(const RID &p_id, const RID &p_other_id) override;
	void update_set_member_of(const RID &p_id, const RID &p_group_id) override;
	void update_set_in_page_link_target(const RID &p_id, const RID &p_other_id) override;
	void update_set_error_message(const RID &p_id, const RID &p_other_id) override;
	void update_set_live(const RID &p_id, AccessibilityServerEnums::AccessibilityLiveMode p_live) override;
	void update_add_action(const RID &p_id, AccessibilityServerEnums::AccessibilityAction p_action, const Callable &p_callable) override;
	void update_add_custom_action(const RID &p_id, int p_action_id, const String &p_action_description) override;
	void update_set_table_row_count(const RID &p_id, int p_count) override;
	void update_set_table_column_count(const RID &p_id, int p_count) override;
	void update_set_table_row_index(const RID &p_id, int p_index) override;
	void update_set_table_column_index(const RID &p_id, int p_index) override;
	void update_set_table_cell_position(const RID &p_id, int p_row_index, int p_column_index) override;
	void update_set_table_cell_span(const RID &p_id, int p_row_span, int p_column_span) override;
	void update_set_list_item_count(const RID &p_id, int p_size) override;
	void update_set_list_item_index(const RID &p_id, int p_index) override;
	void update_set_list_item_level(const RID &p_id, int p_level) override;
	void update_set_list_item_selected(const RID &p_id, bool p_selected) override;
	void update_set_list_item_expanded(const RID &p_id, bool p_expanded) override;
	void update_set_popup_type(const RID &p_id, AccessibilityServerEnums::AccessibilityPopupType p_popup) override;
	void update_set_checked(const RID &p_id, bool p_checekd) override;
	void update_set_num_value(const RID &p_id, double p_position) override;
	void update_set_num_range(const RID &p_id, double p_min, double p_max) override;
	void update_set_num_step(const RID &p_id, double p_step) override;
	void update_set_num_jump(const RID &p_id, double p_jump) override;
	void update_set_scroll_x(const RID &p_id, double p_position) override;
	void update_set_scroll_x_range(const RID &p_id, double p_min, double p_max) override;
	void update_set_scroll_y(const RID &p_id, double p_position) override;
	void update_set_scroll_y_range(const RID &p_id, double p_min, double p_max) override;
	void update_set_text_decorations(const RID &p_id, bool p_underline, bool p_strikethrough, bool p_overline, const Color &p_color) override;
	void update_set_text_align(const RID &p_id, HorizontalAlignment p_align) override;
	void update_set_text_selection(const RID &p_id, const RID &p_text_start_id, int p_start_char, const RID &p_text_end_id, int p_end_char) override;
	void update_set_flag(const RID &p_id, AccessibilityServerEnums::AccessibilityFlags p_flag, bool p_value) override;
	void update_set_classname(const RID &p_id, const String &p_classname) override;
	void update_set_placeholder(const RID &p_id, const String &p_placeholder) override;
	void update_set_language(const RID &p_id, const String &p_language) override;
	void update_set_text_orientation(const RID &p_id, bool p_vertical) override;
	void update_set_list_orientation(const RID &p_id, bool p_vertical) override;
	void update_set_shortcut(const RID &p_id, const String &p_shortcut) override;
	void update_set_url(const RID &p_id, const String &p_url) override;
	void update_set_role_description(const RID &p_id, const String &p_description) override;
	void update_set_state_description(const RID &p_id, const String &p_description) override;
	void update_set_color_value(const RID &p_id, const Color &p_color) override;
	void update_set_background_color(const RID &p_id, const Color &p_color) override;
	void update_set_foreground_color(const RID &p_id, const Color &p_color) override;

	static void register_create_func();

	AccessibilityServerAccessKit();
	~AccessibilityServerAccessKit();
};

#endif // ACCESSKIT_ENABLED
