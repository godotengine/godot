/**************************************************************************/
/*  native_menu_macos.h                                                   */
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

#ifndef NATIVE_MENU_MACOS_H
#define NATIVE_MENU_MACOS_H

#include "core/templates/hash_map.h"
#include "core/templates/rid_owner.h"
#include "servers/display/native_menu.h"
#include "servers/display_server.h"

#import <AppKit/AppKit.h>
#import <ApplicationServices/ApplicationServices.h>

class NativeMenuMacOS : public NativeMenu {
	GDCLASS(NativeMenuMacOS, NativeMenu)

	struct MenuData {
		NSMenu *menu = nullptr;

		Callable open_cb;
		Callable close_cb;
		bool is_open = false;
		bool is_system = false;
	};

	mutable RID_PtrOwner<MenuData> menus;
	HashMap<NSMenu *, RID> menu_lookup;

	NSMenu *main_menu_ns = nullptr;
	NSMenu *application_menu_ns = nullptr;
	NSMenu *window_menu_ns = nullptr;
	NSMenu *help_menu_ns = nullptr;
	NSMenu *dock_menu_ns = nullptr;
	NSMenu *edit_menu_ns = nullptr;
	NSMenu *file_menu_ns = nullptr;

	NSMenuItem *file_menu_item = nullptr;

	RID main_menu;
	RID application_menu;
	RID window_menu;
	RID help_menu;
	RID dock_menu;
	RID edit_menu;
	RID file_menu;

	NSMenuItem *copy_item = nullptr;
	bool can_copy = true;
	Callable copy_cb;

	NSMenuItem *cut_item = nullptr;
	bool can_cut = true;
	Callable cut_cb;

	NSMenuItem *paste_item = nullptr;
	bool can_paste = true;
	Callable paste_cb;

	NSMenuItem *undo_item = nullptr;
	bool can_undo = true;
	Callable undo_cb;
	String undo_description;

	NSMenuItem *redo_item = nullptr;
	bool can_redo = true;
	Callable redo_cb;
	String redo_description;

	int _get_system_menu_start(const NSMenu *p_menu) const;
	int _get_system_menu_count(const NSMenu *p_menu) const;
	bool _is_menu_opened(NSMenu *p_menu) const;
	NSMenuItem *_menu_add_item(NSMenu *p_menu, const String &p_label, Key p_accel, int p_index, int *r_out);

public:
	void _register_system_menus(NSMenu *p_main_menu, NSMenu *p_application_menu, NSMenu *p_window_menu, NSMenu *p_help_menu, NSMenu *p_dock_menu, NSMenu *p_edit_menu, NSMenu *p_file_menu, NSMenuItem *p_file_menu_item, NSMenuItem *p_copy_item, NSMenuItem *p_cut_item, NSMenuItem *p_paste_item, NSMenuItem *p_undo_item, NSMenuItem *p_redo_item);
	NSMenu *_get_dock_menu();

	void _process_ui_event(DisplayServer::WindowID p_window_id, const StringName &p_name);
	void _copy_action(DisplayServer::WindowID p_window_id);
	void _cut_action(DisplayServer::WindowID p_window_id);
	void _paste_action(DisplayServer::WindowID p_window_id);
	void _undo_action(DisplayServer::WindowID p_window_id);
	void _redo_action(DisplayServer::WindowID p_window_id);

	void _menu_need_update(NSMenu *p_menu);
	void _menu_open(NSMenu *p_menu);
	void _menu_close(NSMenu *p_menu);
	void _menu_close_cb(const RID &p_rid);

	virtual bool has_feature(Feature p_feature) const override;

	virtual bool has_system_menu(SystemMenus p_menu_id) const override;
	virtual RID get_system_menu(SystemMenus p_menu_id) const override;

	virtual bool get_system_menu_no_default_items(SystemMenus p_menu_id) const override;
	virtual void set_system_menu_name(SystemMenus p_menu_id, const String &p_string) override;
	virtual void set_system_menu_hidden(SystemMenus p_menu_id, bool p_hidden) override;

	virtual RID create_menu() override;
	virtual bool has_menu(const RID &p_rid) const override;
	virtual void free_menu(const RID &p_rid) override;

	NSMenu *get_native_menu_handle(const RID &p_rid);

	virtual Size2 get_size(const RID &p_rid) const override;
	virtual void popup(const RID &p_rid, const Vector2i &p_position) override;

	virtual void set_interface_direction(const RID &p_rid, bool p_is_rtl) override;
	virtual void set_popup_open_callback(const RID &p_rid, const Callable &p_callback) override;
	virtual Callable get_popup_open_callback(const RID &p_rid) const override;
	virtual void set_popup_close_callback(const RID &p_rid, const Callable &p_callback) override;
	virtual Callable get_popup_close_callback(const RID &p_rid) const override;
	virtual void set_minimum_width(const RID &p_rid, float p_width) override;
	virtual float get_minimum_width(const RID &p_rid) const override;

	virtual bool is_opened(const RID &p_rid) const override;

	virtual int add_submenu_item(const RID &p_rid, const String &p_label, const RID &p_submenu_rid, const Variant &p_tag = Variant(), int p_index = -1) override;
	virtual int add_item(const RID &p_rid, const String &p_label, const Callable &p_callback = Callable(), const Callable &p_key_callback = Callable(), const Variant &p_tag = Variant(), Key p_accel = Key::NONE, int p_index = -1) override;
	virtual int add_check_item(const RID &p_rid, const String &p_label, const Callable &p_callback = Callable(), const Callable &p_key_callback = Callable(), const Variant &p_tag = Variant(), Key p_accel = Key::NONE, int p_index = -1) override;
	virtual int add_icon_item(const RID &p_rid, const Ref<Texture2D> &p_icon, const String &p_label, const Callable &p_callback = Callable(), const Callable &p_key_callback = Callable(), const Variant &p_tag = Variant(), Key p_accel = Key::NONE, int p_index = -1) override;
	virtual int add_icon_check_item(const RID &p_rid, const Ref<Texture2D> &p_icon, const String &p_label, const Callable &p_callback = Callable(), const Callable &p_key_callback = Callable(), const Variant &p_tag = Variant(), Key p_accel = Key::NONE, int p_index = -1) override;
	virtual int add_radio_check_item(const RID &p_rid, const String &p_label, const Callable &p_callback = Callable(), const Callable &p_key_callback = Callable(), const Variant &p_tag = Variant(), Key p_accel = Key::NONE, int p_index = -1) override;
	virtual int add_icon_radio_check_item(const RID &p_rid, const Ref<Texture2D> &p_icon, const String &p_label, const Callable &p_callback = Callable(), const Callable &p_key_callback = Callable(), const Variant &p_tag = Variant(), Key p_accel = Key::NONE, int p_index = -1) override;
	virtual int add_multistate_item(const RID &p_rid, const String &p_label, int p_max_states, int p_default_state, const Callable &p_callback = Callable(), const Callable &p_key_callback = Callable(), const Variant &p_tag = Variant(), Key p_accel = Key::NONE, int p_index = -1) override;
	virtual int add_separator(const RID &p_rid, int p_index = -1) override;

	virtual int find_item_index_with_text(const RID &p_rid, const String &p_text) const override;
	virtual int find_item_index_with_tag(const RID &p_rid, const Variant &p_tag) const override;

	virtual bool is_item_checked(const RID &p_rid, int p_idx) const override;
	virtual bool is_item_checkable(const RID &p_rid, int p_idx) const override;
	virtual bool is_item_radio_checkable(const RID &p_rid, int p_idx) const override;
	virtual Callable get_item_callback(const RID &p_rid, int p_idx) const override;
	virtual Callable get_item_key_callback(const RID &p_rid, int p_idx) const override;
	virtual Variant get_item_tag(const RID &p_rid, int p_idx) const override;
	virtual String get_item_text(const RID &p_rid, int p_idx) const override;
	virtual RID get_item_submenu(const RID &p_rid, int p_idx) const override;
	virtual Key get_item_accelerator(const RID &p_rid, int p_idx) const override;
	virtual bool is_item_disabled(const RID &p_rid, int p_idx) const override;
	virtual bool is_item_hidden(const RID &p_rid, int p_idx) const override;
	virtual String get_item_tooltip(const RID &p_rid, int p_idx) const override;
	virtual int get_item_state(const RID &p_rid, int p_idx) const override;
	virtual int get_item_max_states(const RID &p_rid, int p_idx) const override;
	virtual Ref<Texture2D> get_item_icon(const RID &p_rid, int p_idx) const override;
	virtual int get_item_indentation_level(const RID &p_rid, int p_idx) const override;

	virtual void set_item_checked(const RID &p_rid, int p_idx, bool p_checked) override;
	virtual void set_item_checkable(const RID &p_rid, int p_idx, bool p_checkable) override;
	virtual void set_item_radio_checkable(const RID &p_rid, int p_idx, bool p_checkable) override;
	virtual void set_item_callback(const RID &p_rid, int p_idx, const Callable &p_callback) override;
	virtual void set_item_key_callback(const RID &p_rid, int p_idx, const Callable &p_key_callback) override;
	virtual void set_item_hover_callbacks(const RID &p_rid, int p_idx, const Callable &p_callback) override;
	virtual void set_item_tag(const RID &p_rid, int p_idx, const Variant &p_tag) override;
	virtual void set_item_text(const RID &p_rid, int p_idx, const String &p_text) override;
	virtual void set_item_submenu(const RID &p_rid, int p_idx, const RID &p_submenu_rid) override;
	virtual void set_item_accelerator(const RID &p_rid, int p_idx, Key p_keycode) override;
	virtual void set_item_disabled(const RID &p_rid, int p_idx, bool p_disabled) override;
	virtual void set_item_hidden(const RID &p_rid, int p_idx, bool p_hidden) override;
	virtual void set_item_tooltip(const RID &p_rid, int p_idx, const String &p_tooltip) override;
	virtual void set_item_state(const RID &p_rid, int p_idx, int p_state) override;
	virtual void set_item_max_states(const RID &p_rid, int p_idx, int p_max_states) override;
	virtual void set_item_icon(const RID &p_rid, int p_idx, const Ref<Texture2D> &p_icon) override;
	virtual void set_item_indentation_level(const RID &p_rid, int p_idx, int p_level) override;

	virtual int get_item_count(const RID &p_rid) const override;
	virtual bool is_system_menu(const RID &p_rid) const override;

	virtual void remove_item(const RID &p_rid, int p_idx) override;
	virtual void clear(const RID &p_rid) override;

	virtual void set_can_copy(bool p_enabled) override;
	virtual bool get_can_copy() const override;
	virtual void set_copy_callback(const Callable &p_callback) override;
	virtual Callable get_copy_callback() const override;

	virtual void set_can_cut(bool p_enabled) override;
	virtual bool get_can_cut() const override;
	virtual void set_cut_callback(const Callable &p_callback) override;
	virtual Callable get_cut_callback() const override;

	virtual void set_can_paste(bool p_enabled) override;
	virtual bool get_can_paste() const override;
	virtual void set_paste_callback(const Callable &p_callback) override;
	virtual Callable get_paste_callback() const override;

	virtual void set_can_undo(bool p_enabled) override;
	virtual bool get_can_undo() const override;
	virtual void set_undo_callback(const Callable &p_callback) override;
	virtual Callable get_undo_callback() const override;

	virtual void set_undo_description(const String &p_description) override;
	virtual String get_undo_description() const override;

	virtual void set_can_redo(bool p_enabled) override;
	virtual bool get_can_redo() const override;
	virtual void set_redo_callback(const Callable &p_callback) override;
	virtual Callable get_redo_callback() const override;

	virtual void set_redo_description(const String &p_description) override;
	virtual String get_redo_description() const override;

	NativeMenuMacOS();
	~NativeMenuMacOS();
};

#endif // NATIVE_MENU_MACOS_H
