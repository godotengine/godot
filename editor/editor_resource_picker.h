/**************************************************************************/
/*  editor_resource_picker.h                                              */
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

#ifndef EDITOR_RESOURCE_PICKER_H
#define EDITOR_RESOURCE_PICKER_H

#include "editor_file_dialog.h"
#include "editor_quick_open.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/popup_menu.h"
#include "scene/gui/texture_rect.h"

class EditorResourcePicker : public HBoxContainer {
	GDCLASS(EditorResourcePicker, HBoxContainer);

	static HashMap<StringName, List<StringName>> allowed_types_cache;

	String base_type;
	RES edited_resource;

	bool editable = true;
	bool dropping = false;

	Vector<String> inheritors_array;

	Button *assign_button;
	TextureRect *preview_rect;
	Button *edit_button;
	EditorFileDialog *file_dialog = nullptr;
	EditorQuickOpen *quick_open = nullptr;

	enum MenuOption {
		OBJ_MENU_LOAD,
		OBJ_MENU_QUICKLOAD,
		OBJ_MENU_EDIT,
		OBJ_MENU_CLEAR,
		OBJ_MENU_MAKE_UNIQUE,
		OBJ_MENU_SAVE,
		OBJ_MENU_COPY,
		OBJ_MENU_PASTE,
		OBJ_MENU_SHOW_IN_FILE_SYSTEM,

		TYPE_BASE_ID = 100,
		CONVERT_BASE_ID = 1000,
	};

	PopupMenu *edit_menu;

	void _update_resource();
	void _update_resource_preview(const String &p_path, const Ref<Texture> &p_preview, const Ref<Texture> &p_small_preview, ObjectID p_obj);

	void _resource_selected();
	void _file_quick_selected();
	void _file_selected(const String &p_path);

	void _update_menu();
	void _update_menu_items();
	void _edit_menu_cbk(int p_which);

	void _button_draw();
	void _button_input(const Ref<InputEvent> &p_event);

	void _get_allowed_types(bool p_with_convert, Set<String> *p_vector) const;
	bool _is_drop_valid(const Dictionary &p_drag_data) const;
	bool _is_type_valid(const String p_type_name, Set<String> p_allowed_types) const;

	Variant get_drag_data_fw(const Point2 &p_point, Control *p_from);
	bool can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const;
	void drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from);

protected:
	static void _bind_methods();
	void _notification(int p_what);

public:
	static void clear_caches();

	void set_base_type(const String &p_base_type);
	String get_base_type() const;
	Vector<String> get_allowed_types() const;

	void set_edited_resource(RES p_resource);
	RES get_edited_resource();

	void set_toggle_mode(bool p_enable);
	bool is_toggle_mode() const;
	void set_toggle_pressed(bool p_pressed);

	void set_editable(bool p_editable);
	bool is_editable() const;

	virtual void set_create_options(Object *p_menu_node);
	virtual bool handle_menu_selected(int p_which);

	EditorResourcePicker();
};

class EditorScriptPicker : public EditorResourcePicker {
	GDCLASS(EditorScriptPicker, EditorResourcePicker);

	enum ExtraMenuOption {
		OBJ_MENU_NEW_SCRIPT = 10,
		OBJ_MENU_EXTEND_SCRIPT = 11
	};

	Node *script_owner = nullptr;

protected:
	static void _bind_methods();

public:
	virtual void set_create_options(Object *p_menu_node);
	virtual bool handle_menu_selected(int p_which);

	void set_script_owner(Node *p_owner);
	Node *get_script_owner() const;

	EditorScriptPicker();
};

#endif // EDITOR_RESOURCE_PICKER_H
