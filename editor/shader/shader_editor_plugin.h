/**************************************************************************/
/*  shader_editor_plugin.h                                                */
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

#include "editor/plugins/editor_plugin.h"

class CodeTextEditor;
class EditorDock;
class HSplitContainer;
class ItemList;
class MenuButton;
class ShaderCreateDialog;
class ShaderEditor;
class TabContainer;
class VBoxContainer;
class HBoxContainer;
class WindowWrapper;

class ShaderEditorPlugin : public EditorPlugin {
	GDCLASS(ShaderEditorPlugin, EditorPlugin);

	struct EditedShader {
		Ref<Shader> shader;
		Ref<ShaderInclude> shader_inc;
		ShaderEditor *shader_editor = nullptr;
		String path;
		String name;
	};

	LocalVector<EditedShader> edited_shaders;

	enum FileMenu {
		FILE_MENU_NEW,
		FILE_MENU_NEW_INCLUDE,
		FILE_MENU_OPEN,
		FILE_MENU_OPEN_INCLUDE,
		FILE_MENU_SAVE,
		FILE_MENU_SAVE_AS,
		FILE_MENU_INSPECT,
		FILE_MENU_INSPECT_NATIVE_SHADER_CODE,
		FILE_MENU_CLOSE,
		FILE_MENU_CLOSE_ALL,
		FILE_MENU_CLOSE_OTHER_TABS,
		FILE_MENU_SHOW_IN_FILE_SYSTEM,
		FILE_MENU_COPY_PATH,
		FILE_MENU_TOGGLE_FILES_PANEL,
	};

	enum PopupMenuType {
		FILE,
		CONTEXT,
		CONTEXT_VALID_ITEM,
	};
	HSplitContainer *files_split = nullptr;

	ItemList *shader_list = nullptr;
	TabContainer *shader_tabs = nullptr;

	MenuButton *file_menu = nullptr;
	PopupMenu *context_menu = nullptr;

	EditorDock *shader_dock = nullptr;
	Ref<Shortcut> make_floating_shortcut;

	ShaderCreateDialog *shader_create_dialog = nullptr;

	float text_shader_zoom_factor = 1.0f;
	bool restoring_layout = false;
	bool selected_changed = false;

	Ref<Resource> _get_current_shader();
	void _update_shader_list();
	void _shader_selected(int p_index, bool p_push_item = true);
	void _shader_list_clicked(int p_item, Vector2 p_local_mouse_pos, MouseButton p_mouse_button_index);
	void _setup_popup_menu(PopupMenuType p_type, PopupMenu *p_menu);
	void _make_script_list_context_menu();
	void _menu_item_pressed(int p_index);
	void _resource_saved(Object *obj);
	void _close_shader(int p_index);
	void _close_builtin_shaders_from_scene(const String &p_scene);
	void _file_removed(const String &p_removed_file);
	void _res_saved_callback(const Ref<Resource> &p_res);
	void _set_file_specific_items_disabled(bool p_disabled);

	void _shader_created(Ref<Shader> p_shader);
	void _shader_include_created(Ref<ShaderInclude> p_shader_inc);
	void _update_shader_list_status();
	void _move_shader_tab(int p_from, int p_to);

	Variant get_drag_data_fw(const Point2 &p_point, Control *p_from);
	bool can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const;
	void drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from);

	void _set_text_shader_zoom_factor(float p_zoom_factor);
	void _update_shader_editor_zoom_factor(CodeTextEditor *p_shader_editor) const;

	void _switch_to_editor(ShaderEditor *p_editor, bool p_focus = false);

protected:
	void _notification(int p_what);

	virtual void shortcut_input(const Ref<InputEvent> &p_event) override;

public:
	virtual String get_plugin_name() const override { return "Shader"; }
	virtual void edit(Object *p_object) override;
	virtual bool handles(Object *p_object) const override;
	virtual void make_visible(bool p_visible) override;

	ShaderEditor *get_shader_editor(const Ref<Shader> &p_for_shader);

	virtual void set_window_layout(Ref<ConfigFile> p_layout) override;
	virtual void get_window_layout(Ref<ConfigFile> p_layout) override;

	virtual String get_unsaved_status(const String &p_for_scene) const override;
	virtual void save_external_data() override;
	virtual void apply_changes() override;

	ShaderEditorPlugin();
	~ShaderEditorPlugin();
};
