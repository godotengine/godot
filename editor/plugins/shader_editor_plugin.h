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

#ifndef SHADER_EDITOR_PLUGIN_H
#define SHADER_EDITOR_PLUGIN_H

#include "editor/editor_plugin.h"

class HSplitContainer;
class ItemList;
class MenuButton;
class ShaderCreateDialog;
class TabContainer;
class TextShaderEditor;
class VisualShaderEditor;
class WindowWrapper;

#ifdef MINGW_ENABLED
#undef FILE_OPEN
#endif

class ShaderEditorPlugin : public EditorPlugin {
	GDCLASS(ShaderEditorPlugin, EditorPlugin);

	struct EditedShader {
		Ref<Shader> shader;
		Ref<ShaderInclude> shader_inc;
		TextShaderEditor *shader_editor = nullptr;
		VisualShaderEditor *visual_shader_editor = nullptr;
		String path;
	};

	LocalVector<EditedShader> edited_shaders;

	// Always valid operations come first in the enum, file-specific ones
	// should go after FILE_SAVE which is used to build the menu accordingly.
	enum {
		FILE_NEW,
		FILE_NEW_INCLUDE,
		FILE_OPEN,
		FILE_OPEN_INCLUDE,
		FILE_SAVE,
		FILE_SAVE_AS,
		FILE_INSPECT,
		FILE_CLOSE,
		FILE_MAX
	};

	HSplitContainer *main_split = nullptr;
	ItemList *shader_list = nullptr;
	TabContainer *shader_tabs = nullptr;

	Button *button = nullptr;
	MenuButton *file_menu = nullptr;

	WindowWrapper *window_wrapper = nullptr;
	Button *make_floating = nullptr;

	ShaderCreateDialog *shader_create_dialog = nullptr;

	void _update_shader_list();
	void _shader_selected(int p_index);
	void _shader_list_clicked(int p_item, Vector2 p_local_mouse_pos, MouseButton p_mouse_button_index);
	void _menu_item_pressed(int p_index);
	void _resource_saved(Object *obj);
	void _close_shader(int p_index);
	void _close_builtin_shaders_from_scene(const String &p_scene);
	void _file_removed(const String &p_removed_file);

	void _shader_created(Ref<Shader> p_shader);
	void _shader_include_created(Ref<ShaderInclude> p_shader_inc);
	void _update_shader_list_status();
	void _move_shader_tab(int p_from, int p_to);

	Variant get_drag_data_fw(const Point2 &p_point, Control *p_from);
	bool can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const;
	void drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from);

	void _window_changed(bool p_visible);

protected:
	void _notification(int p_what);

public:
	virtual String get_name() const override { return "Shader"; }
	virtual void edit(Object *p_object) override;
	virtual bool handles(Object *p_object) const override;
	virtual void make_visible(bool p_visible) override;
	virtual void selected_notify() override;

	TextShaderEditor *get_shader_editor(const Ref<Shader> &p_for_shader);
	VisualShaderEditor *get_visual_shader_editor(const Ref<Shader> &p_for_shader);

	virtual void set_window_layout(Ref<ConfigFile> p_layout) override;
	virtual void get_window_layout(Ref<ConfigFile> p_layout) override;

	virtual String get_unsaved_status(const String &p_for_scene) const override;
	virtual void save_external_data() override;
	virtual void apply_changes() override;

	ShaderEditorPlugin();
	~ShaderEditorPlugin();
};

#endif // SHADER_EDITOR_PLUGIN_H
