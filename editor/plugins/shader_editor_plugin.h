/*************************************************************************/
/*  shader_editor_plugin.h                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#ifndef SHADER_EDITOR_PLUGIN_H
#define SHADER_EDITOR_PLUGIN_H

#include "editor/code_editor.h"
#include "editor/editor_plugin.h"
#include "editor/editor_properties.h"
#include "editor/plugins/node_3d_editor_plugin.h"
#include "scene/3d/camera_3d.h"
#include "scene/3d/light_3d.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/subviewport_container.h"
#include "scene/gui/tab_container.h"
#include "scene/gui/text_edit.h"
#include "scene/main/timer.h"
#include "scene/resources/primitive_meshes.h"
#include "scene/resources/shader.h"
#include "servers/rendering/shader_language.h"

class ShaderTextEditor : public CodeTextEditor {
	GDCLASS(ShaderTextEditor, CodeTextEditor);

	Ref<CodeHighlighter> syntax_highlighter;
	Ref<Shader> shader;

	void _check_shader_mode();

protected:
	static void _bind_methods();
	virtual void _load_theme_settings() override;

	virtual void _code_complete_script(const String &p_code, List<ScriptCodeCompletionOption> *r_options) override;
	virtual bool _is_shader_editor() const override { return true; }

public:
	virtual void _validate_script() override;

	void reload_text();

	Ref<Shader> get_edited_shader() const;
	void set_edited_shader(const Ref<Shader> &p_shader);
	ShaderTextEditor();
};

class ShaderEditor : public PanelContainer {
	GDCLASS(ShaderEditor, PanelContainer);

	enum {
		FILE_CLOSE,
		FILE_CLOSE_ALL,
		FILE_CLOSE_OTHER_TABS,
		EDIT_UNDO,
		EDIT_REDO,
		EDIT_CUT,
		EDIT_COPY,
		EDIT_PASTE,
		EDIT_SELECT_ALL,
		EDIT_MOVE_LINE_UP,
		EDIT_MOVE_LINE_DOWN,
		EDIT_INDENT_LEFT,
		EDIT_INDENT_RIGHT,
		EDIT_DELETE_LINE,
		EDIT_CLONE_DOWN,
		EDIT_TOGGLE_COMMENT,
		EDIT_COMPLETE,
		SEARCH_FIND,
		SEARCH_FIND_NEXT,
		SEARCH_FIND_PREV,
		SEARCH_REPLACE,
		SEARCH_GOTO_LINE,
		BOOKMARK_TOGGLE,
		BOOKMARK_GOTO_NEXT,
		BOOKMARK_GOTO_PREV,
		BOOKMARK_REMOVE_ALL,
		HELP_DOCS,
		TOGGLE_SHADERS_PANEL,
	};

	MenuButton *file_menu;
	MenuButton *edit_menu;
	MenuButton *search_menu;
	PopupMenu *bookmarks_menu;
	MenuButton *help_menu;
	PopupMenu *context_menu;
	uint64_t idle;

	GotoLineDialog *goto_line_dialog;
	ConfirmationDialog *erase_tab_confirm;
	ConfirmationDialog *disk_changed;

	// Shaders list
	HSplitContainer *main_splitbox;
	HBoxContainer *main_hbox;
	HSplitContainer *left_splitbox;
	VBoxContainer *left_vbox;
	VBoxContainer *right_vbox;
	VBoxContainer *shaders_vbox;
	LineEdit *filter_shaders;
	ItemList *shader_list;
	ItemList *shader_filtered_list;

	ShaderTextEditor *shader_editor;

	// Scene Viewport
	Node3DEditorViewportContainer *scene_viewport_base = nullptr;
	Node *scene_viewport_parent = nullptr;
	int scene_viewport_pos = -1;

	// Local 3D-Viewport
	SubViewportContainer *local_viewport_base;
	SubViewport *local_viewport;
	Camera3D *camera;
	DirectionalLight3D *light1;
	DirectionalLight3D *light2;
	Ref<ShaderMaterial> material;

	Map<String, Ref<Shader>> recent_shaders_map;

	Ref<QuadMesh> quad_mesh;
	Ref<SphereMesh> sphere_mesh;
	Ref<BoxMesh> box_mesh;
	Ref<CylinderMesh> cylinder_mesh;

	MeshInstance3D *quad_mesh_instance;
	MeshInstance3D *sphere_mesh_instance;
	MeshInstance3D *box_mesh_instance;
	MeshInstance3D *cylinder_mesh_instance;

	HBoxContainer *spatial_hbox;
	Button *scene_button;
	Button *quad_button;
	Button *sphere_button;
	Button *box_button;
	Button *cylinder_button;

	enum PreviewNode3DType {
		NODE3D_SCENE,
		NODE3D_QUAD,
		NODE3D_SPHERE,
		NODE3D_BOX,
		NODE3D_CYLINDER,
	};
	int last_option = 0;
	Map<int, MeshInstance3D *> mesh_instances;
	void _preview_3d_option(int p_option);
	void _preview_viewport_option(int p_viewport_idx);

	void _menu_option(int p_option);
	void _params_changed();
	mutable Ref<Shader> shader;

	void _apply_to_preview();
	void _update_filtered_shader_list();
	void _filter_shaders_text_changed(const String &p_newtext);
	void _editor_settings_changed();

	void _check_for_external_edit();
	void _reload_shader_from_disk();
	void _shader_list_item_selected(int p_which, bool p_filtered);

	static ShaderEditor *shader_editor_singleton;

protected:
	void _notification(int p_what);
	static void _bind_methods();
	void _make_context_menu(bool p_selection, Vector2 p_position);
	void _text_edit_gui_input(const Ref<InputEvent> &ev);

	void _update_bookmark_list();
	void _bookmark_item_pressed(int p_idx);

public:
	static ShaderEditor *get_singleton() { return shader_editor_singleton; }

	bool toggle_shaders_panel();
	bool is_shaders_panel_toggled() const;

	void attach_scene_viewport();
	void detach_scene_viewport();
	void apply_shaders();

	void ensure_select_current();
	void edit(const Ref<Shader> &p_shader);

	void goto_line_selection(int p_line, int p_begin, int p_end);

	virtual Size2 get_minimum_size() const override { return Size2(0, 200); }
	void save_external_data(const String &p_str = "");

	ShaderEditor(EditorNode *p_node);
};

class ShaderEditorPlugin : public EditorPlugin {
	GDCLASS(ShaderEditorPlugin, EditorPlugin);

	bool _2d = false;
	ShaderEditor *shader_editor;
	EditorNode *editor;

public:
	virtual String get_name() const override { return "Shader"; }
	bool has_main_screen() const override { return true; }
	virtual void edit(Object *p_object) override;
	virtual bool handles(Object *p_object) const override;
	virtual void make_visible(bool p_visible) override;
	virtual void selected_notify() override;

	ShaderEditor *get_shader_editor() const { return shader_editor; }

	virtual void save_external_data() override;
	virtual void apply_changes() override;

	ShaderEditorPlugin(EditorNode *p_node);
	~ShaderEditorPlugin();
};

#endif
