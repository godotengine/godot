/*************************************************************************/
/*  shader_editor_plugin.h                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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
#include "scene/gui/margin_container.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/tab_container.h"
#include "scene/gui/text_edit.h"
#include "scene/main/timer.h"
#include "scene/resources/shader.h"
#include "servers/visual/shader_language.h"

class ShaderTextEditor : public CodeTextEditor {
	GDCLASS(ShaderTextEditor, CodeTextEditor);

	Ref<Shader> shader;

	void _check_shader_mode();

protected:
	static void _bind_methods();
	virtual void _load_theme_settings();

	virtual void _code_complete_script(const String &p_code, List<ScriptCodeCompletionOption> *r_options);

public:
	virtual void _validate_script();

	void reload_text();

	Ref<Shader> get_edited_shader() const;
	void set_edited_shader(const Ref<Shader> &p_shader);
	ShaderTextEditor();
};

class ShaderEditor : public MarginContainer {
	GDCLASS(ShaderEditor, MarginContainer);

	enum {

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
		EDIT_DUPLICATE_SELECTION,
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
	};

	MenuButton *edit_menu;
	MenuButton *search_menu;
	PopupMenu *bookmarks_menu;
	MenuButton *help_menu;
	PopupMenu *context_menu;
	uint64_t idle;

	GotoLineDialog *goto_line_dialog;
	ConfirmationDialog *erase_tab_confirm;
	ConfirmationDialog *disk_changed;

	ShaderTextEditor *shader_editor;

	void _menu_option(int p_option);
	void _params_changed();
	mutable Ref<Shader> shader;

	void _editor_settings_changed();

	void _check_for_external_edit();
	void _reload_shader_from_disk();

protected:
	void _notification(int p_what);
	static void _bind_methods();
	void _make_context_menu(bool p_selection, Vector2 p_position);
	void _text_edit_gui_input(const Ref<InputEvent> &ev);

	void _update_bookmark_list();
	void _bookmark_item_pressed(int p_idx);

public:
	void apply_shaders();

	void ensure_select_current();
	void edit(const Ref<Shader> &p_shader);

	void goto_line_selection(int p_line, int p_begin, int p_end);

	virtual Size2 get_minimum_size() const { return Size2(0, 200); }
	void save_external_data(const String &p_str = "");

	ShaderEditor(EditorNode *p_node);
};

class ShaderEditorPlugin : public EditorPlugin {
	GDCLASS(ShaderEditorPlugin, EditorPlugin);

	bool _2d;
	ShaderEditor *shader_editor;
	EditorNode *editor;
	Button *button;

public:
	virtual String get_name() const { return "Shader"; }
	bool has_main_screen() const { return false; }
	virtual void edit(Object *p_object);
	virtual bool handles(Object *p_object) const;
	virtual void make_visible(bool p_visible);
	virtual void selected_notify();

	ShaderEditor *get_shader_editor() const { return shader_editor; }

	virtual void save_external_data();
	virtual void apply_changes();

	ShaderEditorPlugin(EditorNode *p_node);
	~ShaderEditorPlugin();
};

#endif // SHADER_EDITOR_PLUGIN_H
