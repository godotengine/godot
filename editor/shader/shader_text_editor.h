/**************************************************************************/
/*  shader_text_editor.h                                                  */
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

#include "editor/script/script_editor_base.h"
#include "servers/rendering/shader_warnings.h"

class MaterialEditor;
class Environment;
class ShaderMaterial;
class Timer;
class Shader;
class ShaderInclude;

class TextShaderPreview : public VBoxContainer {
	GDCLASS(TextShaderPreview, VBoxContainer);

private:
	Label *error_label = nullptr;
	Button *goto_button = nullptr;
	Button *delete_button = nullptr;
	MarginContainer *surface_container = nullptr;
	MaterialEditor *surface = nullptr;
	Ref<ShaderMaterial> shader_material;
	Ref<Environment> env;

	int line = -1;
	bool in_comment = false;

	static HashMap<String, String> spatial_assignments;
	static HashMap<String, String> canvas_assignments;
	static HashMap<String, String> builtin_spatial_types;
	static HashMap<String, String> builtin_canvas_types;

	String _get_enclosing_function(const PackedStringArray &p_lines, int p_line) const;
	bool _find_statement(const PackedStringArray &p_lines, int p_line, String &r_var_name, int &r_start, int &r_end) const;
	String _find_var_type(const PackedStringArray &p_lines, const String &p_var_name, int p_line, bool p_mode_3d);
	bool _match_uniforms(const Ref<ShaderMaterial> &p_source, const Ref<ShaderMaterial> &p_target) const;
	void _sync_shader_parameters(const Ref<ShaderMaterial> &p_source, Ref<ShaderMaterial> &p_target);
	void _reset_shader_parameters(Ref<ShaderMaterial> &p_target);
	void _show_error(const String &p_error);
	void _goto_pressed();
	void _delete_pressed();
	Ref<ShaderMaterial> _get_source_material() const;

protected:
	static void _bind_methods();
	void _notification(int p_what);

public:
	void set_shader_code(const String &p_code, int p_line, bool p_in_comment);
	void show_shader_compile_error();
	void recompile(const String &p_code);
	void sync_shader_parameters();
	MarginContainer *get_surface_container() const;

	TextShaderPreview();
};

class TextShaderPreviewLineLayer : public Control {
	GDCLASS(TextShaderPreviewLineLayer, Control);

private:
	Color line_color;
	HashMap<int, TextShaderPreview *> *previews = nullptr;
	ScrollContainer *scroll_container = nullptr;
	CodeEdit *code_editor = nullptr;

protected:
	void _notification(int p_what);

public:
	void set_previews(HashMap<int, TextShaderPreview *> &p_previews);
	void set_code_editor(CodeEdit *p_code_editor);
	void set_scroll_container(ScrollContainer *p_scroll_container);

	TextShaderPreviewLineLayer();
};

class ShaderTextEditor : public CodeEditorBase {
	GDCLASS(ShaderTextEditor, CodeEditorBase);

	static ScriptEditorBase *create_editor(const Ref<Resource> &p_resource);

	HashMap<int, TextShaderPreview *> previews;
	TextShaderPreviewLineLayer *preview_line_layer = nullptr;
	VBoxContainer *preview_box = nullptr;
	VBoxContainer *preview_box_child = nullptr;
	Timer *preview_timer = nullptr;
	ScrollContainer *preview_sbox = nullptr;

	bool pending_update_shader_previews = false;

	Error last_compile_result = Error::OK;
	bool compilation_success = true;

	void _check_shader_mode();

	virtual bool _edit_option(int p_option) override;

	void _update_warning_panel();
	void _show_warnings_panel(bool p_show);
	void _update_warnings(bool p_validate);

	bool dependencies_changed = true;

	Color marked_line_color = Color(1, 1, 1);

	struct WarningsComparator {
		_ALWAYS_INLINE_ bool operator()(const ShaderWarning &p_a, const ShaderWarning &p_b) const { return (p_a.get_line() < p_b.get_line()); }
	};

	List<ShaderWarning> warnings;

	bool block_shader_changed = false;
	void _shader_changed();

protected:
	enum {
		PREVIEW_TOGGLE = CODE_ENUM_COUNT,
		PREVIEW_REMOVE_ALL,
		PREVIEW_GOTO_NEXT,
		PREVIEW_GOTO_PREV,
	};

	class EditMenusShTE : public EditMenusCEB {
		GDCLASS(EditMenusShTE, EditMenusCEB);

	protected:
		PopupMenu *previews_menu = nullptr;

		void _shader_preview_item_pressed(int p_idx);
		void _update_shader_preview_list();

	public:
		virtual bool handles(ScriptEditorBase *p_seb) override { return Object::cast_to<ShaderTextEditor>(p_seb); }

		EditMenusShTE(ScriptEditor *p_se);
	};

	static void _bind_methods();

	void _notification(int p_what);

	virtual void _validate_script() override;

	virtual void _load_theme_settings() override;

	TextShaderPreviewLineLayer *get_preview_line_layer() const;
	TextShaderPreview *get_preview(int p_line) const;
	virtual void _code_complete_script(const String &p_code, List<ScriptLanguage::CodeCompletionOption> *r_options, bool &r_force) override;

	void _on_shader_preview_toggled(int p_line);
	void _update_shader_previews();

public:
	virtual String get_doc_url_path() override { return "/tutorials/shaders/shader_reference/index.html"; }

	virtual bool show_members_overview() override { return false; }

	virtual void set_edited_resource(const Ref<Resource> &p_res) override;
	virtual void set_edited_resource(const Ref<Resource> &p_res, const String &p_code);

	virtual void apply_code() override;

	virtual EditMenusBase *create_edit_menu(ScriptEditor *p_se) override { return memnew(EditMenusShTE(p_se)); }

	void focus_preview_line(int p_line);

	static void register_editor();

	void toggle_shader_preview(int p_line);
	void remove_shader_preview(int p_line);
	void goto_shader_preview(int p_line);
	void set_preview_box(Control *p_box);
	void clear_previews();
	void redraw_preview_lines();
	void recompile_previews();
	void update_parameters();

	ShaderTextEditor();
};
