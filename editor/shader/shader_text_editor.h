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

class Shader;
class ShaderInclude;

class ShaderTextEditor : public CodeEditorBase {
	GDCLASS(ShaderTextEditor, CodeEditorBase);

	static ScriptEditorBase *create_editor(const Ref<Resource> &p_resource);

	enum {
		HELP_DOCS = CODE_ENUM_COUNT,
	};
	Error last_compile_result = Error::OK;

	bool compilation_success = true;

	void _check_shader_mode();

	virtual bool _edit_option(int p_option) override;

	void _editor_settings_changed();
	void _project_settings_changed();

	void _update_warning_panel();
	void _show_warnings_panel(bool p_show);
	void _update_warnings(bool p_validate);

	void _script_validated(bool p_valid) {
		compilation_success = p_valid;
		emit_signal(SNAME("validation_changed"));
	}

	uint32_t dependencies_version = 0; // Incremented if deps changed

	Color marked_line_color = Color(1, 1, 1);

	struct WarningsComparator {
		_ALWAYS_INLINE_ bool operator()(const ShaderWarning &p_a, const ShaderWarning &p_b) const { return (p_a.get_line() < p_b.get_line()); }
	};

	List<ShaderWarning> warnings;

	bool block_shader_changed = false;
	void _shader_changed();

protected:
	class EditMenusShTE : public EditMenusCEB {
		GDCLASS(EditMenusShTE, EditMenusCEB);

	public:
		virtual bool handles(ScriptEditorBase *p_seb) override { return Object::cast_to<ShaderTextEditor>(p_seb); }
	};

	static void _bind_methods();

	void _notification(int p_what);

	virtual void _validate_script() override;

	virtual void _load_theme_settings() override;

	virtual void _code_complete_script(const String &p_code, List<ScriptLanguage::CodeCompletionOption> *r_options, bool &r_force) override;

public:
	virtual bool show_members_overview() override { return false; }

	virtual void set_edited_resource(const Ref<Resource> &p_res) override;
	virtual void set_edited_resource(const Ref<Resource> &p_res, const String &p_code);

	virtual void apply_code() override;
	virtual bool is_unsaved() override;

	bool was_compilation_successful() const { return compilation_success; }
	void ensure_select_current() {}

	virtual Size2 get_minimum_size() const override { return Size2(0, 200); }

	virtual EditMenusBase *create_edit_menu() override { return memnew(EditMenusShTE); }

	static void register_editor();
	uint32_t get_dependencies_version() const { return dependencies_version; }

	virtual void reload_text() override;

	Ref<Shader> get_edited_shader() const;
	Ref<ShaderInclude> get_edited_shader_include() const;

	void set_edited_code(const String &p_code);

	ShaderTextEditor();
};
