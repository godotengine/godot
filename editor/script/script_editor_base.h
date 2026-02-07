/**************************************************************************/
/*  script_editor_base.h                                                  */
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

#include "editor/gui/code_editor.h"

class EditorSyntaxHighlighter;
class MenuButton;
class VSplitContainer;

class ScriptEditorBase : public Control {
	GDCLASS(ScriptEditorBase, Control);

protected:
	Ref<Resource> edited_res;

	static void _bind_methods();

public:
	struct EditedFileData {
		String path;
		uint64_t last_modified_time = -1;
	} edited_file_data;

	virtual String get_name();
	virtual Ref<Texture2D> get_theme_icon();

	virtual void set_toggle_list_control(Control *p_toggle_list_control) = 0;
	virtual void update_toggle_files_button() = 0;

	virtual bool show_members_overview() { return false; }

	virtual void set_edited_resource(const Ref<Resource> &p_res) = 0;
	virtual Ref<Resource> get_edited_resource() const { return edited_res; }

	virtual void apply_code() = 0;
	virtual void validate_script() = 0;
	virtual bool is_unsaved() = 0;
	virtual void tag_saved_version();

	virtual void add_syntax_highlighter(Ref<EditorSyntaxHighlighter> p_highlighter) {}
	virtual Control *get_base_editor() const { return nullptr; }
};

typedef ScriptEditorBase *(*CreateScriptEditorFunc)(const Ref<Resource> &p_resource);

class TextEditorBase : public ScriptEditorBase {
	GDCLASS(TextEditorBase, ScriptEditorBase);

	void _post_init();

protected:
	enum {
		EDIT_UNDO,
		EDIT_REDO,
		EDIT_CUT,
		EDIT_COPY,
		EDIT_PASTE,
		EDIT_SELECT_ALL,
		EDIT_TRIM_TRAILING_WHITESAPCE,
		EDIT_TRIM_FINAL_NEWLINES,
		EDIT_CONVERT_INDENT_TO_SPACES,
		EDIT_CONVERT_INDENT_TO_TABS,
		EDIT_MOVE_LINE_UP,
		EDIT_MOVE_LINE_DOWN,
		EDIT_INDENT,
		EDIT_UNINDENT,
		EDIT_DELETE_LINE,
		EDIT_DUPLICATE_SELECTION,
		EDIT_DUPLICATE_LINES,
		EDIT_TO_UPPERCASE,
		EDIT_TO_LOWERCASE,
		EDIT_CAPITALIZE,
		EDIT_TOGGLE_FOLD_LINE,
		EDIT_FOLD_ALL_LINES,
		EDIT_TOGGLE_WORD_WRAP,
		EDIT_UNFOLD_ALL_LINES,
		EDIT_EMOJI_AND_SYMBOL,

		SEARCH_FIND,
		SEARCH_FIND_NEXT,
		SEARCH_FIND_PREV,
		SEARCH_REPLACE,
		SEARCH_IN_FILES,
		SEARCH_GOTO_LINE,

		REPLACE_IN_FILES,

		BOOKMARK_TOGGLE,
		BOOKMARK_GOTO_NEXT,
		BOOKMARK_GOTO_PREV,
		BOOKMARK_REMOVE_ALL,

		BASE_ENUM_COUNT,
	};

	class EditMenus : public HBoxContainer {
		GDCLASS(EditMenus, HBoxContainer);

	protected:
		MenuButton *edit_menu = nullptr;
		MenuButton *search_menu = nullptr;
		MenuButton *goto_menu = nullptr;
		PopupMenu *bookmarks_menu = nullptr;
		PopupMenu *highlighter_menu = nullptr;

		PopupMenu *edit_menu_line = nullptr;
		PopupMenu *edit_menu_fold = nullptr;
		PopupMenu *edit_menu_convert_indent = nullptr;
		PopupMenu *edit_menu_convert = nullptr;

		TextEditorBase *_get_active_editor();
		void _edit_option(int p_op);
		void _prepare_edit_menu();
		void _update_highlighter_menu();
		void _change_syntax_highlighter(int p_idx);
		void _update_bookmark_list();
		void _bookmark_item_pressed(int p_idx);

	public:
		EditMenus();
	};

	static void _popup_move_item(int p_target_id, PopupMenu *r_popup, bool p_move_after = true, int p_idx = -1) {
		int target_idx = r_popup->get_item_index(p_target_id) + p_move_after;
		if (target_idx >= 0 && target_idx < r_popup->get_item_count()) {
			r_popup->set_item_index(p_idx, target_idx);
		}
	}

	static inline EditMenus *edit_menus = nullptr;

	bool editor_enabled = false;
	CodeTextEditor *code_editor = nullptr;
	HBoxContainer *edit_hb = nullptr;

	GotoLinePopup *goto_line_popup = nullptr;

	LocalVector<Ref<EditorSyntaxHighlighter>> highlighters;

	PopupMenu *context_menu = nullptr;
	MenuButton *search_menu = nullptr;

	void _make_context_menu(bool p_selection, bool p_foldable, const Vector2 &p_position = Vector2(0, 0), bool p_show = true);
	void _show_context_menu(const Vector2 &p_position);

	virtual void _text_edit_gui_input(const Ref<InputEvent> &p_ev);
	virtual bool _edit_option(int p_op);

	virtual void _load_theme_settings();
	virtual void _validate_script();

	void _convert_case(CodeTextEditor::CaseStyle p_case) { code_editor->convert_case(p_case); }

public:
	virtual void add_syntax_highlighter(Ref<EditorSyntaxHighlighter> p_highlighter) override;
	virtual void set_syntax_highlighter(Ref<EditorSyntaxHighlighter> p_highlighter);

	virtual void set_edited_resource(const Ref<Resource> &p_res) override;

	virtual bool is_unsaved() override;
	virtual void tag_saved_version() override;

	virtual void reload_text();
	virtual void enable_editor();

	virtual Control *get_edit_menu() = 0;

	virtual Control *get_base_editor() const override { return code_editor->get_text_editor(); }
	virtual CodeTextEditor *get_code_editor() const { return code_editor; }

	virtual void set_tooltip_request_func(const Callable &p_toolip_callback);

	virtual void ensure_focus() { code_editor->get_text_editor()->grab_focus(); }
	virtual void convert_indent() { code_editor->get_text_editor()->convert_indent(); }

	virtual void trim_trailing_whitespace() { code_editor->trim_trailing_whitespace(); }
	virtual void trim_final_newlines() { code_editor->trim_final_newlines(); }
	virtual void insert_final_newline() { code_editor->insert_final_newline(); }

	virtual void goto_line(int p_line, int p_column = 0) { code_editor->goto_line(p_line, p_column); }
	virtual void goto_line_selection(int p_line, int p_begin, int p_end) { code_editor->goto_line_selection(p_line, p_begin, p_end); }
	virtual void goto_line_centered(int p_line, int p_column = 0) { code_editor->goto_line_centered(p_line, p_column); }
	virtual void set_executing_line(int p_line) { code_editor->set_executing_line(p_line); }
	virtual void clear_executing_line() { code_editor->clear_executing_line(); }

	virtual Variant get_edit_state() { return code_editor->get_edit_state(); }
	virtual void set_edit_state(const Variant &p_state);
	virtual Variant get_navigation_state() { return code_editor->get_navigation_state(); }

	virtual void update_settings() { code_editor->update_editor_settings(); }
	virtual void set_find_replace_bar(FindReplaceBar *p_bar) { code_editor->set_find_replace_bar(p_bar); }

	virtual void validate_script() override { code_editor->validate_script(); }

	virtual void set_toggle_list_control(Control *p_toggle_list_control) override {
		code_editor->set_toggle_list_control(p_toggle_list_control);
	}
	virtual void update_toggle_files_button() override { code_editor->update_toggle_files_button(); }

	TextEditorBase();
	~TextEditorBase();
};

class CodeEditorBase : public TextEditorBase {
	GDCLASS(CodeEditorBase, TextEditorBase);

protected:
	enum {
		EDIT_COMPLETE = BASE_ENUM_COUNT,
		EDIT_TOGGLE_COMMENT,

		CODE_ENUM_COUNT,
	};

	class EditMenusCEB : public EditMenus {
		GDCLASS(EditMenusCEB, EditMenus);

	public:
		EditMenusCEB();
	};

	VSplitContainer *editor_box = nullptr;
	RichTextLabel *warnings_panel = nullptr;

	virtual void _code_complete_script(const String &p_code, List<ScriptLanguage::CodeCompletionOption> *r_options, bool &r_force) = 0;
	virtual bool _warning_clicked(const Variant &p_line);

public:
	virtual bool show_members_overview() override { return true; }

	virtual Vector<String> get_functions() { return Vector<String>(); }

	virtual PackedInt32Array get_breakpoints() { return PackedInt32Array(); }
	virtual void set_breakpoint(int p_line, bool p_enabled) {}
	virtual void clear_breakpoints() {}

	CodeEditorBase();
};
