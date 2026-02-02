/**************************************************************************/
/*  script_text_editor.h                                                  */
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
#include "script_editor_plugin.h"

#include "editor/gui/code_editor.h"
#include "scene/gui/color_picker.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/option_button.h"
#include "scene/gui/tree.h"

class RichTextLabel;

class ConnectionInfoDialog : public AcceptDialog {
	GDCLASS(ConnectionInfoDialog, AcceptDialog);

	Label *method = nullptr;
	Tree *tree = nullptr;

	virtual void ok_pressed() override;

public:
	void popup_connections(const String &p_method, const Vector<Node *> &p_nodes);

	ConnectionInfoDialog();
};

class ScriptTextEditor : public CodeEditorBase {
	GDCLASS(ScriptTextEditor, CodeEditorBase);

	Variant pending_state;
	bool script_is_valid = false;

	RichTextLabel *errors_panel = nullptr;

	Vector<String> functions;
	List<ScriptLanguage::Warning> warnings;
	List<ScriptLanguage::ScriptError> errors;
	HashMap<String, List<ScriptLanguage::ScriptError>> depended_errors;
	HashSet<int> safe_lines;

	List<Connection> missing_connections;

	int inline_color_line = -1;
	int inline_color_start = -1;
	int inline_color_end = -1;
	PopupPanel *inline_color_popup = nullptr;
	ColorPicker *inline_color_picker = nullptr;
	OptionButton *inline_color_options = nullptr;
	Ref<Texture2D> color_alpha_texture;

	ScriptEditorQuickOpen *quick_open = nullptr;
	ConnectionInfoDialog *connection_info_dialog = nullptr;

	int connection_gutter = -1;
	void _gutter_clicked(int p_line, int p_gutter);
	void _update_gutter_indexes();

	int line_number_gutter = -1;
	Color default_line_number_color = Color(1, 1, 1);
	Color safe_line_number_color = Color(1, 1, 1);

	Color marked_line_color = Color(1, 1, 1);
	Color warning_line_color = Color(1, 1, 1);
	Color folded_code_region_color = Color(1, 1, 1);
	int previous_line = 0;

	PopupPanel *color_panel = nullptr;
	ColorPicker *color_picker = nullptr;
	Vector3i color_position;
	String color_args;

	bool theme_loaded = false;

	enum {
		EDIT_AUTO_INDENT = CODE_ENUM_COUNT,
		EDIT_PICK_COLOR,
		EDIT_EVALUATE,
		EDIT_CREATE_CODE_REGION,

		SEARCH_LOCATE_FUNCTION,

		DEBUG_TOGGLE_BREAKPOINT,
		DEBUG_REMOVE_ALL_BREAKPOINTS,
		DEBUG_GOTO_NEXT_BREAKPOINT,
		DEBUG_GOTO_PREV_BREAKPOINT,

		SHOW_TOOLTIP_AT_CARET,
		HELP_CONTEXTUAL,
		LOOKUP_SYMBOL,
	};

	enum COLOR_MODE {
		MODE_RGB,
		MODE_STRING,
		MODE_HSV,
		MODE_OKHSL,
		MODE_RGB8,
		MODE_HEX,
		MODE_MAX
	};

	class EditMenusSTE : public EditMenusCEB {
		GDCLASS(EditMenusSTE, EditMenusCEB);
		PopupMenu *breakpoints_menu = nullptr;

		void _update_breakpoint_list();
		void _breakpoint_item_pressed(int p_idx);

	public:
		EditMenusSTE();
	};

	void _enable_code_editor();

	struct DraggedExport {
		ObjectID obj_id;
		String variable_name;
		Variant value;
		String class_name;
	};

	LocalVector<DraggedExport> pending_dragged_exports;
	Vector<ObjectID> _get_objects_for_export_assignment() const;
	String _get_dropped_resource_as_exported_member(const Ref<Resource> &p_resource, const Vector<ObjectID> &p_script_instance_obj_ids);
	void _assign_dragged_export_variables();

	static ScriptEditorBase *create_editor(const Ref<Resource> &p_resource);

protected:
	void _breakpoint_toggled(int p_row);

	void _on_caret_moved();

	void _update_warnings();
	void _update_errors();

	static void _code_complete_scripts(void *p_ud, const String &p_code, List<ScriptLanguage::CodeCompletionOption> *r_options, bool &r_force);
	virtual void _code_complete_script(const String &p_code, List<ScriptLanguage::CodeCompletionOption> *r_options, bool &r_force) override;

	void _set_theme_for_script();
	void _show_errors_panel(bool p_show);
	void _show_warnings_panel(bool p_show);
	void _error_clicked(const Variant &p_line);
	virtual bool _warning_clicked(const Variant &p_line) override;

	bool _is_valid_color_info(const Dictionary &p_info);
	Array _inline_object_parse(const String &p_text);
	void _inline_object_draw(const Dictionary &p_info, const Rect2 &p_rect);
	void _inline_object_handle_click(const Dictionary &p_info, const Rect2 &p_rect);
	String _picker_color_stringify(const Color &p_color, COLOR_MODE p_mode);
	void _picker_color_changed(const Color &p_color);
	void _update_color_constructor_options();
	void _update_background_color();
	void _update_color_text();

	void _notification(int p_what);

	void _edit_option_toggle_inline_comment();
	void _color_changed(const Color &p_color);

	void _lookup_symbol(const String &p_symbol, int p_row, int p_column);
	void _validate_symbol(const String &p_symbol);

	void _show_symbol_tooltip(const String &p_symbol, int p_row, int p_column, bool p_shortcut = false);

	Variant get_drag_data_fw(const Point2 &p_point, Control *p_from);
	bool can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const;
	void drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from);

	String _get_absolute_path(const String &rel_path);

	void _goto_line(int p_line) { goto_line(p_line); }

	void _make_context_menu(bool p_selection, bool p_color, bool p_foldable, bool p_open_docs, bool p_goto_definition, const Vector2 &p_pos);

	virtual void _text_edit_gui_input(const Ref<InputEvent> &p_ev) override;
	virtual bool _edit_option(int p_op) override;

	virtual void _load_theme_settings() override;
	virtual void _validate_script() override;

public:
	void _update_connected_methods();

	virtual void apply_code() override;
	virtual void set_edited_resource(const Ref<Resource> &p_res) override;
	virtual void enable_editor() override;
	virtual Vector<String> get_functions() override;

	virtual Control *get_edit_menu() override;

	virtual Ref<Texture2D> get_theme_icon() override;

	virtual Variant get_edit_state() override;
	virtual void set_edit_state(const Variant &p_state) override;

	virtual PackedInt32Array get_breakpoints() override;
	virtual void set_breakpoint(int p_line, bool p_enabled) override;
	virtual void clear_breakpoints() override;

	virtual void add_callback(const String &p_function, const PackedStringArray &p_args);
	virtual void update_settings() override;

	static void register_editor();

	Variant get_previous_state();
	void store_previous_state();

	ScriptTextEditor();
	~ScriptTextEditor();
};
