/*************************************************************************/
/*  shader_editor_plugin.cpp                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "shader_editor_plugin.h"

#include "core/io/resource_loader.h"
#include "core/io/resource_saver.h"
#include "core/os/keyboard.h"
#include "core/os/os.h"
#include "editor/editor_node.h"
#include "editor/editor_settings.h"
#include "editor/property_editor.h"
#include "scene/resources/shader_graph.h"
#include "servers/visual/shader_types.h"

/*** SHADER SCRIPT EDITOR ****/

Ref<Shader> ShaderTextEditor::get_edited_shader() const {

	return shader;
}
void ShaderTextEditor::set_edited_shader(const Ref<Shader> &p_shader) {

	shader = p_shader;

	_load_theme_settings();

	get_text_edit()->set_text(p_shader->get_code());

	_line_col_changed();
}

void ShaderTextEditor::_load_theme_settings() {

	get_text_edit()->clear_colors();

	/* keyword color */

	get_text_edit()->add_color_override("background_color", EDITOR_DEF("text_editor/highlighting/background_color", Color(0, 0, 0, 0)));
	get_text_edit()->add_color_override("completion_background_color", EDITOR_DEF("text_editor/highlighting/completion_background_color", Color(0, 0, 0, 0)));
	get_text_edit()->add_color_override("completion_selected_color", EDITOR_DEF("text_editor/highlighting/completion_selected_color", Color::html("434244")));
	get_text_edit()->add_color_override("completion_existing_color", EDITOR_DEF("text_editor/highlighting/completion_existing_color", Color::html("21dfdfdf")));
	get_text_edit()->add_color_override("completion_scroll_color", EDITOR_DEF("text_editor/highlighting/completion_scroll_color", Color::html("ffffff")));
	get_text_edit()->add_color_override("completion_font_color", EDITOR_DEF("text_editor/highlighting/completion_font_color", Color::html("aaaaaa")));
	get_text_edit()->add_color_override("font_color", EDITOR_DEF("text_editor/highlighting/text_color", Color(0, 0, 0)));
	get_text_edit()->add_color_override("line_number_color", EDITOR_DEF("text_editor/highlighting/line_number_color", Color(0, 0, 0)));
	get_text_edit()->add_color_override("caret_color", EDITOR_DEF("text_editor/highlighting/caret_color", Color(0, 0, 0)));
	get_text_edit()->add_color_override("caret_background_color", EDITOR_DEF("text_editor/highlighting/caret_background_color", Color(0, 0, 0)));
	get_text_edit()->add_color_override("font_selected_color", EDITOR_DEF("text_editor/highlighting/text_selected_color", Color(1, 1, 1)));
	get_text_edit()->add_color_override("selection_color", EDITOR_DEF("text_editor/highlighting/selection_color", Color(0.2, 0.2, 1)));
	get_text_edit()->add_color_override("brace_mismatch_color", EDITOR_DEF("text_editor/highlighting/brace_mismatch_color", Color(1, 0.2, 0.2)));
	get_text_edit()->add_color_override("current_line_color", EDITOR_DEF("text_editor/highlighting/current_line_color", Color(0.3, 0.5, 0.8, 0.15)));
	get_text_edit()->add_color_override("word_highlighted_color", EDITOR_DEF("text_editor/highlighting/word_highlighted_color", Color(0.8, 0.9, 0.9, 0.15)));
	get_text_edit()->add_color_override("number_color", EDITOR_DEF("text_editor/highlighting/number_color", Color(0.9, 0.6, 0.0, 2)));
	get_text_edit()->add_color_override("function_color", EDITOR_DEF("text_editor/highlighting/function_color", Color(0.4, 0.6, 0.8)));
	get_text_edit()->add_color_override("member_variable_color", EDITOR_DEF("text_editor/highlighting/member_variable_color", Color(0.9, 0.3, 0.3)));
	get_text_edit()->add_color_override("mark_color", EDITOR_DEF("text_editor/highlighting/mark_color", Color(1.0, 0.4, 0.4, 0.4)));
	get_text_edit()->add_color_override("breakpoint_color", EDITOR_DEF("text_editor/highlighting/breakpoint_color", Color(0.8, 0.8, 0.4, 0.2)));
	get_text_edit()->add_color_override("search_result_color", EDITOR_DEF("text_editor/highlighting/search_result_color", Color(0.05, 0.25, 0.05, 1)));
	get_text_edit()->add_color_override("search_result_border_color", EDITOR_DEF("text_editor/highlighting/search_result_border_color", Color(0.1, 0.45, 0.1, 1)));
	get_text_edit()->add_color_override("symbol_color", EDITOR_DEF("text_editor/highlighting/symbol_color", Color::hex(0x005291ff)));

	Color keyword_color = EDITOR_DEF("text_editor/highlighting/keyword_color", Color(0.5, 0.0, 0.2));

	List<String> keywords;
	ShaderLanguage::get_keyword_list(&keywords);

	if (shader.is_valid()) {

		for (const Map<StringName, ShaderLanguage::FunctionInfo>::Element *E = ShaderTypes::get_singleton()->get_functions(VisualServer::ShaderMode(shader->get_mode())).front(); E; E = E->next()) {

			for (const Map<StringName, ShaderLanguage::DataType>::Element *F = E->get().built_ins.front(); F; F = F->next()) {
				keywords.push_back(F->key());
			}
		}

		for (const Set<String>::Element *E = ShaderTypes::get_singleton()->get_modes(VisualServer::ShaderMode(shader->get_mode())).front(); E; E = E->next()) {

			keywords.push_back(E->get());
		}
	}

	for (List<String>::Element *E = keywords.front(); E; E = E->next()) {

		get_text_edit()->add_keyword_color(E->get(), keyword_color);
	}

	//colorize core types
	//Color basetype_color= EDITOR_DEF("text_editor/base_type_color",Color(0.3,0.3,0.0));

	//colorize comments
	Color comment_color = EDITOR_DEF("text_editor/highlighting/comment_color", Color::hex(0x797e7eff));

	get_text_edit()->add_color_region("/*", "*/", comment_color, false);
	get_text_edit()->add_color_region("//", "", comment_color, false);

	/*//colorize strings
	Color string_color = EDITOR_DEF("text_editor/string_color",Color::hex(0x6b6f00ff));

	List<String> strings;
	shader->get_shader_mode()->get_string_delimiters(&strings);

	for (List<String>::Element *E=strings.front();E;E=E->next()) {

		String string = E->get();
		String beg = string.get_slice(" ",0);
		String end = string.get_slice_count(" ")>1?string.get_slice(" ",1):String();
		get_text_edit()->add_color_region(beg,end,string_color,end=="");
	}*/
}

void ShaderTextEditor::_check_shader_mode() {

	String type = ShaderLanguage::get_shader_type(get_text_edit()->get_text());

	print_line("type is: " + type);
	Shader::Mode mode;

	if (type == "canvas_item") {
		mode = Shader::MODE_CANVAS_ITEM;
	} else if (type == "particles") {
		mode = Shader::MODE_PARTICLES;
	} else {
		mode = Shader::MODE_SPATIAL;
	}

	if (shader->get_mode() != mode) {
		shader->set_code(get_text_edit()->get_text());
		_load_theme_settings();
	}
}

void ShaderTextEditor::_code_complete_script(const String &p_code, List<String> *r_options) {

	_check_shader_mode();

	ShaderLanguage sl;
	String calltip;

	Error err = sl.complete(p_code, ShaderTypes::get_singleton()->get_functions(VisualServer::ShaderMode(shader->get_mode())), ShaderTypes::get_singleton()->get_modes(VisualServer::ShaderMode(shader->get_mode())), ShaderTypes::get_singleton()->get_types(), r_options, calltip);
	if (err != OK)
		ERR_PRINT("Shaderlang complete failed");

	if (calltip != "") {
		get_text_edit()->set_code_hint(calltip);
	}
}

void ShaderTextEditor::_validate_script() {

	_check_shader_mode();

	String code = get_text_edit()->get_text();
	//List<StringName> params;
	//shader->get_param_list(&params);

	ShaderLanguage sl;

	Error err = sl.compile(code, ShaderTypes::get_singleton()->get_functions(VisualServer::ShaderMode(shader->get_mode())), ShaderTypes::get_singleton()->get_modes(VisualServer::ShaderMode(shader->get_mode())), ShaderTypes::get_singleton()->get_types());

	if (err != OK) {
		String error_text = "error(" + itos(sl.get_error_line()) + "): " + sl.get_error_text();
		set_error(error_text);
		get_text_edit()->set_line_as_marked(sl.get_error_line() - 1, true);

	} else {
		for (int i = 0; i < get_text_edit()->get_line_count(); i++)
			get_text_edit()->set_line_as_marked(i, false);
		set_error("");
	}

	emit_signal("script_changed");
}

void ShaderTextEditor::_bind_methods() {
}

ShaderTextEditor::ShaderTextEditor() {
}

/*** SCRIPT EDITOR ******/

void ShaderEditor::_menu_option(int p_option) {

	ShaderTextEditor *current = shader_editor;

	switch (p_option) {
		case EDIT_UNDO: {

			current->get_text_edit()->undo();
		} break;
		case EDIT_REDO: {
			current->get_text_edit()->redo();

		} break;
		case EDIT_CUT: {

			current->get_text_edit()->cut();
		} break;
		case EDIT_COPY: {
			current->get_text_edit()->copy();

		} break;
		case EDIT_PASTE: {
			current->get_text_edit()->paste();

		} break;
		case EDIT_SELECT_ALL: {

			current->get_text_edit()->select_all();

		} break;
		case SEARCH_FIND: {

			current->get_find_replace_bar()->popup_search();
		} break;
		case SEARCH_FIND_NEXT: {

			current->get_find_replace_bar()->search_next();
		} break;
		case SEARCH_FIND_PREV: {

			current->get_find_replace_bar()->search_prev();
		} break;
		case SEARCH_REPLACE: {

			current->get_find_replace_bar()->popup_replace();
		} break;
		case SEARCH_GOTO_LINE: {

			goto_line_dialog->popup_find_line(current->get_text_edit());
		} break;
	}
}

void ShaderEditor::_notification(int p_what) {

	if (p_what == NOTIFICATION_ENTER_TREE) {
	}
	if (p_what == NOTIFICATION_DRAW) {

		RID ci = get_canvas_item();
		Ref<StyleBox> style = get_stylebox("panel", "Panel");
		style->draw(ci, Rect2(Point2(), get_size()));
	}
}

void ShaderEditor::_params_changed() {

	shader_editor->_validate_script();
}

void ShaderEditor::_editor_settings_changed() {

	shader_editor->get_text_edit()->set_auto_brace_completion(EditorSettings::get_singleton()->get("text_editor/completion/auto_brace_complete"));
	shader_editor->get_text_edit()->set_scroll_pass_end_of_file(EditorSettings::get_singleton()->get("text_editor/cursor/scroll_past_end_of_file"));
	shader_editor->get_text_edit()->set_indent_size(EditorSettings::get_singleton()->get("text_editor/indent/size"));
	shader_editor->get_text_edit()->set_indent_using_spaces(EditorSettings::get_singleton()->get("text_editor/indent/type"));
	shader_editor->get_text_edit()->set_auto_indent(EditorSettings::get_singleton()->get("text_editor/indent/auto_indent"));
	shader_editor->get_text_edit()->set_draw_tabs(EditorSettings::get_singleton()->get("text_editor/indent/draw_tabs"));
	shader_editor->get_text_edit()->set_show_line_numbers(EditorSettings::get_singleton()->get("text_editor/line_numbers/show_line_numbers"));
	shader_editor->get_text_edit()->set_syntax_coloring(EditorSettings::get_singleton()->get("text_editor/highlighting/syntax_highlighting"));
	shader_editor->get_text_edit()->set_highlight_all_occurrences(EditorSettings::get_singleton()->get("text_editor/highlighting/highlight_all_occurrences"));
	shader_editor->get_text_edit()->cursor_set_blink_enabled(EditorSettings::get_singleton()->get("text_editor/cursor/caret_blink"));
	shader_editor->get_text_edit()->cursor_set_blink_speed(EditorSettings::get_singleton()->get("text_editor/cursor/caret_blink_speed"));
	shader_editor->get_text_edit()->add_constant_override("line_spacing", EditorSettings::get_singleton()->get("text_editor/theme/line_spacing"));
	shader_editor->get_text_edit()->cursor_set_block_mode(EditorSettings::get_singleton()->get("text_editor/cursor/block_caret"));
	shader_editor->get_text_edit()->set_smooth_scroll_enabled(EditorSettings::get_singleton()->get("text_editor/open_scripts/smooth_scrolling"));
	shader_editor->get_text_edit()->set_v_scroll_speed(EditorSettings::get_singleton()->get("text_editor/open_scripts/v_scroll_speed"));
}

void ShaderEditor::_bind_methods() {

	ClassDB::bind_method("_editor_settings_changed", &ShaderEditor::_editor_settings_changed);

	ClassDB::bind_method("_menu_option", &ShaderEditor::_menu_option);
	ClassDB::bind_method("_params_changed", &ShaderEditor::_params_changed);
	ClassDB::bind_method("apply_shaders", &ShaderEditor::apply_shaders);
	//ClassDB::bind_method("_close_current_tab",&ShaderEditor::_close_current_tab);
}

void ShaderEditor::ensure_select_current() {

	/*
	if (tab_container->get_child_count() && tab_container->get_current_tab()>=0) {

		ShaderTextEditor *ste = Object::cast_to<ShaderTextEditor>(tab_container->get_child(tab_container->get_current_tab()));
		if (!ste)
			return;
		Ref<Shader> shader = ste->get_edited_shader();
		get_scene()->get_root_node()->call("_resource_selected",shader);
	}*/
}

void ShaderEditor::edit(const Ref<Shader> &p_shader) {

	if (p_shader.is_null())
		return;

	shader = p_shader;

	shader_editor->set_edited_shader(p_shader);

	//vertex_editor->set_edited_shader(shader,ShaderLanguage::SHADER_MATERIAL_VERTEX);
	// see if already has it
}

void ShaderEditor::save_external_data() {

	if (shader.is_null())
		return;
	apply_shaders();

	if (shader->get_path() != "" && shader->get_path().find("local://") == -1 && shader->get_path().find("::") == -1) {
		//external shader, save it
		ResourceSaver::save(shader->get_path(), shader);
	}
}

void ShaderEditor::apply_shaders() {

	if (shader.is_valid()) {
		shader->set_code(shader_editor->get_text_edit()->get_text());
		shader->set_edited(true);
	}
}

ShaderEditor::ShaderEditor() {

	HBoxContainer *hbc = memnew(HBoxContainer);

	add_child(hbc);

	edit_menu = memnew(MenuButton);
	hbc->add_child(edit_menu);
	edit_menu->set_position(Point2(5, -1));
	edit_menu->set_text(TTR("Edit"));
	edit_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/undo", TTR("Undo"), KEY_MASK_CMD | KEY_Z), EDIT_UNDO);
	edit_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/redo", TTR("Redo"), KEY_MASK_CMD | KEY_Y), EDIT_REDO);
	edit_menu->get_popup()->add_separator();
	edit_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/cut", TTR("Cut"), KEY_MASK_CMD | KEY_X), EDIT_CUT);
	edit_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/copy", TTR("Copy"), KEY_MASK_CMD | KEY_C), EDIT_COPY);
	edit_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/paste", TTR("Paste"), KEY_MASK_CMD | KEY_V), EDIT_PASTE);
	edit_menu->get_popup()->add_separator();
	edit_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/select_all", TTR("Select All"), KEY_MASK_CMD | KEY_A), EDIT_SELECT_ALL);
	edit_menu->get_popup()->connect("id_pressed", this, "_menu_option");

	search_menu = memnew(MenuButton);
	hbc->add_child(search_menu);
	search_menu->set_position(Point2(38, -1));
	search_menu->set_text(TTR("Search"));
	search_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/find", TTR("Find.."), KEY_MASK_CMD | KEY_F), SEARCH_FIND);
	search_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/find_next", TTR("Find Next"), KEY_F3), SEARCH_FIND_NEXT);
	search_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/find_previous", TTR("Find Previous"), KEY_MASK_SHIFT | KEY_F3), SEARCH_FIND_PREV);
	search_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/replace", TTR("Replace.."), KEY_MASK_CMD | KEY_R), SEARCH_REPLACE);
	search_menu->get_popup()->add_separator();
	//search_menu->get_popup()->add_item("Locate Symbol..",SEARCH_LOCATE_SYMBOL,KEY_MASK_CMD|KEY_K);
	search_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/goto_line", TTR("Goto Line.."), KEY_MASK_CMD | KEY_L), SEARCH_GOTO_LINE);
	search_menu->get_popup()->connect("id_pressed", this, "_menu_option");

	goto_line_dialog = memnew(GotoLineDialog);
	add_child(goto_line_dialog);

	shader_editor = memnew(ShaderTextEditor);
	add_child(shader_editor);
	shader_editor->set_v_size_flags(SIZE_EXPAND_FILL);

	shader_editor->connect("script_changed", this, "apply_shaders");
	EditorSettings::get_singleton()->connect("settings_changed", this, "_editor_settings_changed");

	_editor_settings_changed();
}

void ShaderEditorPlugin::edit(Object *p_object) {

	Shader *s = Object::cast_to<Shader>(p_object);
	shader_editor->edit(s);
}

bool ShaderEditorPlugin::handles(Object *p_object) const {

	Shader *shader = Object::cast_to<Shader>(p_object);
	return shader != NULL;
}

void ShaderEditorPlugin::make_visible(bool p_visible) {

	if (p_visible) {
		button->show();
		editor->make_bottom_panel_item_visible(shader_editor);

	} else {

		button->hide();
		if (shader_editor->is_visible_in_tree())
			editor->hide_bottom_panel();
		shader_editor->apply_shaders();
	}
}

void ShaderEditorPlugin::selected_notify() {

	shader_editor->ensure_select_current();
}

void ShaderEditorPlugin::save_external_data() {

	shader_editor->save_external_data();
}

void ShaderEditorPlugin::apply_changes() {

	shader_editor->apply_shaders();
}

ShaderEditorPlugin::ShaderEditorPlugin(EditorNode *p_node) {

	editor = p_node;
	shader_editor = memnew(ShaderEditor);

	shader_editor->set_custom_minimum_size(Size2(0, 300));
	button = editor->add_bottom_panel_item(TTR("Shader"), shader_editor);
	button->hide();
}

ShaderEditorPlugin::~ShaderEditorPlugin() {
}
