/*************************************************************************/
/*  shader_editor_plugin.cpp                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "editor/editor_settings.h"

#include "editor/editor_node.h"
#include "editor/property_editor.h"
#include "io/resource_loader.h"
#include "io/resource_saver.h"
#include "os/keyboard.h"
#include "os/os.h"
#include "scene/resources/shader_graph.h"
#include "spatial_editor_plugin.h"

/*** SETTINGS EDITOR ****/

/*** SCRIPT EDITOR ****/

Ref<Shader> ShaderTextEditor::get_edited_shader() const {

	return shader;
}
void ShaderTextEditor::set_edited_shader(const Ref<Shader> &p_shader, ShaderLanguage::ShaderType p_type) {

	shader = p_shader;
	type = p_type;

	_load_theme_settings();

	if (p_type == ShaderLanguage::SHADER_MATERIAL_LIGHT || p_type == ShaderLanguage::SHADER_CANVAS_ITEM_LIGHT)
		get_text_edit()->set_text(shader->get_light_code());
	else if (p_type == ShaderLanguage::SHADER_MATERIAL_VERTEX || p_type == ShaderLanguage::SHADER_CANVAS_ITEM_VERTEX)
		get_text_edit()->set_text(shader->get_vertex_code());
	else
		get_text_edit()->set_text(shader->get_fragment_code());

	_line_col_changed();
}

void ShaderTextEditor::_load_theme_settings() {

	get_text_edit()->clear_colors();

	/* keyword color */

	get_text_edit()->set_custom_bg_color(EDITOR_DEF("text_editor/background_color", Color(0, 0, 0, 0)));
	get_text_edit()->add_color_override("completion_background_color", EDITOR_DEF("text_editor/completion_background_color", Color(0, 0, 0, 0)));
	get_text_edit()->add_color_override("completion_selected_color", EDITOR_DEF("text_editor/completion_selected_color", Color::html("434244")));
	get_text_edit()->add_color_override("completion_existing_color", EDITOR_DEF("text_editor/completion_existing_color", Color::html("21dfdfdf")));
	get_text_edit()->add_color_override("completion_scroll_color", EDITOR_DEF("text_editor/completion_scroll_color", Color::html("ffffff")));
	get_text_edit()->add_color_override("completion_font_color", EDITOR_DEF("text_editor/completion_font_color", Color::html("aaaaaa")));
	get_text_edit()->add_color_override("font_color", EDITOR_DEF("text_editor/text_color", Color(0, 0, 0)));
	get_text_edit()->add_color_override("line_number_color", EDITOR_DEF("text_editor/line_number_color", Color(0, 0, 0)));
	get_text_edit()->add_color_override("caret_color", EDITOR_DEF("text_editor/caret_color", Color(0, 0, 0)));
	get_text_edit()->add_color_override("caret_background_color", EDITOR_DEF("text_editor/caret_background_color", Color(0, 0, 0)));
	get_text_edit()->add_color_override("font_selected_color", EDITOR_DEF("text_editor/text_selected_color", Color(1, 1, 1)));
	get_text_edit()->add_color_override("selection_color", EDITOR_DEF("text_editor/selection_color", Color(0.2, 0.2, 1)));
	get_text_edit()->add_color_override("brace_mismatch_color", EDITOR_DEF("text_editor/brace_mismatch_color", Color(1, 0.2, 0.2)));
	get_text_edit()->add_color_override("current_line_color", EDITOR_DEF("text_editor/current_line_color", Color(0.3, 0.5, 0.8, 0.15)));
	get_text_edit()->add_color_override("word_highlighted_color", EDITOR_DEF("text_editor/word_highlighted_color", Color(0.8, 0.9, 0.9, 0.15)));
	get_text_edit()->add_color_override("number_color", EDITOR_DEF("text_editor/number_color", Color(0.9, 0.6, 0.0, 2)));
	get_text_edit()->add_color_override("function_color", EDITOR_DEF("text_editor/function_color", Color(0.4, 0.6, 0.8)));
	get_text_edit()->add_color_override("member_variable_color", EDITOR_DEF("text_editor/member_variable_color", Color(0.9, 0.3, 0.3)));
	get_text_edit()->add_color_override("mark_color", EDITOR_DEF("text_editor/mark_color", Color(1.0, 0.4, 0.4, 0.4)));
	get_text_edit()->add_color_override("breakpoint_color", EDITOR_DEF("text_editor/breakpoint_color", Color(0.8, 0.8, 0.4, 0.2)));
	get_text_edit()->add_color_override("search_result_color", EDITOR_DEF("text_editor/search_result_color", Color(0.05, 0.25, 0.05, 1)));
	get_text_edit()->add_color_override("search_result_border_color", EDITOR_DEF("text_editor/search_result_border_color", Color(0.1, 0.45, 0.1, 1)));

	Color keyword_color = EDITOR_DEF("text_editor/keyword_color", Color(0.5, 0.0, 0.2));

	List<String> keywords;
	ShaderLanguage::get_keyword_list(type, &keywords);

	for (List<String>::Element *E = keywords.front(); E; E = E->next()) {

		get_text_edit()->add_keyword_color(E->get(), keyword_color);
	}

	//colorize core types
	//	Color basetype_color= EDITOR_DEF("text_editor/base_type_color",Color(0.3,0.3,0.0));

	//colorize comments
	Color comment_color = EDITOR_DEF("text_editor/comment_color", Color::hex(0x797e7eff));

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

	//colorize symbols
	Color symbol_color = EDITOR_DEF("text_editor/symbol_color", Color::hex(0x005291ff));
	get_text_edit()->set_symbol_color(symbol_color);
}

void ShaderTextEditor::_validate_script() {

	String errortxt;
	int line, col;

	String code = get_text_edit()->get_text();
	//List<StringName> params;
	//shader->get_param_list(&params);

	Error err = ShaderLanguage::compile(code, type, NULL, NULL, &errortxt, &line, &col);

	if (err != OK) {
		String error_text = "error(" + itos(line + 1) + "," + itos(col + 1) + "): " + errortxt;
		set_error(error_text);
		get_text_edit()->set_line_as_marked(line, true);

	} else {
		for (int i = 0; i < get_text_edit()->get_line_count(); i++)
			get_text_edit()->set_line_as_marked(i, false);
		set_error("");
	}

	emit_signal("script_changed");
}

void ShaderTextEditor::_bind_methods() {

	//ADD_SIGNAL( MethodInfo("script_changed") );
}

ShaderTextEditor::ShaderTextEditor() {
}

/*** SCRIPT EDITOR ******/

void ShaderEditor::_menu_option(int p_option) {

	ShaderTextEditor *current = tab_container->get_current_tab_control()->cast_to<ShaderTextEditor>();
	if (!current)
		return;

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
		//		case SEARCH_LOCATE_SYMBOL: {

		//		} break;
		case SEARCH_GOTO_LINE: {

			goto_line_dialog->popup_find_line(current->get_text_edit());
		} break;
	}
}

void ShaderEditor::_tab_changed(int p_which) {

	ShaderTextEditor *shader_editor = tab_container->get_tab_control(p_which)->cast_to<ShaderTextEditor>();

	if (shader_editor && is_inside_tree())
		shader_editor->get_text_edit()->grab_focus();

	ensure_select_current();
}

void ShaderEditor::_notification(int p_what) {

	if (p_what == NOTIFICATION_ENTER_TREE) {

		close->set_normal_texture(get_icon("Close", "EditorIcons"));
		close->set_hover_texture(get_icon("CloseHover", "EditorIcons"));
		close->set_pressed_texture(get_icon("Close", "EditorIcons"));
		close->connect("pressed", this, "_close_callback");
	}
	if (p_what == NOTIFICATION_DRAW) {

		RID ci = get_canvas_item();
		Ref<StyleBox> style = get_stylebox("panel", "Panel");
		style->draw(ci, Rect2(Point2(), get_size()));
	}
}

Dictionary ShaderEditor::get_state() const {
#if 0
	apply_shaders();

	Dictionary state;

	Array paths;
	int open=-1;

	for(int i=0;i<tab_container->get_child_count();i++) {

		ShaderTextEditor *ste = tab_container->get_child(i)->cast_to<ShaderTextEditor>();
		if (!ste)
			continue;


		Ref<Shader> shader = ste->get_edited_shader();
		if (shader->get_path()!="" && shader->get_path().find("local://")==-1 && shader->get_path().find("::")==-1) {

			paths.push_back(shader->get_path());
		} else {


			const Node *owner = _find_node_with_shader(get_root_node(),shader.get_ref_ptr());
			if (owner)
				paths.push_back(owner->get_path());

		}

		if (i==tab_container->get_current_tab())
			open=i;
	}

	if (paths.size())
		state["sources"]=paths;
	if (open!=-1)
		state["current"]=open;


	return state;
#endif
	return Dictionary();
}
void ShaderEditor::set_state(const Dictionary &p_state) {
#if 0
	print_line("setting state..");
	if (!p_state.has("sources"))
		return; //bleh

	Array sources = p_state["sources"];
	for(int i=0;i<sources.size();i++) {

		Variant source=sources[i];

		Ref<Shader> shader;

		if (source.get_type()==Variant::NODE_PATH) {

			print_line("cain find owner at path "+String(source));
			Node *owner=get_root_node()->get_node(source);
			if (!owner)
				continue;

			shader = owner->get_shader();
		} else if (source.get_type()==Variant::STRING) {

			print_line("loading at path "+String(source));
			shader = ResourceLoader::load(source,"Shader");
		}

		print_line("found shader at "+String(source)+"? - "+itos(shader.is_null()));
		if (shader.is_null()) //ah well..
			continue;

		get_scene()->get_root_node()->call("_resource_selected",shader);
	}

	if (p_state.has("current"))
	tab_container->set_current_tab(p_state["current"]);
#endif
}
void ShaderEditor::clear() {
}

void ShaderEditor::_params_changed() {

	fragment_editor->_validate_script();
	vertex_editor->_validate_script();
	light_editor->_validate_script();
}

void ShaderEditor::_editor_settings_changed() {

	vertex_editor->update_editor_settings();
	fragment_editor->update_editor_settings();
	light_editor->update_editor_settings();
}

void ShaderEditor::_bind_methods() {

	ObjectTypeDB::bind_method("_editor_settings_changed", &ShaderEditor::_editor_settings_changed);
	ObjectTypeDB::bind_method("_tab_changed", &ShaderEditor::_tab_changed);
	ObjectTypeDB::bind_method("_menu_option", &ShaderEditor::_menu_option);
	ObjectTypeDB::bind_method("_params_changed", &ShaderEditor::_params_changed);
	ObjectTypeDB::bind_method("_close_callback", &ShaderEditor::_close_callback);
	ObjectTypeDB::bind_method("apply_shaders", &ShaderEditor::apply_shaders);
	//	ObjectTypeDB::bind_method("_close_current_tab",&ShaderEditor::_close_current_tab);
}

void ShaderEditor::ensure_select_current() {

	/*
	if (tab_container->get_child_count() && tab_container->get_current_tab()>=0) {

		ShaderTextEditor *ste = tab_container->get_child(tab_container->get_current_tab())->cast_to<ShaderTextEditor>();
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

	if (shader->get_mode() == Shader::MODE_MATERIAL) {
		vertex_editor->set_edited_shader(p_shader, ShaderLanguage::SHADER_MATERIAL_VERTEX);
		fragment_editor->set_edited_shader(p_shader, ShaderLanguage::SHADER_MATERIAL_FRAGMENT);
		light_editor->set_edited_shader(shader, ShaderLanguage::SHADER_MATERIAL_LIGHT);
	} else if (shader->get_mode() == Shader::MODE_CANVAS_ITEM) {

		vertex_editor->set_edited_shader(p_shader, ShaderLanguage::SHADER_CANVAS_ITEM_VERTEX);
		fragment_editor->set_edited_shader(p_shader, ShaderLanguage::SHADER_CANVAS_ITEM_FRAGMENT);
		light_editor->set_edited_shader(shader, ShaderLanguage::SHADER_CANVAS_ITEM_LIGHT);
	}

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
		shader->set_code(vertex_editor->get_text_edit()->get_text(), fragment_editor->get_text_edit()->get_text(), light_editor->get_text_edit()->get_text(), 0, 0);
		shader->set_edited(true);
	}
}

void ShaderEditor::_close_callback() {

	hide();
}

ShaderEditor::ShaderEditor() {

	tab_container = memnew(TabContainer);
	add_child(tab_container);
	tab_container->set_area_as_parent_rect();
	tab_container->set_begin(Point2(0, 0));
	//tab_container->set_begin(Point2(0,0));

	close = memnew(TextureButton);
	close->set_anchor_and_margin(MARGIN_LEFT, ANCHOR_END, 20);
	close->set_anchor_and_margin(MARGIN_RIGHT, ANCHOR_END, 4);
	close->set_anchor_and_margin(MARGIN_TOP, ANCHOR_BEGIN, 2);
	add_child(close);

	edit_menu = memnew(MenuButton);
	add_child(edit_menu);
	edit_menu->set_pos(Point2(5, -1));
	edit_menu->set_text(TTR("Edit"));
	edit_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/undo", TTR("Undo"), KEY_MASK_CMD | KEY_Z), EDIT_UNDO);
	edit_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/redo", TTR("Redo"), KEY_MASK_CMD | KEY_Y), EDIT_REDO);
	edit_menu->get_popup()->add_separator();
	edit_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/cut", TTR("Cut"), KEY_MASK_CMD | KEY_X), EDIT_CUT);
	edit_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/copy", TTR("Copy"), KEY_MASK_CMD | KEY_C), EDIT_COPY);
	edit_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/paste", TTR("Paste"), KEY_MASK_CMD | KEY_V), EDIT_PASTE);
	edit_menu->get_popup()->add_separator();
	edit_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/select_all", TTR("Select All"), KEY_MASK_CMD | KEY_A), EDIT_SELECT_ALL);
	edit_menu->get_popup()->connect("item_pressed", this, "_menu_option");

	search_menu = memnew(MenuButton);
	add_child(search_menu);
	search_menu->set_pos(Point2(38, -1));
	search_menu->set_text(TTR("Search"));
	search_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/find", TTR("Find.."), KEY_MASK_CMD | KEY_F), SEARCH_FIND);
	search_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/find_next", TTR("Find Next"), KEY_F3), SEARCH_FIND_NEXT);
	search_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/find_previous", TTR("Find Previous"), KEY_MASK_SHIFT | KEY_F3), SEARCH_FIND_PREV);
	search_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/replace", TTR("Replace.."), KEY_MASK_CMD | KEY_R), SEARCH_REPLACE);
	search_menu->get_popup()->add_separator();
	//	search_menu->get_popup()->add_item("Locate Symbol..",SEARCH_LOCATE_SYMBOL,KEY_MASK_CMD|KEY_K);
	search_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/goto_line", TTR("Goto Line.."), KEY_MASK_CMD | KEY_L), SEARCH_GOTO_LINE);
	search_menu->get_popup()->connect("item_pressed", this, "_menu_option");

	tab_container->connect("tab_changed", this, "_tab_changed");

	erase_tab_confirm = memnew(ConfirmationDialog);
	add_child(erase_tab_confirm);
	erase_tab_confirm->connect("confirmed", this, "_close_current_tab");

	goto_line_dialog = memnew(GotoLineDialog);
	add_child(goto_line_dialog);

	vertex_editor = memnew(ShaderTextEditor);
	tab_container->add_child(vertex_editor);
	vertex_editor->set_name(TTR("Vertex"));

	fragment_editor = memnew(ShaderTextEditor);
	tab_container->add_child(fragment_editor);
	fragment_editor->set_name(TTR("Fragment"));

	light_editor = memnew(ShaderTextEditor);
	tab_container->add_child(light_editor);
	light_editor->set_name(TTR("Lighting"));

	tab_container->set_current_tab(1);

	vertex_editor->connect("script_changed", this, "apply_shaders");
	fragment_editor->connect("script_changed", this, "apply_shaders");
	light_editor->connect("script_changed", this, "apply_shaders");
	EditorSettings::get_singleton()->connect("settings_changed", this, "_editor_settings_changed");

	_editor_settings_changed();
}

void ShaderEditorPlugin::edit(Object *p_object) {

	Shader *s = p_object->cast_to<Shader>();
	if (!s || s->cast_to<ShaderGraph>()) {
		shader_editor->hide(); //Dont edit ShaderGraph
		return;
	}

	if (_2d && s->get_mode() == Shader::MODE_CANVAS_ITEM)
		shader_editor->edit(s);
	else if (!_2d && s->get_mode() == Shader::MODE_MATERIAL)
		shader_editor->edit(s);
}

bool ShaderEditorPlugin::handles(Object *p_object) const {

	bool handles = true;
	Shader *shader = p_object->cast_to<Shader>();
	if (!shader || shader->cast_to<ShaderGraph>()) // Dont handle ShaderGraph's
		handles = false;
	if (handles && _2d)
		handles = shader->get_mode() == Shader::MODE_CANVAS_ITEM;
	else if (handles && !_2d)
		return shader->get_mode() == Shader::MODE_MATERIAL;

	if (!handles)
		shader_editor->hide();
	return handles;
}

void ShaderEditorPlugin::make_visible(bool p_visible) {

	if (p_visible) {
		shader_editor->show();
	} else {
		shader_editor->apply_shaders();
	}
}

void ShaderEditorPlugin::selected_notify() {

	shader_editor->ensure_select_current();
}

Dictionary ShaderEditorPlugin::get_state() const {

	return shader_editor->get_state();
}

void ShaderEditorPlugin::set_state(const Dictionary &p_state) {

	shader_editor->set_state(p_state);
}
void ShaderEditorPlugin::clear() {

	shader_editor->clear();
}

void ShaderEditorPlugin::save_external_data() {

	shader_editor->save_external_data();
}

void ShaderEditorPlugin::apply_changes() {

	shader_editor->apply_shaders();
}

ShaderEditorPlugin::ShaderEditorPlugin(EditorNode *p_node, bool p_2d) {

	editor = p_node;
	shader_editor = memnew(ShaderEditor);
	_2d = p_2d;
	if (p_2d)
		add_control_to_container(CONTAINER_CANVAS_EDITOR_BOTTOM, shader_editor);
	else
		add_control_to_container(CONTAINER_SPATIAL_EDITOR_BOTTOM, shader_editor);
	//	editor->get_viewport()->add_child(shader_editor);
	//	shader_editor->set_area_as_parent_rect();

	shader_editor->hide();
}

ShaderEditorPlugin::~ShaderEditorPlugin() {
}
