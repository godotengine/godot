/*************************************************************************/
/*  shader_editor_plugin.cpp                                             */
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

#include "shader_editor_plugin.h"

#include "core/io/resource_loader.h"
#include "core/io/resource_saver.h"
#include "core/os/keyboard.h"
#include "core/os/os.h"
#include "editor/editor_feature_profile.h"
#include "editor/editor_node.h"
#include "editor/editor_scale.h"
#include "editor/editor_settings.h"
#include "editor/property_editor.h"
#include "servers/display_server.h"
#include "servers/rendering/shader_types.h"

/*** SHADER SCRIPT EDITOR ****/

Ref<Shader> ShaderTextEditor::get_edited_shader() const {
	return shader;
}

void ShaderTextEditor::set_edited_shader(const Ref<Shader> &p_shader) {
	if (shader == p_shader) {
		return;
	}
	shader = p_shader;

	_load_theme_settings();

	get_text_editor()->set_text(p_shader->get_code());
	get_text_editor()->clear_undo_history();

	_validate_script();
	_line_col_changed();
}

void ShaderTextEditor::reload_text() {
	ERR_FAIL_COND(shader.is_null());

	CodeEdit *te = get_text_editor();
	int column = te->cursor_get_column();
	int row = te->cursor_get_line();
	int h = te->get_h_scroll();
	int v = te->get_v_scroll();

	te->set_text(shader->get_code());
	te->cursor_set_line(row);
	te->cursor_set_column(column);
	te->set_h_scroll(h);
	te->set_v_scroll(v);

	te->tag_saved_version();

	update_line_and_column();
}

void ShaderTextEditor::_load_theme_settings() {
	Color background_color = EDITOR_GET("text_editor/highlighting/background_color");
	Color completion_background_color = EDITOR_GET("text_editor/highlighting/completion_background_color");
	Color completion_selected_color = EDITOR_GET("text_editor/highlighting/completion_selected_color");
	Color completion_existing_color = EDITOR_GET("text_editor/highlighting/completion_existing_color");
	Color completion_scroll_color = EDITOR_GET("text_editor/highlighting/completion_scroll_color");
	Color completion_font_color = EDITOR_GET("text_editor/highlighting/completion_font_color");
	Color text_color = EDITOR_GET("text_editor/highlighting/text_color");
	Color line_number_color = EDITOR_GET("text_editor/highlighting/line_number_color");
	Color caret_color = EDITOR_GET("text_editor/highlighting/caret_color");
	Color caret_background_color = EDITOR_GET("text_editor/highlighting/caret_background_color");
	Color text_selected_color = EDITOR_GET("text_editor/highlighting/text_selected_color");
	Color selection_color = EDITOR_GET("text_editor/highlighting/selection_color");
	Color brace_mismatch_color = EDITOR_GET("text_editor/highlighting/brace_mismatch_color");
	Color current_line_color = EDITOR_GET("text_editor/highlighting/current_line_color");
	Color line_length_guideline_color = EDITOR_GET("text_editor/highlighting/line_length_guideline_color");
	Color word_highlighted_color = EDITOR_GET("text_editor/highlighting/word_highlighted_color");
	Color mark_color = EDITOR_GET("text_editor/highlighting/mark_color");
	Color bookmark_color = EDITOR_GET("text_editor/highlighting/bookmark_color");
	Color breakpoint_color = EDITOR_GET("text_editor/highlighting/breakpoint_color");
	Color executing_line_color = EDITOR_GET("text_editor/highlighting/executing_line_color");
	Color code_folding_color = EDITOR_GET("text_editor/highlighting/code_folding_color");
	Color search_result_color = EDITOR_GET("text_editor/highlighting/search_result_color");
	Color search_result_border_color = EDITOR_GET("text_editor/highlighting/search_result_border_color");

	get_text_editor()->add_theme_color_override("background_color", background_color);
	get_text_editor()->add_theme_color_override("completion_background_color", completion_background_color);
	get_text_editor()->add_theme_color_override("completion_selected_color", completion_selected_color);
	get_text_editor()->add_theme_color_override("completion_existing_color", completion_existing_color);
	get_text_editor()->add_theme_color_override("completion_scroll_color", completion_scroll_color);
	get_text_editor()->add_theme_color_override("completion_font_color", completion_font_color);
	get_text_editor()->add_theme_color_override("font_color", text_color);
	get_text_editor()->add_theme_color_override("line_number_color", line_number_color);
	get_text_editor()->add_theme_color_override("caret_color", caret_color);
	get_text_editor()->add_theme_color_override("caret_background_color", caret_background_color);
	get_text_editor()->add_theme_color_override("font_color_selected", text_selected_color);
	get_text_editor()->add_theme_color_override("selection_color", selection_color);
	get_text_editor()->add_theme_color_override("brace_mismatch_color", brace_mismatch_color);
	get_text_editor()->add_theme_color_override("current_line_color", current_line_color);
	get_text_editor()->add_theme_color_override("line_length_guideline_color", line_length_guideline_color);
	get_text_editor()->add_theme_color_override("word_highlighted_color", word_highlighted_color);
	get_text_editor()->add_theme_color_override("mark_color", mark_color);
	get_text_editor()->add_theme_color_override("bookmark_color", bookmark_color);
	get_text_editor()->add_theme_color_override("breakpoint_color", breakpoint_color);
	get_text_editor()->add_theme_color_override("executing_line_color", executing_line_color);
	get_text_editor()->add_theme_color_override("code_folding_color", code_folding_color);
	get_text_editor()->add_theme_color_override("search_result_color", search_result_color);
	get_text_editor()->add_theme_color_override("search_result_border_color", search_result_border_color);

	syntax_highlighter->set_number_color(EDITOR_GET("text_editor/highlighting/number_color"));
	syntax_highlighter->set_symbol_color(EDITOR_GET("text_editor/highlighting/symbol_color"));
	syntax_highlighter->set_function_color(EDITOR_GET("text_editor/highlighting/function_color"));
	syntax_highlighter->set_member_variable_color(EDITOR_GET("text_editor/highlighting/member_variable_color"));

	syntax_highlighter->clear_keyword_colors();

	List<String> keywords;
	ShaderLanguage::get_keyword_list(&keywords);
	const Color keyword_color = EDITOR_GET("text_editor/highlighting/keyword_color");

	for (List<String>::Element *E = keywords.front(); E; E = E->next()) {
		syntax_highlighter->add_keyword_color(E->get(), keyword_color);
	}

	// Colorize built-ins like `COLOR` differently to make them easier
	// to distinguish from keywords at a quick glance.

	List<String> built_ins;
	if (shader.is_valid()) {
		for (const Map<StringName, ShaderLanguage::FunctionInfo>::Element *E = ShaderTypes::get_singleton()->get_functions(RenderingServer::ShaderMode(shader->get_mode())).front(); E; E = E->next()) {
			for (const Map<StringName, ShaderLanguage::BuiltInInfo>::Element *F = E->get().built_ins.front(); F; F = F->next()) {
				built_ins.push_back(F->key());
			}
		}

		for (int i = 0; i < ShaderTypes::get_singleton()->get_modes(RenderingServer::ShaderMode(shader->get_mode())).size(); i++) {
			built_ins.push_back(ShaderTypes::get_singleton()->get_modes(RenderingServer::ShaderMode(shader->get_mode()))[i]);
		}
	}

	const Color member_variable_color = EDITOR_GET("text_editor/highlighting/member_variable_color");

	for (List<String>::Element *E = built_ins.front(); E; E = E->next()) {
		syntax_highlighter->add_keyword_color(E->get(), member_variable_color);
	}

	// Colorize comments.
	const Color comment_color = EDITOR_GET("text_editor/highlighting/comment_color");
	syntax_highlighter->clear_color_regions();
	syntax_highlighter->add_color_region("/*", "*/", comment_color, false);
	syntax_highlighter->add_color_region("//", "", comment_color, true);
}

void ShaderTextEditor::_check_shader_mode() {
	String type = ShaderLanguage::get_shader_type(get_text_editor()->get_text());

	Shader::Mode mode;

	if (type == "canvas_item") {
		mode = Shader::MODE_CANVAS_ITEM;
	} else if (type == "particles") {
		mode = Shader::MODE_PARTICLES;
	} else {
		mode = Shader::MODE_SPATIAL;
	}

	if (shader->get_mode() != mode) {
		shader->set_code(get_text_editor()->get_text());
		_load_theme_settings();
	}
}

static ShaderLanguage::DataType _get_global_variable_type(const StringName &p_variable) {
	RS::GlobalVariableType gvt = RS::get_singleton()->global_variable_get_type(p_variable);
	return RS::global_variable_type_get_shader_datatype(gvt);
}

void ShaderTextEditor::_code_complete_script(const String &p_code, List<ScriptCodeCompletionOption> *r_options) {
	_check_shader_mode();

	ShaderLanguage sl;
	String calltip;

	sl.complete(p_code, ShaderTypes::get_singleton()->get_functions(RenderingServer::ShaderMode(shader->get_mode())), ShaderTypes::get_singleton()->get_modes(RenderingServer::ShaderMode(shader->get_mode())), ShaderTypes::get_singleton()->get_types(), _get_global_variable_type, r_options, calltip);

	get_text_editor()->set_code_hint(calltip);
}

void ShaderTextEditor::_validate_script() {
	_check_shader_mode();

	String code = get_text_editor()->get_text();
	//List<StringName> params;
	//shader->get_param_list(&params);

	ShaderLanguage sl;

	Error err = sl.compile(code, ShaderTypes::get_singleton()->get_functions(RenderingServer::ShaderMode(shader->get_mode())), ShaderTypes::get_singleton()->get_modes(RenderingServer::ShaderMode(shader->get_mode())), ShaderTypes::get_singleton()->get_types(), _get_global_variable_type);

	if (err != OK) {
		String error_text = "error(" + itos(sl.get_error_line()) + "): " + sl.get_error_text();
		set_error(error_text);
		set_error_pos(sl.get_error_line() - 1, 0);
		for (int i = 0; i < get_text_editor()->get_line_count(); i++) {
			get_text_editor()->set_line_as_marked(i, false);
		}
		get_text_editor()->set_line_as_marked(sl.get_error_line() - 1, true);

	} else {
		for (int i = 0; i < get_text_editor()->get_line_count(); i++) {
			get_text_editor()->set_line_as_marked(i, false);
		}
		set_error("");
	}

	emit_signal("script_changed");
}

void ShaderTextEditor::_bind_methods() {
}

ShaderTextEditor::ShaderTextEditor() {
	syntax_highlighter.instance();
	get_text_editor()->set_syntax_highlighter(syntax_highlighter);
}

/*** SHADER EDITOR ******/

ShaderEditor *ShaderEditor::shader_editor_singleton = nullptr;

void ShaderEditor::_preview_3d_option(int p_option) {
	if (last_option == p_option) {
		return;
	}
	if (last_option == NODE3D_QUAD) {
		camera->set_transform(Transform(Basis(), Vector3(0, 0, 10))); // restore camera transform for standard mesh
	}
	last_option = p_option;

	if (p_option == NODE3D_SCENE) {
		local_viewport_base->hide();
		scene_viewport_base->show();
	} else {
		local_viewport_base->show();
		scene_viewport_base->hide();

		if (p_option == NODE3D_QUAD) {
			camera->set_transform(Transform(Basis(), Vector3(0, 0, 2)));
		}
		for (Map<int, MeshInstance3D *>::Element *E = mesh_instances.front(); E; E = E->next()) {
			E->get()->set_visible(p_option == E->key());
		}
	}
}

void ShaderEditor::_menu_option(int p_option) {
	switch (p_option) {
		case FILE_CLOSE: {
			if (shader.is_valid()) {
				int index = shader_list->get_selected_items()[0];

				String id = shader_list->get_item_metadata(index);
				shader_list->remove_item(index);

				recent_shaders_map.erase(id);

				if (shader_filtered_list->is_visible()) {
					_update_filtered_shader_list();
				}
				if (shader_list->get_item_count() > 0) {
					edit(recent_shaders_map[shader_list->get_item_metadata(shader_list->get_item_count() - 1)]);
				} else {
					edit(nullptr);
				}
			}
		} break;
		case FILE_CLOSE_ALL: {
			shader_list->clear();
			shader_filtered_list->clear();
			recent_shaders_map.clear();
			edit(nullptr);
		} break;
		case FILE_CLOSE_OTHER_TABS: {
			if (shader.is_valid()) {
				String tab_name = shader_list->get_item_text(shader_list->get_selected_items()[0]);
				shader_list->clear();
				recent_shaders_map.clear();
				recent_shaders_map.insert(shader->get_path(), shader);
				shader_list->add_item(tab_name, get_theme_icon("Shader", "EditorIcons"));
				shader_list->set_item_metadata(0, shader->get_path());
				shader_list->select(0);

				if (shader_filtered_list->is_visible()) {
					_update_filtered_shader_list();
				}
			}
		} break;
		case EDIT_UNDO: {
			shader_editor->get_text_editor()->undo();
		} break;
		case EDIT_REDO: {
			shader_editor->get_text_editor()->redo();
		} break;
		case EDIT_CUT: {
			shader_editor->get_text_editor()->cut();
		} break;
		case EDIT_COPY: {
			shader_editor->get_text_editor()->copy();
		} break;
		case EDIT_PASTE: {
			shader_editor->get_text_editor()->paste();
		} break;
		case EDIT_SELECT_ALL: {
			shader_editor->get_text_editor()->select_all();
		} break;
		case EDIT_MOVE_LINE_UP: {
			shader_editor->move_lines_up();
		} break;
		case EDIT_MOVE_LINE_DOWN: {
			shader_editor->move_lines_down();
		} break;
		case EDIT_INDENT_LEFT: {
			if (shader.is_null()) {
				return;
			}

			CodeEdit *tx = shader_editor->get_text_editor();
			tx->indent_left();

		} break;
		case EDIT_INDENT_RIGHT: {
			if (shader.is_null()) {
				return;
			}

			CodeEdit *tx = shader_editor->get_text_editor();
			tx->indent_right();

		} break;
		case EDIT_DELETE_LINE: {
			shader_editor->delete_lines();
		} break;
		case EDIT_CLONE_DOWN: {
			shader_editor->clone_lines_down();
		} break;
		case EDIT_TOGGLE_COMMENT: {
			if (shader.is_null()) {
				return;
			}

			shader_editor->toggle_inline_comment("//");

		} break;
		case EDIT_COMPLETE: {
			shader_editor->get_text_editor()->query_code_comple();
		} break;
		case SEARCH_FIND: {
			shader_editor->get_find_replace_bar()->popup_search();
		} break;
		case SEARCH_FIND_NEXT: {
			shader_editor->get_find_replace_bar()->search_next();
		} break;
		case SEARCH_FIND_PREV: {
			shader_editor->get_find_replace_bar()->search_prev();
		} break;
		case SEARCH_REPLACE: {
			shader_editor->get_find_replace_bar()->popup_replace();
		} break;
		case SEARCH_GOTO_LINE: {
			goto_line_dialog->popup_find_line(shader_editor->get_text_editor());
		} break;
		case BOOKMARK_TOGGLE: {
			shader_editor->toggle_bookmark();
		} break;
		case BOOKMARK_GOTO_NEXT: {
			shader_editor->goto_next_bookmark();
		} break;
		case BOOKMARK_GOTO_PREV: {
			shader_editor->goto_prev_bookmark();
		} break;
		case BOOKMARK_REMOVE_ALL: {
			shader_editor->remove_all_bookmarks();
		} break;
		case HELP_DOCS: {
			OS::get_singleton()->shell_open("https://docs.godotengine.org/en/latest/tutorials/shaders/shader_reference/index.html");
		} break;
		case TOGGLE_SHADERS_PANEL: {
			toggle_shaders_panel();
			shader_editor->update_toggle_scripts_button();
		} break;
	}
	if (p_option != SEARCH_FIND && p_option != SEARCH_REPLACE && p_option != SEARCH_GOTO_LINE) {
		shader_editor->get_text_editor()->call_deferred("grab_focus");
	}
}

bool ShaderEditor::toggle_shaders_panel() {
	shaders_vbox->set_visible(!shaders_vbox->is_visible());
	return shaders_vbox->is_visible();
}

bool ShaderEditor::is_shaders_panel_toggled() const {
	return shaders_vbox->is_visible();
}

void ShaderEditor::_notification(int p_what) {
	if (p_what == NOTIFICATION_READY || p_what == NOTIFICATION_THEME_CHANGED) {
		quad_button->set_icon(get_theme_icon("QuadMesh", "EditorIcons"));
		sphere_button->set_icon(get_theme_icon("SphereMesh", "EditorIcons"));
		box_button->set_icon(get_theme_icon("BoxMesh", "EditorIcons"));
		cylinder_button->set_icon(get_theme_icon("CylinderMesh", "EditorIcons"));
	}
	if (p_what == NOTIFICATION_WM_WINDOW_FOCUS_IN) {
		_check_for_external_edit();
	}
}

void ShaderEditor::_params_changed() {
	shader_editor->_validate_script();
}

void ShaderEditor::_editor_settings_changed() {
	shader_editor->update_editor_settings();

	shader_editor->get_text_editor()->add_theme_constant_override("line_spacing", EditorSettings::get_singleton()->get("text_editor/theme/line_spacing"));
	shader_editor->get_text_editor()->set_draw_breakpoints_gutter(false);
	shader_editor->get_text_editor()->set_draw_executing_lines_gutter(false);
}

void ShaderEditor::_bind_methods() {
	ClassDB::bind_method("_params_changed", &ShaderEditor::_params_changed);
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

void ShaderEditor::goto_line_selection(int p_line, int p_begin, int p_end) {
	shader_editor->goto_line_selection(p_line, p_begin, p_end);
}

void ShaderEditor::_check_for_external_edit() {
	if (shader.is_null() || !shader.is_valid()) {
		return;
	}

	// internal shader.
	if (shader->get_path() == "" || shader->get_path().find("local://") != -1 || shader->get_path().find("::") != -1) {
		return;
	}

	bool use_autoreload = bool(EDITOR_DEF("text_editor/files/auto_reload_scripts_on_external_change", false));
	if (shader->get_last_modified_time() != FileAccess::get_modified_time(shader->get_path())) {
		if (use_autoreload) {
			_reload_shader_from_disk();
		} else {
			disk_changed->call_deferred("popup_centered");
		}
	}
}

void ShaderEditor::_reload_shader_from_disk() {
	Ref<Shader> rel_shader = ResourceLoader::load(shader->get_path(), shader->get_class(), true);
	ERR_FAIL_COND(!rel_shader.is_valid());

	shader->set_code(rel_shader->get_code());
	shader->set_last_modified_time(rel_shader->get_last_modified_time());
	shader_editor->reload_text();
}

void ShaderEditor::edit(const Ref<Shader> &p_shader) {
	if (p_shader.is_null() || !p_shader->is_text_shader()) {
		shader = Ref<Shader>();
		main_splitbox->hide();
		return;
	}
	main_splitbox->show();

	if (shader == p_shader) {
		return;
	}

	shader = p_shader;

	shader_editor->set_edited_shader(p_shader);
	_apply_to_preview();

	if (!recent_shaders_map.has(p_shader->get_path())) {
		String shader_name = p_shader->get_path().get_file();
		int count = 0;
		for (Map<String, Ref<Shader>>::Element *E = recent_shaders_map.front(); E; E = E->next()) {
			if (E->key() == p_shader->get_path()) {
				count++;
			}
		}
		if (count > 0) {
			shader_name += ":" + itos(count);
		}
		recent_shaders_map.insert(p_shader->get_path(), p_shader);

		shader_list->add_item(shader_name, get_theme_icon("Shader", "EditorIcons"));
		shader_list->set_item_metadata(shader_list->get_item_count() - 1, p_shader->get_path());
		shader_list->select(shader_list->get_item_count() - 1);

		if (shader_filtered_list->is_visible()) {
			_update_filtered_shader_list();
		}
	} else {
		for (int i = 0; i < shader_list->get_item_count(); i++) {
			if (shader_list->get_item_metadata(i) == p_shader->get_path()) {
				shader_list->select(i);
				break;
			}
		}
	}

	//vertex_editor->set_edited_shader(shader,ShaderLanguage::SHADER_MATERIAL_VERTEX);
	// see if already has it
}

void ShaderEditor::_update_filtered_shader_list() {
	_filter_shaders_text_changed(filter_shaders->get_text());
}

void ShaderEditor::_filter_shaders_text_changed(const String &p_newtext) {
	if (p_newtext == "") {
		shader_list->show();
		shader_filtered_list->hide();
		return;
	}

	shader_list->hide();
	shader_filtered_list->show();
	shader_filtered_list->clear();

	for (int i = 0; i < shader_list->get_item_count(); i++) {
		String shader_name = shader_list->get_item_text(i);
		if (shader_name.get_basename().find(p_newtext) != -1) {
			shader_filtered_list->add_item(shader_name);
			shader_filtered_list->set_item_metadata(shader_filtered_list->get_item_count() - 1, shader_list->get_item_metadata(i));
		}
	}
}

void ShaderEditor::save_external_data(const String &p_str) {
	if (shader.is_null()) {
		disk_changed->hide();
		return;
	}

	apply_shaders();
	if (shader->get_path() != "" && shader->get_path().find("local://") == -1 && shader->get_path().find("::") == -1) {
		//external shader, save it
		ResourceSaver::save(shader->get_path(), shader);
	}

	disk_changed->hide();
}

void ShaderEditor::attach_scene_viewport() {
	Ref<EditorFeatureProfile> profile = EditorFeatureProfileManager::get_singleton()->get_current_profile();
	if (profile.is_valid()) {
		if (profile->is_feature_disabled(EditorFeatureProfile::FEATURE_3D)) {
			spatial_hbox->hide();
			return;
		}
	}

	spatial_hbox->show();
	scene_viewport_base = Node3DEditor::get_singleton()->get_editor_viewport_base();
	scene_viewport_parent = scene_viewport_base->get_parent();

	if (last_option != NODE3D_SCENE) {
		scene_viewport_base->hide();
	}

	Node3DEditor::get_singleton()->set_viewports_reattaching(true);

	scene_viewport_pos = scene_viewport_base->get_index();
	scene_viewport_parent->remove_child(scene_viewport_base);
	right_vbox->add_child(scene_viewport_base);

	Node3DEditor::get_singleton()->set_viewports_reattaching(false);
}

void ShaderEditor::detach_scene_viewport() {
	if (!scene_viewport_base) {
		return;
	}

	Node3DEditor::get_singleton()->set_viewports_reattaching(true);

	right_vbox->remove_child(scene_viewport_base);
	scene_viewport_parent->add_child(scene_viewport_base);
	scene_viewport_parent->move_child(scene_viewport_base, scene_viewport_pos);
	scene_viewport_base->show();

	Node3DEditor::get_singleton()->set_viewports_reattaching(false);
	scene_viewport_base = nullptr;
}

void ShaderEditor::_apply_to_preview() {
	material->set_shader(shader);
	for (Map<int, MeshInstance3D *>::Element *E = mesh_instances.front(); E; E = E->next()) {
		E->get()->set_material_override(material);
	}
}

void ShaderEditor::apply_shaders() {
	if (shader.is_valid()) {
		String shader_code = shader->get_code();
		String editor_code = shader_editor->get_text_editor()->get_text();
		if (shader_code != editor_code) {
			_apply_to_preview();
			shader->set_code(editor_code);
			shader->set_edited(true);
		}
	}
}

void ShaderEditor::_text_edit_gui_input(const Ref<InputEvent> &ev) {
	Ref<InputEventMouseButton> mb = ev;

	if (mb.is_valid()) {
		if (mb->get_button_index() == BUTTON_RIGHT && mb->is_pressed()) {
			int col, row;
			CodeEdit *tx = shader_editor->get_text_editor();
			tx->_get_mouse_pos(mb->get_global_position() - tx->get_global_position(), row, col);
			tx->set_right_click_moves_caret(EditorSettings::get_singleton()->get("text_editor/cursor/right_click_moves_caret"));

			if (tx->is_right_click_moving_caret()) {
				if (tx->is_selection_active()) {
					int from_line = tx->get_selection_from_line();
					int to_line = tx->get_selection_to_line();
					int from_column = tx->get_selection_from_column();
					int to_column = tx->get_selection_to_column();

					if (row < from_line || row > to_line || (row == from_line && col < from_column) || (row == to_line && col > to_column)) {
						// Right click is outside the selected text
						tx->deselect();
					}
				}
				if (!tx->is_selection_active()) {
					tx->cursor_set_line(row, true, false);
					tx->cursor_set_column(col);
				}
			}
			_make_context_menu(tx->is_selection_active(), get_local_mouse_position());
		}
	}

	Ref<InputEventKey> k = ev;
	if (k.is_valid() && k->is_pressed() && k->get_keycode() == KEY_MENU) {
		CodeEdit *tx = shader_editor->get_text_editor();
		_make_context_menu(tx->is_selection_active(), (get_global_transform().inverse() * tx->get_global_transform()).xform(tx->_get_cursor_pixel_pos()));
		context_menu->grab_focus();
	}
}

void ShaderEditor::_update_bookmark_list() {
	bookmarks_menu->clear();

	bookmarks_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/toggle_bookmark"), BOOKMARK_TOGGLE);
	bookmarks_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/remove_all_bookmarks"), BOOKMARK_REMOVE_ALL);
	bookmarks_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/goto_next_bookmark"), BOOKMARK_GOTO_NEXT);
	bookmarks_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/goto_previous_bookmark"), BOOKMARK_GOTO_PREV);

	Array bookmark_list = shader_editor->get_text_editor()->get_bookmarked_lines();
	if (bookmark_list.size() == 0) {
		return;
	}

	bookmarks_menu->add_separator();

	for (int i = 0; i < bookmark_list.size(); i++) {
		String line = shader_editor->get_text_editor()->get_line(bookmark_list[i]).strip_edges();
		// Limit the size of the line if too big.
		if (line.length() > 50) {
			line = line.substr(0, 50);
		}

		bookmarks_menu->add_item(String::num((int)bookmark_list[i] + 1) + " - \"" + line + "\"");
		bookmarks_menu->set_item_metadata(bookmarks_menu->get_item_count() - 1, bookmark_list[i]);
	}
}

void ShaderEditor::_bookmark_item_pressed(int p_idx) {
	if (p_idx < 4) { // Any item before the separator.
		_menu_option(bookmarks_menu->get_item_id(p_idx));
	} else {
		shader_editor->goto_line(bookmarks_menu->get_item_metadata(p_idx));
	}
}

void ShaderEditor::_shader_list_item_selected(int p_which, bool p_filtered) {
	String id;
	if (!p_filtered) {
		id = shader_list->get_item_metadata(p_which);
	} else {
		id = shader_filtered_list->get_item_metadata(p_which);
	}
	edit(recent_shaders_map[id]);
}

void ShaderEditor::_make_context_menu(bool p_selection, Vector2 p_position) {
	context_menu->clear();
	if (p_selection) {
		context_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/cut"), EDIT_CUT);
		context_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/copy"), EDIT_COPY);
	}

	context_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/paste"), EDIT_PASTE);
	context_menu->add_separator();
	context_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/select_all"), EDIT_SELECT_ALL);
	context_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/undo"), EDIT_UNDO);
	context_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/redo"), EDIT_REDO);

	context_menu->add_separator();
	context_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/indent_left"), EDIT_INDENT_LEFT);
	context_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/indent_right"), EDIT_INDENT_RIGHT);
	context_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/toggle_comment"), EDIT_TOGGLE_COMMENT);
	context_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/toggle_bookmark"), BOOKMARK_TOGGLE);

	context_menu->set_position(get_global_transform().xform(p_position));
	context_menu->set_size(Vector2(1, 1));
	context_menu->popup();
}

ShaderEditor::ShaderEditor(EditorNode *p_node) {
	shader_editor_singleton = this;

	shader_editor = memnew(ShaderTextEditor);
	shader_editor->set_h_size_flags(SIZE_EXPAND_FILL);
	shader_editor->set_v_size_flags(SIZE_EXPAND_FILL);
	shader_editor->add_theme_constant_override("separation", 0);
	shader_editor->set_anchors_and_offsets_preset(Control::PRESET_WIDE);

	shader_editor->connect("script_changed", callable_mp(this, &ShaderEditor::apply_shaders));
	EditorSettings::get_singleton()->connect("settings_changed", callable_mp(this, &ShaderEditor::_editor_settings_changed));

	shader_editor->get_text_editor()->set_callhint_settings(
			EditorSettings::get_singleton()->get("text_editor/completion/put_callhint_tooltip_below_current_line"),
			EditorSettings::get_singleton()->get("text_editor/completion/callhint_tooltip_offset"));

	shader_editor->get_text_editor()->set_select_identifiers_on_hover(true);
	shader_editor->get_text_editor()->set_context_menu_enabled(false);
	shader_editor->get_text_editor()->connect("gui_input", callable_mp(this, &ShaderEditor::_text_edit_gui_input));
	shader_editor->show_toggle_scripts_button();

	shader_editor->update_editor_settings();

	context_menu = memnew(PopupMenu);
	add_child(context_menu);
	context_menu->connect("id_pressed", callable_mp(this, &ShaderEditor::_menu_option));

	VBoxContainer *main_container = memnew(VBoxContainer);
	HBoxContainer *hbc = memnew(HBoxContainer);

	file_menu = memnew(MenuButton);
	file_menu->set_shortcut_context(this);
	file_menu->set_text(TTR("File"));
	file_menu->set_switch_on_hover(true);
	file_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/close", TTR("Close"), KEY_MASK_CMD | KEY_W), FILE_CLOSE);
	file_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/close_all", TTR("Close All")), FILE_CLOSE_ALL);
	file_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/close_other_tabs", TTR("Close Other Tabs")), FILE_CLOSE_OTHER_TABS);
	file_menu->get_popup()->add_separator();
	file_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/toggle_scripts_panel", TTR("Toggle Shaders Panel"), KEY_MASK_CMD | KEY_BACKSLASH), TOGGLE_SHADERS_PANEL);
	file_menu->get_popup()->connect("id_pressed", callable_mp(this, &ShaderEditor::_menu_option));

	edit_menu = memnew(MenuButton);
	edit_menu->set_shortcut_context(this);
	edit_menu->set_text(TTR("Edit"));
	edit_menu->set_switch_on_hover(true);

	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/undo"), EDIT_UNDO);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/redo"), EDIT_REDO);
	edit_menu->get_popup()->add_separator();
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/cut"), EDIT_CUT);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/copy"), EDIT_COPY);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/paste"), EDIT_PASTE);
	edit_menu->get_popup()->add_separator();
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/select_all"), EDIT_SELECT_ALL);
	edit_menu->get_popup()->add_separator();
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/move_up"), EDIT_MOVE_LINE_UP);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/move_down"), EDIT_MOVE_LINE_DOWN);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/indent_left"), EDIT_INDENT_LEFT);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/indent_right"), EDIT_INDENT_RIGHT);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/delete_line"), EDIT_DELETE_LINE);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/toggle_comment"), EDIT_TOGGLE_COMMENT);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/clone_down"), EDIT_CLONE_DOWN);
	edit_menu->get_popup()->add_separator();
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/complete_symbol"), EDIT_COMPLETE);
	edit_menu->get_popup()->connect("id_pressed", callable_mp(this, &ShaderEditor::_menu_option));

	search_menu = memnew(MenuButton);
	search_menu->set_shortcut_context(this);
	search_menu->set_text(TTR("Search"));
	search_menu->set_switch_on_hover(true);

	search_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/find"), SEARCH_FIND);
	search_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/find_next"), SEARCH_FIND_NEXT);
	search_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/find_previous"), SEARCH_FIND_PREV);
	search_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/replace"), SEARCH_REPLACE);
	search_menu->get_popup()->connect("id_pressed", callable_mp(this, &ShaderEditor::_menu_option));

	MenuButton *goto_menu = memnew(MenuButton);
	goto_menu->set_shortcut_context(this);
	goto_menu->set_text(TTR("Go To"));
	goto_menu->set_switch_on_hover(true);
	goto_menu->get_popup()->connect("id_pressed", callable_mp(this, &ShaderEditor::_menu_option));

	goto_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/goto_line"), SEARCH_GOTO_LINE);
	goto_menu->get_popup()->add_separator();

	bookmarks_menu = memnew(PopupMenu);
	bookmarks_menu->set_name("Bookmarks");
	goto_menu->get_popup()->add_child(bookmarks_menu);
	goto_menu->get_popup()->add_submenu_item(TTR("Bookmarks"), "Bookmarks");
	_update_bookmark_list();
	bookmarks_menu->connect("about_to_popup", callable_mp(this, &ShaderEditor::_update_bookmark_list));
	bookmarks_menu->connect("index_pressed", callable_mp(this, &ShaderEditor::_bookmark_item_pressed));

	help_menu = memnew(MenuButton);
	help_menu->set_text(TTR("Help"));
	help_menu->set_switch_on_hover(true);
	help_menu->get_popup()->add_icon_item(p_node->get_gui_base()->get_theme_icon("Instance", "EditorIcons"), TTR("Online Docs"), HELP_DOCS);
	help_menu->get_popup()->connect("id_pressed", callable_mp(this, &ShaderEditor::_menu_option));

	/////////////////////////////////////////////

	add_child(main_container);

	main_splitbox = memnew(HSplitContainer);
	main_splitbox->set_v_size_flags(SIZE_EXPAND_FILL);
	main_splitbox->set_split_offset(600 * EDSCALE);
	main_container->add_child(main_splitbox);

	left_vbox = memnew(VBoxContainer);
	main_splitbox->add_child(left_vbox);

	right_vbox = memnew(VBoxContainer);
	main_splitbox->add_child(right_vbox);

	hbc->add_child(file_menu);
	hbc->add_child(search_menu);
	hbc->add_child(edit_menu);
	hbc->add_child(goto_menu);
	hbc->add_child(help_menu);
	hbc->add_theme_style_override("panel", p_node->get_gui_base()->get_theme_stylebox("ScriptEditorPanel", "EditorStyles"));
	left_vbox->add_child(hbc);

	HBoxContainer *hbc2 = memnew(HBoxContainer);
	right_vbox->add_child(hbc2);
	hbc2->add_theme_style_override("panel", p_node->get_gui_base()->get_theme_stylebox("ScriptEditorPanel", "EditorStyles"));

	spatial_hbox = memnew(HBoxContainer);
	hbc2->add_child(spatial_hbox);

	Ref<ButtonGroup> preview_btn_group;
	preview_btn_group.instance();

	scene_button = memnew(Button);
	spatial_hbox->add_child(scene_button);
	scene_button->set_button_group(preview_btn_group);
	scene_button->set_toggle_mode(true);
	scene_button->set_text(TTR("Scene"));
	scene_button->set_tooltip(TTR("Show scene viewport."));
	scene_button->set_pressed(true);
	scene_button->connect("pressed", callable_mp(this, &ShaderEditor::_preview_3d_option), varray(NODE3D_SCENE));

	spatial_hbox->add_child(memnew(VSeparator));

	quad_button = memnew(Button);
	spatial_hbox->add_child(quad_button);
	quad_button->set_button_group(preview_btn_group);
	quad_button->set_toggle_mode(true);
	quad_button->set_tooltip(TTR("Show quad mesh in the viewport."));
	quad_button->connect("pressed", callable_mp(this, &ShaderEditor::_preview_3d_option), varray(NODE3D_QUAD));

	sphere_button = memnew(Button);
	spatial_hbox->add_child(sphere_button);
	sphere_button->set_button_group(preview_btn_group);
	sphere_button->set_toggle_mode(true);
	sphere_button->set_tooltip(TTR("Show sphere mesh in the viewport."));
	sphere_button->connect("pressed", callable_mp(this, &ShaderEditor::_preview_3d_option), varray(NODE3D_SPHERE));

	box_button = memnew(Button);
	spatial_hbox->add_child(box_button);
	box_button->set_button_group(preview_btn_group);
	box_button->set_toggle_mode(true);
	box_button->set_tooltip(TTR("Show box mesh in the viewport."));
	box_button->connect("pressed", callable_mp(this, &ShaderEditor::_preview_3d_option), varray(NODE3D_BOX));

	cylinder_button = memnew(Button);
	spatial_hbox->add_child(cylinder_button);
	cylinder_button->set_button_group(preview_btn_group);
	cylinder_button->set_toggle_mode(true);
	cylinder_button->set_tooltip(TTR("Show cylinder mesh in the viewport."));
	cylinder_button->connect("pressed", callable_mp(this, &ShaderEditor::_preview_3d_option), varray(NODE3D_CYLINDER));

	spatial_hbox->add_child(memnew(VSeparator));
	hbc2->add_spacer();

	left_splitbox = memnew(HSplitContainer);
	left_splitbox->set_v_size_flags(SIZE_EXPAND_FILL);
	left_vbox->add_child(left_splitbox);

	shaders_vbox = memnew(VBoxContainer);
	shaders_vbox->set_v_size_flags(SIZE_EXPAND_FILL);

	filter_shaders = memnew(LineEdit);
	filter_shaders->set_placeholder(TTR("Filter shaders"));
	filter_shaders->set_clear_button_enabled(true);
	filter_shaders->connect("text_changed", callable_mp(this, &ShaderEditor::_filter_shaders_text_changed));
	shaders_vbox->add_child(filter_shaders);

	shader_list = memnew(ItemList);
	shaders_vbox->add_child(shader_list);
	shader_list->set_custom_minimum_size(Size2(150, 60) * EDSCALE); //need to give a bit of limit to avoid it from disappearing
	shader_list->set_v_size_flags(SIZE_EXPAND_FILL);
	shader_list->connect("item_selected", callable_mp(this, &ShaderEditor::_shader_list_item_selected), varray(false));

	shader_filtered_list = memnew(ItemList);
	shaders_vbox->add_child(shader_filtered_list);
	shader_filtered_list->set_custom_minimum_size(Size2(150, 60) * EDSCALE); //need to give a bit of limit to avoid it from disappearing
	shader_filtered_list->set_v_size_flags(SIZE_EXPAND_FILL);
	shader_filtered_list->connect("item_selected", callable_mp(this, &ShaderEditor::_shader_list_item_selected), varray(true));
	shader_filtered_list->hide();

	left_splitbox->add_child(shaders_vbox);
	left_splitbox->add_child(shader_editor);

	local_viewport_base = memnew(SubViewportContainer);
	local_viewport_base->set_stretch(true);
	right_vbox->add_child(local_viewport_base);
	local_viewport_base->set_custom_minimum_size(Size2(250, 200) * EDSCALE);
	local_viewport_base->set_v_size_flags(SIZE_EXPAND_FILL);
	local_viewport_base->hide();

	material.instance();

	/////////////////////////////////////////////

	local_viewport = memnew(SubViewport);
	local_viewport_base->add_child(local_viewport);

	Ref<World3D> world_3d;
	world_3d.instance();
	world_3d->set_environment(Node3DEditor::get_singleton()->get_viewport_environment());
	local_viewport->set_world_3d(world_3d); //use own world
	local_viewport->set_handle_input_locally(true);
	local_viewport->set_transparent_background(true);

	camera = memnew(Camera3D);
	camera->set_transform(Transform(Basis(), Vector3(0, 0, 10)));
	camera->set_perspective(45, 0.01, 1000.0);
	camera->make_current();
	local_viewport->add_child(camera);

	light1 = memnew(DirectionalLight3D);
	light1->set_transform(Transform().looking_at(Vector3(-1, -1, -1), Vector3(0, 1, 0)));
	local_viewport->add_child(light1);

	light2 = memnew(DirectionalLight3D);
	light2->set_transform(Transform().looking_at(Vector3(0, 1, 0), Vector3(0, 0, 1)));
	light2->set_color(Color(0.7, 0.7, 0.7));
	local_viewport->add_child(light2);

	quad_mesh_instance = memnew(MeshInstance3D);
	local_viewport->add_child(quad_mesh_instance);

	sphere_mesh_instance = memnew(MeshInstance3D);
	local_viewport->add_child(sphere_mesh_instance);

	box_mesh_instance = memnew(MeshInstance3D);
	local_viewport->add_child(box_mesh_instance);

	cylinder_mesh_instance = memnew(MeshInstance3D);
	local_viewport->add_child(cylinder_mesh_instance);

	quad_mesh.instance();
	quad_mesh_instance->set_mesh(quad_mesh);
	quad_mesh_instance->hide();
	mesh_instances.insert(NODE3D_QUAD, quad_mesh_instance);

	Transform initial_transform;
	initial_transform.basis.rotate(Vector3(1, 0, 0), Math::deg2rad(45.0));
	initial_transform.basis = initial_transform.basis * Basis().rotated(Vector3(0, 1, 0), Math::deg2rad(-45.0));
	initial_transform.basis.scale(Vector3(0.8, 0.8, 0.8));
	initial_transform.origin.y = 0.2;

	sphere_mesh.instance();
	sphere_mesh_instance->set_transform(initial_transform);
	sphere_mesh_instance->set_mesh(sphere_mesh);
	sphere_mesh_instance->hide();
	mesh_instances.insert(NODE3D_SPHERE, sphere_mesh_instance);

	box_mesh.instance();
	box_mesh_instance->set_transform(initial_transform);
	box_mesh_instance->set_mesh(box_mesh);
	box_mesh_instance->hide();
	mesh_instances.insert(NODE3D_BOX, box_mesh_instance);

	cylinder_mesh.instance();
	cylinder_mesh_instance->set_transform(initial_transform);
	cylinder_mesh_instance->set_mesh(cylinder_mesh);
	cylinder_mesh_instance->hide();
	mesh_instances.insert(NODE3D_CYLINDER, cylinder_mesh_instance);

	/////////////////////////////////////////////

	goto_line_dialog = memnew(GotoLineDialog);
	add_child(goto_line_dialog);

	disk_changed = memnew(ConfirmationDialog);

	VBoxContainer *vbc = memnew(VBoxContainer);
	disk_changed->add_child(vbc);

	Label *dl = memnew(Label);
	dl->set_text(TTR("This shader has been modified on on disk.\nWhat action should be taken?"));
	vbc->add_child(dl);

	disk_changed->connect("confirmed", callable_mp(this, &ShaderEditor::_reload_shader_from_disk));
	disk_changed->get_ok_button()->set_text(TTR("Reload"));

	disk_changed->add_button(TTR("Resave"), !DisplayServer::get_singleton()->get_swap_cancel_ok(), "resave");
	disk_changed->connect("custom_action", callable_mp(this, &ShaderEditor::save_external_data));

	add_child(disk_changed);

	_editor_settings_changed();
}

void ShaderEditorPlugin::edit(Object *p_object) {
	Shader *s = Object::cast_to<Shader>(p_object);
	shader_editor->edit(s);
}

bool ShaderEditorPlugin::handles(Object *p_object) const {
	Shader *shader = Object::cast_to<Shader>(p_object);
	return shader != nullptr && shader->is_text_shader();
}

void ShaderEditorPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		shader_editor->attach_scene_viewport();
		shader_editor->show();
		shader_editor->set_process(true);
		shader_editor->ensure_select_current();
	} else {
		shader_editor->detach_scene_viewport();
		shader_editor->hide();
		shader_editor->set_process(false);
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
	shader_editor = memnew(ShaderEditor(p_node));
	editor->get_main_control()->add_child(shader_editor);
	shader_editor->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	shader_editor->hide();
}

ShaderEditorPlugin::~ShaderEditorPlugin() {
}
