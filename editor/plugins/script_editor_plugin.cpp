/*************************************************************************/
/*  script_editor_plugin.cpp                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
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
#include "script_editor_plugin.h"

#include "editor/editor_node.h"
#include "editor/editor_settings.h"
#include "editor/script_editor_debugger.h"
#include "global_config.h"
#include "io/resource_loader.h"
#include "io/resource_saver.h"
#include "os/file_access.h"
#include "os/input.h"
#include "os/keyboard.h"
#include "os/keyboard.h"
#include "os/os.h"
#include "scene/main/viewport.h"

/*** SCRIPT EDITOR ****/

void ScriptEditorBase::_bind_methods() {

	ADD_SIGNAL(MethodInfo("name_changed"));
	ADD_SIGNAL(MethodInfo("request_help_search", PropertyInfo(Variant::STRING, "topic")));
	ADD_SIGNAL(MethodInfo("request_open_script_at_line", PropertyInfo(Variant::OBJECT, "script"), PropertyInfo(Variant::INT, "line")));
	ADD_SIGNAL(MethodInfo("request_save_history"));
	ADD_SIGNAL(MethodInfo("go_to_help", PropertyInfo(Variant::STRING, "what")));
}

static bool _can_open_in_editor(Script *p_script) {

	String path = p_script->get_path();

	if (path.find("::") != -1) {
		//refuse handling this if it can't be edited

		bool valid = false;
		for (int i = 0; i < EditorNode::get_singleton()->get_editor_data().get_edited_scene_count(); i++) {
			if (path.begins_with(EditorNode::get_singleton()->get_editor_data().get_scene_path(i))) {
				valid = true;
				break;
			}
		}

		return valid;
	}

	return true;
}

class EditorScriptCodeCompletionCache : public ScriptCodeCompletionCache {

	struct Cache {
		uint64_t time_loaded;
		RES cache;
	};

	Map<String, Cache> cached;

public:
	uint64_t max_time_cache;
	int max_cache_size;

	void cleanup() {

		List<Map<String, Cache>::Element *> to_clean;

		Map<String, Cache>::Element *I = cached.front();
		while (I) {
			if ((OS::get_singleton()->get_ticks_msec() - I->get().time_loaded) > max_time_cache) {
				to_clean.push_back(I);
			}
			I = I->next();
		}

		while (to_clean.front()) {
			cached.erase(to_clean.front()->get());
			to_clean.pop_front();
		}
	}

	RES get_cached_resource(const String &p_path) {

		Map<String, Cache>::Element *E = cached.find(p_path);
		if (!E) {

			Cache c;
			c.cache = ResourceLoader::load(p_path);
			E = cached.insert(p_path, c);
		}

		E->get().time_loaded = OS::get_singleton()->get_ticks_msec();

		if (cached.size() > max_cache_size) {
			uint64_t older;
			Map<String, Cache>::Element *O = cached.front();
			older = O->get().time_loaded;
			Map<String, Cache>::Element *I = O;
			while (I) {
				if (I->get().time_loaded < older) {
					older = I->get().time_loaded;
					O = I;
				}
				I = I->next();
			}

			if (O != E) { //should never heppane..
				cached.erase(O);
			}
		}

		return E->get().cache;
	}

	EditorScriptCodeCompletionCache() {

		max_cache_size = 128;
		max_time_cache = 5 * 60 * 1000; //minutes, five
	}
};

#define SORT_SCRIPT_LIST

void ScriptEditorQuickOpen::popup(const Vector<String> &p_functions, bool p_dontclear) {

	popup_centered_ratio(0.6);
	if (p_dontclear)
		search_box->select_all();
	else
		search_box->clear();
	search_box->grab_focus();
	functions = p_functions;
	_update_search();
}

void ScriptEditorQuickOpen::_text_changed(const String &p_newtext) {

	_update_search();
}

void ScriptEditorQuickOpen::_sbox_input(const InputEvent &p_ie) {

	if (p_ie.type == InputEvent::KEY && (p_ie.key.scancode == KEY_UP ||
												p_ie.key.scancode == KEY_DOWN ||
												p_ie.key.scancode == KEY_PAGEUP ||
												p_ie.key.scancode == KEY_PAGEDOWN)) {

		search_options->call("_gui_input", p_ie);
		search_box->accept_event();
	}
}

void ScriptEditorQuickOpen::_update_search() {

	search_options->clear();
	TreeItem *root = search_options->create_item();

	for (int i = 0; i < functions.size(); i++) {

		String file = functions[i];
		if ((search_box->get_text() == "" || file.findn(search_box->get_text()) != -1)) {

			TreeItem *ti = search_options->create_item(root);
			ti->set_text(0, file);
			if (root->get_children() == ti)
				ti->select(0);
		}
	}

	get_ok()->set_disabled(root->get_children() == NULL);
}

void ScriptEditorQuickOpen::_confirmed() {

	TreeItem *ti = search_options->get_selected();
	if (!ti)
		return;
	int line = ti->get_text(0).get_slice(":", 1).to_int();

	emit_signal("goto_line", line - 1);
	hide();
}

void ScriptEditorQuickOpen::_notification(int p_what) {

	if (p_what == NOTIFICATION_ENTER_TREE) {

		connect("confirmed", this, "_confirmed");
	}
}

void ScriptEditorQuickOpen::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_text_changed"), &ScriptEditorQuickOpen::_text_changed);
	ClassDB::bind_method(D_METHOD("_confirmed"), &ScriptEditorQuickOpen::_confirmed);
	ClassDB::bind_method(D_METHOD("_sbox_input"), &ScriptEditorQuickOpen::_sbox_input);

	ADD_SIGNAL(MethodInfo("goto_line", PropertyInfo(Variant::INT, "line")));
}

ScriptEditorQuickOpen::ScriptEditorQuickOpen() {

	VBoxContainer *vbc = memnew(VBoxContainer);
	add_child(vbc);
	//set_child_rect(vbc);
	search_box = memnew(LineEdit);
	vbc->add_margin_child(TTR("Search:"), search_box);
	search_box->connect("text_changed", this, "_text_changed");
	search_box->connect("gui_input", this, "_sbox_input");
	search_options = memnew(Tree);
	vbc->add_margin_child(TTR("Matches:"), search_options, true);
	get_ok()->set_text(TTR("Open"));
	get_ok()->set_disabled(true);
	register_text_enter(search_box);
	set_hide_on_ok(false);
	search_options->connect("item_activated", this, "_confirmed");
	search_options->set_hide_root(true);
}

/////////////////////////////////

ScriptEditor *ScriptEditor::script_editor = NULL;

/*** SCRIPT EDITOR ******/

String ScriptEditor::_get_debug_tooltip(const String &p_text, Node *_se) {

	//ScriptEditorBase *se=_se->cast_to<ScriptEditorBase>();

	String val = debugger->get_var_value(p_text);
	if (val != String()) {
		return p_text + ": " + val;
	} else {

		return String();
	}
}

void ScriptEditor::_breaked(bool p_breaked, bool p_can_debug) {

	if (bool(EditorSettings::get_singleton()->get("text_editor/external/use_external_editor"))) {
		return;
	}

	debug_menu->get_popup()->set_item_disabled(debug_menu->get_popup()->get_item_index(DEBUG_NEXT), !(p_breaked && p_can_debug));
	debug_menu->get_popup()->set_item_disabled(debug_menu->get_popup()->get_item_index(DEBUG_STEP), !(p_breaked && p_can_debug));
	debug_menu->get_popup()->set_item_disabled(debug_menu->get_popup()->get_item_index(DEBUG_BREAK), p_breaked);
	debug_menu->get_popup()->set_item_disabled(debug_menu->get_popup()->get_item_index(DEBUG_CONTINUE), !p_breaked);

	for (int i = 0; i < tab_container->get_child_count(); i++) {

		ScriptEditorBase *se = tab_container->get_child(i)->cast_to<ScriptEditorBase>();
		if (!se) {

			continue;
		}

		se->set_debugger_active(p_breaked);
	}
}

void ScriptEditor::_show_debugger(bool p_show) {

	//debug_menu->get_popup()->set_item_checked( debug_menu->get_popup()->get_item_index(DEBUG_SHOW), p_show);
}

void ScriptEditor::_script_created(Ref<Script> p_script) {
	editor->push_item(p_script.operator->());
}

void ScriptEditor::_goto_script_line2(int p_line) {

	int selected = tab_container->get_current_tab();
	if (selected < 0 || selected >= tab_container->get_child_count())
		return;

	ScriptEditorBase *current = tab_container->get_child(selected)->cast_to<ScriptEditorBase>();
	if (!current)
		return;

	current->goto_line(p_line);
}

void ScriptEditor::_goto_script_line(REF p_script, int p_line) {

	editor->push_item(p_script.ptr());

	int selected = tab_container->get_current_tab();
	if (selected < 0 || selected >= tab_container->get_child_count())
		return;

	ScriptEditorBase *current = tab_container->get_child(selected)->cast_to<ScriptEditorBase>();
	if (!current)
		return;

	current->goto_line(p_line, true);
}

void ScriptEditor::_update_history_arrows() {

	script_back->set_disabled(history_pos <= 0);
	script_forward->set_disabled(history_pos >= history.size() - 1);
}

void ScriptEditor::_save_history() {

	if (history_pos >= 0 && history_pos < history.size() && history[history_pos].control == tab_container->get_current_tab_control()) {

		Node *n = tab_container->get_current_tab_control();

		if (n->cast_to<ScriptEditorBase>()) {

			history[history_pos].state = n->cast_to<ScriptEditorBase>()->get_edit_state();
		}
		if (n->cast_to<EditorHelp>()) {

			history[history_pos].state = n->cast_to<EditorHelp>()->get_scroll();
		}
	}

	history.resize(history_pos + 1);
	ScriptHistory sh;
	sh.control = tab_container->get_current_tab_control();
	sh.state = Variant();

	history.push_back(sh);
	history_pos++;

	_update_history_arrows();
}

void ScriptEditor::_go_to_tab(int p_idx) {

	Node *cn = tab_container->get_child(p_idx);
	if (!cn)
		return;
	Control *c = cn->cast_to<Control>();
	if (!c)
		return;

	if (history_pos >= 0 && history_pos < history.size() && history[history_pos].control == tab_container->get_current_tab_control()) {

		Node *n = tab_container->get_current_tab_control();

		if (n->cast_to<ScriptEditorBase>()) {

			history[history_pos].state = n->cast_to<ScriptEditorBase>()->get_edit_state();
		}
		if (n->cast_to<EditorHelp>()) {

			history[history_pos].state = n->cast_to<EditorHelp>()->get_scroll();
		}
	}

	history.resize(history_pos + 1);
	ScriptHistory sh;
	sh.control = c;
	sh.state = Variant();

	history.push_back(sh);
	history_pos++;

	tab_container->set_current_tab(p_idx);

	c = tab_container->get_current_tab_control();

	if (c->cast_to<ScriptEditorBase>()) {

		script_name_label->set_text(c->cast_to<ScriptEditorBase>()->get_name());
		script_icon->set_texture(c->cast_to<ScriptEditorBase>()->get_icon());
		if (is_visible_in_tree())
			c->cast_to<ScriptEditorBase>()->ensure_focus();
	}
	if (c->cast_to<EditorHelp>()) {

		script_name_label->set_text(c->cast_to<EditorHelp>()->get_class());
		script_icon->set_texture(get_icon("Help", "EditorIcons"));
		if (is_visible_in_tree())
			c->cast_to<EditorHelp>()->set_focused();
	}

	c->set_meta("__editor_pass", ++edit_pass);
	_update_history_arrows();
	_update_script_colors();
	_update_selected_editor_menu();
}

void ScriptEditor::_close_tab(int p_idx, bool p_save) {

	int selected = p_idx;
	if (selected < 0 || selected >= tab_container->get_child_count())
		return;

	Node *tselected = tab_container->get_child(selected);
	ScriptEditorBase *current = tab_container->get_child(selected)->cast_to<ScriptEditorBase>();
	if (current) {
		if (p_save) {
			apply_scripts();
		}
		if (current->get_edit_menu()) {
			memdelete(current->get_edit_menu());
		}
	}

	//remove from history
	history.resize(history_pos + 1);

	for (int i = 0; i < history.size(); i++) {
		if (history[i].control == tselected) {
			history.remove(i);
			i--;
			history_pos--;
		}
	}

	if (history_pos >= history.size()) {
		history_pos = history.size() - 1;
	}

	int idx = tab_container->get_current_tab();
	memdelete(tselected);
	if (idx >= tab_container->get_child_count())
		idx = tab_container->get_child_count() - 1;
	if (idx >= 0) {

		if (history_pos >= 0) {
			idx = history[history_pos].control->get_index();
		}
		tab_container->set_current_tab(idx);

		//script_list->select(idx);
	}

	_update_history_arrows();

	_update_script_names();
	_save_layout();
}

void ScriptEditor::_close_current_tab() {

	_close_tab(tab_container->get_current_tab());
}

void ScriptEditor::_close_discard_current_tab(const String &p_str) {
	_close_tab(tab_container->get_current_tab(), false);
	erase_tab_confirm->hide();
}

void ScriptEditor::_close_docs_tab() {

	int child_count = tab_container->get_child_count();
	for (int i = child_count - 1; i >= 0; i--) {

		EditorHelp *se = tab_container->get_child(i)->cast_to<EditorHelp>();

		if (se) {
			_close_tab(i);
		}
	}
}

void ScriptEditor::_close_all_tabs() {

	int child_count = tab_container->get_child_count();
	for (int i = child_count - 1; i >= 0; i--) {

		tab_container->set_current_tab(i);
		ScriptEditorBase *se = tab_container->get_child(i)->cast_to<ScriptEditorBase>();

		if (se) {

			// Maybe there are unsaved changes
			if (se->is_unsaved()) {
				_ask_close_current_unsaved_tab(se);
				continue;
			}
		}

		_close_current_tab();
	}
}

void ScriptEditor::_ask_close_current_unsaved_tab(ScriptEditorBase *current) {
	erase_tab_confirm->set_text("Close and save changes?\n\"" + current->get_name() + "\"");
	erase_tab_confirm->popup_centered_minsize();
}

void ScriptEditor::_resave_scripts(const String &p_str) {

	apply_scripts();

	for (int i = 0; i < tab_container->get_child_count(); i++) {

		ScriptEditorBase *se = tab_container->get_child(i)->cast_to<ScriptEditorBase>();
		if (!se)
			continue;

		Ref<Script> script = se->get_edited_script();

		if (script->get_path() == "" || script->get_path().find("local://") != -1 || script->get_path().find("::") != -1)
			continue; //internal script, who cares

		if (trim_trailing_whitespace_on_save) {
			se->trim_trailing_whitespace();
		}
		editor->save_resource(script);
		se->tag_saved_version();
	}

	disk_changed->hide();
}

void ScriptEditor::_reload_scripts() {

	for (int i = 0; i < tab_container->get_child_count(); i++) {

		ScriptEditorBase *se = tab_container->get_child(i)->cast_to<ScriptEditorBase>();
		if (!se) {

			continue;
		}

		Ref<Script> script = se->get_edited_script();

		if (script->get_path() == "" || script->get_path().find("local://") != -1 || script->get_path().find("::") != -1) {

			continue; //internal script, who cares
		}

		uint64_t last_date = script->get_last_modified_time();
		uint64_t date = FileAccess::get_modified_time(script->get_path());

		//printf("last date: %lli vs date: %lli\n",last_date,date);
		if (last_date == date) {
			continue;
		}

		Ref<Script> rel_script = ResourceLoader::load(script->get_path(), script->get_class(), true);
		ERR_CONTINUE(!rel_script.is_valid());
		script->set_source_code(rel_script->get_source_code());
		script->set_last_modified_time(rel_script->get_last_modified_time());
		script->reload();
		se->reload_text();
	}

	disk_changed->hide();
	_update_script_names();
}

void ScriptEditor::_res_saved_callback(const Ref<Resource> &p_res) {

	for (int i = 0; i < tab_container->get_child_count(); i++) {

		ScriptEditorBase *se = tab_container->get_child(i)->cast_to<ScriptEditorBase>();
		if (!se) {

			continue;
		}

		Ref<Script> script = se->get_edited_script();

		if (script->get_path() == "" || script->get_path().find("local://") != -1 || script->get_path().find("::") != -1) {
			continue; //internal script, who cares
		}

		if (script == p_res) {

			se->tag_saved_version();
		}
	}

	_update_script_names();

	if (!pending_auto_reload && auto_reload_running_scripts) {
		call_deferred("_live_auto_reload_running_scripts");
		pending_auto_reload = true;
	}
}

void ScriptEditor::_live_auto_reload_running_scripts() {
	pending_auto_reload = false;
	debugger->reload_scripts();
}

bool ScriptEditor::_test_script_times_on_disk(Ref<Script> p_for_script) {

	disk_changed_list->clear();
	TreeItem *r = disk_changed_list->create_item();
	disk_changed_list->set_hide_root(true);

	bool need_ask = false;
	bool need_reload = false;
	bool use_autoreload = bool(EDITOR_DEF("text_editor/files/auto_reload_scripts_on_external_change", false));

	for (int i = 0; i < tab_container->get_child_count(); i++) {

		ScriptEditorBase *se = tab_container->get_child(i)->cast_to<ScriptEditorBase>();
		if (se) {

			Ref<Script> script = se->get_edited_script();

			if (p_for_script.is_valid() && p_for_script != script)
				continue;

			if (script->get_path() == "" || script->get_path().find("local://") != -1 || script->get_path().find("::") != -1)
				continue; //internal script, who cares

			uint64_t last_date = script->get_last_modified_time();
			uint64_t date = FileAccess::get_modified_time(script->get_path());

			//printf("last date: %lli vs date: %lli\n",last_date,date);
			if (last_date != date) {

				TreeItem *ti = disk_changed_list->create_item(r);
				ti->set_text(0, script->get_path().get_file());

				if (!use_autoreload || se->is_unsaved()) {
					need_ask = true;
				}
				need_reload = true;
				//r->set_metadata(0,);
			}
		}
	}

	if (need_reload) {
		if (!need_ask) {
			script_editor->_reload_scripts();
			need_reload = false;
		} else {
			disk_changed->call_deferred("popup_centered_ratio", 0.5);
		}
	}

	return need_reload;
}

void ScriptEditor::_file_dialog_action(String p_file) {

	switch (file_dialog_option) {
		case FILE_SAVE_THEME_AS: {
			if (!EditorSettings::get_singleton()->save_text_editor_theme_as(p_file)) {
				editor->show_warning(TTR("Error while saving theme"), TTR("Error saving"));
			}
		} break;
		case FILE_IMPORT_THEME: {
			if (!EditorSettings::get_singleton()->import_text_editor_theme(p_file)) {
				editor->show_warning(TTR("Error importing theme"), TTR("Error importing"));
			}
		} break;
	}
	file_dialog_option = -1;
}

void ScriptEditor::_menu_option(int p_option) {

	switch (p_option) {
		case FILE_NEW: {
			script_create_dialog->config("Node", ".gd");
			script_create_dialog->popup_centered(Size2(300, 300) * EDSCALE);
		} break;
		case FILE_OPEN: {

			editor->open_resource("Script");
			return;
		} break;
		case FILE_SAVE_ALL: {

			if (_test_script_times_on_disk())
				return;

			save_all_scripts();
		} break;
		case FILE_IMPORT_THEME: {
			file_dialog->set_mode(EditorFileDialog::MODE_OPEN_FILE);
			file_dialog->set_access(EditorFileDialog::ACCESS_FILESYSTEM);
			file_dialog_option = FILE_IMPORT_THEME;
			file_dialog->clear_filters();
			file_dialog->add_filter("*.tet");
			file_dialog->popup_centered_ratio();
			file_dialog->set_title(TTR("Import Theme"));
		} break;
		case FILE_RELOAD_THEME: {
			EditorSettings::get_singleton()->load_text_editor_theme();
		} break;
		case FILE_SAVE_THEME: {
			if (!EditorSettings::get_singleton()->save_text_editor_theme()) {
				editor->show_warning(TTR("Error while saving theme"), TTR("Error saving"));
			}
		} break;
		case FILE_SAVE_THEME_AS: {
			file_dialog->set_mode(EditorFileDialog::MODE_SAVE_FILE);
			file_dialog->set_access(EditorFileDialog::ACCESS_FILESYSTEM);
			file_dialog_option = FILE_SAVE_THEME_AS;
			file_dialog->clear_filters();
			file_dialog->add_filter("*.tet");
			file_dialog->set_current_path(EditorSettings::get_singleton()->get_settings_path() + "/text_editor_themes/" + EditorSettings::get_singleton()->get("text_editor/theme/color_theme"));
			file_dialog->popup_centered_ratio();
			file_dialog->set_title(TTR("Save Theme As.."));
		} break;
		case SEARCH_HELP: {

			help_search_dialog->popup();
		} break;
		case SEARCH_CLASSES: {

			String current;

			if (tab_container->get_tab_count() > 0) {
				EditorHelp *eh = tab_container->get_child(tab_container->get_current_tab())->cast_to<EditorHelp>();
				if (eh) {
					current = eh->get_class();
				}
			}

			help_index->popup();

			if (current != "") {
				help_index->call_deferred("select_class", current);
			}
		} break;
		case SEARCH_WEBSITE: {

			OS::get_singleton()->shell_open("http://docs.godotengine.org/");
		} break;

		case WINDOW_NEXT: {

			_history_forward();
		} break;
		case WINDOW_PREV: {
			_history_back();
		} break;
		case DEBUG_SHOW: {
			if (debugger) {
				bool visible = debug_menu->get_popup()->is_item_checked(debug_menu->get_popup()->get_item_index(DEBUG_SHOW));
				debug_menu->get_popup()->set_item_checked(debug_menu->get_popup()->get_item_index(DEBUG_SHOW), !visible);
				if (visible)
					debugger->hide();
				else
					debugger->show();
			}
		} break;
		case DEBUG_SHOW_KEEP_OPEN: {
			bool visible = debug_menu->get_popup()->is_item_checked(debug_menu->get_popup()->get_item_index(DEBUG_SHOW_KEEP_OPEN));
			if (debugger)
				debugger->set_hide_on_stop(visible);
			debug_menu->get_popup()->set_item_checked(debug_menu->get_popup()->get_item_index(DEBUG_SHOW_KEEP_OPEN), !visible);
		} break;
	}

	int selected = tab_container->get_current_tab();
	if (selected < 0 || selected >= tab_container->get_child_count())
		return;

	ScriptEditorBase *current = tab_container->get_child(selected)->cast_to<ScriptEditorBase>();
	if (current) {

		switch (p_option) {
			case FILE_NEW: {
				script_create_dialog->config("Node", ".gd");
				script_create_dialog->popup_centered(Size2(300, 300) * EDSCALE);
			} break;
			case FILE_SAVE: {

				if (_test_script_times_on_disk())
					return;

				if (trim_trailing_whitespace_on_save)
					current->trim_trailing_whitespace();
				editor->save_resource(current->get_edited_script());

			} break;
			case FILE_SAVE_AS: {

				current->trim_trailing_whitespace();
				editor->push_item(current->get_edited_script()->cast_to<Object>());
				editor->save_resource_as(current->get_edited_script());

			} break;

			case FILE_TOOL_RELOAD:
			case FILE_TOOL_RELOAD_SOFT: {

				current->reload(p_option == FILE_TOOL_RELOAD_SOFT);

			} break;

			case FILE_CLOSE: {
				if (current->is_unsaved()) {
					_ask_close_current_unsaved_tab(current);
				} else {
					_close_current_tab();
				}
			} break;
			case CLOSE_DOCS: {
				_close_docs_tab();
			} break;
			case CLOSE_ALL: {
				_close_all_tabs();
			} break;
			case DEBUG_NEXT: {

				if (debugger)
					debugger->debug_next();
			} break;
			case DEBUG_STEP: {

				if (debugger)
					debugger->debug_step();

			} break;
			case DEBUG_BREAK: {

				if (debugger)
					debugger->debug_break();

			} break;
			case DEBUG_CONTINUE: {

				if (debugger)
					debugger->debug_continue();

			} break;
			case WINDOW_MOVE_LEFT: {

				if (tab_container->get_current_tab() > 0) {
					tab_container->call_deferred("set_current_tab", tab_container->get_current_tab() - 1);
					script_list->call_deferred("select", tab_container->get_current_tab() - 1);
					tab_container->move_child(current, tab_container->get_current_tab() - 1);
					_update_script_names();
				}
			} break;
			case WINDOW_MOVE_RIGHT: {

				if (tab_container->get_current_tab() < tab_container->get_child_count() - 1) {
					tab_container->call_deferred("set_current_tab", tab_container->get_current_tab() + 1);
					script_list->call_deferred("select", tab_container->get_current_tab() + 1);
					tab_container->move_child(current, tab_container->get_current_tab() + 1);
					_update_script_names();
				}

			} break;

			default: {

				if (p_option >= WINDOW_SELECT_BASE) {

					tab_container->set_current_tab(p_option - WINDOW_SELECT_BASE);
					script_list->select(p_option - WINDOW_SELECT_BASE);
				}
			}
		}
	}

	EditorHelp *help = tab_container->get_current_tab_control()->cast_to<EditorHelp>();
	if (help) {

		switch (p_option) {

			case HELP_SEARCH_FIND: {
				help->popup_search();
			} break;
			case HELP_SEARCH_FIND_NEXT: {
				help->search_again();
			} break;
			case FILE_CLOSE: {
				_close_current_tab();
			} break;
			case CLOSE_DOCS: {
				_close_docs_tab();
			} break;
			case CLOSE_ALL: {
				_close_all_tabs();
			} break;
		}
	}
}

void ScriptEditor::_tab_changed(int p_which) {

	ensure_select_current();
}

void ScriptEditor::_notification(int p_what) {

	if (p_what == NOTIFICATION_ENTER_TREE) {

		editor->connect("play_pressed", this, "_editor_play");
		editor->connect("pause_pressed", this, "_editor_pause");
		editor->connect("stop_pressed", this, "_editor_stop");
		editor->connect("script_add_function_request", this, "_add_callback");
		editor->connect("resource_saved", this, "_res_saved_callback");
		script_list->connect("item_selected", this, "_script_selected");
		script_split->connect("dragged", this, "_script_split_dragged");
		autosave_timer->connect("timeout", this, "_autosave_scripts");
		{
			float autosave_time = EditorSettings::get_singleton()->get("text_editor/files/autosave_interval_secs");
			if (autosave_time > 0) {
				autosave_timer->set_wait_time(autosave_time);
				autosave_timer->start();
			} else {
				autosave_timer->stop();
			}
		}

		EditorSettings::get_singleton()->connect("settings_changed", this, "_editor_settings_changed");
		help_search->set_icon(get_icon("Help", "EditorIcons"));
		site_search->set_icon(get_icon("Godot", "EditorIcons"));
		class_search->set_icon(get_icon("ClassList", "EditorIcons"));

		script_forward->set_icon(get_icon("Forward", "EditorIcons"));
		script_back->set_icon(get_icon("Back", "EditorIcons"));
	}

	if (p_what == NOTIFICATION_READY) {

		get_tree()->connect("tree_changed", this, "_tree_changed");
		editor->connect("request_help", this, "_request_help");
	}

	if (p_what == NOTIFICATION_EXIT_TREE) {

		editor->disconnect("play_pressed", this, "_editor_play");
		editor->disconnect("pause_pressed", this, "_editor_pause");
		editor->disconnect("stop_pressed", this, "_editor_stop");
	}

	if (p_what == MainLoop::NOTIFICATION_WM_FOCUS_IN) {

		_test_script_times_on_disk();
		_update_modified_scripts_for_external_editor();
	}

	if (p_what == NOTIFICATION_PROCESS) {
	}
}

bool ScriptEditor::can_take_away_focus() const {

	int selected = tab_container->get_current_tab();
	if (selected < 0 || selected >= tab_container->get_child_count())
		return true;

	ScriptEditorBase *current = tab_container->get_child(selected)->cast_to<ScriptEditorBase>();
	if (!current)
		return true;

	return current->can_lose_focus_on_node_selection();
}

void ScriptEditor::close_builtin_scripts_from_scene(const String &p_scene) {

	for (int i = 0; i < tab_container->get_child_count(); i++) {

		ScriptEditorBase *se = tab_container->get_child(i)->cast_to<ScriptEditorBase>();

		if (se) {

			Ref<Script> script = se->get_edited_script();
			if (!script.is_valid())
				continue;

			if (script->get_path().find("::") != -1 && script->get_path().begins_with(p_scene)) { //is an internal script and belongs to scene being closed
				_close_tab(i);
				i--;
			}
		}
	}
}

void ScriptEditor::edited_scene_changed() {

	_update_modified_scripts_for_external_editor();
}

static const Node *_find_node_with_script(const Node *p_node, const RefPtr &p_script) {

	if (p_node->get_script() == p_script)
		return p_node;

	for (int i = 0; i < p_node->get_child_count(); i++) {

		const Node *result = _find_node_with_script(p_node->get_child(i), p_script);
		if (result)
			return result;
	}

	return NULL;
}

Dictionary ScriptEditor::get_state() const {

	//apply_scripts();

	Dictionary state;
#if 0
	Array paths;
	int open=-1;

	for(int i=0;i<tab_container->get_child_count();i++) {

		ScriptTextEditor *se = tab_container->get_child(i)->cast_to<ScriptTextEditor>();
		if (!se)
			continue;


		Ref<Script> script = se->get_edited_script();
		if (script->get_path()!="" && script->get_path().find("local://")==-1 && script->get_path().find("::")==-1) {

			paths.push_back(script->get_path());
		} else {


			const Node *owner = _find_node_with_script(get_tree()->get_root(),script.get_ref_ptr());
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

#endif
	return state;
}
void ScriptEditor::set_state(const Dictionary &p_state) {

#if 0
	print_line("attempt set state: "+String(Variant(p_state)));

	if (!p_state.has("sources"))
		return; //bleh

	Array sources = p_state["sources"];
	for(int i=0;i<sources.size();i++) {

		Variant source=sources[i];

		Ref<Script> script;

		if (source.get_type()==Variant::NODE_PATH) {


			Node *owner=get_tree()->get_root()->get_node(source);
			if (!owner)
				continue;

			script = owner->get_script();
		} else if (source.get_type()==Variant::STRING) {


			script = ResourceLoader::load(source,"Script");
		}


		if (script.is_null()) //ah well..
			continue;

		editor->call("_resource_selected",script);
	}

	if (p_state.has("current")) {
		tab_container->set_current_tab(p_state["current"]);
	}
#endif
}
void ScriptEditor::clear() {
#if 0
	List<ScriptTextEditor*> stes;
	for(int i=0;i<tab_container->get_child_count();i++) {

		ScriptTextEditor *se = tab_container->get_child(i)->cast_to<ScriptTextEditor>();
		if (!se)
			continue;
		stes.push_back(se);

	}

	while(stes.size()) {

		memdelete(stes.front()->get());
		stes.pop_front();
	}

	int idx = tab_container->get_current_tab();
	if (idx>=tab_container->get_child_count())
		idx=tab_container->get_child_count()-1;
	if (idx>=0) {
		tab_container->set_current_tab(idx);
		script_list->select( script_list->find_metadata(idx) );
	}

#endif
}

void ScriptEditor::get_breakpoints(List<String> *p_breakpoints) {

	for (int i = 0; i < tab_container->get_child_count(); i++) {

		ScriptEditorBase *se = tab_container->get_child(i)->cast_to<ScriptEditorBase>();
		if (!se)
			continue;

		List<int> bpoints;
		se->get_breakpoints(&bpoints);
		Ref<Script> script = se->get_edited_script();
		String base = script->get_path();
		ERR_CONTINUE(base.begins_with("local://") || base == "");

		for (List<int>::Element *E = bpoints.front(); E; E = E->next()) {

			p_breakpoints->push_back(base + ":" + itos(E->get() + 1));
		}
	}
}

void ScriptEditor::ensure_focus_current() {

	if (!is_inside_tree())
		return;

	int cidx = tab_container->get_current_tab();
	if (cidx < 0 || cidx >= tab_container->get_tab_count())
		;
	Control *c = tab_container->get_child(cidx)->cast_to<Control>();
	if (!c)
		return;
	ScriptEditorBase *se = c->cast_to<ScriptEditorBase>();
	if (!se)
		return;
	se->ensure_focus();
}

void ScriptEditor::_script_selected(int p_idx) {

	grab_focus_block = !Input::get_singleton()->is_mouse_button_pressed(1); //amazing hack, simply amazing

	_go_to_tab(script_list->get_item_metadata(p_idx));
	grab_focus_block = false;
}

void ScriptEditor::ensure_select_current() {

	if (tab_container->get_child_count() && tab_container->get_current_tab() >= 0) {

		Node *current = tab_container->get_child(tab_container->get_current_tab());

		ScriptEditorBase *se = current->cast_to<ScriptEditorBase>();
		if (se) {

			Ref<Script> script = se->get_edited_script();

			if (!grab_focus_block && is_visible_in_tree())
				se->ensure_focus();

			//edit_menu->show();
			//search_menu->show();
		}

		EditorHelp *eh = current->cast_to<EditorHelp>();

		if (eh) {
			//edit_menu->hide();
			//search_menu->hide();
			//script_search_menu->show();
		}
	}

	_update_selected_editor_menu();
}

void ScriptEditor::_find_scripts(Node *p_base, Node *p_current, Set<Ref<Script> > &used) {
	if (p_current != p_base && p_current->get_owner() != p_base)
		return;

	if (p_current->get_script_instance()) {
		Ref<Script> scr = p_current->get_script();
		if (scr.is_valid())
			used.insert(scr);
	}

	for (int i = 0; i < p_current->get_child_count(); i++) {
		_find_scripts(p_base, p_current->get_child(i), used);
	}
}

struct _ScriptEditorItemData {

	String name;
	String sort_key;
	Ref<Texture> icon;
	int index;
	String tooltip;
	bool used;
	int category;

	bool operator<(const _ScriptEditorItemData &id) const {

		return category == id.category ? sort_key < id.sort_key : category < id.category;
	}
};

void ScriptEditor::_update_script_colors() {

	bool script_temperature_enabled = EditorSettings::get_singleton()->get("text_editor/open_scripts/script_temperature_enabled");
	bool highlight_current = EditorSettings::get_singleton()->get("text_editor/open_scripts/highlight_current_script");

	int hist_size = EditorSettings::get_singleton()->get("text_editor/open_scripts/script_temperature_history_size");
	Color hot_color = EditorSettings::get_singleton()->get("text_editor/open_scripts/script_temperature_hot_color");
	Color cold_color = EditorSettings::get_singleton()->get("text_editor/open_scripts/script_temperature_cold_color");

	for (int i = 0; i < script_list->get_item_count(); i++) {

		int c = script_list->get_item_metadata(i);
		Node *n = tab_container->get_child(c);
		if (!n)
			continue;

		script_list->set_item_custom_bg_color(i, Color(0, 0, 0, 0));

		bool current = tab_container->get_current_tab() == c;
		if (current && highlight_current) {
			script_list->set_item_custom_bg_color(i, EditorSettings::get_singleton()->get("text_editor/open_scripts/current_script_background_color"));

		} else if (script_temperature_enabled) {

			if (!n->has_meta("__editor_pass")) {
				continue;
			}

			int pass = n->get_meta("__editor_pass");
			int h = edit_pass - pass;
			if (h > hist_size) {
				continue;
			}
			int non_zero_hist_size = (hist_size == 0) ? 1 : hist_size;
			float v = Math::ease((edit_pass - pass) / float(non_zero_hist_size), 0.4);

			script_list->set_item_custom_bg_color(i, hot_color.linear_interpolate(cold_color, v));
		}
	}
}

void ScriptEditor::_update_script_names() {

	if (restoring_layout)
		return;

	waiting_update_names = false;
	Set<Ref<Script> > used;
	Node *edited = EditorNode::get_singleton()->get_edited_scene();
	if (edited) {
		_find_scripts(edited, edited, used);
	}

	script_list->clear();
	bool split_script_help = EditorSettings::get_singleton()->get("text_editor/open_scripts/group_help_pages");
	ScriptSortBy sort_by = (ScriptSortBy)(int)EditorSettings::get_singleton()->get("text_editor/open_scripts/sort_scripts_by");
	ScriptListName display_as = (ScriptListName)(int)EditorSettings::get_singleton()->get("text_editor/open_scripts/list_script_names_as");

	Vector<_ScriptEditorItemData> sedata;

	for (int i = 0; i < tab_container->get_child_count(); i++) {

		ScriptEditorBase *se = tab_container->get_child(i)->cast_to<ScriptEditorBase>();
		if (se) {

			String name = se->get_name();
			Ref<Texture> icon = se->get_icon();
			String path = se->get_edited_script()->get_path();

			_ScriptEditorItemData sd;
			sd.icon = icon;
			sd.name = name;
			sd.tooltip = path;
			sd.index = i;
			sd.used = used.has(se->get_edited_script());
			sd.category = 0;

			switch (sort_by) {
				case SORT_BY_NAME: {
					sd.sort_key = name.to_lower();
				} break;
				case SORT_BY_PATH: {
					sd.sort_key = path;
				} break;
			}

			switch (display_as) {
				case DISPLAY_NAME: {
					sd.name = name;
				} break;
				case DISPLAY_DIR_AND_NAME: {
					if (!path.get_base_dir().get_file().empty()) {
						sd.name = path.get_base_dir().get_file() + "/" + name;
					} else {
						sd.name = name;
					}
				} break;
				case DISPLAY_FULL_PATH: {
					sd.name = path;
				} break;
			}

			sedata.push_back(sd);
		}

		EditorHelp *eh = tab_container->get_child(i)->cast_to<EditorHelp>();
		if (eh) {

			String name = eh->get_class();
			Ref<Texture> icon = get_icon("Help", "EditorIcons");
			String tooltip = name + " Class Reference";

			_ScriptEditorItemData sd;
			sd.icon = icon;
			sd.name = name;
			sd.sort_key = name;
			sd.tooltip = tooltip;
			sd.index = i;
			sd.used = false;
			sd.category = split_script_help ? 1 : 0;
			sedata.push_back(sd);
		}
	}

	sedata.sort();

	for (int i = 0; i < sedata.size(); i++) {

		script_list->add_item(sedata[i].name, sedata[i].icon);
		int index = script_list->get_item_count() - 1;
		script_list->set_item_tooltip(index, sedata[i].tooltip);
		script_list->set_item_metadata(index, sedata[i].index);
		if (sedata[i].used) {

			script_list->set_item_custom_bg_color(index, Color(88 / 255.0, 88 / 255.0, 60 / 255.0));
		}
		if (tab_container->get_current_tab() == sedata[i].index) {
			script_list->select(index);
			script_name_label->set_text(sedata[i].name);
			script_icon->set_texture(sedata[i].icon);
		}
	}

	_update_script_colors();
}

void ScriptEditor::edit(const Ref<Script> &p_script, bool p_grab_focus) {

	if (p_script.is_null())
		return;

	// refuse to open built-in if scene is not loaded

	// see if already has it

	bool open_dominant = EditorSettings::get_singleton()->get("text_editor/files/open_dominant_script_on_scene_change");

	if (p_script->get_path().is_resource_file() && bool(EditorSettings::get_singleton()->get("text_editor/external/use_external_editor"))) {

		String path = EditorSettings::get_singleton()->get("text_editor/external/exec_path");
		String flags = EditorSettings::get_singleton()->get("text_editor/external/exec_flags");
		List<String> args;
		flags = flags.strip_edges();
		if (flags != String()) {
			Vector<String> flagss = flags.split(" ", false);
			for (int i = 0; i < flagss.size(); i++)
				args.push_back(flagss[i]);
		}

		args.push_back(GlobalConfig::get_singleton()->globalize_path(p_script->get_path()));
		Error err = OS::get_singleton()->execute(path, args, false);
		if (err == OK)
			return;
		WARN_PRINT("Couldn't open external text editor, using internal");
	}

	for (int i = 0; i < tab_container->get_child_count(); i++) {

		ScriptEditorBase *se = tab_container->get_child(i)->cast_to<ScriptEditorBase>();
		if (!se)
			continue;

		if (se->get_edited_script() == p_script) {

			if (open_dominant || !EditorNode::get_singleton()->is_changing_scene()) {
				if (tab_container->get_current_tab() != i) {
					_go_to_tab(i);
					script_list->select(script_list->find_metadata(i));
				}
				if (is_visible_in_tree())
					se->ensure_focus();
			}
			return;
		}
	}

	// doesn't have it, make a new one

	ScriptEditorBase *se;

	for (int i = script_editor_func_count - 1; i >= 0; i--) {
		se = script_editor_funcs[i](p_script);
		if (se)
			break;
	}
	ERR_FAIL_COND(!se);
	tab_container->add_child(se);

	se->set_edited_script(p_script);
	se->set_tooltip_request_func("_get_debug_tooltip", this);
	if (se->get_edit_menu()) {
		se->get_edit_menu()->hide();
		menu_hb->add_child(se->get_edit_menu());
		menu_hb->move_child(se->get_edit_menu(), 1);
	}

	if (p_grab_focus) {
		_go_to_tab(tab_container->get_tab_count() - 1);
	}

	_update_script_names();
	_save_layout();
	se->connect("name_changed", this, "_update_script_names");
	se->connect("request_help_search", this, "_help_search");
	se->connect("request_open_script_at_line", this, "_goto_script_line");
	se->connect("go_to_help", this, "_help_class_goto");
	se->connect("request_save_history", this, "_save_history");

	//test for modification, maybe the script was not edited but was loaded

	_test_script_times_on_disk(p_script);
	_update_modified_scripts_for_external_editor(p_script);
}

void ScriptEditor::save_all_scripts() {

	for (int i = 0; i < tab_container->get_child_count(); i++) {

		ScriptEditorBase *se = tab_container->get_child(i)->cast_to<ScriptEditorBase>();
		if (!se)
			continue;

		if (!se->is_unsaved())
			continue;

		if (trim_trailing_whitespace_on_save) {
			se->trim_trailing_whitespace();
		}

		Ref<Script> script = se->get_edited_script();
		if (script.is_valid())
			se->apply_code();

		if (script->get_path() != "" && script->get_path().find("local://") == -1 && script->get_path().find("::") == -1) {
			//external script, save it

			editor->save_resource(script);
			//ResourceSaver::save(script->get_path(),script);
		}
	}

	_update_script_names();
}

void ScriptEditor::apply_scripts() const {

	for (int i = 0; i < tab_container->get_child_count(); i++) {

		ScriptEditorBase *se = tab_container->get_child(i)->cast_to<ScriptEditorBase>();
		if (!se)
			continue;
		se->apply_code();
	}
}

void ScriptEditor::_editor_play() {

	debugger->start();
	debug_menu->get_popup()->grab_focus();
	debug_menu->get_popup()->set_item_disabled(debug_menu->get_popup()->get_item_index(DEBUG_NEXT), true);
	debug_menu->get_popup()->set_item_disabled(debug_menu->get_popup()->get_item_index(DEBUG_STEP), true);
	debug_menu->get_popup()->set_item_disabled(debug_menu->get_popup()->get_item_index(DEBUG_BREAK), false);
	debug_menu->get_popup()->set_item_disabled(debug_menu->get_popup()->get_item_index(DEBUG_CONTINUE), true);

	//debugger_gui->start_listening(Globals::get_singleton()->get("debug/debug_port"));
}

void ScriptEditor::_editor_pause() {
}
void ScriptEditor::_editor_stop() {

	debugger->stop();
	debug_menu->get_popup()->set_item_disabled(debug_menu->get_popup()->get_item_index(DEBUG_NEXT), true);
	debug_menu->get_popup()->set_item_disabled(debug_menu->get_popup()->get_item_index(DEBUG_STEP), true);
	debug_menu->get_popup()->set_item_disabled(debug_menu->get_popup()->get_item_index(DEBUG_BREAK), true);
	debug_menu->get_popup()->set_item_disabled(debug_menu->get_popup()->get_item_index(DEBUG_CONTINUE), true);

	for (int i = 0; i < tab_container->get_child_count(); i++) {

		ScriptEditorBase *se = tab_container->get_child(i)->cast_to<ScriptEditorBase>();
		if (!se) {

			continue;
		}

		se->set_debugger_active(false);
	}
}

void ScriptEditor::_add_callback(Object *p_obj, const String &p_function, const PoolStringArray &p_args) {

	//print_line("add callback! hohoho"); kinda sad to remove this
	ERR_FAIL_COND(!p_obj);
	Ref<Script> script = p_obj->get_script();
	ERR_FAIL_COND(!script.is_valid());

	editor->push_item(script.ptr());

	for (int i = 0; i < tab_container->get_child_count(); i++) {

		ScriptEditorBase *se = tab_container->get_child(i)->cast_to<ScriptEditorBase>();
		if (!se)
			continue;
		if (se->get_edited_script() != script)
			continue;

		se->add_callback(p_function, p_args);

		_go_to_tab(i);

		script_list->select(script_list->find_metadata(i));

		break;
	}
}

void ScriptEditor::_save_layout() {

	if (restoring_layout) {
		return;
	}

	editor->save_layout();
}

void ScriptEditor::_editor_settings_changed() {

	trim_trailing_whitespace_on_save = EditorSettings::get_singleton()->get("text_editor/files/trim_trailing_whitespace_on_save");
	float autosave_time = EditorSettings::get_singleton()->get("text_editor/files/autosave_interval_secs");
	if (autosave_time > 0) {
		autosave_timer->set_wait_time(autosave_time);
		autosave_timer->start();
	} else {
		autosave_timer->stop();
	}

	if (current_theme == "") {
		current_theme = EditorSettings::get_singleton()->get("text_editor/theme/color_theme");
	} else if (current_theme != EditorSettings::get_singleton()->get("text_editor/theme/color_theme")) {
		current_theme = EditorSettings::get_singleton()->get("text_editor/theme/color_theme");
		EditorSettings::get_singleton()->load_text_editor_theme();
	}

	for (int i = 0; i < tab_container->get_child_count(); i++) {

		ScriptEditorBase *se = tab_container->get_child(i)->cast_to<ScriptEditorBase>();
		if (!se)
			continue;

		se->update_settings();
	}
	_update_script_colors();
	_update_script_names();

	ScriptServer::set_reload_scripts_on_save(EDITOR_DEF("text_editor/files/auto_reload_and_parse_scripts_on_save", true));
}

void ScriptEditor::_autosave_scripts() {

	save_all_scripts();
}

void ScriptEditor::_tree_changed() {

	if (waiting_update_names)
		return;

	waiting_update_names = true;
	call_deferred("_update_script_names");
}

void ScriptEditor::_script_split_dragged(float) {

	_save_layout();
}

void ScriptEditor::_unhandled_input(const InputEvent &p_event) {
	if (p_event.key.pressed || !is_visible_in_tree()) return;
	if (ED_IS_SHORTCUT("script_editor/next_script", p_event)) {
		int next_tab = script_list->get_current() + 1;
		next_tab %= script_list->get_item_count();
		_go_to_tab(script_list->get_item_metadata(next_tab));
		_update_script_names();
	}
	if (ED_IS_SHORTCUT("script_editor/prev_script", p_event)) {
		int next_tab = script_list->get_current() - 1;
		next_tab = next_tab >= 0 ? next_tab : script_list->get_item_count() - 1;
		_go_to_tab(script_list->get_item_metadata(next_tab));
		_update_script_names();
	}
}

void ScriptEditor::set_window_layout(Ref<ConfigFile> p_layout) {

	if (!bool(EDITOR_DEF("text_editor/files/restore_scripts_on_load", true))) {
		return;
	}

	if (!p_layout->has_section_key("ScriptEditor", "open_scripts") && !p_layout->has_section_key("ScriptEditor", "open_help"))
		return;

	Array scripts = p_layout->get_value("ScriptEditor", "open_scripts");
	Array helps;
	if (p_layout->has_section_key("ScriptEditor", "open_help"))
		helps = p_layout->get_value("ScriptEditor", "open_help");

	restoring_layout = true;

	for (int i = 0; i < scripts.size(); i++) {

		String path = scripts[i];
		if (!FileAccess::exists(path))
			continue;
		Ref<Script> scr = ResourceLoader::load(path);
		if (scr.is_valid()) {
			edit(scr);
		}
	}

	for (int i = 0; i < helps.size(); i++) {

		String path = helps[i];
		_help_class_open(path);
	}

	for (int i = 0; i < tab_container->get_child_count(); i++) {
		tab_container->get_child(i)->set_meta("__editor_pass", Variant());
	}

	if (p_layout->has_section_key("ScriptEditor", "split_offset")) {
		script_split->set_split_offset(p_layout->get_value("ScriptEditor", "split_offset"));
	}

	restoring_layout = false;

	_update_script_names();
}

void ScriptEditor::get_window_layout(Ref<ConfigFile> p_layout) {

	Array scripts;
	Array helps;

	for (int i = 0; i < tab_container->get_child_count(); i++) {

		ScriptEditorBase *se = tab_container->get_child(i)->cast_to<ScriptEditorBase>();
		if (se) {

			String path = se->get_edited_script()->get_path();
			if (!path.is_resource_file())
				continue;

			scripts.push_back(path);
		}

		EditorHelp *eh = tab_container->get_child(i)->cast_to<EditorHelp>();

		if (eh) {

			helps.push_back(eh->get_class());
		}
	}

	p_layout->set_value("ScriptEditor", "open_scripts", scripts);
	p_layout->set_value("ScriptEditor", "open_help", helps);
	p_layout->set_value("ScriptEditor", "split_offset", script_split->get_split_offset());
}

void ScriptEditor::_help_class_open(const String &p_class) {

	if (p_class == "")
		return;

	for (int i = 0; i < tab_container->get_child_count(); i++) {

		EditorHelp *eh = tab_container->get_child(i)->cast_to<EditorHelp>();

		if (eh && eh->get_class() == p_class) {

			_go_to_tab(i);
			_update_script_names();
			return;
		}
	}

	EditorHelp *eh = memnew(EditorHelp);

	eh->set_name(p_class);
	tab_container->add_child(eh);
	_go_to_tab(tab_container->get_tab_count() - 1);
	eh->go_to_class(p_class, 0);
	eh->connect("go_to_help", this, "_help_class_goto");
	_update_script_names();
	_save_layout();
}

void ScriptEditor::_help_class_goto(const String &p_desc) {

	String cname = p_desc.get_slice(":", 1);

	for (int i = 0; i < tab_container->get_child_count(); i++) {

		EditorHelp *eh = tab_container->get_child(i)->cast_to<EditorHelp>();

		if (eh && eh->get_class() == cname) {

			_go_to_tab(i);
			eh->go_to_help(p_desc);
			_update_script_names();
			return;
		}
	}

	EditorHelp *eh = memnew(EditorHelp);

	eh->set_name(cname);
	tab_container->add_child(eh);
	_go_to_tab(tab_container->get_tab_count() - 1);
	eh->go_to_help(p_desc);
	eh->connect("go_to_help", this, "_help_class_goto");
	_update_script_names();
	_save_layout();
}

void ScriptEditor::_update_selected_editor_menu() {

	for (int i = 0; i < tab_container->get_child_count(); i++) {

		bool current = tab_container->get_current_tab() == i;

		ScriptEditorBase *se = tab_container->get_child(i)->cast_to<ScriptEditorBase>();
		if (se && se->get_edit_menu()) {

			if (current)
				se->get_edit_menu()->show();
			else
				se->get_edit_menu()->hide();
		}
	}

	EditorHelp *eh = tab_container->get_current_tab_control()->cast_to<EditorHelp>();
	if (eh) {
		script_search_menu->show();
	} else {
		script_search_menu->hide();
	}
}

void ScriptEditor::_update_history_pos(int p_new_pos) {

	Node *n = tab_container->get_current_tab_control();

	if (n->cast_to<ScriptEditorBase>()) {

		history[history_pos].state = n->cast_to<ScriptEditorBase>()->get_edit_state();
	}
	if (n->cast_to<EditorHelp>()) {

		history[history_pos].state = n->cast_to<EditorHelp>()->get_scroll();
	}

	history_pos = p_new_pos;
	tab_container->set_current_tab(history[history_pos].control->get_index());

	n = history[history_pos].control;

	if (n->cast_to<ScriptEditorBase>()) {

		n->cast_to<ScriptEditorBase>()->set_edit_state(history[history_pos].state);
		n->cast_to<ScriptEditorBase>()->ensure_focus();
	}

	if (n->cast_to<EditorHelp>()) {

		n->cast_to<EditorHelp>()->set_scroll(history[history_pos].state);
		n->cast_to<EditorHelp>()->set_focused();
	}

	n->set_meta("__editor_pass", ++edit_pass);
	_update_script_names();
	_update_history_arrows();
	_update_selected_editor_menu();
}

void ScriptEditor::_history_forward() {

	if (history_pos < history.size() - 1) {
		_update_history_pos(history_pos + 1);
	}
}

void ScriptEditor::_history_back() {

	if (history_pos > 0) {
		_update_history_pos(history_pos - 1);
	}
}
void ScriptEditor::set_scene_root_script(Ref<Script> p_script) {

	bool open_dominant = EditorSettings::get_singleton()->get("text_editor/files/open_dominant_script_on_scene_change");

	if (bool(EditorSettings::get_singleton()->get("text_editor/external/use_external_editor")))
		return;

	if (open_dominant && p_script.is_valid() && _can_open_in_editor(p_script.ptr())) {
		edit(p_script);
	}
}

bool ScriptEditor::script_go_to_method(Ref<Script> p_script, const String &p_method) {

	for (int i = 0; i < tab_container->get_child_count(); i++) {
		ScriptEditorBase *current = tab_container->get_child(i)->cast_to<ScriptEditorBase>();

		if (current && current->get_edited_script() == p_script) {
			if (current->goto_method(p_method)) {
				edit(p_script);
				return true;
			}
			break;
		}
	}
	return false;
}

void ScriptEditor::set_live_auto_reload_running_scripts(bool p_enabled) {

	auto_reload_running_scripts = p_enabled;
}

void ScriptEditor::_help_search(String p_text) {
	help_search_dialog->popup(p_text);
}

void ScriptEditor::_open_script_request(const String &p_path) {

	Ref<Script> script = ResourceLoader::load(p_path);
	if (script.is_valid()) {
		script_editor->edit(script, false);
	}
}

int ScriptEditor::script_editor_func_count = 0;
CreateScriptEditorFunc ScriptEditor::script_editor_funcs[ScriptEditor::SCRIPT_EDITOR_FUNC_MAX];

void ScriptEditor::register_create_script_editor_function(CreateScriptEditorFunc p_func) {

	ERR_FAIL_COND(script_editor_func_count == SCRIPT_EDITOR_FUNC_MAX);
	script_editor_funcs[script_editor_func_count++] = p_func;
}

void ScriptEditor::_bind_methods() {

	ClassDB::bind_method("_file_dialog_action", &ScriptEditor::_file_dialog_action);
	ClassDB::bind_method("_tab_changed", &ScriptEditor::_tab_changed);
	ClassDB::bind_method("_menu_option", &ScriptEditor::_menu_option);
	ClassDB::bind_method("_close_current_tab", &ScriptEditor::_close_current_tab);
	ClassDB::bind_method("_close_discard_current_tab", &ScriptEditor::_close_discard_current_tab);
	ClassDB::bind_method("_close_docs_tab", &ScriptEditor::_close_docs_tab);
	ClassDB::bind_method("_close_all_tabs", &ScriptEditor::_close_all_tabs);
	ClassDB::bind_method("_editor_play", &ScriptEditor::_editor_play);
	ClassDB::bind_method("_editor_pause", &ScriptEditor::_editor_pause);
	ClassDB::bind_method("_editor_stop", &ScriptEditor::_editor_stop);
	ClassDB::bind_method("_add_callback", &ScriptEditor::_add_callback);
	ClassDB::bind_method("_reload_scripts", &ScriptEditor::_reload_scripts);
	ClassDB::bind_method("_resave_scripts", &ScriptEditor::_resave_scripts);
	ClassDB::bind_method("_res_saved_callback", &ScriptEditor::_res_saved_callback);
	ClassDB::bind_method("_goto_script_line", &ScriptEditor::_goto_script_line);
	ClassDB::bind_method("_goto_script_line2", &ScriptEditor::_goto_script_line2);
	ClassDB::bind_method("_help_search", &ScriptEditor::_help_search);
	ClassDB::bind_method("_save_history", &ScriptEditor::_save_history);

	ClassDB::bind_method("_breaked", &ScriptEditor::_breaked);
	ClassDB::bind_method("_show_debugger", &ScriptEditor::_show_debugger);
	ClassDB::bind_method("_get_debug_tooltip", &ScriptEditor::_get_debug_tooltip);
	ClassDB::bind_method("_autosave_scripts", &ScriptEditor::_autosave_scripts);
	ClassDB::bind_method("_editor_settings_changed", &ScriptEditor::_editor_settings_changed);
	ClassDB::bind_method("_update_script_names", &ScriptEditor::_update_script_names);
	ClassDB::bind_method("_tree_changed", &ScriptEditor::_tree_changed);
	ClassDB::bind_method("_script_selected", &ScriptEditor::_script_selected);
	ClassDB::bind_method("_script_created", &ScriptEditor::_script_created);
	ClassDB::bind_method("_script_split_dragged", &ScriptEditor::_script_split_dragged);
	ClassDB::bind_method("_help_class_open", &ScriptEditor::_help_class_open);
	ClassDB::bind_method("_help_class_goto", &ScriptEditor::_help_class_goto);
	ClassDB::bind_method("_request_help", &ScriptEditor::_help_class_open);
	ClassDB::bind_method("_history_forward", &ScriptEditor::_history_forward);
	ClassDB::bind_method("_history_back", &ScriptEditor::_history_back);
	ClassDB::bind_method("_live_auto_reload_running_scripts", &ScriptEditor::_live_auto_reload_running_scripts);
	ClassDB::bind_method("_unhandled_input", &ScriptEditor::_unhandled_input);
}

ScriptEditor::ScriptEditor(EditorNode *p_editor) {

	current_theme = "";

	completion_cache = memnew(EditorScriptCodeCompletionCache);
	restoring_layout = false;
	waiting_update_names = false;
	pending_auto_reload = false;
	auto_reload_running_scripts = false;
	editor = p_editor;

	menu_hb = memnew(HBoxContainer);
	add_child(menu_hb);

	script_split = memnew(HSplitContainer);
	add_child(script_split);
	script_split->set_v_size_flags(SIZE_EXPAND_FILL);

	script_list = memnew(ItemList);
	script_split->add_child(script_list);
	script_list->set_custom_minimum_size(Size2(0, 0));
	script_split->set_split_offset(140);

	tab_container = memnew(TabContainer);
	tab_container->set_tabs_visible(false);
	script_split->add_child(tab_container);

	tab_container->set_h_size_flags(SIZE_EXPAND_FILL);

	ED_SHORTCUT("script_editor/next_script", TTR("Next script"), KEY_MASK_CMD | KEY_MASK_SHIFT | KEY_GREATER);
	ED_SHORTCUT("script_editor/prev_script", TTR("Previous script"), KEY_MASK_CMD | KEY_LESS);
	set_process_unhandled_input(true);

	file_menu = memnew(MenuButton);
	menu_hb->add_child(file_menu);
	file_menu->set_text(TTR("File"));
	file_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/new", TTR("New")), FILE_NEW);
	file_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/open", TTR("Open")), FILE_OPEN);
	file_menu->get_popup()->add_separator();
	file_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/save", TTR("Save"), KEY_MASK_ALT | KEY_MASK_CMD | KEY_S), FILE_SAVE);
	file_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/save_as", TTR("Save As..")), FILE_SAVE_AS);
	file_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/save_all", TTR("Save All"), KEY_MASK_CMD | KEY_MASK_SHIFT | KEY_MASK_ALT | KEY_S), FILE_SAVE_ALL);
	file_menu->get_popup()->add_separator();
	file_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/reload_script_soft", TTR("Soft Reload Script"), KEY_MASK_CMD | KEY_MASK_SHIFT | KEY_R), FILE_TOOL_RELOAD_SOFT);
	file_menu->get_popup()->add_separator();

	file_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/history_previous", TTR("History Prev"), KEY_MASK_ALT | KEY_LEFT), WINDOW_PREV);
	file_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/history_next", TTR("History Next"), KEY_MASK_ALT | KEY_RIGHT), WINDOW_NEXT);
	file_menu->get_popup()->add_separator();
	file_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/import_theme", TTR("Import Theme")), FILE_IMPORT_THEME);
	file_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/reload_theme", TTR("Reload Theme")), FILE_RELOAD_THEME);
	file_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/save_theme", TTR("Save Theme")), FILE_SAVE_THEME);
	file_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/save_theme_as", TTR("Save Theme As")), FILE_SAVE_THEME_AS);
	file_menu->get_popup()->add_separator();
	file_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/close_docs", TTR("Close Docs")), CLOSE_DOCS);
	file_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/close_file", TTR("Close"), KEY_MASK_CMD | KEY_W), FILE_CLOSE);
	file_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/close_all", TTR("Close All")), CLOSE_ALL);
	file_menu->get_popup()->connect("id_pressed", this, "_menu_option");

	script_search_menu = memnew(MenuButton);
	menu_hb->add_child(script_search_menu);
	script_search_menu->set_text(TTR("Search"));
	script_search_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/find", TTR("Find.."), KEY_MASK_CMD | KEY_F), HELP_SEARCH_FIND);
	script_search_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/find_next", TTR("Find Next"), KEY_F3), HELP_SEARCH_FIND_NEXT);
	script_search_menu->get_popup()->connect("id_pressed", this, "_menu_option");
	script_search_menu->hide();

	debug_menu = memnew(MenuButton);
	menu_hb->add_child(debug_menu);
	debug_menu->set_text(TTR("Debug"));
	debug_menu->get_popup()->add_separator();
	debug_menu->get_popup()->add_shortcut(ED_SHORTCUT("debugger/step_over", TTR("Step Over"), KEY_F10), DEBUG_NEXT);
	debug_menu->get_popup()->add_shortcut(ED_SHORTCUT("debugger/step_into", TTR("Step Into"), KEY_F11), DEBUG_STEP);
	debug_menu->get_popup()->add_separator();
	debug_menu->get_popup()->add_shortcut(ED_SHORTCUT("debugger/break", TTR("Break")), DEBUG_BREAK);
	debug_menu->get_popup()->add_shortcut(ED_SHORTCUT("debugger/continue", TTR("Continue")), DEBUG_CONTINUE);
	debug_menu->get_popup()->add_separator();
	//debug_menu->get_popup()->add_check_item("Show Debugger",DEBUG_SHOW);
	debug_menu->get_popup()->add_check_shortcut(ED_SHORTCUT("debugger/keep_debugger_open", TTR("Keep Debugger Open")), DEBUG_SHOW_KEEP_OPEN);
	debug_menu->get_popup()->connect("id_pressed", this, "_menu_option");

	debug_menu->get_popup()->set_item_disabled(debug_menu->get_popup()->get_item_index(DEBUG_NEXT), true);
	debug_menu->get_popup()->set_item_disabled(debug_menu->get_popup()->get_item_index(DEBUG_STEP), true);
	debug_menu->get_popup()->set_item_disabled(debug_menu->get_popup()->get_item_index(DEBUG_BREAK), true);
	debug_menu->get_popup()->set_item_disabled(debug_menu->get_popup()->get_item_index(DEBUG_CONTINUE), true);

#if 0
	window_menu = memnew( MenuButton );
	menu_hb->add_child(window_menu);
	window_menu->set_text(TTR("Window"));
	window_menu->get_popup()->add_item(TTR("Close"),WINDOW_CLOSE,KEY_MASK_CMD|KEY_W);
	window_menu->get_popup()->add_separator();
	window_menu->get_popup()->add_item(TTR("Move Left"),WINDOW_MOVE_LEFT,KEY_MASK_CMD|KEY_LEFT);
	window_menu->get_popup()->add_item(TTR("Move Right"),WINDOW_MOVE_RIGHT,KEY_MASK_CMD|KEY_RIGHT);
	window_menu->get_popup()->add_separator();
	window_menu->get_popup()->connect("id_pressed", this,"_menu_option");

#endif

	menu_hb->add_spacer();

	script_icon = memnew(TextureRect);
	menu_hb->add_child(script_icon);
	script_name_label = memnew(Label);
	menu_hb->add_child(script_name_label);

	script_icon->hide();
	script_name_label->hide();

	menu_hb->add_spacer();

	site_search = memnew(ToolButton);
	site_search->set_text(TTR("Tutorials"));
	site_search->connect("pressed", this, "_menu_option", varray(SEARCH_WEBSITE));
	menu_hb->add_child(site_search);
	site_search->set_tooltip(TTR("Open https://godotengine.org at tutorials section."));

	class_search = memnew(ToolButton);
	class_search->set_text(TTR("Classes"));
	class_search->connect("pressed", this, "_menu_option", varray(SEARCH_CLASSES));
	menu_hb->add_child(class_search);
	class_search->set_tooltip(TTR("Search the class hierarchy."));

	help_search = memnew(ToolButton);
	help_search->set_text(TTR("Search Help"));
	help_search->connect("pressed", this, "_menu_option", varray(SEARCH_HELP));
	menu_hb->add_child(help_search);
	help_search->set_tooltip(TTR("Search the reference documentation."));

	menu_hb->add_child(memnew(VSeparator));

	script_back = memnew(ToolButton);
	script_back->connect("pressed", this, "_history_back");
	menu_hb->add_child(script_back);
	script_back->set_disabled(true);
	script_back->set_tooltip(TTR("Go to previous edited document."));

	script_forward = memnew(ToolButton);
	script_forward->connect("pressed", this, "_history_forward");
	menu_hb->add_child(script_forward);
	script_forward->set_disabled(true);
	script_forward->set_tooltip(TTR("Go to next edited document."));

	tab_container->connect("tab_changed", this, "_tab_changed");

	erase_tab_confirm = memnew(ConfirmationDialog);
	erase_tab_confirm->get_ok()->set_text(TTR("Save"));
	erase_tab_confirm->add_button(TTR("Discard"), OS::get_singleton()->get_swap_ok_cancel(), "discard");
	erase_tab_confirm->connect("confirmed", this, "_close_current_tab");
	erase_tab_confirm->connect("custom_action", this, "_close_discard_current_tab");
	add_child(erase_tab_confirm);

	script_create_dialog = memnew(ScriptCreateDialog);
	script_create_dialog->set_title(TTR("Create Script"));
	add_child(script_create_dialog);
	script_create_dialog->connect("script_created", this, "_script_created");

	file_dialog_option = -1;
	file_dialog = memnew(EditorFileDialog);
	add_child(file_dialog);
	file_dialog->connect("file_selected", this, "_file_dialog_action");

	debugger = memnew(ScriptEditorDebugger(editor));
	debugger->connect("goto_script_line", this, "_goto_script_line");
	debugger->connect("show_debugger", this, "_show_debugger");

	disk_changed = memnew(ConfirmationDialog);
	{
		VBoxContainer *vbc = memnew(VBoxContainer);
		disk_changed->add_child(vbc);
		//disk_changed->set_child_rect(vbc);

		Label *dl = memnew(Label);
		dl->set_text(TTR("The following files are newer on disk.\nWhat action should be taken?:"));
		vbc->add_child(dl);

		disk_changed_list = memnew(Tree);
		vbc->add_child(disk_changed_list);
		disk_changed_list->set_v_size_flags(SIZE_EXPAND_FILL);

		disk_changed->connect("confirmed", this, "_reload_scripts");
		disk_changed->get_ok()->set_text(TTR("Reload"));

		disk_changed->add_button(TTR("Resave"), !OS::get_singleton()->get_swap_ok_cancel(), "resave");
		disk_changed->connect("custom_action", this, "_resave_scripts");
	}

	add_child(disk_changed);

	script_editor = this;

	Button *db = EditorNode::get_singleton()->add_bottom_panel_item(TTR("Debugger"), debugger);
	debugger->set_tool_button(db);

	debugger->connect("breaked", this, "_breaked");

	autosave_timer = memnew(Timer);
	autosave_timer->set_one_shot(false);
	add_child(autosave_timer);

	grab_focus_block = false;

	help_search_dialog = memnew(EditorHelpSearch);
	add_child(help_search_dialog);
	help_search_dialog->connect("go_to_help", this, "_help_class_goto");

	help_index = memnew(EditorHelpIndex);
	add_child(help_index);
	help_index->connect("open_class", this, "_help_class_open");

	history_pos = -1;
	//debugger_gui->hide();

	edit_pass = 0;
	trim_trailing_whitespace_on_save = false;

	ScriptServer::edit_request_func = _open_script_request;
}

ScriptEditor::~ScriptEditor() {

	memdelete(completion_cache);
}

void ScriptEditorPlugin::edit(Object *p_object) {

	if (!p_object->cast_to<Script>())
		return;

	script_editor->edit(p_object->cast_to<Script>());
}

bool ScriptEditorPlugin::handles(Object *p_object) const {

	if (p_object->cast_to<Script>()) {

		bool valid = _can_open_in_editor(p_object->cast_to<Script>());

		if (!valid) { //user tried to open it by clicking
			EditorNode::get_singleton()->show_warning(TTR("Built-in scripts can only be edited when the scene they belong to is loaded"));
		}
		return valid;
	}

	return p_object->is_class("Script");
}

void ScriptEditorPlugin::make_visible(bool p_visible) {

	if (p_visible) {
		script_editor->show();
		script_editor->set_process(true);
		script_editor->ensure_select_current();
	} else {

		script_editor->hide();
		script_editor->set_process(false);
	}
}

void ScriptEditorPlugin::selected_notify() {

	script_editor->ensure_select_current();
}

Dictionary ScriptEditorPlugin::get_state() const {

	return script_editor->get_state();
}

void ScriptEditorPlugin::set_state(const Dictionary &p_state) {

	script_editor->set_state(p_state);
}
void ScriptEditorPlugin::clear() {

	script_editor->clear();
}

void ScriptEditorPlugin::save_external_data() {

	script_editor->save_all_scripts();
}

void ScriptEditorPlugin::apply_changes() {

	script_editor->apply_scripts();
}

void ScriptEditorPlugin::restore_global_state() {
}

void ScriptEditorPlugin::save_global_state() {
}

void ScriptEditorPlugin::set_window_layout(Ref<ConfigFile> p_layout) {

	script_editor->set_window_layout(p_layout);
}

void ScriptEditorPlugin::get_window_layout(Ref<ConfigFile> p_layout) {

	script_editor->get_window_layout(p_layout);
}

void ScriptEditorPlugin::get_breakpoints(List<String> *p_breakpoints) {

	return script_editor->get_breakpoints(p_breakpoints);
}

void ScriptEditorPlugin::edited_scene_changed() {

	script_editor->edited_scene_changed();
}

ScriptEditorPlugin::ScriptEditorPlugin(EditorNode *p_node) {

	editor = p_node;
	script_editor = memnew(ScriptEditor(p_node));
	editor->get_viewport()->add_child(script_editor);
	script_editor->set_v_size_flags(Control::SIZE_EXPAND_FILL);

	script_editor->hide();

	EDITOR_DEF("text_editor/files/auto_reload_scripts_on_external_change", true);
	ScriptServer::set_reload_scripts_on_save(EDITOR_DEF("text_editor/files/auto_reload_and_parse_scripts_on_save", true));
	EDITOR_DEF("text_editor/files/open_dominant_script_on_scene_change", true);
	EDITOR_DEF("text_editor/external/use_external_editor", false);
	EDITOR_DEF("text_editor/external/exec_path", "");
	EDITOR_DEF("text_editor/open_scripts/script_temperature_enabled", true);
	EDITOR_DEF("text_editor/open_scripts/highlight_current_script", true);
	EDITOR_DEF("text_editor/open_scripts/script_temperature_history_size", 15);
	EDITOR_DEF("text_editor/open_scripts/script_temperature_hot_color", Color(1, 0, 0, 0.3));
	EDITOR_DEF("text_editor/open_scripts/script_temperature_cold_color", Color(0, 0, 1, 0.3));
	EDITOR_DEF("text_editor/open_scripts/current_script_background_color", Color(0.81, 0.81, 0.14, 0.63));
	EDITOR_DEF("text_editor/open_scripts/group_help_pages", true);
	EditorSettings::get_singleton()->add_property_hint(PropertyInfo(Variant::INT, "text_editor/open_scripts/sort_scripts_by", PROPERTY_HINT_ENUM, "Name,Path"));
	EDITOR_DEF("text_editor/open_scripts/sort_scripts_by", 0);
	EditorSettings::get_singleton()->add_property_hint(PropertyInfo(Variant::INT, "text_editor/open_scripts/list_script_names_as", PROPERTY_HINT_ENUM, "Name,Parent Directory And Name,Full Path"));
	EDITOR_DEF("text_editor/open_scripts/list_script_names_as", 0);
	EditorSettings::get_singleton()->add_property_hint(PropertyInfo(Variant::STRING, "text_editor/external/exec_path", PROPERTY_HINT_GLOBAL_FILE));
	EDITOR_DEF("text_editor/external/exec_flags", "");
}

ScriptEditorPlugin::~ScriptEditorPlugin() {
}
