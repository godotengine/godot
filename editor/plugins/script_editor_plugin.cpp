/*************************************************************************/
/*  script_editor_plugin.cpp                                             */
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
#include "script_editor_plugin.h"

#include "core/io/resource_loader.h"
#include "core/io/resource_saver.h"
#include "core/os/file_access.h"
#include "core/os/input.h"
#include "core/os/keyboard.h"
#include "core/os/os.h"
#include "core/project_settings.h"
#include "editor/editor_node.h"
#include "editor/editor_settings.h"
#include "editor/node_dock.h"
#include "editor/script_editor_debugger.h"
#include "scene/main/viewport.h"

/*** SCRIPT EDITOR ****/

void ScriptEditorBase::_bind_methods() {

	ADD_SIGNAL(MethodInfo("name_changed"));
	ADD_SIGNAL(MethodInfo("edited_script_changed"));
	ADD_SIGNAL(MethodInfo("request_help_search", PropertyInfo(Variant::STRING, "topic")));
	ADD_SIGNAL(MethodInfo("request_help_index"));
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

void ScriptEditorQuickOpen::_sbox_input(const Ref<InputEvent> &p_ie) {

	Ref<InputEventKey> k = p_ie;

	if (k.is_valid() && (k->get_scancode() == KEY_UP ||
								k->get_scancode() == KEY_DOWN ||
								k->get_scancode() == KEY_PAGEUP ||
								k->get_scancode() == KEY_PAGEDOWN)) {

		search_options->call("_gui_input", k);
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

	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {

			connect("confirmed", this, "_confirmed");
		} break;
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

	//ScriptEditorBase *se=Object::cast_to<ScriptEditorBase>(_se);

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

		ScriptEditorBase *se = Object::cast_to<ScriptEditorBase>(tab_container->get_child(i));
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

	ScriptEditorBase *current = Object::cast_to<ScriptEditorBase>(tab_container->get_child(selected));
	if (!current)
		return;

	current->goto_line(p_line);
}

void ScriptEditor::_goto_script_line(REF p_script, int p_line) {

	editor->push_item(p_script.ptr());

	if (bool(EditorSettings::get_singleton()->get("text_editor/external/use_external_editor"))) {

		Ref<Script> script = Object::cast_to<Script>(*p_script);
		if (!script.is_null() && script->get_path().is_resource_file())
			edit(p_script, p_line, 0);
	}

	int selected = tab_container->get_current_tab();
	if (selected < 0 || selected >= tab_container->get_child_count())
		return;

	ScriptEditorBase *current = Object::cast_to<ScriptEditorBase>(tab_container->get_child(selected));
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

		if (Object::cast_to<ScriptEditorBase>(n)) {

			history[history_pos].state = Object::cast_to<ScriptEditorBase>(n)->get_edit_state();
		}
		if (Object::cast_to<EditorHelp>(n)) {

			history[history_pos].state = Object::cast_to<EditorHelp>(n)->get_scroll();
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

	Control *c = Object::cast_to<Control>(tab_container->get_child(p_idx));
	if (!c)
		return;

	if (history_pos >= 0 && history_pos < history.size() && history[history_pos].control == tab_container->get_current_tab_control()) {

		Node *n = tab_container->get_current_tab_control();

		if (Object::cast_to<ScriptEditorBase>(n)) {

			history[history_pos].state = Object::cast_to<ScriptEditorBase>(n)->get_edit_state();
		}
		if (Object::cast_to<EditorHelp>(n)) {

			history[history_pos].state = Object::cast_to<EditorHelp>(n)->get_scroll();
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

	if (Object::cast_to<ScriptEditorBase>(c)) {

		script_name_label->set_text(Object::cast_to<ScriptEditorBase>(c)->get_name());
		script_icon->set_texture(Object::cast_to<ScriptEditorBase>(c)->get_icon());
		if (is_visible_in_tree())
			Object::cast_to<ScriptEditorBase>(c)->ensure_focus();

		notify_script_changed(Object::cast_to<ScriptEditorBase>(c)->get_edited_script());
	}
	if (Object::cast_to<EditorHelp>(c)) {

		script_name_label->set_text(Object::cast_to<EditorHelp>(c)->get_class());
		script_icon->set_texture(get_icon("Help", "EditorIcons"));
		if (is_visible_in_tree())
			Object::cast_to<EditorHelp>(c)->set_focused();
	}

	c->set_meta("__editor_pass", ++edit_pass);
	_update_history_arrows();
	_update_script_colors();
	_update_members_overview();
	_update_help_overview();
	_update_selected_editor_menu();
	_update_members_overview_visibility();
	_update_help_overview_visibility();
}

void ScriptEditor::_add_recent_script(String p_path) {

	if (p_path.empty()) {
		return;
	}

	// remove if already stored
	int already_recent = previous_scripts.find(p_path);
	if (already_recent >= 0) {
		previous_scripts.remove(already_recent);
	}

	// add to list
	previous_scripts.insert(0, p_path);

	_update_recent_scripts();
}

void ScriptEditor::_update_recent_scripts() {

	// make sure we don't exceed max size
	const int max_history = EDITOR_DEF("text_editor/files/maximum_recent_files", 20);
	if (previous_scripts.size() > max_history) {
		previous_scripts.resize(max_history);
	}

	recent_scripts->clear();

	recent_scripts->add_shortcut(ED_SHORTCUT("script_editor/open_recent", TTR("Open Recent"), KEY_MASK_CMD | KEY_MASK_SHIFT | KEY_T));
	recent_scripts->add_separator();

	const int max_shown = 8;
	for (int i = 0; i < previous_scripts.size() && i <= max_shown; i++) {
		String path = previous_scripts.get(i);
		// just show script name and last dir
		recent_scripts->add_item(path.get_slice("/", path.get_slice_count("/") - 2) + "/" + path.get_file());
	}

	recent_scripts->add_separator();
	recent_scripts->add_shortcut(ED_SHORTCUT("script_editor/clear_recent", TTR("Clear Recent Files")));

	recent_scripts->set_as_minsize();
}

void ScriptEditor::_open_recent_script(int p_idx) {

	// clear button
	if (p_idx == recent_scripts->get_item_count() - 1) {
		previous_scripts.clear();
		call_deferred("_update_recent_scripts");
		return;
	}

	// take two for the open recent button
	if (p_idx > 0) {
		p_idx -= 2;
	}

	if (p_idx < previous_scripts.size() && p_idx >= 0) {

		String path = previous_scripts.get(p_idx);
		// if its not on disk its a help file or deleted
		if (FileAccess::exists(path)) {
			Ref<Script> script = ResourceLoader::load(path);
			if (script.is_valid()) {
				edit(script, true);
			}
			// if it's a path then its most likely a delted file not help
		} else if (!path.is_resource_file()) {
			_help_class_open(path);
		}
		previous_scripts.remove(p_idx);
		_update_recent_scripts();
	}
}

void ScriptEditor::_close_tab(int p_idx, bool p_save) {

	int selected = p_idx;
	if (selected < 0 || selected >= tab_container->get_child_count())
		return;

	Node *tselected = tab_container->get_child(selected);
	ScriptEditorBase *current = Object::cast_to<ScriptEditorBase>(tab_container->get_child(selected));
	if (current) {
		_add_recent_script(current->get_edited_script()->get_path());
		if (p_save) {
			apply_scripts();
		}
		current->clear_edit_menu();
		notify_script_close(current->get_edited_script());
	} else {
		EditorHelp *help = Object::cast_to<EditorHelp>(tab_container->get_child(selected));
		_add_recent_script(help->get_class());
	}

	// roll back to previous tab
	_history_back();

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
	_update_members_overview_visibility();
	_update_help_overview_visibility();
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

		EditorHelp *se = Object::cast_to<EditorHelp>(tab_container->get_child(i));

		if (se) {
			_close_tab(i);
		}
	}
}

void ScriptEditor::_copy_script_path() {
	ScriptEditorBase *se = Object::cast_to<ScriptEditorBase>(tab_container->get_child(tab_container->get_current_tab()));
	Ref<Script> script = se->get_edited_script();
	OS::get_singleton()->set_clipboard(script->get_path());
}

void ScriptEditor::_close_other_tabs() {

	int child_count = tab_container->get_child_count();
	int current_idx = tab_container->get_current_tab();
	for (int i = child_count - 1; i >= 0; i--) {

		if (i == current_idx) {
			continue;
		}

		tab_container->set_current_tab(i);
		ScriptEditorBase *se = Object::cast_to<ScriptEditorBase>(tab_container->get_child(i));

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

void ScriptEditor::_close_all_tabs() {

	int child_count = tab_container->get_child_count();
	for (int i = child_count - 1; i >= 0; i--) {

		tab_container->set_current_tab(i);
		ScriptEditorBase *se = Object::cast_to<ScriptEditorBase>(tab_container->get_child(i));

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
	erase_tab_confirm->set_text(TTR("Close and save changes?\n\"") + current->get_name() + "\"");
	erase_tab_confirm->popup_centered_minsize();
}

void ScriptEditor::_resave_scripts(const String &p_str) {

	apply_scripts();

	for (int i = 0; i < tab_container->get_child_count(); i++) {

		ScriptEditorBase *se = Object::cast_to<ScriptEditorBase>(tab_container->get_child(i));
		if (!se)
			continue;

		Ref<Script> script = se->get_edited_script();

		if (script->get_path() == "" || script->get_path().find("local://") != -1 || script->get_path().find("::") != -1)
			continue; //internal script, who cares

		if (trim_trailing_whitespace_on_save) {
			se->trim_trailing_whitespace();
		}

		if (convert_indent_on_save) {
			if (use_space_indentation) {
				se->convert_indent_to_spaces();
			} else {
				se->convert_indent_to_tabs();
			}
		}

		editor->save_resource(script);
		se->tag_saved_version();
	}

	disk_changed->hide();
}

void ScriptEditor::_reload_scripts() {

	for (int i = 0; i < tab_container->get_child_count(); i++) {

		ScriptEditorBase *se = Object::cast_to<ScriptEditorBase>(tab_container->get_child(i));
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

		ScriptEditorBase *se = Object::cast_to<ScriptEditorBase>(tab_container->get_child(i));
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

		ScriptEditorBase *se = Object::cast_to<ScriptEditorBase>(tab_container->get_child(i));
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

Ref<Script> ScriptEditor::_get_current_script() {

	int selected = tab_container->get_current_tab();
	if (selected < 0 || selected >= tab_container->get_child_count())
		return NULL;

	ScriptEditorBase *current = Object::cast_to<ScriptEditorBase>(tab_container->get_child(selected));
	if (current) {
		return current->get_edited_script();
	} else {
		return NULL;
	}
}

Array ScriptEditor::_get_open_scripts() const {

	Array ret;
	Vector<Ref<Script> > scripts = get_open_scripts();
	int scrits_amount = scripts.size();
	for (int idx_script = 0; idx_script < scrits_amount; idx_script++) {
		ret.push_back(scripts[idx_script]);
	}
	return ret;
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
			file_dialog->set_current_path(EditorSettings::get_singleton()->get_text_editor_themes_dir().plus_file(EditorSettings::get_singleton()->get("text_editor/theme/color_theme")));
			file_dialog->popup_centered_ratio();
			file_dialog->set_title(TTR("Save Theme As.."));
		} break;
		case SEARCH_HELP: {

			help_search_dialog->popup();
		} break;
		case SEARCH_CLASSES: {

			help_index->popup();
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
		case WINDOW_SORT: {
			_sort_list_on_update = true;
			_update_script_names();
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
		case DEBUG_WITH_EXTERNAL_EDITOR: {
			bool debug_with_external_editor = !debug_menu->get_popup()->is_item_checked(debug_menu->get_popup()->get_item_index(DEBUG_WITH_EXTERNAL_EDITOR));
			debugger->set_debug_with_external_editor(debug_with_external_editor);
			debug_menu->get_popup()->set_item_checked(debug_menu->get_popup()->get_item_index(DEBUG_WITH_EXTERNAL_EDITOR), debug_with_external_editor);
		} break;
		case TOGGLE_SCRIPTS_PANEL: {
			list_split->set_visible(!list_split->is_visible());
		}
	}

	int selected = tab_container->get_current_tab();
	if (selected < 0 || selected >= tab_container->get_child_count())
		return;

	ScriptEditorBase *current = Object::cast_to<ScriptEditorBase>(tab_container->get_child(selected));
	if (current) {

		switch (p_option) {
			case FILE_SAVE: {

				if (_test_script_times_on_disk())
					return;

				if (trim_trailing_whitespace_on_save)
					current->trim_trailing_whitespace();

				if (convert_indent_on_save) {
					if (use_space_indentation) {
						current->convert_indent_to_spaces();
					} else {
						current->convert_indent_to_tabs();
					}
				}
				editor->save_resource(current->get_edited_script());

			} break;
			case FILE_SAVE_AS: {

				current->trim_trailing_whitespace();

				if (convert_indent_on_save) {
					if (use_space_indentation) {
						current->convert_indent_to_spaces();
					} else {
						current->convert_indent_to_tabs();
					}
				}
				editor->push_item(Object::cast_to<Object>(current->get_edited_script().ptr()));
				editor->save_resource_as(current->get_edited_script());

			} break;

			case FILE_TOOL_RELOAD:
			case FILE_TOOL_RELOAD_SOFT: {

				current->reload(p_option == FILE_TOOL_RELOAD_SOFT);

			} break;
			case FILE_RUN: {

				Ref<Script> scr = current->get_edited_script();
				if (scr.is_null()) {
					EditorNode::get_singleton()->show_warning("Can't obtain the script for running");
					break;
				}

				current->apply_code();
				Error err = scr->reload(false); //hard reload script before running always

				if (err != OK) {
					EditorNode::get_singleton()->show_warning("Script failed reloading, check console for errors.");
					return;
				}
				if (!scr->is_tool()) {

					EditorNode::get_singleton()->show_warning("Script is not in tool mode, will not be able to run");
					return;
				}

				if (!ClassDB::is_parent_class(scr->get_instance_base_type(), "EditorScript")) {

					EditorNode::get_singleton()->show_warning("To run this script, it must inherit EditorScript and be set to tool mode");
					return;
				}

				Ref<EditorScript> es = memnew(EditorScript);
				es->set_script(scr.get_ref_ptr());
				es->set_editor(EditorNode::get_singleton());

				es->_run();

				EditorNode::get_undo_redo()->clear_history();
			} break;
			case FILE_CLOSE: {
				if (current->is_unsaved()) {
					_ask_close_current_unsaved_tab(current);
				} else {
					_close_current_tab();
				}
			} break;
			case FILE_COPY_PATH: {
				_copy_script_path();
			} break;
			case CLOSE_DOCS: {
				_close_docs_tab();
			} break;
			case CLOSE_OTHER_TABS: {
				_close_other_tabs();
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
			case WINDOW_MOVE_UP: {

				if (tab_container->get_current_tab() > 0) {
					tab_container->move_child(current, tab_container->get_current_tab() - 1);
					tab_container->set_current_tab(tab_container->get_current_tab() - 1);
					_update_script_names();
				}
			} break;
			case WINDOW_MOVE_DOWN: {

				if (tab_container->get_current_tab() < tab_container->get_child_count() - 1) {
					tab_container->move_child(current, tab_container->get_current_tab() + 1);
					tab_container->set_current_tab(tab_container->get_current_tab() + 1);
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
	} else {

		EditorHelp *help = Object::cast_to<EditorHelp>(tab_container->get_current_tab_control());
		if (help) {

			switch (p_option) {

				case SEARCH_CLASSES: {

					help_index->popup();
					help_index->call_deferred("select_class", help->get_class());
				} break;
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
				case CLOSE_OTHER_TABS: {
					_close_other_tabs();
				} break;
				case CLOSE_ALL: {
					_close_all_tabs();
				} break;
				case WINDOW_MOVE_UP: {

					if (tab_container->get_current_tab() > 0) {
						tab_container->move_child(help, tab_container->get_current_tab() - 1);
						tab_container->set_current_tab(tab_container->get_current_tab() - 1);
						_update_script_names();
					}
				} break;
				case WINDOW_MOVE_DOWN: {

					if (tab_container->get_current_tab() < tab_container->get_child_count() - 1) {
						tab_container->move_child(help, tab_container->get_current_tab() + 1);
						tab_container->set_current_tab(tab_container->get_current_tab() + 1);
						_update_script_names();
					}
				} break;
			}
		}
	}
}

void ScriptEditor::_tab_changed(int p_which) {

	ensure_select_current();
}

void ScriptEditor::_notification(int p_what) {

	switch (p_what) {

		case NOTIFICATION_ENTER_TREE: {

			editor->connect("play_pressed", this, "_editor_play");
			editor->connect("pause_pressed", this, "_editor_pause");
			editor->connect("stop_pressed", this, "_editor_stop");
			editor->connect("script_add_function_request", this, "_add_callback");
			editor->connect("resource_saved", this, "_res_saved_callback");
			script_list->connect("item_selected", this, "_script_selected");

			members_overview->connect("item_selected", this, "_members_overview_selected");
			help_overview->connect("item_selected", this, "_help_overview_selected");
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
			help_search->set_icon(get_icon("HelpSearch", "EditorIcons"));
			site_search->set_icon(get_icon("Instance", "EditorIcons"));
			class_search->set_icon(get_icon("ClassList", "EditorIcons"));

			script_forward->set_icon(get_icon("Forward", "EditorIcons"));
			script_back->set_icon(get_icon("Back", "EditorIcons"));
		} break;

		case NOTIFICATION_READY: {

			get_tree()->connect("tree_changed", this, "_tree_changed");
			editor->connect("request_help", this, "_request_help");
			editor->connect("request_help_search", this, "_help_search");
			editor->connect("request_help_index", this, "_help_index");
		} break;

		case NOTIFICATION_EXIT_TREE: {

			editor->disconnect("play_pressed", this, "_editor_play");
			editor->disconnect("pause_pressed", this, "_editor_pause");
			editor->disconnect("stop_pressed", this, "_editor_stop");
		} break;

		case MainLoop::NOTIFICATION_WM_FOCUS_IN: {

			_test_script_times_on_disk();
			_update_modified_scripts_for_external_editor();
		} break;

		case NOTIFICATION_PROCESS: {
		} break;

		case EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED: {

			help_search->set_icon(get_icon("HelpSearch", "EditorIcons"));
			site_search->set_icon(get_icon("Instance", "EditorIcons"));
			class_search->set_icon(get_icon("ClassList", "EditorIcons"));

			script_forward->set_icon(get_icon("Forward", "EditorIcons"));
			script_back->set_icon(get_icon("Back", "EditorIcons"));

			recent_scripts->set_as_minsize();
		} break;

		default:
			break;
	}
}

bool ScriptEditor::can_take_away_focus() const {

	int selected = tab_container->get_current_tab();
	if (selected < 0 || selected >= tab_container->get_child_count())
		return true;

	ScriptEditorBase *current = Object::cast_to<ScriptEditorBase>(tab_container->get_child(selected));
	if (!current)
		return true;

	return current->can_lose_focus_on_node_selection();
}

void ScriptEditor::close_builtin_scripts_from_scene(const String &p_scene) {

	for (int i = 0; i < tab_container->get_child_count(); i++) {

		ScriptEditorBase *se = Object::cast_to<ScriptEditorBase>(tab_container->get_child(i));

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

void ScriptEditor::notify_script_close(const Ref<Script> &p_script) {
	emit_signal("script_close", p_script);
}

void ScriptEditor::notify_script_changed(const Ref<Script> &p_script) {
	emit_signal("editor_script_changed", p_script);
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

void ScriptEditor::get_breakpoints(List<String> *p_breakpoints) {

	for (int i = 0; i < tab_container->get_child_count(); i++) {

		ScriptEditorBase *se = Object::cast_to<ScriptEditorBase>(tab_container->get_child(i));
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
		return;

	Control *c = Object::cast_to<Control>(tab_container->get_child(cidx));
	ScriptEditorBase *se = Object::cast_to<ScriptEditorBase>(c);
	if (!se)
		return;
	se->ensure_focus();
}

void ScriptEditor::_members_overview_selected(int p_idx) {
	Node *current = tab_container->get_child(tab_container->get_current_tab());
	ScriptEditorBase *se = Object::cast_to<ScriptEditorBase>(current);
	if (!se) {
		return;
	}
	// Go to the member's line and reset the cursor column. We can't just change scroll_position
	// directly, since code might be folded.
	se->goto_line(members_overview->get_item_metadata(p_idx));
	Dictionary state = se->get_edit_state();
	state["column"] = 0;
	se->set_edit_state(state);
}

void ScriptEditor::_help_overview_selected(int p_idx) {
	Node *current = tab_container->get_child(tab_container->get_current_tab());
	EditorHelp *se = Object::cast_to<EditorHelp>(current);
	if (!se) {
		return;
	}
	se->scroll_to_section(help_overview->get_item_metadata(p_idx));
}

void ScriptEditor::_script_selected(int p_idx) {

	grab_focus_block = !Input::get_singleton()->is_mouse_button_pressed(1); //amazing hack, simply amazing

	_go_to_tab(script_list->get_item_metadata(p_idx));
	grab_focus_block = false;
}

void ScriptEditor::ensure_select_current() {

	if (tab_container->get_child_count() && tab_container->get_current_tab() >= 0) {

		Node *current = tab_container->get_child(tab_container->get_current_tab());

		ScriptEditorBase *se = Object::cast_to<ScriptEditorBase>(current);
		if (se) {

			Ref<Script> script = se->get_edited_script();

			if (!grab_focus_block && is_visible_in_tree())
				se->ensure_focus();

			//edit_menu->show();
			//search_menu->show();
		}

		EditorHelp *eh = Object::cast_to<EditorHelp>(current);

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
	Node *ref;

	bool operator<(const _ScriptEditorItemData &id) const {

		return category == id.category ? sort_key < id.sort_key : category < id.category;
	}
};

void ScriptEditor::_update_members_overview_visibility() {

	int selected = tab_container->get_current_tab();
	if (selected < 0 || selected >= tab_container->get_child_count())
		return;

	Node *current = tab_container->get_child(tab_container->get_current_tab());
	ScriptEditorBase *se = Object::cast_to<ScriptEditorBase>(current);
	if (!se) {
		members_overview->set_visible(false);
		return;
	}

	if (members_overview_enabled && se->show_members_overview()) {
		members_overview->set_visible(true);
	} else {
		members_overview->set_visible(false);
	}
}

void ScriptEditor::_update_members_overview() {
	members_overview->clear();

	int selected = tab_container->get_current_tab();
	if (selected < 0 || selected >= tab_container->get_child_count())
		return;

	Node *current = tab_container->get_child(tab_container->get_current_tab());
	ScriptEditorBase *se = Object::cast_to<ScriptEditorBase>(current);
	if (!se) {
		return;
	}

	Vector<String> functions = se->get_functions();
	for (int i = 0; i < functions.size(); i++) {
		members_overview->add_item(functions[i].get_slice(":", 0));
		members_overview->set_item_metadata(i, functions[i].get_slice(":", 1).to_int() - 1);
	}
}

void ScriptEditor::_update_help_overview_visibility() {

	int selected = tab_container->get_current_tab();
	if (selected < 0 || selected >= tab_container->get_child_count()) {
		help_overview->set_visible(false);
		return;
	}

	Node *current = tab_container->get_child(tab_container->get_current_tab());
	EditorHelp *se = Object::cast_to<EditorHelp>(current);
	if (!se) {
		help_overview->set_visible(false);
		return;
	}

	if (help_overview_enabled) {
		help_overview->set_visible(true);
	} else {
		help_overview->set_visible(false);
	}
}

void ScriptEditor::_update_help_overview() {
	help_overview->clear();

	int selected = tab_container->get_current_tab();
	if (selected < 0 || selected >= tab_container->get_child_count())
		return;

	Node *current = tab_container->get_child(tab_container->get_current_tab());
	EditorHelp *se = Object::cast_to<EditorHelp>(current);
	if (!se) {
		return;
	}

	Vector<Pair<String, int> > sections = se->get_sections();
	for (int i = 0; i < sections.size(); i++) {
		help_overview->add_item(sections[i].first);
		help_overview->set_item_metadata(i, sections[i].second);
	}
}

void ScriptEditor::_update_script_colors() {

	bool script_temperature_enabled = EditorSettings::get_singleton()->get("text_editor/open_scripts/script_temperature_enabled");
	bool highlight_current = EditorSettings::get_singleton()->get("text_editor/open_scripts/highlight_current_script");

	int hist_size = EditorSettings::get_singleton()->get("text_editor/open_scripts/script_temperature_history_size");
	Color hot_color = get_color("accent_color", "Editor");
	Color cold_color = get_color("font_color", "Editor");

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

			script_list->set_item_custom_fg_color(i, hot_color.linear_interpolate(cold_color, v));
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

		ScriptEditorBase *se = Object::cast_to<ScriptEditorBase>(tab_container->get_child(i));
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
			sd.ref = se;

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

		EditorHelp *eh = Object::cast_to<EditorHelp>(tab_container->get_child(i));
		if (eh) {

			String name = eh->get_class();
			Ref<Texture> icon = get_icon("Help", "EditorIcons");
			String tooltip = name + TTR(" Class Reference");

			_ScriptEditorItemData sd;
			sd.icon = icon;
			sd.name = name;
			sd.sort_key = name.to_lower();
			sd.tooltip = tooltip;
			sd.index = i;
			sd.used = false;
			sd.category = split_script_help ? 1 : 0;
			sd.ref = eh;

			sedata.push_back(sd);
		}
	}

	if (_sort_list_on_update && !sedata.empty()) {
		sedata.sort();

		// change actual order of tab_container so that the order can be rearranged by user
		int cur_tab = tab_container->get_current_tab();
		int prev_tab = tab_container->get_previous_tab();
		int new_cur_tab = -1;
		int new_prev_tab = -1;
		for (int i = 0; i < sedata.size(); i++) {
			tab_container->move_child(sedata[i].ref, i);
			if (new_prev_tab == -1 && sedata[i].index == prev_tab) {
				new_prev_tab = i;
			}
			if (new_cur_tab == -1 && sedata[i].index == cur_tab) {
				new_cur_tab = i;
			}
		}
		tab_container->call_deferred("set_current_tab", new_prev_tab);
		tab_container->call_deferred("set_current_tab", new_cur_tab);
		_sort_list_on_update = false;
	}

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

	_update_members_overview();
	_update_help_overview();
	_update_members_overview_visibility();
	_update_help_overview_visibility();
	_update_script_colors();
}

bool ScriptEditor::edit(const Ref<Script> &p_script, int p_line, int p_col, bool p_grab_focus) {

	if (p_script.is_null())
		return false;

	// refuse to open built-in if scene is not loaded

	// see if already has it

	bool open_dominant = EditorSettings::get_singleton()->get("text_editor/files/open_dominant_script_on_scene_change");

	if (p_script->get_language()->overrides_external_editor()) {
		Error err = p_script->get_language()->open_in_external_editor(p_script, p_line >= 0 ? p_line : 0, p_col);
		if (err != OK)
			ERR_PRINT("Couldn't open script in the overridden external text editor");
		return false;
	}

	if ((debugger->get_dump_stack_script() != p_script || debugger->get_debug_with_external_editor()) &&
			p_script->get_path().is_resource_file() &&
			bool(EditorSettings::get_singleton()->get("text_editor/external/use_external_editor"))) {

		String path = EditorSettings::get_singleton()->get("text_editor/external/exec_path");
		String flags = EditorSettings::get_singleton()->get("text_editor/external/exec_flags");

		Dictionary keys;
		keys["project"] = ProjectSettings::get_singleton()->get_resource_path();
		keys["file"] = ProjectSettings::get_singleton()->globalize_path(p_script->get_path());
		keys["line"] = p_line >= 0 ? p_line : 0;
		keys["col"] = p_col;

		flags = flags.format(keys).strip_edges().replace("\\\\", "\\");

		List<String> args;

		if (flags.size()) {
			int from = 0, to = 0;
			bool inside_quotes = false;
			for (int i = 0; i < flags.size(); i++) {
				if (flags[i] == '"' && (!i || flags[i - 1] != '\\')) {
					inside_quotes = !inside_quotes;
				} else if (flags[i] == '\0' || (!inside_quotes && flags[i] == ' ')) {
					args.push_back(flags.substr(from, to));
					from = i + 1;
					to = 0;
				} else {
					to++;
				}
			}
		}

		Error err = OS::get_singleton()->execute(path, args, false);
		if (err == OK)
			return false;
		WARN_PRINT("Couldn't open external text editor, using internal");
	}

	for (int i = 0; i < tab_container->get_child_count(); i++) {

		ScriptEditorBase *se = Object::cast_to<ScriptEditorBase>(tab_container->get_child(i));
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

				if (p_line >= 0)
					se->goto_line(p_line - 1);
			}
			return true;
		}
	}

	// doesn't have it, make a new one

	ScriptEditorBase *se;

	for (int i = script_editor_func_count - 1; i >= 0; i--) {
		se = script_editor_funcs[i](p_script);
		if (se)
			break;
	}
	ERR_FAIL_COND_V(!se, false);

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
	se->connect("edited_script_changed", this, "_script_changed");
	se->connect("request_help_search", this, "_help_search");
	se->connect("request_open_script_at_line", this, "_goto_script_line");
	se->connect("go_to_help", this, "_help_class_goto");
	se->connect("request_save_history", this, "_save_history");

	//test for modification, maybe the script was not edited but was loaded

	_test_script_times_on_disk(p_script);
	_update_modified_scripts_for_external_editor(p_script);

	if (p_line >= 0)
		se->goto_line(p_line - 1);

	notify_script_changed(p_script);
	return true;
}

void ScriptEditor::save_all_scripts() {

	for (int i = 0; i < tab_container->get_child_count(); i++) {

		ScriptEditorBase *se = Object::cast_to<ScriptEditorBase>(tab_container->get_child(i));
		if (!se)
			continue;

		if (convert_indent_on_save) {
			if (use_space_indentation) {
				se->convert_indent_to_spaces();
			} else {
				se->convert_indent_to_tabs();
			}
		}

		if (trim_trailing_whitespace_on_save) {
			se->trim_trailing_whitespace();
		}

		if (!se->is_unsaved())
			continue;

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

		ScriptEditorBase *se = Object::cast_to<ScriptEditorBase>(tab_container->get_child(i));
		if (!se)
			continue;
		se->apply_code();
	}
}

void ScriptEditor::open_script_create_dialog(const String &p_base_name, const String &p_base_path) {
	_menu_option(FILE_NEW);
	script_create_dialog->config(p_base_name, p_base_path);
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

		ScriptEditorBase *se = Object::cast_to<ScriptEditorBase>(tab_container->get_child(i));
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

		ScriptEditorBase *se = Object::cast_to<ScriptEditorBase>(tab_container->get_child(i));
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
	convert_indent_on_save = EditorSettings::get_singleton()->get("text_editor/indent/convert_indent_on_save");
	use_space_indentation = EditorSettings::get_singleton()->get("text_editor/indent/type");

	members_overview_enabled = EditorSettings::get_singleton()->get("text_editor/open_scripts/show_members_overview");
	help_overview_enabled = EditorSettings::get_singleton()->get("text_editor/help/show_help_index");
	_update_members_overview_visibility();
	_update_help_overview_visibility();

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

		ScriptEditorBase *se = Object::cast_to<ScriptEditorBase>(tab_container->get_child(i));
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

Variant ScriptEditor::get_drag_data_fw(const Point2 &p_point, Control *p_from) {

	// return Variant(); // return this if drag disabled

	Node *cur_node = tab_container->get_child(tab_container->get_current_tab());

	HBoxContainer *drag_preview = memnew(HBoxContainer);
	String preview_name = "";
	Ref<Texture> preview_icon;

	ScriptEditorBase *se = Object::cast_to<ScriptEditorBase>(cur_node);
	if (se) {
		preview_name = se->get_name();
		preview_icon = se->get_icon();
	}
	EditorHelp *eh = Object::cast_to<EditorHelp>(cur_node);
	if (eh) {
		preview_name = eh->get_class();
		preview_icon = get_icon("Help", "EditorIcons");
	}

	if (!preview_icon.is_null()) {
		TextureRect *tf = memnew(TextureRect);
		tf->set_texture(preview_icon);
		drag_preview->add_child(tf);
	}
	Label *label = memnew(Label(preview_name));
	drag_preview->add_child(label);
	set_drag_preview(drag_preview);

	Dictionary drag_data;
	drag_data["type"] = "script_list_element"; // using a custom type because node caused problems when dragging to scene tree
	drag_data["script_list_element"] = cur_node;

	return drag_data;
}

bool ScriptEditor::can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const {

	Dictionary d = p_data;
	if (!d.has("type"))
		return false;

	if (String(d["type"]) == "script_list_element") {

		Node *node = d["script_list_element"];

		ScriptEditorBase *se = Object::cast_to<ScriptEditorBase>(node);
		if (se) {
			return true;
		}
		EditorHelp *eh = Object::cast_to<EditorHelp>(node);
		if (eh) {
			return true;
		}
	}

	if (String(d["type"]) == "nodes") {

		Array nodes = d["nodes"];
		if (nodes.size() == 0)
			return false;
		Node *node = get_node((nodes[0]));

		ScriptEditorBase *se = Object::cast_to<ScriptEditorBase>(node);
		if (se) {
			return true;
		}
		EditorHelp *eh = Object::cast_to<EditorHelp>(node);
		if (eh) {
			return true;
		}
	}

	if (String(d["type"]) == "files") {

		Vector<String> files = d["files"];

		if (files.size() == 0)
			return false; //weird

		for (int i = 0; i < files.size(); i++) {
			String file = files[i];
			if (file == "" || !FileAccess::exists(file))
				continue;
			Ref<Script> scr = ResourceLoader::load(file);
			if (scr.is_valid()) {
				return true;
			}
		}
		return true;
	}

	return false;
}

void ScriptEditor::drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) {

	if (!can_drop_data_fw(p_point, p_data, p_from))
		return;

	Dictionary d = p_data;
	if (!d.has("type"))
		return;

	if (String(d["type"]) == "script_list_element") {

		Node *node = d["script_list_element"];

		ScriptEditorBase *se = Object::cast_to<ScriptEditorBase>(node);
		EditorHelp *eh = Object::cast_to<EditorHelp>(node);
		if (se || eh) {
			int new_index = script_list->get_item_at_position(p_point);
			tab_container->move_child(node, new_index);
			tab_container->set_current_tab(new_index);
			_update_script_names();
		}
	}

	if (String(d["type"]) == "nodes") {

		Array nodes = d["nodes"];
		if (nodes.size() == 0)
			return;
		Node *node = get_node(nodes[0]);

		ScriptEditorBase *se = Object::cast_to<ScriptEditorBase>(node);
		EditorHelp *eh = Object::cast_to<EditorHelp>(node);
		if (se || eh) {
			int new_index = script_list->get_item_at_position(p_point);
			tab_container->move_child(node, new_index);
			tab_container->set_current_tab(new_index);
			_update_script_names();
		}
	}

	if (String(d["type"]) == "files") {

		Vector<String> files = d["files"];

		int new_index = script_list->get_item_at_position(p_point);
		int num_tabs_before = tab_container->get_child_count();
		for (int i = 0; i < files.size(); i++) {
			String file = files[i];
			if (file == "" || !FileAccess::exists(file))
				continue;
			Ref<Script> scr = ResourceLoader::load(file);
			if (scr.is_valid()) {
				edit(scr);
				if (tab_container->get_child_count() > num_tabs_before) {
					tab_container->move_child(tab_container->get_child(tab_container->get_child_count() - 1), new_index);
					num_tabs_before = tab_container->get_child_count();
				} else {
					tab_container->move_child(tab_container->get_child(tab_container->get_current_tab()), new_index);
				}
			}
		}
		tab_container->set_current_tab(new_index);
		_update_script_names();
	}
}

void ScriptEditor::_unhandled_input(const Ref<InputEvent> &p_event) {
	if (!is_visible_in_tree() || !p_event->is_pressed() || p_event->is_echo())
		return;
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
	if (ED_IS_SHORTCUT("script_editor/window_move_up", p_event)) {
		_menu_option(WINDOW_MOVE_UP);
	}
	if (ED_IS_SHORTCUT("script_editor/window_move_down", p_event)) {
		_menu_option(WINDOW_MOVE_DOWN);
	}
}

void ScriptEditor::_script_list_gui_input(const Ref<InputEvent> &ev) {

	Ref<InputEventMouseButton> mb = ev;
	if (mb.is_valid() && mb->get_button_index() == BUTTON_RIGHT && mb->is_pressed()) {

		_make_script_list_context_menu();
	}
}

void ScriptEditor::_make_script_list_context_menu() {

	context_menu->clear();

	int selected = tab_container->get_current_tab();
	if (selected < 0 || selected >= tab_container->get_child_count())
		return;

	ScriptEditorBase *se = Object::cast_to<ScriptEditorBase>(tab_container->get_child(selected));
	if (se) {
		context_menu->add_shortcut(ED_GET_SHORTCUT("script_editor/save"), FILE_SAVE);
		context_menu->add_shortcut(ED_GET_SHORTCUT("script_editor/save_as"), FILE_SAVE_AS);
		context_menu->add_shortcut(ED_GET_SHORTCUT("script_editor/close_file"), FILE_CLOSE);
		context_menu->add_shortcut(ED_GET_SHORTCUT("script_editor/close_all"), CLOSE_ALL);
		context_menu->add_shortcut(ED_GET_SHORTCUT("script_editor/close_other_tabs"), CLOSE_OTHER_TABS);
		context_menu->add_separator();
		context_menu->add_shortcut(ED_GET_SHORTCUT("script_editor/copy_path"), FILE_COPY_PATH);
		context_menu->add_shortcut(ED_GET_SHORTCUT("script_editor/reload_script_soft"), FILE_TOOL_RELOAD_SOFT);

		Ref<Script> scr = se->get_edited_script();
		if (!scr.is_null() && scr->is_tool()) {
			context_menu->add_separator();
			context_menu->add_shortcut(ED_GET_SHORTCUT("script_editor/run_file"), FILE_RUN);
		}
	} else {
		context_menu->add_shortcut(ED_GET_SHORTCUT("script_editor/close_file"), FILE_CLOSE);
	}

	EditorHelp *eh = Object::cast_to<EditorHelp>(tab_container->get_child(selected));
	if (eh) {
		// nothing
	}

	context_menu->add_separator();
	context_menu->add_shortcut(ED_GET_SHORTCUT("script_editor/window_move_up"), WINDOW_MOVE_UP);
	context_menu->add_shortcut(ED_GET_SHORTCUT("script_editor/window_move_down"), WINDOW_MOVE_DOWN);
	context_menu->add_shortcut(ED_GET_SHORTCUT("script_editor/window_sort"), WINDOW_SORT);
	context_menu->add_shortcut(ED_GET_SHORTCUT("script_editor/toggle_scripts_panel"), TOGGLE_SCRIPTS_PANEL);

	context_menu->set_position(get_global_transform().xform(get_local_mouse_position()));
	context_menu->set_size(Vector2(1, 1));
	context_menu->popup();
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
		if (path == "") { // invalid, skip
			continue;
		}
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

		ScriptEditorBase *se = Object::cast_to<ScriptEditorBase>(tab_container->get_child(i));
		if (se) {

			String path = se->get_edited_script()->get_path();
			if (!path.is_resource_file())
				continue;

			scripts.push_back(path);
		}

		EditorHelp *eh = Object::cast_to<EditorHelp>(tab_container->get_child(i));

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

		EditorHelp *eh = Object::cast_to<EditorHelp>(tab_container->get_child(i));

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

		EditorHelp *eh = Object::cast_to<EditorHelp>(tab_container->get_child(i));

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

		ScriptEditorBase *se = Object::cast_to<ScriptEditorBase>(tab_container->get_child(i));
		if (se && se->get_edit_menu()) {

			if (current)
				se->get_edit_menu()->show();
			else
				se->get_edit_menu()->hide();
		}
	}

	EditorHelp *eh = Object::cast_to<EditorHelp>(tab_container->get_current_tab_control());
	if (eh) {
		script_search_menu->show();
	} else {
		script_search_menu->hide();
	}
}

void ScriptEditor::_update_history_pos(int p_new_pos) {

	Node *n = tab_container->get_current_tab_control();

	if (Object::cast_to<ScriptEditorBase>(n)) {

		history[history_pos].state = Object::cast_to<ScriptEditorBase>(n)->get_edit_state();
	}
	if (Object::cast_to<EditorHelp>(n)) {

		history[history_pos].state = Object::cast_to<EditorHelp>(n)->get_scroll();
	}

	history_pos = p_new_pos;
	tab_container->set_current_tab(history[history_pos].control->get_index());

	n = history[history_pos].control;

	if (Object::cast_to<ScriptEditorBase>(n)) {

		Object::cast_to<ScriptEditorBase>(n)->set_edit_state(history[history_pos].state);
		Object::cast_to<ScriptEditorBase>(n)->ensure_focus();

		notify_script_changed(Object::cast_to<ScriptEditorBase>(n)->get_edited_script());
	}

	if (Object::cast_to<EditorHelp>(n)) {

		Object::cast_to<EditorHelp>(n)->set_scroll(history[history_pos].state);
		Object::cast_to<EditorHelp>(n)->set_focused();
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

Vector<Ref<Script> > ScriptEditor::get_open_scripts() const {

	Vector<Ref<Script> > out_scripts = Vector<Ref<Script> >();

	for (int i = 0; i < tab_container->get_child_count(); i++) {
		ScriptEditorBase *se = Object::cast_to<ScriptEditorBase>(tab_container->get_child(i));
		if (!se)
			continue;
		out_scripts.push_back(se->get_edited_script());
	}

	return out_scripts;
}

void ScriptEditor::set_scene_root_script(Ref<Script> p_script) {

	bool open_dominant = EditorSettings::get_singleton()->get("text_editor/files/open_dominant_script_on_scene_change");

	if (bool(EditorSettings::get_singleton()->get("text_editor/external/use_external_editor")))
		return;

	if (open_dominant && p_script.is_valid() && _can_open_in_editor(p_script.ptr())) {
		edit(p_script);
	}
}

bool ScriptEditor::script_goto_method(Ref<Script> p_script, const String &p_method) {

	int line = p_script->get_member_line(p_method);

	if (line == -1)
		return false;

	return edit(p_script, line, 0);
}

void ScriptEditor::set_live_auto_reload_running_scripts(bool p_enabled) {

	auto_reload_running_scripts = p_enabled;
}

void ScriptEditor::_help_index(String p_text) {
	help_index->popup();
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

void ScriptEditor::_script_changed() {

	NodeDock::singleton->update_lists();
}

void ScriptEditor::_bind_methods() {

	ClassDB::bind_method("_file_dialog_action", &ScriptEditor::_file_dialog_action);
	ClassDB::bind_method("_tab_changed", &ScriptEditor::_tab_changed);
	ClassDB::bind_method("_menu_option", &ScriptEditor::_menu_option);
	ClassDB::bind_method("_close_current_tab", &ScriptEditor::_close_current_tab);
	ClassDB::bind_method("_close_discard_current_tab", &ScriptEditor::_close_discard_current_tab);
	ClassDB::bind_method("_close_docs_tab", &ScriptEditor::_close_docs_tab);
	ClassDB::bind_method("_close_all_tabs", &ScriptEditor::_close_all_tabs);
	ClassDB::bind_method("_close_other_tabs", &ScriptEditor::_close_other_tabs);
	ClassDB::bind_method("_open_recent_script", &ScriptEditor::_open_recent_script);
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
	ClassDB::bind_method("_help_index", &ScriptEditor::_help_index);
	ClassDB::bind_method("_save_history", &ScriptEditor::_save_history);
	ClassDB::bind_method("_copy_script_path", &ScriptEditor::_copy_script_path);

	ClassDB::bind_method("_breaked", &ScriptEditor::_breaked);
	ClassDB::bind_method("_show_debugger", &ScriptEditor::_show_debugger);
	ClassDB::bind_method("_get_debug_tooltip", &ScriptEditor::_get_debug_tooltip);
	ClassDB::bind_method("_autosave_scripts", &ScriptEditor::_autosave_scripts);
	ClassDB::bind_method("_editor_settings_changed", &ScriptEditor::_editor_settings_changed);
	ClassDB::bind_method("_update_script_names", &ScriptEditor::_update_script_names);
	ClassDB::bind_method("_tree_changed", &ScriptEditor::_tree_changed);
	ClassDB::bind_method("_members_overview_selected", &ScriptEditor::_members_overview_selected);
	ClassDB::bind_method("_help_overview_selected", &ScriptEditor::_help_overview_selected);
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
	ClassDB::bind_method("_script_list_gui_input", &ScriptEditor::_script_list_gui_input);
	ClassDB::bind_method("_script_changed", &ScriptEditor::_script_changed);
	ClassDB::bind_method("_update_recent_scripts", &ScriptEditor::_update_recent_scripts);

	ClassDB::bind_method(D_METHOD("get_drag_data_fw"), &ScriptEditor::get_drag_data_fw);
	ClassDB::bind_method(D_METHOD("can_drop_data_fw"), &ScriptEditor::can_drop_data_fw);
	ClassDB::bind_method(D_METHOD("drop_data_fw"), &ScriptEditor::drop_data_fw);

	ClassDB::bind_method(D_METHOD("get_current_script"), &ScriptEditor::_get_current_script);
	ClassDB::bind_method(D_METHOD("get_open_scripts"), &ScriptEditor::_get_open_scripts);
	ClassDB::bind_method(D_METHOD("open_script_create_dialog", "base_name", "base_path"), &ScriptEditor::open_script_create_dialog);

	ADD_SIGNAL(MethodInfo("editor_script_changed", PropertyInfo(Variant::OBJECT, "script", PROPERTY_HINT_RESOURCE_TYPE, "Script")));
	ADD_SIGNAL(MethodInfo("script_close", PropertyInfo(Variant::OBJECT, "script", PROPERTY_HINT_RESOURCE_TYPE, "Script")));
}

ScriptEditor::ScriptEditor(EditorNode *p_editor) {

	current_theme = "";

	completion_cache = memnew(EditorScriptCodeCompletionCache);
	restoring_layout = false;
	waiting_update_names = false;
	pending_auto_reload = false;
	auto_reload_running_scripts = false;
	members_overview_enabled = true;
	help_overview_enabled = true;
	editor = p_editor;

	VBoxContainer *main_container = memnew(VBoxContainer);
	add_child(main_container);

	menu_hb = memnew(HBoxContainer);
	main_container->add_child(menu_hb);

	script_split = memnew(HSplitContainer);
	main_container->add_child(script_split);
	script_split->set_v_size_flags(SIZE_EXPAND_FILL);

	list_split = memnew(VSplitContainer);
	script_split->add_child(list_split);
	list_split->set_v_size_flags(SIZE_EXPAND_FILL);

	script_list = memnew(ItemList);
	list_split->add_child(script_list);
	script_list->set_custom_minimum_size(Size2(150 * EDSCALE, 100)); //need to give a bit of limit to avoid it from disappearing
	script_list->set_v_size_flags(SIZE_EXPAND_FILL);
	script_split->set_split_offset(140);
	//list_split->set_split_offset(500);
	_sort_list_on_update = true;
	script_list->connect("gui_input", this, "_script_list_gui_input");
	script_list->set_allow_rmb_select(true);
	script_list->set_drag_forwarding(this);

	context_menu = memnew(PopupMenu);
	add_child(context_menu);
	context_menu->connect("id_pressed", this, "_menu_option");

	members_overview = memnew(ItemList);
	list_split->add_child(members_overview);
	members_overview->set_custom_minimum_size(Size2(0, 100)); //need to give a bit of limit to avoid it from disappearing
	members_overview->set_v_size_flags(SIZE_EXPAND_FILL);

	help_overview = memnew(ItemList);
	list_split->add_child(help_overview);
	help_overview->set_custom_minimum_size(Size2(0, 100)); //need to give a bit of limit to avoid it from disappearing
	help_overview->set_v_size_flags(SIZE_EXPAND_FILL);

	tab_container = memnew(TabContainer);
	tab_container->set_tabs_visible(false);
	script_split->add_child(tab_container);

	tab_container->set_h_size_flags(SIZE_EXPAND_FILL);

	ED_SHORTCUT("script_editor/window_sort", TTR("Sort"));
	ED_SHORTCUT("script_editor/window_move_up", TTR("Move Up"), KEY_MASK_SHIFT | KEY_MASK_ALT | KEY_UP);
	ED_SHORTCUT("script_editor/window_move_down", TTR("Move Down"), KEY_MASK_SHIFT | KEY_MASK_ALT | KEY_DOWN);
	ED_SHORTCUT("script_editor/next_script", TTR("Next script"), KEY_MASK_CMD | KEY_MASK_SHIFT | KEY_PERIOD); // these should be KEY_GREATER and KEY_LESS but those don't work
	ED_SHORTCUT("script_editor/prev_script", TTR("Previous script"), KEY_MASK_CMD | KEY_MASK_SHIFT | KEY_COLON);
	set_process_unhandled_input(true);

	file_menu = memnew(MenuButton);
	menu_hb->add_child(file_menu);
	file_menu->set_text(TTR("File"));
	file_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/new", TTR("New")), FILE_NEW);
	file_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/open", TTR("Open")), FILE_OPEN);
	file_menu->get_popup()->add_submenu_item(TTR("Open Recent"), "RecentScripts", FILE_OPEN_RECENT);

	recent_scripts = memnew(PopupMenu);
	recent_scripts->set_name("RecentScripts");
	file_menu->get_popup()->add_child(recent_scripts);
	recent_scripts->connect("id_pressed", this, "_open_recent_script");
	_update_recent_scripts();

	file_menu->get_popup()->add_separator();
	file_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/save", TTR("Save"), KEY_MASK_ALT | KEY_MASK_CMD | KEY_S), FILE_SAVE);
	file_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/save_as", TTR("Save As..")), FILE_SAVE_AS);
	file_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/save_all", TTR("Save All"), KEY_MASK_CMD | KEY_MASK_SHIFT | KEY_MASK_ALT | KEY_S), FILE_SAVE_ALL);
	file_menu->get_popup()->add_separator();
	file_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/reload_script_soft", TTR("Soft Reload Script"), KEY_MASK_CMD | KEY_MASK_SHIFT | KEY_R), FILE_TOOL_RELOAD_SOFT);
	file_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/copy_path", TTR("Copy Script Path")), FILE_COPY_PATH);
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
	file_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/close_other_tabs", TTR("Close Other Tabs")), CLOSE_OTHER_TABS);
	file_menu->get_popup()->add_separator();
	file_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/run_file", TTR("Run"), KEY_MASK_CMD | KEY_MASK_SHIFT | KEY_X), FILE_RUN);
	file_menu->get_popup()->add_separator();
	file_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/toggle_scripts_panel", TTR("Toggle Scripts Panel"), KEY_MASK_CMD | KEY_BACKSLASH), TOGGLE_SCRIPTS_PANEL);
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
	debug_menu->get_popup()->add_shortcut(ED_SHORTCUT("debugger/step_over", TTR("Step Over"), KEY_F10), DEBUG_NEXT);
	debug_menu->get_popup()->add_shortcut(ED_SHORTCUT("debugger/step_into", TTR("Step Into"), KEY_F11), DEBUG_STEP);
	debug_menu->get_popup()->add_separator();
	debug_menu->get_popup()->add_shortcut(ED_SHORTCUT("debugger/break", TTR("Break")), DEBUG_BREAK);
	debug_menu->get_popup()->add_shortcut(ED_SHORTCUT("debugger/continue", TTR("Continue"), KEY_F12), DEBUG_CONTINUE);
	debug_menu->get_popup()->add_separator();
	//debug_menu->get_popup()->add_check_item("Show Debugger",DEBUG_SHOW);
	debug_menu->get_popup()->add_check_shortcut(ED_SHORTCUT("debugger/keep_debugger_open", TTR("Keep Debugger Open")), DEBUG_SHOW_KEEP_OPEN);
	debug_menu->get_popup()->add_check_shortcut(ED_SHORTCUT("debugger/debug_with_exteral_editor", TTR("Debug with external editor")), DEBUG_WITH_EXTERNAL_EDITOR);
	debug_menu->get_popup()->connect("id_pressed", this, "_menu_option");

	debug_menu->get_popup()->set_item_disabled(debug_menu->get_popup()->get_item_index(DEBUG_NEXT), true);
	debug_menu->get_popup()->set_item_disabled(debug_menu->get_popup()->get_item_index(DEBUG_STEP), true);
	debug_menu->get_popup()->set_item_disabled(debug_menu->get_popup()->get_item_index(DEBUG_BREAK), true);
	debug_menu->get_popup()->set_item_disabled(debug_menu->get_popup()->get_item_index(DEBUG_CONTINUE), true);

	menu_hb->add_spacer();

	script_icon = memnew(TextureRect);
	menu_hb->add_child(script_icon);
	script_name_label = memnew(Label);
	menu_hb->add_child(script_name_label);

	script_icon->hide();
	script_name_label->hide();

	menu_hb->add_spacer();

	site_search = memnew(ToolButton);
	site_search->set_text(TTR("Online Docs"));
	site_search->connect("pressed", this, "_menu_option", varray(SEARCH_WEBSITE));
	menu_hb->add_child(site_search);
	site_search->set_tooltip(TTR("Open Godot online documentation"));

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
	convert_indent_on_save = false;
	use_space_indentation = false;

	ScriptServer::edit_request_func = _open_script_request;

	add_style_override("panel", editor->get_gui_base()->get_stylebox("ScriptEditorPanel", "EditorStyles"));
	tab_container->add_style_override("panel", editor->get_gui_base()->get_stylebox("ScriptEditor", "EditorStyles"));
}

ScriptEditor::~ScriptEditor() {

	memdelete(completion_cache);
}

void ScriptEditorPlugin::edit(Object *p_object) {

	if (!Object::cast_to<Script>(p_object))
		return;

	script_editor->edit(Object::cast_to<Script>(p_object));
}

bool ScriptEditorPlugin::handles(Object *p_object) const {

	if (Object::cast_to<Script>(p_object)) {

		bool valid = _can_open_in_editor(Object::cast_to<Script>(p_object));

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
	EDITOR_DEF("text_editor/open_scripts/current_script_background_color", Color(1, 1, 1, 0.3));
	EDITOR_DEF("text_editor/open_scripts/group_help_pages", true);
	EditorSettings::get_singleton()->add_property_hint(PropertyInfo(Variant::INT, "text_editor/open_scripts/sort_scripts_by", PROPERTY_HINT_ENUM, "Name,Path"));
	EDITOR_DEF("text_editor/open_scripts/sort_scripts_by", 0);
	EditorSettings::get_singleton()->add_property_hint(PropertyInfo(Variant::INT, "text_editor/open_scripts/list_script_names_as", PROPERTY_HINT_ENUM, "Name,Parent Directory And Name,Full Path"));
	EDITOR_DEF("text_editor/open_scripts/list_script_names_as", 0);
	EditorSettings::get_singleton()->add_property_hint(PropertyInfo(Variant::STRING, "text_editor/external/exec_path", PROPERTY_HINT_GLOBAL_FILE));
	EDITOR_DEF("text_editor/external/exec_flags", "");

	ED_SHORTCUT("script_editor/open_recent", TTR("Open Recent"), KEY_MASK_CMD | KEY_MASK_SHIFT | KEY_T);
	ED_SHORTCUT("script_editor/clear_recent", TTR("Clear Recent Files"));
}

ScriptEditorPlugin::~ScriptEditorPlugin() {
}
