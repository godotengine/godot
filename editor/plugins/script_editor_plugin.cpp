/*************************************************************************/
/*  script_editor_plugin.cpp                                             */
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

#include "script_editor_plugin.h"

#include "core/config/project_settings.h"
#include "core/input/input.h"
#include "core/io/file_access.h"
#include "core/io/resource_loader.h"
#include "core/os/keyboard.h"
#include "core/os/os.h"
#include "editor/debugger/editor_debugger_node.h"
#include "editor/debugger/script_editor_debugger.h"
#include "editor/editor_node.h"
#include "editor/editor_run_script.h"
#include "editor/editor_scale.h"
#include "editor/editor_settings.h"
#include "editor/filesystem_dock.h"
#include "editor/find_in_files.h"
#include "editor/node_dock.h"
#include "editor/plugins/shader_editor_plugin.h"
#include "modules/visual_script/editor/visual_script_editor.h"
#include "scene/main/window.h"
#include "scene/scene_string_names.h"
#include "script_text_editor.h"
#include "servers/display_server.h"
#include "text_editor.h"

/*** SYNTAX HIGHLIGHTER ****/

String EditorSyntaxHighlighter::_get_name() const {
	String ret;
	if (GDVIRTUAL_CALL(_get_name, ret)) {
		return ret;
	}
	return "Unnamed";
}

Array EditorSyntaxHighlighter::_get_supported_languages() const {
	Array ret;
	if (GDVIRTUAL_CALL(_get_supported_languages, ret)) {
		return ret;
	}
	return Array();
}

Ref<EditorSyntaxHighlighter> EditorSyntaxHighlighter::_create() const {
	Ref<EditorSyntaxHighlighter> syntax_highlighter;
	syntax_highlighter.instantiate();
	if (get_script_instance()) {
		syntax_highlighter->set_script(get_script_instance()->get_script());
	}
	return syntax_highlighter;
}

void EditorSyntaxHighlighter::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_get_edited_resource"), &EditorSyntaxHighlighter::_get_edited_resource);

	GDVIRTUAL_BIND(_get_name)
	GDVIRTUAL_BIND(_get_supported_languages)
}

////

void EditorStandardSyntaxHighlighter::_update_cache() {
	highlighter->set_text_edit(text_edit);
	highlighter->clear_keyword_colors();
	highlighter->clear_member_keyword_colors();
	highlighter->clear_color_regions();

	highlighter->set_symbol_color(EDITOR_GET("text_editor/theme/highlighting/symbol_color"));
	highlighter->set_function_color(EDITOR_GET("text_editor/theme/highlighting/function_color"));
	highlighter->set_number_color(EDITOR_GET("text_editor/theme/highlighting/number_color"));
	highlighter->set_member_variable_color(EDITOR_GET("text_editor/theme/highlighting/member_variable_color"));

	/* Engine types. */
	const Color type_color = EDITOR_GET("text_editor/theme/highlighting/engine_type_color");
	List<StringName> types;
	ClassDB::get_class_list(&types);
	for (const StringName &E : types) {
		highlighter->add_keyword_color(E, type_color);
	}

	/* User types. */
	const Color usertype_color = EDITOR_GET("text_editor/theme/highlighting/user_type_color");
	List<StringName> global_classes;
	ScriptServer::get_global_class_list(&global_classes);
	for (const StringName &E : global_classes) {
		highlighter->add_keyword_color(E, usertype_color);
	}

	/* Autoloads. */
	OrderedHashMap<StringName, ProjectSettings::AutoloadInfo> autoloads = ProjectSettings::get_singleton()->get_autoload_list();
	for (OrderedHashMap<StringName, ProjectSettings::AutoloadInfo>::Element E = autoloads.front(); E; E = E.next()) {
		const ProjectSettings::AutoloadInfo &info = E.value();
		if (info.is_singleton) {
			highlighter->add_keyword_color(info.name, usertype_color);
		}
	}

	const Ref<Script> script = _get_edited_resource();
	if (script.is_valid()) {
		/* Core types. */
		const Color basetype_color = EDITOR_GET("text_editor/theme/highlighting/base_type_color");
		List<String> core_types;
		script->get_language()->get_core_type_words(&core_types);
		for (const String &E : core_types) {
			highlighter->add_keyword_color(E, basetype_color);
		}

		/* Reserved words. */
		const Color keyword_color = EDITOR_GET("text_editor/theme/highlighting/keyword_color");
		const Color control_flow_keyword_color = EDITOR_GET("text_editor/theme/highlighting/control_flow_keyword_color");
		List<String> keywords;
		script->get_language()->get_reserved_words(&keywords);
		for (const String &E : keywords) {
			if (script->get_language()->is_control_flow_keyword(E)) {
				highlighter->add_keyword_color(E, control_flow_keyword_color);
			} else {
				highlighter->add_keyword_color(E, keyword_color);
			}
		}

		/* Member types. */
		const Color member_variable_color = EDITOR_GET("text_editor/theme/highlighting/member_variable_color");
		StringName instance_base = script->get_instance_base_type();
		if (instance_base != StringName()) {
			List<PropertyInfo> plist;
			ClassDB::get_property_list(instance_base, &plist);
			for (const PropertyInfo &E : plist) {
				String name = E.name;
				if (E.usage & PROPERTY_USAGE_CATEGORY || E.usage & PROPERTY_USAGE_GROUP || E.usage & PROPERTY_USAGE_SUBGROUP) {
					continue;
				}
				if (name.find("/") != -1) {
					continue;
				}
				highlighter->add_member_keyword_color(name, member_variable_color);
			}

			List<String> clist;
			ClassDB::get_integer_constant_list(instance_base, &clist);
			for (const String &E : clist) {
				highlighter->add_member_keyword_color(E, member_variable_color);
			}
		}

		/* Comments */
		const Color comment_color = EDITOR_GET("text_editor/theme/highlighting/comment_color");
		List<String> comments;
		script->get_language()->get_comment_delimiters(&comments);
		for (const String &comment : comments) {
			String beg = comment.get_slice(" ", 0);
			String end = comment.get_slice_count(" ") > 1 ? comment.get_slice(" ", 1) : String();
			highlighter->add_color_region(beg, end, comment_color, end.is_empty());
		}

		/* Strings */
		const Color string_color = EDITOR_GET("text_editor/theme/highlighting/string_color");
		List<String> strings;
		script->get_language()->get_string_delimiters(&strings);
		for (const String &string : strings) {
			String beg = string.get_slice(" ", 0);
			String end = string.get_slice_count(" ") > 1 ? string.get_slice(" ", 1) : String();
			highlighter->add_color_region(beg, end, string_color, end.is_empty());
		}
	}
}

Ref<EditorSyntaxHighlighter> EditorStandardSyntaxHighlighter::_create() const {
	Ref<EditorStandardSyntaxHighlighter> syntax_highlighter;
	syntax_highlighter.instantiate();
	return syntax_highlighter;
}

////

Ref<EditorSyntaxHighlighter> EditorPlainTextSyntaxHighlighter::_create() const {
	Ref<EditorPlainTextSyntaxHighlighter> syntax_highlighter;
	syntax_highlighter.instantiate();
	return syntax_highlighter;
}

////////////////////////////////////////////////////////////////////////////////

/*** SCRIPT EDITOR ****/

void ScriptEditorBase::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_base_editor"), &ScriptEditorBase::get_base_editor);
	ClassDB::bind_method(D_METHOD("add_syntax_highlighter", "highlighter"), &ScriptEditorBase::add_syntax_highlighter);

	ADD_SIGNAL(MethodInfo("name_changed"));
	ADD_SIGNAL(MethodInfo("edited_script_changed"));
	ADD_SIGNAL(MethodInfo("request_help", PropertyInfo(Variant::STRING, "topic")));
	ADD_SIGNAL(MethodInfo("request_open_script_at_line", PropertyInfo(Variant::OBJECT, "script"), PropertyInfo(Variant::INT, "line")));
	ADD_SIGNAL(MethodInfo("request_save_history"));
	ADD_SIGNAL(MethodInfo("go_to_help", PropertyInfo(Variant::STRING, "what")));
	// TODO: This signal is no use for VisualScript.
	ADD_SIGNAL(MethodInfo("search_in_files_requested", PropertyInfo(Variant::STRING, "text")));
	ADD_SIGNAL(MethodInfo("replace_in_files_requested", PropertyInfo(Variant::STRING, "text")));
}

class EditorScriptCodeCompletionCache : public ScriptCodeCompletionCache {
	struct Cache {
		uint64_t time_loaded = 0;
		RES cache;
	};

	Map<String, Cache> cached;

public:
	uint64_t max_time_cache = 5 * 60 * 1000; //minutes, five
	int max_cache_size = 128;

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

	virtual RES get_cached_resource(const String &p_path) {
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

			if (O != E) { //should never happen..
				cached.erase(O);
			}
		}

		return E->get().cache;
	}

	virtual ~EditorScriptCodeCompletionCache() {}
};

void ScriptEditorQuickOpen::popup_dialog(const Vector<String> &p_functions, bool p_dontclear) {
	popup_centered_ratio(0.6);
	if (p_dontclear) {
		search_box->select_all();
	} else {
		search_box->clear();
	}
	search_box->grab_focus();
	functions = p_functions;
	_update_search();
}

void ScriptEditorQuickOpen::_text_changed(const String &p_newtext) {
	_update_search();
}

void ScriptEditorQuickOpen::_sbox_input(const Ref<InputEvent> &p_ie) {
	Ref<InputEventKey> k = p_ie;

	if (k.is_valid() && (k->get_keycode() == Key::UP || k->get_keycode() == Key::DOWN || k->get_keycode() == Key::PAGEUP || k->get_keycode() == Key::PAGEDOWN)) {
		search_options->gui_input(k);
		search_box->accept_event();
	}
}

void ScriptEditorQuickOpen::_update_search() {
	search_options->clear();
	TreeItem *root = search_options->create_item();

	for (int i = 0; i < functions.size(); i++) {
		String file = functions[i];
		if ((search_box->get_text().is_empty() || file.findn(search_box->get_text()) != -1)) {
			TreeItem *ti = search_options->create_item(root);
			ti->set_text(0, file);
			if (root->get_first_child() == ti) {
				ti->select(0);
			}
		}
	}

	get_ok_button()->set_disabled(root->get_first_child() == nullptr);
}

void ScriptEditorQuickOpen::_confirmed() {
	TreeItem *ti = search_options->get_selected();
	if (!ti) {
		return;
	}
	int line = ti->get_text(0).get_slice(":", 1).to_int();

	emit_signal(SNAME("goto_line"), line - 1);
	hide();
}

void ScriptEditorQuickOpen::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			connect("confirmed", callable_mp(this, &ScriptEditorQuickOpen::_confirmed));

			search_box->set_clear_button_enabled(true);
			[[fallthrough]];
		}
		case NOTIFICATION_VISIBILITY_CHANGED: {
			search_box->set_right_icon(search_options->get_theme_icon(SNAME("Search"), SNAME("EditorIcons")));
		} break;
		case NOTIFICATION_EXIT_TREE: {
			disconnect("confirmed", callable_mp(this, &ScriptEditorQuickOpen::_confirmed));
		} break;
	}
}

void ScriptEditorQuickOpen::_bind_methods() {
	ADD_SIGNAL(MethodInfo("goto_line", PropertyInfo(Variant::INT, "line")));
}

ScriptEditorQuickOpen::ScriptEditorQuickOpen() {
	VBoxContainer *vbc = memnew(VBoxContainer);
	add_child(vbc);
	search_box = memnew(LineEdit);
	vbc->add_margin_child(TTR("Search:"), search_box);
	search_box->connect("text_changed", callable_mp(this, &ScriptEditorQuickOpen::_text_changed));
	search_box->connect("gui_input", callable_mp(this, &ScriptEditorQuickOpen::_sbox_input));
	search_options = memnew(Tree);
	vbc->add_margin_child(TTR("Matches:"), search_options, true);
	get_ok_button()->set_text(TTR("Open"));
	get_ok_button()->set_disabled(true);
	register_text_enter(search_box);
	set_hide_on_ok(false);
	search_options->connect("item_activated", callable_mp(this, &ScriptEditorQuickOpen::_confirmed));
	search_options->set_hide_root(true);
	search_options->set_hide_folding(true);
	search_options->add_theme_constant_override("draw_guides", 1);
}

/////////////////////////////////

ScriptEditor *ScriptEditor::script_editor = nullptr;

/*** SCRIPT EDITOR ******/

String ScriptEditor::_get_debug_tooltip(const String &p_text, Node *_se) {
	String val = EditorDebuggerNode::get_singleton()->get_var_value(p_text);
	if (!val.is_empty()) {
		return p_text + ": " + val;
	} else {
		return String();
	}
}

void ScriptEditor::_breaked(bool p_breaked, bool p_can_debug) {
	if (bool(EditorSettings::get_singleton()->get("text_editor/external/use_external_editor"))) {
		return;
	}

	for (int i = 0; i < tab_container->get_child_count(); i++) {
		ScriptEditorBase *se = Object::cast_to<ScriptEditorBase>(tab_container->get_child(i));
		if (!se) {
			continue;
		}

		se->set_debugger_active(p_breaked);
	}
}

void ScriptEditor::_script_created(Ref<Script> p_script) {
	editor->push_item(p_script.operator->());
}

void ScriptEditor::_goto_script_line2(int p_line) {
	ScriptEditorBase *current = _get_current_editor();
	if (current) {
		current->goto_line(p_line);
	}
}

void ScriptEditor::_goto_script_line(REF p_script, int p_line) {
	Ref<Script> script = Object::cast_to<Script>(*p_script);
	if (script.is_valid() && (script->has_source_code() || script->get_path().is_resource_file())) {
		if (edit(p_script, p_line, 0)) {
			editor->push_item(p_script.ptr());

			ScriptEditorBase *current = _get_current_editor();
			if (ScriptTextEditor *script_text_editor = Object::cast_to<ScriptTextEditor>(current)) {
				script_text_editor->goto_line_centered(p_line);
			} else if (current) {
				current->goto_line(p_line, true);
			}
		}
	}
}

void ScriptEditor::_set_execution(REF p_script, int p_line) {
	Ref<Script> script = Object::cast_to<Script>(*p_script);
	if (script.is_valid() && (script->has_source_code() || script->get_path().is_resource_file())) {
		for (int i = 0; i < tab_container->get_child_count(); i++) {
			ScriptEditorBase *se = Object::cast_to<ScriptEditorBase>(tab_container->get_child(i));
			if (!se) {
				continue;
			}

			if ((script != nullptr && se->get_edited_resource() == p_script) || se->get_edited_resource()->get_path() == script->get_path()) {
				se->set_executing_line(p_line);
			}
		}
	}
}

void ScriptEditor::_clear_execution(REF p_script) {
	Ref<Script> script = Object::cast_to<Script>(*p_script);
	if (script.is_valid() && (script->has_source_code() || script->get_path().is_resource_file())) {
		for (int i = 0; i < tab_container->get_child_count(); i++) {
			ScriptEditorBase *se = Object::cast_to<ScriptEditorBase>(tab_container->get_child(i));
			if (!se) {
				continue;
			}

			if ((script != nullptr && se->get_edited_resource() == p_script) || se->get_edited_resource()->get_path() == script->get_path()) {
				se->clear_executing_line();
			}
		}
	}
}

void ScriptEditor::_set_breakpoint(REF p_script, int p_line, bool p_enabled) {
	Ref<Script> script = Object::cast_to<Script>(*p_script);
	if (script.is_valid() && (script->has_source_code() || script->get_path().is_resource_file())) {
		// Update if open.
		for (int i = 0; i < tab_container->get_child_count(); i++) {
			ScriptEditorBase *se = Object::cast_to<ScriptEditorBase>(tab_container->get_child(i));
			if (se && se->get_edited_resource()->get_path() == script->get_path()) {
				se->set_breakpoint(p_line, p_enabled);
				return;
			}
		}

		// Handle closed.
		Dictionary state = script_editor_cache->get_value(script->get_path(), "state");
		Array breakpoints;
		if (state.has("breakpoints")) {
			breakpoints = state["breakpoints"];
		}

		if (breakpoints.has(p_line)) {
			if (!p_enabled) {
				breakpoints.erase(p_line);
			}
		} else if (p_enabled) {
			breakpoints.push_back(p_line);
		}
		state["breakpoints"] = breakpoints;
		script_editor_cache->set_value(script->get_path(), "state", state);
		EditorDebuggerNode::get_singleton()->set_breakpoint(script->get_path(), p_line + 1, false);
	}
}

void ScriptEditor::_clear_breakpoints() {
	for (int i = 0; i < tab_container->get_child_count(); i++) {
		ScriptEditorBase *se = Object::cast_to<ScriptEditorBase>(tab_container->get_child(i));
		if (se) {
			se->clear_breakpoints();
		}
	}

	// Clear from closed scripts.
	List<String> cached_editors;
	script_editor_cache->get_sections(&cached_editors);
	for (const String &E : cached_editors) {
		Array breakpoints = _get_cached_breakpoints_for_script(E);
		for (int i = 0; i < breakpoints.size(); i++) {
			EditorDebuggerNode::get_singleton()->set_breakpoint(E, (int)breakpoints[i] + 1, false);
		}

		if (breakpoints.size() > 0) {
			Dictionary state = script_editor_cache->get_value(E, "state");
			state["breakpoints"] = Array();
			script_editor_cache->set_value(E, "state", state);
		}
	}
}

Array ScriptEditor::_get_cached_breakpoints_for_script(const String &p_path) const {
	if (!ResourceLoader::exists(p_path, "Script") || p_path.begins_with("local://") || !script_editor_cache->has_section_key(p_path, "state")) {
		return Array();
	}

	Dictionary state = script_editor_cache->get_value(p_path, "state");
	if (!state.has("breakpoints")) {
		return Array();
	}
	return state["breakpoints"];
}

ScriptEditorBase *ScriptEditor::_get_current_editor() const {
	int selected = tab_container->get_current_tab();
	if (selected < 0 || selected >= tab_container->get_child_count()) {
		return nullptr;
	}

	return Object::cast_to<ScriptEditorBase>(tab_container->get_child(selected));
}

void ScriptEditor::_update_history_arrows() {
	script_back->set_disabled(history_pos <= 0);
	script_forward->set_disabled(history_pos >= history.size() - 1);
}

void ScriptEditor::_save_history() {
	if (history_pos >= 0 && history_pos < history.size() && history[history_pos].control == tab_container->get_current_tab_control()) {
		Node *n = tab_container->get_current_tab_control();

		if (Object::cast_to<ScriptEditorBase>(n)) {
			history.write[history_pos].state = Object::cast_to<ScriptEditorBase>(n)->get_edit_state();
		}
		if (Object::cast_to<EditorHelp>(n)) {
			history.write[history_pos].state = Object::cast_to<EditorHelp>(n)->get_scroll();
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
	ScriptEditorBase *current = _get_current_editor();
	if (current) {
		if (current->is_unsaved()) {
			current->apply_code();
		}
	}

	Control *c = Object::cast_to<Control>(tab_container->get_child(p_idx));
	if (!c) {
		return;
	}

	if (history_pos >= 0 && history_pos < history.size() && history[history_pos].control == tab_container->get_current_tab_control()) {
		Node *n = tab_container->get_current_tab_control();

		if (Object::cast_to<ScriptEditorBase>(n)) {
			history.write[history_pos].state = Object::cast_to<ScriptEditorBase>(n)->get_edit_state();
		}
		if (Object::cast_to<EditorHelp>(n)) {
			history.write[history_pos].state = Object::cast_to<EditorHelp>(n)->get_scroll();
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
		script_icon->set_texture(Object::cast_to<ScriptEditorBase>(c)->get_theme_icon());
		if (is_visible_in_tree()) {
			Object::cast_to<ScriptEditorBase>(c)->ensure_focus();
		}

		Ref<Script> script = Object::cast_to<ScriptEditorBase>(c)->get_edited_resource();
		if (script != nullptr) {
			notify_script_changed(script);
		}

		Object::cast_to<ScriptEditorBase>(c)->validate();
	}
	if (Object::cast_to<EditorHelp>(c)) {
		script_name_label->set_text(Object::cast_to<EditorHelp>(c)->get_class());
		script_icon->set_texture(get_theme_icon(SNAME("Help"), SNAME("EditorIcons")));
		if (is_visible_in_tree()) {
			Object::cast_to<EditorHelp>(c)->set_focused();
		}
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
	if (p_path.is_empty()) {
		return;
	}

	Array rc = EditorSettings::get_singleton()->get_project_metadata("recent_files", "scripts", Array());
	if (rc.find(p_path) != -1) {
		rc.erase(p_path);
	}
	rc.push_front(p_path);
	if (rc.size() > 10) {
		rc.resize(10);
	}

	EditorSettings::get_singleton()->set_project_metadata("recent_files", "scripts", rc);
	_update_recent_scripts();
}

void ScriptEditor::_update_recent_scripts() {
	Array rc = EditorSettings::get_singleton()->get_project_metadata("recent_files", "scripts", Array());
	recent_scripts->clear();

	String path;
	for (int i = 0; i < rc.size(); i++) {
		path = rc[i];
		recent_scripts->add_item(path.replace("res://", ""));
	}

	recent_scripts->add_separator();
	recent_scripts->add_shortcut(ED_SHORTCUT("script_editor/clear_recent", TTR("Clear Recent Files")));

	recent_scripts->set_as_minsize();
}

void ScriptEditor::_open_recent_script(int p_idx) {
	// clear button
	if (p_idx == recent_scripts->get_item_count() - 1) {
		EditorSettings::get_singleton()->set_project_metadata("recent_files", "scripts", Array());
		call_deferred(SNAME("_update_recent_scripts"));
		return;
	}

	Array rc = EditorSettings::get_singleton()->get_project_metadata("recent_files", "scripts", Array());
	ERR_FAIL_INDEX(p_idx, rc.size());

	String path = rc[p_idx];
	// if its not on disk its a help file or deleted
	if (FileAccess::exists(path)) {
		List<String> extensions;
		ResourceLoader::get_recognized_extensions_for_type("Script", &extensions);

		if (extensions.find(path.get_extension())) {
			Ref<Script> script = ResourceLoader::load(path);
			if (script.is_valid()) {
				edit(script, true);
				return;
			}
		}

		Error err;
		Ref<TextFile> text_file = _load_text_file(path, &err);
		if (text_file.is_valid()) {
			edit(text_file, true);
			return;
		}
		// if it's a path then it's most likely a deleted file not help
	} else if (path.find("::") != -1) {
		// built-in script
		String res_path = path.get_slice("::", 0);
		if (ResourceLoader::get_resource_type(res_path) == "PackedScene") {
			if (!EditorNode::get_singleton()->is_scene_open(res_path)) {
				EditorNode::get_singleton()->load_scene(res_path);
			}
		} else {
			EditorNode::get_singleton()->load_resource(res_path);
		}
		Ref<Script> script = ResourceLoader::load(path);
		if (script.is_valid()) {
			edit(script, true);
			return;
		}
	} else if (!path.is_resource_file()) {
		_help_class_open(path);
		return;
	}

	rc.remove_at(p_idx);
	EditorSettings::get_singleton()->set_project_metadata("recent_files", "scripts", rc);
	_update_recent_scripts();
	_show_error_dialog(path);
}

void ScriptEditor::_show_error_dialog(String p_path) {
	error_dialog->set_text(vformat(TTR("Can't open '%s'. The file could have been moved or deleted."), p_path));
	error_dialog->popup_centered();
}

void ScriptEditor::_close_tab(int p_idx, bool p_save, bool p_history_back) {
	int selected = p_idx;
	if (selected < 0 || selected >= tab_container->get_child_count()) {
		return;
	}

	Node *tselected = tab_container->get_child(selected);

	ScriptEditorBase *current = Object::cast_to<ScriptEditorBase>(tselected);
	if (current) {
		RES file = current->get_edited_resource();
		if (p_save && file.is_valid()) {
			// Do not try to save internal scripts, but prompt to save in-memory
			// scripts which are not saved to disk yet (have empty path).
			if (file->is_built_in()) {
				save_current_script();
			}
		}
		if (file.is_valid()) {
			if (!file->get_path().is_empty()) {
				// Only saved scripts can be restored.
				previous_scripts.push_back(file->get_path());
			}

			Ref<Script> script = file;
			if (script.is_valid()) {
				notify_script_close(script);
			}
		}
	}

	// roll back to previous tab
	if (p_history_back) {
		_history_back();
	}

	//remove from history
	history.resize(history_pos + 1);

	for (int i = 0; i < history.size(); i++) {
		if (history[i].control == tselected) {
			history.remove_at(i);
			i--;
			history_pos--;
		}
	}

	if (history_pos >= history.size()) {
		history_pos = history.size() - 1;
	}

	int idx = tab_container->get_current_tab();
	if (current) {
		current->clear_edit_menu();
		_save_editor_state(current);
	}
	memdelete(tselected);
	if (idx >= tab_container->get_child_count()) {
		idx = tab_container->get_child_count() - 1;
	}
	if (idx >= 0) {
		if (history_pos >= 0) {
			idx = history[history_pos].control->get_index();
		}
		tab_container->set_current_tab(idx);
	} else {
		_update_selected_editor_menu();
	}

	_update_history_arrows();

	_update_script_names();
	_update_members_overview_visibility();
	_update_help_overview_visibility();
	_save_layout();
	_update_find_replace_bar();
}

void ScriptEditor::_close_current_tab(bool p_save) {
	_close_tab(tab_container->get_current_tab(), p_save);
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
			_close_tab(i, true, false);
		}
	}
}

void ScriptEditor::_copy_script_path() {
	ScriptEditorBase *se = _get_current_editor();
	if (se) {
		RES script = se->get_edited_resource();
		DisplayServer::get_singleton()->clipboard_set(script->get_path());
	}
}

void ScriptEditor::_close_other_tabs() {
	int current_idx = tab_container->get_current_tab();
	for (int i = tab_container->get_child_count() - 1; i >= 0; i--) {
		if (i != current_idx) {
			script_close_queue.push_back(i);
		}
	}
	_queue_close_tabs();
}

void ScriptEditor::_close_all_tabs() {
	for (int i = tab_container->get_child_count() - 1; i >= 0; i--) {
		script_close_queue.push_back(i);
	}
	_queue_close_tabs();
}

void ScriptEditor::_queue_close_tabs() {
	while (!script_close_queue.is_empty()) {
		int idx = script_close_queue.front()->get();
		script_close_queue.pop_front();

		tab_container->set_current_tab(idx);
		ScriptEditorBase *se = Object::cast_to<ScriptEditorBase>(tab_container->get_child(idx));
		if (se) {
			// Maybe there are unsaved changes.
			if (se->is_unsaved()) {
				_ask_close_current_unsaved_tab(se);
				erase_tab_confirm->connect(SceneStringNames::get_singleton()->visibility_changed, callable_mp(this, &ScriptEditor::_queue_close_tabs), varray(), CONNECT_ONESHOT);
				break;
			}
		}

		_close_current_tab(false);
	}
	_update_find_replace_bar();
}

void ScriptEditor::_ask_close_current_unsaved_tab(ScriptEditorBase *current) {
	erase_tab_confirm->set_text(TTR("Close and save changes?") + "\n\"" + current->get_name() + "\"");
	erase_tab_confirm->popup_centered();
}

void ScriptEditor::_resave_scripts(const String &p_str) {
	apply_scripts();

	for (int i = 0; i < tab_container->get_child_count(); i++) {
		ScriptEditorBase *se = Object::cast_to<ScriptEditorBase>(tab_container->get_child(i));
		if (!se) {
			continue;
		}

		RES script = se->get_edited_resource();

		if (script->is_built_in()) {
			continue; //internal script, who cares
		}

		if (trim_trailing_whitespace_on_save) {
			se->trim_trailing_whitespace();
		}

		se->insert_final_newline();

		if (convert_indent_on_save) {
			if (use_space_indentation) {
				se->convert_indent_to_spaces();
			} else {
				se->convert_indent_to_tabs();
			}
		}

		Ref<TextFile> text_file = script;
		if (text_file != nullptr) {
			se->apply_code();
			_save_text_file(text_file, text_file->get_path());
			break;
		} else {
			editor->save_resource(script);
		}
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

		RES edited_res = se->get_edited_resource();

		if (edited_res->is_built_in()) {
			continue; //internal script, who cares
		}

		uint64_t last_date = edited_res->get_last_modified_time();
		uint64_t date = FileAccess::get_modified_time(edited_res->get_path());

		if (last_date == date) {
			continue;
		}

		Ref<Script> script = edited_res;
		if (script != nullptr) {
			Ref<Script> rel_script = ResourceLoader::load(script->get_path(), script->get_class(), ResourceFormatLoader::CACHE_MODE_IGNORE);
			ERR_CONTINUE(!rel_script.is_valid());
			script->set_source_code(rel_script->get_source_code());
			script->set_last_modified_time(rel_script->get_last_modified_time());
			script->reload();
		}

		Ref<TextFile> text_file = edited_res;
		if (text_file != nullptr) {
			Error err;
			Ref<TextFile> rel_text_file = _load_text_file(text_file->get_path(), &err);
			ERR_CONTINUE(!rel_text_file.is_valid());
			text_file->set_text(rel_text_file->get_text());
			text_file->set_last_modified_time(rel_text_file->get_last_modified_time());
		}
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

		RES script = se->get_edited_resource();

		if (script == p_res) {
			se->tag_saved_version();
		}
	}

	_update_script_names();
	_trigger_live_script_reload();
}

void ScriptEditor::_scene_saved_callback(const String &p_path) {
	// If scene was saved, mark all built-in scripts from that scene as saved.
	for (int i = 0; i < tab_container->get_child_count(); i++) {
		ScriptEditorBase *se = Object::cast_to<ScriptEditorBase>(tab_container->get_child(i));
		if (!se) {
			continue;
		}

		RES edited_res = se->get_edited_resource();

		if (!edited_res->is_built_in()) {
			continue; // External script, who cares.
		}

		if (edited_res->get_path().get_slice("::", 0) == p_path) {
			se->tag_saved_version();
		}

		Ref<Script> scr = edited_res;
		if (scr.is_valid() && scr->is_tool()) {
			scr->reload(true);
		}
	}
}

void ScriptEditor::_trigger_live_script_reload() {
	if (!pending_auto_reload && auto_reload_running_scripts) {
		call_deferred(SNAME("_live_auto_reload_running_scripts"));
		pending_auto_reload = true;
	}
}

void ScriptEditor::_live_auto_reload_running_scripts() {
	pending_auto_reload = false;
	EditorDebuggerNode::get_singleton()->reload_scripts();
}

bool ScriptEditor::_test_script_times_on_disk(RES p_for_script) {
	disk_changed_list->clear();
	TreeItem *r = disk_changed_list->create_item();
	disk_changed_list->set_hide_root(true);

	bool need_ask = false;
	bool need_reload = false;
	bool use_autoreload = bool(EDITOR_DEF("text_editor/behavior/files/auto_reload_scripts_on_external_change", false));

	for (int i = 0; i < tab_container->get_child_count(); i++) {
		ScriptEditorBase *se = Object::cast_to<ScriptEditorBase>(tab_container->get_child(i));
		if (se) {
			RES edited_res = se->get_edited_resource();
			if (p_for_script.is_valid() && edited_res.is_valid() && p_for_script != edited_res) {
				continue;
			}

			if (edited_res->is_built_in()) {
				continue; //internal script, who cares
			}

			uint64_t last_date = edited_res->get_last_modified_time();
			uint64_t date = FileAccess::get_modified_time(edited_res->get_path());

			if (last_date != date) {
				TreeItem *ti = disk_changed_list->create_item(r);
				ti->set_text(0, edited_res->get_path().get_file());

				if (!use_autoreload || se->is_unsaved()) {
					need_ask = true;
				}
				need_reload = true;
			}
		}
	}

	if (need_reload) {
		if (!need_ask) {
			script_editor->_reload_scripts();
			need_reload = false;
		} else {
			disk_changed->call_deferred(SNAME("popup_centered_ratio"), 0.5);
		}
	}

	return need_reload;
}

void ScriptEditor::_file_dialog_action(String p_file) {
	switch (file_dialog_option) {
		case FILE_NEW_TEXTFILE: {
			Error err;
			FileAccess *file = FileAccess::open(p_file, FileAccess::WRITE, &err);
			if (err) {
				editor->show_warning(TTR("Error writing TextFile:") + "\n" + p_file, TTR("Error!"));
				break;
			}
			file->close();
			memdelete(file);

			if (EditorFileSystem::get_singleton()) {
				if (textfile_extensions.has(p_file.get_extension())) {
					EditorFileSystem::get_singleton()->update_file(p_file);
				}
			}

			if (!open_textfile_after_create) {
				return;
			}
			[[fallthrough]];
		}
		case FILE_OPEN: {
			open_file(p_file);
			file_dialog_option = -1;
		} break;
		case FILE_SAVE_AS: {
			ScriptEditorBase *current = _get_current_editor();
			if (current) {
				RES resource = current->get_edited_resource();
				String path = ProjectSettings::get_singleton()->localize_path(p_file);
				Error err = _save_text_file(resource, path);

				if (err != OK) {
					editor->show_accept(TTR("Error saving file!"), TTR("OK"));
					return;
				}

				resource->set_path(path);
				_update_script_names();
			}
		} break;
		case THEME_SAVE_AS: {
			if (!EditorSettings::get_singleton()->save_text_editor_theme_as(p_file)) {
				editor->show_warning(TTR("Error while saving theme."), TTR("Error Saving"));
			}
		} break;
		case THEME_IMPORT: {
			if (!EditorSettings::get_singleton()->import_text_editor_theme(p_file)) {
				editor->show_warning(TTR("Error importing theme."), TTR("Error Importing"));
			}
		} break;
	}
	file_dialog_option = -1;
}

Ref<Script> ScriptEditor::_get_current_script() {
	ScriptEditorBase *current = _get_current_editor();

	if (current) {
		Ref<Script> script = current->get_edited_resource();
		return script != nullptr ? script : nullptr;
	} else {
		return nullptr;
	}
}

Array ScriptEditor::_get_open_scripts() const {
	Array ret;
	Vector<Ref<Script>> scripts = get_open_scripts();
	int scrits_amount = scripts.size();
	for (int idx_script = 0; idx_script < scrits_amount; idx_script++) {
		ret.push_back(scripts[idx_script]);
	}
	return ret;
}

bool ScriptEditor::toggle_scripts_panel() {
	list_split->set_visible(!list_split->is_visible());
	return list_split->is_visible();
}

bool ScriptEditor::is_scripts_panel_toggled() {
	return list_split->is_visible();
}

void ScriptEditor::_menu_option(int p_option) {
	ScriptEditorBase *current = _get_current_editor();
	switch (p_option) {
		case FILE_NEW: {
			script_create_dialog->config("Node", "new_script", false, false);
			script_create_dialog->popup_centered();
		} break;
		case FILE_NEW_TEXTFILE: {
			file_dialog->set_file_mode(EditorFileDialog::FILE_MODE_SAVE_FILE);
			file_dialog->set_access(EditorFileDialog::ACCESS_FILESYSTEM);
			file_dialog_option = FILE_NEW_TEXTFILE;

			file_dialog->clear_filters();
			for (const String &E : textfile_extensions) {
				file_dialog->add_filter("*." + E + " ; " + E.to_upper());
			}
			file_dialog->popup_file_dialog();
			file_dialog->set_title(TTR("New Text File..."));
			open_textfile_after_create = true;
		} break;
		case FILE_OPEN: {
			file_dialog->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_FILE);
			file_dialog->set_access(EditorFileDialog::ACCESS_FILESYSTEM);
			file_dialog_option = FILE_OPEN;

			List<String> extensions;
			ResourceLoader::get_recognized_extensions_for_type("Script", &extensions);
			file_dialog->clear_filters();
			for (int i = 0; i < extensions.size(); i++) {
				file_dialog->add_filter("*." + extensions[i] + " ; " + extensions[i].to_upper());
			}

			for (const String &E : textfile_extensions) {
				file_dialog->add_filter("*." + E + " ; " + E.to_upper());
			}

			file_dialog->popup_file_dialog();
			file_dialog->set_title(TTR("Open File"));
			return;
		} break;
		case FILE_REOPEN_CLOSED: {
			if (previous_scripts.is_empty()) {
				return;
			}

			String path = previous_scripts.back()->get();
			previous_scripts.pop_back();

			List<String> extensions;
			ResourceLoader::get_recognized_extensions_for_type("Script", &extensions);
			bool built_in = !path.is_resource_file();

			if (extensions.find(path.get_extension()) || built_in) {
				if (built_in) {
					String res_path = path.get_slice("::", 0);
					if (ResourceLoader::get_resource_type(res_path) == "PackedScene") {
						if (!EditorNode::get_singleton()->is_scene_open(res_path)) {
							EditorNode::get_singleton()->load_scene(res_path);
							script_editor->call_deferred(SNAME("_menu_option"), p_option);
							previous_scripts.push_back(path); //repeat the operation
							return;
						}
					} else {
						EditorNode::get_singleton()->load_resource(res_path);
					}
				}

				Ref<Script> scr = ResourceLoader::load(path);
				if (!scr.is_valid()) {
					editor->show_warning(TTR("Could not load file at:") + "\n\n" + path, TTR("Error!"));
					file_dialog_option = -1;
					return;
				}

				edit(scr);
				file_dialog_option = -1;
				return;
			} else {
				Error error;
				Ref<TextFile> text_file = _load_text_file(path, &error);
				if (error != OK) {
					editor->show_warning(TTR("Could not load file at:") + "\n\n" + path, TTR("Error!"));
				}

				if (text_file.is_valid()) {
					edit(text_file);
					file_dialog_option = -1;
					return;
				}
			}
		} break;
		case FILE_SAVE_ALL: {
			if (_test_script_times_on_disk()) {
				return;
			}

			save_all_scripts();
		} break;
		case SEARCH_IN_FILES: {
			_on_find_in_files_requested("");
		} break;
		case REPLACE_IN_FILES: {
			_on_replace_in_files_requested("");
		} break;
		case SEARCH_HELP: {
			help_search_dialog->popup_dialog();
		} break;
		case SEARCH_WEBSITE: {
			OS::get_singleton()->shell_open("https://docs.godotengine.org/");
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
		case TOGGLE_SCRIPTS_PANEL: {
			toggle_scripts_panel();
			if (current) {
				current->update_toggle_scripts_button();
			} else {
				Control *tab = tab_container->get_current_tab_control();
				EditorHelp *editor_help = Object::cast_to<EditorHelp>(tab);
				if (editor_help) {
					editor_help->update_toggle_scripts_button();
				}
			}
		}
	}

	if (current) {
		switch (p_option) {
			case FILE_SAVE: {
				save_current_script();
			} break;
			case FILE_SAVE_AS: {
				if (trim_trailing_whitespace_on_save) {
					current->trim_trailing_whitespace();
				}

				current->insert_final_newline();

				if (convert_indent_on_save) {
					if (use_space_indentation) {
						current->convert_indent_to_spaces();
					} else {
						current->convert_indent_to_tabs();
					}
				}

				RES resource = current->get_edited_resource();
				Ref<TextFile> text_file = resource;
				Ref<Script> script = resource;

				if (text_file != nullptr) {
					file_dialog->set_file_mode(EditorFileDialog::FILE_MODE_SAVE_FILE);
					file_dialog->set_access(EditorFileDialog::ACCESS_FILESYSTEM);
					file_dialog_option = FILE_SAVE_AS;

					List<String> extensions;
					ResourceLoader::get_recognized_extensions_for_type("Script", &extensions);
					file_dialog->clear_filters();
					file_dialog->set_current_dir(text_file->get_path().get_base_dir());
					file_dialog->set_current_file(text_file->get_path().get_file());
					file_dialog->popup_file_dialog();
					file_dialog->set_title(TTR("Save File As..."));
					break;
				}

				if (script != nullptr) {
					const Vector<DocData::ClassDoc> &documentations = script->get_documentation();
					for (int j = 0; j < documentations.size(); j++) {
						const DocData::ClassDoc &doc = documentations.get(j);
						if (EditorHelp::get_doc_data()->has_doc(doc.name)) {
							EditorHelp::get_doc_data()->remove_doc(doc.name);
						}
					}
				}

				editor->push_item(resource.ptr());
				editor->save_resource_as(resource);

				if (script != nullptr) {
					const Vector<DocData::ClassDoc> &documentations = script->get_documentation();
					for (int j = 0; j < documentations.size(); j++) {
						const DocData::ClassDoc &doc = documentations.get(j);
						EditorHelp::get_doc_data()->add_doc(doc);
						update_doc(doc.name);
					}
				}
			} break;

			case FILE_TOOL_RELOAD:
			case FILE_TOOL_RELOAD_SOFT: {
				current->reload(p_option == FILE_TOOL_RELOAD_SOFT);

			} break;
			case FILE_RUN: {
				Ref<Script> scr = current->get_edited_resource();
				if (scr == nullptr || scr.is_null()) {
					EditorNode::get_singleton()->show_warning(TTR("Can't obtain the script for running."));
					break;
				}

				current->apply_code();
				Error err = scr->reload(false); //hard reload script before running always

				if (err != OK) {
					EditorNode::get_singleton()->show_warning(TTR("Script failed reloading, check console for errors."));
					return;
				}
				if (!scr->is_tool()) {
					EditorNode::get_singleton()->show_warning(TTR("Script is not in tool mode, will not be able to run."));
					return;
				}

				if (!ClassDB::is_parent_class(scr->get_instance_base_type(), "EditorScript")) {
					EditorNode::get_singleton()->show_warning(TTR("To run this script, it must inherit EditorScript and be set to tool mode."));
					return;
				}

				Ref<EditorScript> es = memnew(EditorScript);
				es->set_script(scr);
				es->set_editor(EditorNode::get_singleton());

				es->_run();

				EditorNode::get_undo_redo()->clear_history();
			} break;
			case FILE_CLOSE: {
				if (current->is_unsaved()) {
					_ask_close_current_unsaved_tab(current);
				} else {
					_close_current_tab(false);
				}
			} break;
			case FILE_COPY_PATH: {
				_copy_script_path();
			} break;
			case SHOW_IN_FILE_SYSTEM: {
				const RES script = current->get_edited_resource();
				String path = script->get_path();
				if (!path.is_empty()) {
					if (script->is_built_in()) {
						path = path.get_slice("::", 0); // Show the scene instead.
					}

					FileSystemDock *file_system_dock = EditorNode::get_singleton()->get_filesystem_dock();
					file_system_dock->navigate_to_path(path);
					// Ensure that the FileSystem dock is visible.
					TabContainer *tab_container = (TabContainer *)file_system_dock->get_parent_control();
					tab_container->set_current_tab(file_system_dock->get_index());
				}
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
					_update_script_names();
				}
			}
		}
	} else {
		EditorHelp *help = Object::cast_to<EditorHelp>(tab_container->get_current_tab_control());
		if (help) {
			switch (p_option) {
				case HELP_SEARCH_FIND: {
					help->popup_search();
				} break;
				case HELP_SEARCH_FIND_NEXT: {
					help->search_again();
				} break;
				case HELP_SEARCH_FIND_PREVIOUS: {
					help->search_again(true);
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

void ScriptEditor::_theme_option(int p_option) {
	switch (p_option) {
		case THEME_IMPORT: {
			file_dialog->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_FILE);
			file_dialog->set_access(EditorFileDialog::ACCESS_FILESYSTEM);
			file_dialog_option = THEME_IMPORT;
			file_dialog->clear_filters();
			file_dialog->add_filter("*.tet");
			file_dialog->popup_file_dialog();
			file_dialog->set_title(TTR("Import Theme"));
		} break;
		case THEME_RELOAD: {
			EditorSettings::get_singleton()->load_text_editor_theme();
		} break;
		case THEME_SAVE: {
			if (EditorSettings::get_singleton()->is_default_text_editor_theme()) {
				ScriptEditor::_show_save_theme_as_dialog();
			} else if (!EditorSettings::get_singleton()->save_text_editor_theme()) {
				editor->show_warning(TTR("Error while saving theme"), TTR("Error saving"));
			}
		} break;
		case THEME_SAVE_AS: {
			ScriptEditor::_show_save_theme_as_dialog();
		} break;
	}
}

void ScriptEditor::_show_save_theme_as_dialog() {
	file_dialog->set_file_mode(EditorFileDialog::FILE_MODE_SAVE_FILE);
	file_dialog->set_access(EditorFileDialog::ACCESS_FILESYSTEM);
	file_dialog_option = THEME_SAVE_AS;
	file_dialog->clear_filters();
	file_dialog->add_filter("*.tet");
	file_dialog->set_current_path(EditorSettings::get_singleton()->get_text_editor_themes_dir().plus_file(EditorSettings::get_singleton()->get("text_editor/theme/color_theme")));
	file_dialog->popup_file_dialog();
	file_dialog->set_title(TTR("Save Theme As..."));
}

void ScriptEditor::_tab_changed(int p_which) {
	ensure_select_current();
}

void ScriptEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			editor->connect("stop_pressed", callable_mp(this, &ScriptEditor::_editor_stop));
			editor->connect("script_add_function_request", callable_mp(this, &ScriptEditor::_add_callback));
			editor->connect("resource_saved", callable_mp(this, &ScriptEditor::_res_saved_callback));
			editor->connect("scene_saved", callable_mp(this, &ScriptEditor::_scene_saved_callback));
			editor->get_filesystem_dock()->connect("files_moved", callable_mp(this, &ScriptEditor::_files_moved));
			editor->get_filesystem_dock()->connect("file_removed", callable_mp(this, &ScriptEditor::_file_removed));
			script_list->connect("item_selected", callable_mp(this, &ScriptEditor::_script_selected));

			members_overview->connect("item_selected", callable_mp(this, &ScriptEditor::_members_overview_selected));
			help_overview->connect("item_selected", callable_mp(this, &ScriptEditor::_help_overview_selected));
			script_split->connect("dragged", callable_mp(this, &ScriptEditor::_script_split_dragged));

			EditorSettings::get_singleton()->connect("settings_changed", callable_mp(this, &ScriptEditor::_editor_settings_changed));
			EditorFileSystem::get_singleton()->connect("filesystem_changed", callable_mp(this, &ScriptEditor::_filesystem_changed));
			_editor_settings_changed();
			[[fallthrough]];
		}
		case NOTIFICATION_TRANSLATION_CHANGED:
		case NOTIFICATION_LAYOUT_DIRECTION_CHANGED:
		case NOTIFICATION_THEME_CHANGED: {
			help_search->set_icon(get_theme_icon(SNAME("HelpSearch"), SNAME("EditorIcons")));
			site_search->set_icon(get_theme_icon(SNAME("Instance"), SNAME("EditorIcons")));

			if (is_layout_rtl()) {
				script_forward->set_icon(get_theme_icon(SNAME("Back"), SNAME("EditorIcons")));
				script_back->set_icon(get_theme_icon(SNAME("Forward"), SNAME("EditorIcons")));
			} else {
				script_forward->set_icon(get_theme_icon(SNAME("Forward"), SNAME("EditorIcons")));
				script_back->set_icon(get_theme_icon(SNAME("Back"), SNAME("EditorIcons")));
			}

			members_overview_alphabeta_sort_button->set_icon(get_theme_icon(SNAME("Sort"), SNAME("EditorIcons")));

			filter_scripts->set_right_icon(get_theme_icon(SNAME("Search"), SNAME("EditorIcons")));
			filter_methods->set_right_icon(get_theme_icon(SNAME("Search"), SNAME("EditorIcons")));

			filename->add_theme_style_override("normal", editor->get_gui_base()->get_theme_stylebox(SNAME("normal"), SNAME("LineEdit")));

			recent_scripts->set_as_minsize();

			if (is_inside_tree()) {
				_update_script_colors();
				_update_script_names();
			}
		} break;

		case NOTIFICATION_READY: {
			get_tree()->connect("tree_changed", callable_mp(this, &ScriptEditor::_tree_changed));
			editor->get_inspector_dock()->connect("request_help", callable_mp(this, &ScriptEditor::_help_class_open));
			editor->connect("request_help_search", callable_mp(this, &ScriptEditor::_help_search));
		} break;

		case NOTIFICATION_EXIT_TREE: {
			editor->disconnect("stop_pressed", callable_mp(this, &ScriptEditor::_editor_stop));
		} break;

		case NOTIFICATION_WM_WINDOW_FOCUS_IN: {
			_test_script_times_on_disk();
			_update_modified_scripts_for_external_editor();
		} break;

		case CanvasItem::NOTIFICATION_VISIBILITY_CHANGED: {
			if (is_visible()) {
				find_in_files_button->show();
			} else {
				if (find_in_files->is_visible_in_tree()) {
					editor->hide_bottom_panel();
				}
				find_in_files_button->hide();
			}

		} break;

		default:
			break;
	}
}

bool ScriptEditor::can_take_away_focus() const {
	ScriptEditorBase *current = _get_current_editor();
	if (current) {
		return current->can_lose_focus_on_node_selection();
	} else {
		return true;
	}
}

void ScriptEditor::close_builtin_scripts_from_scene(const String &p_scene) {
	for (int i = 0; i < tab_container->get_child_count(); i++) {
		ScriptEditorBase *se = Object::cast_to<ScriptEditorBase>(tab_container->get_child(i));

		if (se) {
			Ref<Script> script = se->get_edited_resource();
			if (script == nullptr || !script.is_valid()) {
				continue;
			}

			if (script->is_built_in() && script->get_path().begins_with(p_scene)) { //is an internal script and belongs to scene being closed
				_close_tab(i, false);
				i--;
			}
		}
	}
}

void ScriptEditor::edited_scene_changed() {
	_update_modified_scripts_for_external_editor();
}

void ScriptEditor::notify_script_close(const Ref<Script> &p_script) {
	emit_signal(SNAME("script_close"), p_script);
}

void ScriptEditor::notify_script_changed(const Ref<Script> &p_script) {
	emit_signal(SNAME("editor_script_changed"), p_script);
}

void ScriptEditor::get_breakpoints(List<String> *p_breakpoints) {
	Set<String> loaded_scripts;
	for (int i = 0; i < tab_container->get_child_count(); i++) {
		ScriptEditorBase *se = Object::cast_to<ScriptEditorBase>(tab_container->get_child(i));
		if (!se) {
			continue;
		}

		Ref<Script> script = se->get_edited_resource();
		if (script == nullptr) {
			continue;
		}

		String base = script->get_path();
		loaded_scripts.insert(base);
		if (base.begins_with("local://") || base.is_empty()) {
			continue;
		}

		Array bpoints = se->get_breakpoints();
		for (int j = 0; j < bpoints.size(); j++) {
			p_breakpoints->push_back(base + ":" + itos((int)bpoints[j] + 1));
		}
	}

	// Load breakpoints that are in closed scripts.
	List<String> cached_editors;
	script_editor_cache->get_sections(&cached_editors);
	for (const String &E : cached_editors) {
		if (loaded_scripts.has(E)) {
			continue;
		}

		Array breakpoints = _get_cached_breakpoints_for_script(E);
		for (int i = 0; i < breakpoints.size(); i++) {
			p_breakpoints->push_back(E + ":" + itos((int)breakpoints[i] + 1));
		}
	}
}

void ScriptEditor::_members_overview_selected(int p_idx) {
	ScriptEditorBase *se = _get_current_editor();
	if (!se) {
		return;
	}
	// Go to the member's line and reset the cursor column. We can't change scroll_position
	// directly until we have gone to the line first, since code might be folded.
	se->goto_line(members_overview->get_item_metadata(p_idx));
	Dictionary state = se->get_edit_state();
	state["column"] = 0;
	state["scroll_position"] = members_overview->get_item_metadata(p_idx);
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
	grab_focus_block = !Input::get_singleton()->is_mouse_button_pressed(MouseButton::LEFT); //amazing hack, simply amazing

	_go_to_tab(script_list->get_item_metadata(p_idx));
	grab_focus_block = false;
}

void ScriptEditor::ensure_select_current() {
	if (tab_container->get_child_count() && tab_container->get_current_tab() >= 0) {
		ScriptEditorBase *se = _get_current_editor();
		if (se) {
			se->enable_editor();

			if (!grab_focus_block && is_visible_in_tree()) {
				se->ensure_focus();
			}
		}
	}
	_update_find_replace_bar();

	_update_selected_editor_menu();
}

void ScriptEditor::_find_scripts(Node *p_base, Node *p_current, Set<Ref<Script>> &used) {
	if (p_current != p_base && p_current->get_owner() != p_base) {
		return;
	}

	if (p_current->get_script_instance()) {
		Ref<Script> scr = p_current->get_script();
		if (scr.is_valid()) {
			used.insert(scr);
		}
	}

	for (int i = 0; i < p_current->get_child_count(); i++) {
		_find_scripts(p_base, p_current->get_child(i), used);
	}
}

struct _ScriptEditorItemData {
	String name;
	String sort_key;
	Ref<Texture2D> icon;
	int index = 0;
	String tooltip;
	bool used = false;
	int category = 0;
	Node *ref = nullptr;

	bool operator<(const _ScriptEditorItemData &id) const {
		if (category == id.category) {
			if (sort_key == id.sort_key) {
				return index < id.index;
			} else {
				return sort_key.naturalnocasecmp_to(id.sort_key) < 0;
			}
		} else {
			return category < id.category;
		}
	}
};

void ScriptEditor::_update_members_overview_visibility() {
	ScriptEditorBase *se = _get_current_editor();
	if (!se) {
		members_overview_alphabeta_sort_button->set_visible(false);
		members_overview->set_visible(false);
		overview_vbox->set_visible(false);
		return;
	}

	if (members_overview_enabled && se->show_members_overview()) {
		members_overview_alphabeta_sort_button->set_visible(true);
		members_overview->set_visible(true);
		overview_vbox->set_visible(true);
	} else {
		members_overview_alphabeta_sort_button->set_visible(false);
		members_overview->set_visible(false);
		overview_vbox->set_visible(false);
	}
}

void ScriptEditor::_toggle_members_overview_alpha_sort(bool p_alphabetic_sort) {
	EditorSettings::get_singleton()->set("text_editor/script_list/sort_members_outline_alphabetically", p_alphabetic_sort);
	_update_members_overview();
}

void ScriptEditor::_update_members_overview() {
	members_overview->clear();

	ScriptEditorBase *se = _get_current_editor();
	if (!se) {
		return;
	}

	Vector<String> functions = se->get_functions();
	if (EditorSettings::get_singleton()->get("text_editor/script_list/sort_members_outline_alphabetically")) {
		functions.sort();
	}

	for (int i = 0; i < functions.size(); i++) {
		String filter = filter_methods->get_text();
		String name = functions[i].get_slice(":", 0);
		if (filter.is_empty() || filter.is_subsequence_ofi(name)) {
			members_overview->add_item(name);
			members_overview->set_item_metadata(members_overview->get_item_count() - 1, functions[i].get_slice(":", 1).to_int() - 1);
		}
	}

	String path = se->get_edited_resource()->get_path();
	bool built_in = !path.is_resource_file();
	String name = built_in ? path.get_file() : se->get_name();
	filename->set_text(name);
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
		members_overview_alphabeta_sort_button->set_visible(false);
		help_overview->set_visible(true);
		overview_vbox->set_visible(true);
		filename->set_text(se->get_name());
	} else {
		help_overview->set_visible(false);
		overview_vbox->set_visible(false);
	}
}

void ScriptEditor::_update_help_overview() {
	help_overview->clear();

	int selected = tab_container->get_current_tab();
	if (selected < 0 || selected >= tab_container->get_child_count()) {
		return;
	}

	Node *current = tab_container->get_child(tab_container->get_current_tab());
	EditorHelp *se = Object::cast_to<EditorHelp>(current);
	if (!se) {
		return;
	}

	Vector<Pair<String, int>> sections = se->get_sections();
	for (int i = 0; i < sections.size(); i++) {
		help_overview->add_item(sections[i].first);
		help_overview->set_item_metadata(i, sections[i].second);
	}
}

void ScriptEditor::_update_script_colors() {
	bool script_temperature_enabled = EditorSettings::get_singleton()->get("text_editor/script_list/script_temperature_enabled");

	int hist_size = EditorSettings::get_singleton()->get("text_editor/script_list/script_temperature_history_size");
	Color hot_color = get_theme_color(SNAME("accent_color"), SNAME("Editor"));
	Color cold_color = get_theme_color(SNAME("font_color"), SNAME("Editor"));

	for (int i = 0; i < script_list->get_item_count(); i++) {
		int c = script_list->get_item_metadata(i);
		Node *n = tab_container->get_child(c);
		if (!n) {
			continue;
		}

		script_list->set_item_custom_bg_color(i, Color(0, 0, 0, 0));

		if (script_temperature_enabled) {
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

			script_list->set_item_custom_fg_color(i, hot_color.lerp(cold_color, v));
		}
	}
}

void ScriptEditor::_update_script_names() {
	if (restoring_layout) {
		return;
	}

	Set<Ref<Script>> used;
	Node *edited = EditorNode::get_singleton()->get_edited_scene();
	if (edited) {
		_find_scripts(edited, edited, used);
	}

	script_list->clear();
	bool split_script_help = EditorSettings::get_singleton()->get("text_editor/script_list/group_help_pages");
	ScriptSortBy sort_by = (ScriptSortBy)(int)EditorSettings::get_singleton()->get("text_editor/script_list/sort_scripts_by");
	ScriptListName display_as = (ScriptListName)(int)EditorSettings::get_singleton()->get("text_editor/script_list/list_script_names_as");

	Vector<_ScriptEditorItemData> sedata;

	for (int i = 0; i < tab_container->get_child_count(); i++) {
		ScriptEditorBase *se = Object::cast_to<ScriptEditorBase>(tab_container->get_child(i));
		if (se) {
			Ref<Texture2D> icon = se->get_theme_icon();
			String path = se->get_edited_resource()->get_path();
			bool saved = !path.is_empty();
			if (saved) {
				// The script might be deleted, moved, or renamed, so make sure
				// to update original path to previously edited resource.
				se->set_meta("_edit_res_path", path);
			}
			String name = se->get_name();

			_ScriptEditorItemData sd;
			sd.icon = icon;
			sd.name = name;
			sd.tooltip = saved ? path : TTR("Unsaved file.");
			sd.index = i;
			sd.used = used.has(se->get_edited_resource());
			sd.category = 0;
			sd.ref = se;

			switch (sort_by) {
				case SORT_BY_NAME: {
					sd.sort_key = name.to_lower();
				} break;
				case SORT_BY_PATH: {
					sd.sort_key = path;
				} break;
				case SORT_BY_NONE: {
					sd.sort_key = "";
				} break;
			}

			switch (display_as) {
				case DISPLAY_NAME: {
					sd.name = name;
				} break;
				case DISPLAY_DIR_AND_NAME: {
					if (!path.get_base_dir().get_file().is_empty()) {
						sd.name = path.get_base_dir().get_file().plus_file(name);
					} else {
						sd.name = name;
					}
				} break;
				case DISPLAY_FULL_PATH: {
					sd.name = path;
				} break;
			}
			if (!saved) {
				sd.name = se->get_name();
			}

			sedata.push_back(sd);
		}

		Vector<String> disambiguated_script_names;
		Vector<String> full_script_paths;
		for (int j = 0; j < sedata.size(); j++) {
			String name = sedata[j].name.replace("(*)", "");
			ScriptListName script_display = (ScriptListName)(int)EditorSettings::get_singleton()->get("text_editor/script_list/list_script_names_as");
			switch (script_display) {
				case DISPLAY_NAME: {
					name = name.get_file();
				} break;
				case DISPLAY_DIR_AND_NAME: {
					name = name.get_base_dir().get_file().plus_file(name.get_file());
				} break;
				default:
					break;
			}

			disambiguated_script_names.append(name);
			full_script_paths.append(sedata[j].tooltip);
		}

		EditorNode::disambiguate_filenames(full_script_paths, disambiguated_script_names);

		for (int j = 0; j < sedata.size(); j++) {
			if (sedata[j].name.ends_with("(*)")) {
				sedata.write[j].name = disambiguated_script_names[j] + "(*)";
			} else {
				sedata.write[j].name = disambiguated_script_names[j];
			}
		}

		EditorHelp *eh = Object::cast_to<EditorHelp>(tab_container->get_child(i));
		if (eh) {
			String name = eh->get_class();
			Ref<Texture2D> icon = get_theme_icon(SNAME("Help"), SNAME("EditorIcons"));
			String tooltip = vformat(TTR("%s Class Reference"), name);

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

	if (_sort_list_on_update && !sedata.is_empty()) {
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
			// Update index of sd entries for sorted order
			_ScriptEditorItemData sd = sedata[i];
			sd.index = i;
			sedata.set(i, sd);
		}
		tab_container->set_current_tab(new_prev_tab);
		tab_container->set_current_tab(new_cur_tab);
		_sort_list_on_update = false;
	}

	Vector<_ScriptEditorItemData> sedata_filtered;
	for (int i = 0; i < sedata.size(); i++) {
		String filter = filter_scripts->get_text();
		if (filter.is_empty() || filter.is_subsequence_ofi(sedata[i].name)) {
			sedata_filtered.push_back(sedata[i]);
		}
	}

	for (int i = 0; i < sedata_filtered.size(); i++) {
		script_list->add_item(sedata_filtered[i].name, sedata_filtered[i].icon);
		int index = script_list->get_item_count() - 1;
		script_list->set_item_tooltip(index, sedata_filtered[i].tooltip);
		script_list->set_item_metadata(index, sedata_filtered[i].index); /* Saving as metadata the script's index in the tab container and not the filtered one */
		if (sedata_filtered[i].used) {
			script_list->set_item_custom_bg_color(index, Color(88 / 255.0, 88 / 255.0, 60 / 255.0));
		}
		if (tab_container->get_current_tab() == sedata_filtered[i].index) {
			script_list->select(index);
			script_name_label->set_text(sedata_filtered[i].name);
			script_icon->set_texture(sedata_filtered[i].icon);
			ScriptEditorBase *se = _get_current_editor();
			if (se) {
				se->enable_editor();
				_update_selected_editor_menu();
			}
		}
	}

	if (!waiting_update_names) {
		_update_members_overview();
		_update_help_overview();
	} else {
		waiting_update_names = false;
	}
	_update_members_overview_visibility();
	_update_help_overview_visibility();
	_update_script_colors();

	file_menu->get_popup()->set_item_disabled(file_menu->get_popup()->get_item_index(FILE_REOPEN_CLOSED), previous_scripts.is_empty());
}

void ScriptEditor::_update_script_connections() {
	for (int i = 0; i < tab_container->get_child_count(); i++) {
		ScriptTextEditor *ste = Object::cast_to<ScriptTextEditor>(tab_container->get_child(i));
		if (!ste) {
			continue;
		}
		ste->_update_connected_methods();
	}
}

Ref<TextFile> ScriptEditor::_load_text_file(const String &p_path, Error *r_error) const {
	if (r_error) {
		*r_error = ERR_FILE_CANT_OPEN;
	}

	String local_path = ProjectSettings::get_singleton()->localize_path(p_path);
	String path = ResourceLoader::path_remap(local_path);

	TextFile *text_file = memnew(TextFile);
	Ref<TextFile> text_res(text_file);
	Error err = text_file->load_text(path);

	ERR_FAIL_COND_V_MSG(err != OK, RES(), "Cannot load text file '" + path + "'.");

	text_file->set_file_path(local_path);
	text_file->set_path(local_path, true);

	if (ResourceLoader::get_timestamp_on_load()) {
		text_file->set_last_modified_time(FileAccess::get_modified_time(path));
	}

	if (r_error) {
		*r_error = OK;
	}

	return text_res;
}

Error ScriptEditor::_save_text_file(Ref<TextFile> p_text_file, const String &p_path) {
	Ref<TextFile> sqscr = p_text_file;
	ERR_FAIL_COND_V(sqscr.is_null(), ERR_INVALID_PARAMETER);

	String source = sqscr->get_text();

	Error err;
	FileAccess *file = FileAccess::open(p_path, FileAccess::WRITE, &err);

	ERR_FAIL_COND_V_MSG(err, err, "Cannot save text file '" + p_path + "'.");

	file->store_string(source);
	if (file->get_error() != OK && file->get_error() != ERR_FILE_EOF) {
		memdelete(file);
		return ERR_CANT_CREATE;
	}
	file->close();
	memdelete(file);

	if (ResourceSaver::get_timestamp_on_save()) {
		p_text_file->set_last_modified_time(FileAccess::get_modified_time(p_path));
	}

	_res_saved_callback(sqscr);
	return OK;
}

bool ScriptEditor::edit(const RES &p_resource, int p_line, int p_col, bool p_grab_focus) {
	if (p_resource.is_null()) {
		return false;
	}

	Ref<Script> script = p_resource;

	// Don't open dominant script if using an external editor.
	bool use_external_editor =
			EditorSettings::get_singleton()->get("text_editor/external/use_external_editor") ||
			(script.is_valid() && script->get_language()->overrides_external_editor());
	use_external_editor = use_external_editor && !(script.is_valid() && script->is_built_in()); // Ignore external editor for built-in scripts.
	const bool open_dominant = EditorSettings::get_singleton()->get("text_editor/behavior/files/open_dominant_script_on_scene_change");

	const bool should_open = (open_dominant && !use_external_editor) || !EditorNode::get_singleton()->is_changing_scene();

	if (script.is_valid() && script->get_language()->overrides_external_editor()) {
		if (should_open) {
			Error err = script->get_language()->open_in_external_editor(script, p_line >= 0 ? p_line : 0, p_col);
			if (err != OK) {
				ERR_PRINT("Couldn't open script in the overridden external text editor");
			}
		}
		return false;
	}

	if (use_external_editor &&
			(EditorDebuggerNode::get_singleton()->get_dump_stack_script() != p_resource || EditorDebuggerNode::get_singleton()->get_debug_with_external_editor()) &&
			p_resource->get_path().is_resource_file() &&
			p_resource->get_class_name() != StringName("VisualScript")) {
		String path = EditorSettings::get_singleton()->get("text_editor/external/exec_path");
		String flags = EditorSettings::get_singleton()->get("text_editor/external/exec_flags");

		List<String> args;
		bool has_file_flag = false;
		String script_path = ProjectSettings::get_singleton()->globalize_path(p_resource->get_path());

		if (flags.size()) {
			String project_path = ProjectSettings::get_singleton()->get_resource_path();

			flags = flags.replacen("{line}", itos(p_line > 0 ? p_line : 0));
			flags = flags.replacen("{col}", itos(p_col));
			flags = flags.strip_edges().replace("\\\\", "\\");

			int from = 0;
			int num_chars = 0;
			bool inside_quotes = false;

			for (int i = 0; i < flags.size(); i++) {
				if (flags[i] == '"' && (!i || flags[i - 1] != '\\')) {
					if (!inside_quotes) {
						from++;
					}
					inside_quotes = !inside_quotes;

				} else if (flags[i] == '\0' || (!inside_quotes && flags[i] == ' ')) {
					String arg = flags.substr(from, num_chars);
					if (arg.find("{file}") != -1) {
						has_file_flag = true;
					}

					// do path replacement here, else there will be issues with spaces and quotes
					arg = arg.replacen("{project}", project_path);
					arg = arg.replacen("{file}", script_path);
					args.push_back(arg);

					from = i + 1;
					num_chars = 0;
				} else {
					num_chars++;
				}
			}
		}

		// Default to passing script path if no {file} flag is specified.
		if (!has_file_flag) {
			args.push_back(script_path);
		}

		Error err = OS::get_singleton()->create_process(path, args);
		if (err == OK) {
			return false;
		}
		WARN_PRINT("Couldn't open external text editor, using internal");
	}

	for (int i = 0; i < tab_container->get_child_count(); i++) {
		ScriptEditorBase *se = Object::cast_to<ScriptEditorBase>(tab_container->get_child(i));
		if (!se) {
			continue;
		}

		if ((script != nullptr && se->get_edited_resource() == p_resource) || se->get_edited_resource()->get_path() == p_resource->get_path()) {
			if (should_open) {
				se->enable_editor();

				if (tab_container->get_current_tab() != i) {
					_go_to_tab(i);
					_update_script_names();
				}
				if (is_visible_in_tree()) {
					se->ensure_focus();
				}

				if (p_line > 0) {
					se->goto_line(p_line - 1);
				}
			}
			_update_script_names();
			script_list->ensure_current_is_visible();
			return true;
		}
	}

	// doesn't have it, make a new one

	ScriptEditorBase *se = nullptr;

	for (int i = script_editor_func_count - 1; i >= 0; i--) {
		se = script_editor_funcs[i](p_resource);
		if (se) {
			break;
		}
	}
	ERR_FAIL_COND_V(!se, false);

	se->set_edited_resource(p_resource);

	if (p_resource->get_class_name() != StringName("VisualScript")) {
		bool highlighter_set = false;
		for (int i = 0; i < syntax_highlighters.size(); i++) {
			Ref<EditorSyntaxHighlighter> highlighter = syntax_highlighters[i]->_create();
			if (highlighter.is_null()) {
				continue;
			}
			se->add_syntax_highlighter(highlighter);

			if (script != nullptr && !highlighter_set) {
				Array languages = highlighter->_get_supported_languages();
				if (languages.find(script->get_language()->get_name()) > -1) {
					se->set_syntax_highlighter(highlighter);
					highlighter_set = true;
				}
			}
		}
	}

	tab_container->add_child(se);

	if (p_grab_focus) {
		se->enable_editor();
	}

	// If we delete a script within the filesystem, the original resource path
	// is lost, so keep it as metadata to figure out the exact tab to delete.
	se->set_meta("_edit_res_path", p_resource->get_path());
	se->set_tooltip_request_func("_get_debug_tooltip", this);
	if (se->get_edit_menu()) {
		se->get_edit_menu()->hide();
		menu_hb->add_child(se->get_edit_menu());
		menu_hb->move_child(se->get_edit_menu(), 1);
	}

	if (p_grab_focus) {
		_go_to_tab(tab_container->get_tab_count() - 1);
		_add_recent_script(p_resource->get_path());
	}

	if (script_editor_cache->has_section(p_resource->get_path())) {
		se->set_edit_state(script_editor_cache->get_value(p_resource->get_path(), "state"));
	}

	_sort_list_on_update = true;
	_update_script_names();
	_save_layout();
	se->connect("name_changed", callable_mp(this, &ScriptEditor::_update_script_names));
	se->connect("edited_script_changed", callable_mp(this, &ScriptEditor::_script_changed));
	se->connect("request_help", callable_mp(this, &ScriptEditor::_help_search));
	se->connect("request_open_script_at_line", callable_mp(this, &ScriptEditor::_goto_script_line));
	se->connect("go_to_help", callable_mp(this, &ScriptEditor::_help_class_goto));
	se->connect("request_save_history", callable_mp(this, &ScriptEditor::_save_history));
	se->connect("search_in_files_requested", callable_mp(this, &ScriptEditor::_on_find_in_files_requested));
	se->connect("replace_in_files_requested", callable_mp(this, &ScriptEditor::_on_replace_in_files_requested));

	//test for modification, maybe the script was not edited but was loaded

	_test_script_times_on_disk(p_resource);
	_update_modified_scripts_for_external_editor(p_resource);

	if (p_line > 0) {
		se->goto_line(p_line - 1);
	}

	notify_script_changed(p_resource);
	return true;
}

void ScriptEditor::save_current_script() {
	ScriptEditorBase *current = _get_current_editor();
	if (!current || _test_script_times_on_disk()) {
		return;
	}

	if (trim_trailing_whitespace_on_save) {
		current->trim_trailing_whitespace();
	}

	current->insert_final_newline();

	if (convert_indent_on_save) {
		if (use_space_indentation) {
			current->convert_indent_to_spaces();
		} else {
			current->convert_indent_to_tabs();
		}
	}

	RES resource = current->get_edited_resource();
	Ref<TextFile> text_file = resource;
	Ref<Script> script = resource;

	if (text_file != nullptr) {
		current->apply_code();
		_save_text_file(text_file, text_file->get_path());
		return;
	}

	if (script != nullptr) {
		const Vector<DocData::ClassDoc> &documentations = script->get_documentation();
		for (int j = 0; j < documentations.size(); j++) {
			const DocData::ClassDoc &doc = documentations.get(j);
			if (EditorHelp::get_doc_data()->has_doc(doc.name)) {
				EditorHelp::get_doc_data()->remove_doc(doc.name);
			}
		}
	}

	if (resource->is_built_in()) {
		// If built-in script, save the scene instead.
		const String scene_path = resource->get_path().get_slice("::", 0);
		if (!scene_path.is_empty()) {
			Vector<String> scene_to_save;
			scene_to_save.push_back(scene_path);
			editor->save_scene_list(scene_to_save);
		}
	} else {
		editor->save_resource(resource);
	}

	if (script != nullptr) {
		const Vector<DocData::ClassDoc> &documentations = script->get_documentation();
		for (int j = 0; j < documentations.size(); j++) {
			const DocData::ClassDoc &doc = documentations.get(j);
			EditorHelp::get_doc_data()->add_doc(doc);
			update_doc(doc.name);
		}
	}
}

void ScriptEditor::save_all_scripts() {
	Vector<String> scenes_to_save;

	for (int i = 0; i < tab_container->get_child_count(); i++) {
		ScriptEditorBase *se = Object::cast_to<ScriptEditorBase>(tab_container->get_child(i));
		if (!se) {
			continue;
		}

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

		se->insert_final_newline();

		if (!se->is_unsaved()) {
			continue;
		}

		RES edited_res = se->get_edited_resource();
		if (edited_res.is_valid()) {
			se->apply_code();
		}

		if (!edited_res->is_built_in()) {
			Ref<TextFile> text_file = edited_res;
			Ref<Script> script = edited_res;

			if (text_file != nullptr) {
				_save_text_file(text_file, text_file->get_path());
				continue;
			}

			if (script != nullptr) {
				const Vector<DocData::ClassDoc> &documentations = script->get_documentation();
				for (int j = 0; j < documentations.size(); j++) {
					const DocData::ClassDoc &doc = documentations.get(j);
					if (EditorHelp::get_doc_data()->has_doc(doc.name)) {
						EditorHelp::get_doc_data()->remove_doc(doc.name);
					}
				}
			}

			editor->save_resource(edited_res); //external script, save it

			if (script != nullptr) {
				const Vector<DocData::ClassDoc> &documentations = script->get_documentation();
				for (int j = 0; j < documentations.size(); j++) {
					const DocData::ClassDoc &doc = documentations.get(j);
					EditorHelp::get_doc_data()->add_doc(doc);
					update_doc(doc.name);
				}
			}
		} else {
			// For built-in scripts, save their scenes instead.
			const String scene_path = edited_res->get_path().get_slice("::", 0);
			if (!scenes_to_save.has(scene_path)) {
				scenes_to_save.push_back(scene_path);
			}
		}
	}

	if (!scenes_to_save.is_empty()) {
		editor->save_scene_list(scenes_to_save);
	}

	_update_script_names();
	EditorFileSystem::get_singleton()->update_script_classes();
}

void ScriptEditor::apply_scripts() const {
	for (int i = 0; i < tab_container->get_child_count(); i++) {
		ScriptEditorBase *se = Object::cast_to<ScriptEditorBase>(tab_container->get_child(i));
		if (!se) {
			continue;
		}
		se->apply_code();
	}
}

void ScriptEditor::open_script_create_dialog(const String &p_base_name, const String &p_base_path) {
	_menu_option(FILE_NEW);
	script_create_dialog->config(p_base_name, p_base_path);
}

void ScriptEditor::open_text_file_create_dialog(const String &p_base_path, const String &p_base_name) {
	file_dialog->set_current_file(p_base_name);
	file_dialog->set_current_dir(p_base_path);
	_menu_option(FILE_NEW_TEXTFILE);
	open_textfile_after_create = false;
}

RES ScriptEditor::open_file(const String &p_file) {
	List<String> extensions;
	ResourceLoader::get_recognized_extensions_for_type("Script", &extensions);
	if (extensions.find(p_file.get_extension())) {
		Ref<Script> scr = ResourceLoader::load(p_file);
		if (!scr.is_valid()) {
			editor->show_warning(TTR("Could not load file at:") + "\n\n" + p_file, TTR("Error!"));
			return RES();
		}

		edit(scr);
		return scr;
	}

	Error error;
	Ref<TextFile> text_file = _load_text_file(p_file, &error);
	if (error != OK) {
		editor->show_warning(TTR("Could not load file at:") + "\n\n" + p_file, TTR("Error!"));
		return RES();
	}

	if (text_file.is_valid()) {
		edit(text_file);
		return text_file;
	}
	return RES();
}

void ScriptEditor::_editor_stop() {
	for (int i = 0; i < tab_container->get_child_count(); i++) {
		ScriptEditorBase *se = Object::cast_to<ScriptEditorBase>(tab_container->get_child(i));
		if (!se) {
			continue;
		}

		se->set_debugger_active(false);
	}
}

void ScriptEditor::_add_callback(Object *p_obj, const String &p_function, const PackedStringArray &p_args) {
	ERR_FAIL_COND(!p_obj);
	Ref<Script> script = p_obj->get_script();
	ERR_FAIL_COND(!script.is_valid());

	editor->push_item(script.ptr());

	for (int i = 0; i < tab_container->get_child_count(); i++) {
		ScriptEditorBase *se = Object::cast_to<ScriptEditorBase>(tab_container->get_child(i));
		if (!se) {
			continue;
		}
		if (se->get_edited_resource() != script) {
			continue;
		}

		se->add_callback(p_function, p_args);

		_go_to_tab(i);

		script_list->select(script_list->find_metadata(i));

		// Save the current script so the changes can be picked up by an external editor.
		if (!script.ptr()->is_built_in()) { // But only if it's not built-in script.
			save_current_script();
		}

		break;
	}
}

void ScriptEditor::_save_editor_state(ScriptEditorBase *p_editor) {
	if (restoring_layout) {
		return;
	}

	const String &path = p_editor->get_edited_resource()->get_path();
	if (!path.is_resource_file()) {
		return;
	}

	script_editor_cache->set_value(path, "state", p_editor->get_edit_state());
	// This is saved later when we save the editor layout.
}

void ScriptEditor::_save_layout() {
	if (restoring_layout) {
		return;
	}

	editor->save_layout();
}

void ScriptEditor::_editor_settings_changed() {
	textfile_extensions.clear();
	const Vector<String> textfile_ext = ((String)(EditorSettings::get_singleton()->get("docks/filesystem/textfile_extensions"))).split(",", false);
	for (const String &E : textfile_ext) {
		textfile_extensions.insert(E);
	}

	trim_trailing_whitespace_on_save = EditorSettings::get_singleton()->get("text_editor/behavior/files/trim_trailing_whitespace_on_save");
	convert_indent_on_save = EditorSettings::get_singleton()->get("text_editor/behavior/files/convert_indent_on_save");
	use_space_indentation = EditorSettings::get_singleton()->get("text_editor/behavior/indent/type");

	members_overview_enabled = EditorSettings::get_singleton()->get("text_editor/script_list/show_members_overview");
	help_overview_enabled = EditorSettings::get_singleton()->get("text_editor/help/show_help_index");
	_update_members_overview_visibility();
	_update_help_overview_visibility();

	_update_autosave_timer();

	if (current_theme.is_empty()) {
		current_theme = EditorSettings::get_singleton()->get("text_editor/theme/color_theme");
	} else if (current_theme != String(EditorSettings::get_singleton()->get("text_editor/theme/color_theme"))) {
		current_theme = EditorSettings::get_singleton()->get("text_editor/theme/color_theme");
		EditorSettings::get_singleton()->load_text_editor_theme();
	}

	for (int i = 0; i < tab_container->get_child_count(); i++) {
		ScriptEditorBase *se = Object::cast_to<ScriptEditorBase>(tab_container->get_child(i));
		if (!se) {
			continue;
		}

		se->update_settings();
	}
	_update_script_colors();
	_update_script_names();

	ScriptServer::set_reload_scripts_on_save(EDITOR_DEF("text_editor/behavior/files/auto_reload_and_parse_scripts_on_save", true));
}

void ScriptEditor::_filesystem_changed() {
	_update_script_names();
}

void ScriptEditor::_files_moved(const String &p_old_file, const String &p_new_file) {
	if (!script_editor_cache->has_section(p_old_file)) {
		return;
	}
	Variant state = script_editor_cache->get_value(p_old_file, "state");
	script_editor_cache->erase_section(p_old_file);
	script_editor_cache->set_value(p_new_file, "state", state);

	// If Script, update breakpoints with debugger.
	Array breakpoints = _get_cached_breakpoints_for_script(p_new_file);
	for (int i = 0; i < breakpoints.size(); i++) {
		int line = (int)breakpoints[i] + 1;
		EditorDebuggerNode::get_singleton()->set_breakpoint(p_old_file, line, false);
		if (!p_new_file.begins_with("local://") && ResourceLoader::exists(p_new_file, "Script")) {
			EditorDebuggerNode::get_singleton()->set_breakpoint(p_new_file, line, true);
		}
	}
	// This is saved later when we save the editor layout.
}

void ScriptEditor::_file_removed(const String &p_removed_file) {
	for (int i = 0; i < tab_container->get_child_count(); i++) {
		ScriptEditorBase *se = Object::cast_to<ScriptEditorBase>(tab_container->get_child(i));
		if (!se) {
			continue;
		}
		if (se->get_meta("_edit_res_path") == p_removed_file) {
			// The script is deleted with no undo, so just close the tab.
			_close_tab(i, false, false);
		}
	}

	// Check closed.
	if (script_editor_cache->has_section(p_removed_file)) {
		Array breakpoints = _get_cached_breakpoints_for_script(p_removed_file);
		for (int i = 0; i < breakpoints.size(); i++) {
			EditorDebuggerNode::get_singleton()->set_breakpoint(p_removed_file, (int)breakpoints[i] + 1, false);
		}
		script_editor_cache->erase_section(p_removed_file);
	}
}

void ScriptEditor::_update_find_replace_bar() {
	ScriptEditorBase *se = _get_current_editor();
	if (se) {
		se->set_find_replace_bar(find_replace_bar);
	} else {
		find_replace_bar->set_text_edit(nullptr);
		find_replace_bar->hide();
	}
}

void ScriptEditor::_autosave_scripts() {
	save_all_scripts();
}

void ScriptEditor::_update_autosave_timer() {
	if (!autosave_timer->is_inside_tree()) {
		return;
	}

	float autosave_time = EditorSettings::get_singleton()->get("text_editor/behavior/files/autosave_interval_secs");
	if (autosave_time > 0) {
		autosave_timer->set_wait_time(autosave_time);
		autosave_timer->start();
	} else {
		autosave_timer->stop();
	}
}

void ScriptEditor::_tree_changed() {
	if (waiting_update_names) {
		return;
	}

	waiting_update_names = true;
	call_deferred(SNAME("_update_script_names"));
	call_deferred(SNAME("_update_script_connections"));
}

void ScriptEditor::_script_split_dragged(float) {
	_save_layout();
}

Variant ScriptEditor::get_drag_data_fw(const Point2 &p_point, Control *p_from) {
	if (tab_container->get_child_count() == 0) {
		return Variant();
	}

	Node *cur_node = tab_container->get_child(tab_container->get_current_tab());

	HBoxContainer *drag_preview = memnew(HBoxContainer);
	String preview_name = "";
	Ref<Texture2D> preview_icon;

	ScriptEditorBase *se = Object::cast_to<ScriptEditorBase>(cur_node);
	if (se) {
		preview_name = se->get_name();
		preview_icon = se->get_theme_icon();
	}
	EditorHelp *eh = Object::cast_to<EditorHelp>(cur_node);
	if (eh) {
		preview_name = eh->get_class();
		preview_icon = get_theme_icon(SNAME("Help"), SNAME("EditorIcons"));
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
	if (!d.has("type")) {
		return false;
	}

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
		if (nodes.size() == 0) {
			return false;
		}
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

		if (files.size() == 0) {
			return false; //weird
		}

		for (int i = 0; i < files.size(); i++) {
			String file = files[i];
			if (file.is_empty() || !FileAccess::exists(file)) {
				continue;
			}
			if (ResourceLoader::exists(file, "Script")) {
				Ref<Script> scr = ResourceLoader::load(file);
				if (scr.is_valid()) {
					return true;
				}
			}

			if (textfile_extensions.has(file.get_extension())) {
				Error err;
				Ref<TextFile> text_file = _load_text_file(file, &err);
				if (text_file.is_valid() && err == OK) {
					return true;
				}
			}
		}
		return false;
	}

	return false;
}

void ScriptEditor::drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) {
	if (!can_drop_data_fw(p_point, p_data, p_from)) {
		return;
	}

	Dictionary d = p_data;
	if (!d.has("type")) {
		return;
	}

	if (String(d["type"]) == "script_list_element") {
		Node *node = d["script_list_element"];

		ScriptEditorBase *se = Object::cast_to<ScriptEditorBase>(node);
		EditorHelp *eh = Object::cast_to<EditorHelp>(node);
		if (se || eh) {
			int new_index = 0;
			if (script_list->get_item_count() > 0) {
				new_index = script_list->get_item_metadata(script_list->get_item_at_position(p_point));
			}
			tab_container->move_child(node, new_index);
			tab_container->set_current_tab(new_index);
			_update_script_names();
		}
	}

	if (String(d["type"]) == "nodes") {
		Array nodes = d["nodes"];
		if (nodes.size() == 0) {
			return;
		}
		Node *node = get_node(nodes[0]);

		ScriptEditorBase *se = Object::cast_to<ScriptEditorBase>(node);
		EditorHelp *eh = Object::cast_to<EditorHelp>(node);
		if (se || eh) {
			int new_index = 0;
			if (script_list->get_item_count() > 0) {
				new_index = script_list->get_item_metadata(script_list->get_item_at_position(p_point));
			}
			tab_container->move_child(node, new_index);
			tab_container->set_current_tab(new_index);
			_update_script_names();
		}
	}

	if (String(d["type"]) == "files") {
		Vector<String> files = d["files"];

		int new_index = 0;
		if (script_list->get_item_count() > 0) {
			new_index = script_list->get_item_metadata(script_list->get_item_at_position(p_point));
		}
		int num_tabs_before = tab_container->get_child_count();
		for (int i = 0; i < files.size(); i++) {
			String file = files[i];
			if (file.is_empty() || !FileAccess::exists(file)) {
				continue;
			}

			if (!ResourceLoader::exists(file, "Script") && !textfile_extensions.has(file.get_extension())) {
				continue;
			}

			RES res = open_file(file);
			if (res.is_valid()) {
				if (tab_container->get_child_count() > num_tabs_before) {
					tab_container->move_child(tab_container->get_child(tab_container->get_child_count() - 1), new_index);
					num_tabs_before = tab_container->get_child_count();
				} else { /* Maybe script was already open */
					tab_container->move_child(tab_container->get_child(tab_container->get_current_tab()), new_index);
				}
			}
		}
		tab_container->set_current_tab(new_index);
		_update_script_names();
	}
}

void ScriptEditor::input(const Ref<InputEvent> &p_event) {
	// This is implemented in `input()` rather than `unhandled_input()` to allow
	// the shortcut to be used regardless of the click location.
	// This feature can be disabled to avoid interfering with other uses of the additional
	// mouse buttons, such as push-to-talk in a VoIP program.
	if (EDITOR_GET("interface/editor/mouse_extra_buttons_navigate_history")) {
		const Ref<InputEventMouseButton> mb = p_event;

		// Navigate the script history using additional mouse buttons present on some mice.
		// This must be hardcoded as the editor shortcuts dialog doesn't allow assigning
		// more than one shortcut per action.
		if (mb.is_valid() && mb->is_pressed() && is_visible_in_tree()) {
			if (mb->get_button_index() == MouseButton::MB_XBUTTON1) {
				_history_back();
			}

			if (mb->get_button_index() == MouseButton::MB_XBUTTON2) {
				_history_forward();
			}
		}
	}
}

void ScriptEditor::unhandled_key_input(const Ref<InputEvent> &p_event) {
	ERR_FAIL_COND(p_event.is_null());

	if (!is_visible_in_tree() || !p_event->is_pressed() || p_event->is_echo()) {
		return;
	}
	if (ED_IS_SHORTCUT("script_editor/next_script", p_event)) {
		if (script_list->get_item_count() > 1) {
			int next_tab = script_list->get_current() + 1;
			next_tab %= script_list->get_item_count();
			_go_to_tab(script_list->get_item_metadata(next_tab));
			_update_script_names();
		}
	}
	if (ED_IS_SHORTCUT("script_editor/prev_script", p_event)) {
		if (script_list->get_item_count() > 1) {
			int next_tab = script_list->get_current() - 1;
			next_tab = next_tab >= 0 ? next_tab : script_list->get_item_count() - 1;
			_go_to_tab(script_list->get_item_metadata(next_tab));
			_update_script_names();
		}
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
	if (mb.is_valid() && mb->is_pressed()) {
		switch (mb->get_button_index()) {
			case MouseButton::MIDDLE: {
				// Right-click selects automatically; middle-click does not.
				int idx = script_list->get_item_at_position(mb->get_position(), true);
				if (idx >= 0) {
					script_list->select(idx);
					_script_selected(idx);
					_menu_option(FILE_CLOSE);
				}
			} break;

			case MouseButton::RIGHT: {
				_make_script_list_context_menu();
			} break;
			default:
				break;
		}
	}
}

void ScriptEditor::_make_script_list_context_menu() {
	context_menu->clear();

	int selected = tab_container->get_current_tab();
	if (selected < 0 || selected >= tab_container->get_child_count()) {
		return;
	}

	ScriptEditorBase *se = Object::cast_to<ScriptEditorBase>(tab_container->get_child(selected));
	if (se) {
		context_menu->add_shortcut(ED_GET_SHORTCUT("script_editor/save"), FILE_SAVE);
		context_menu->add_shortcut(ED_GET_SHORTCUT("script_editor/save_as"), FILE_SAVE_AS);
	}
	context_menu->add_shortcut(ED_GET_SHORTCUT("script_editor/close_file"), FILE_CLOSE);
	context_menu->add_shortcut(ED_GET_SHORTCUT("script_editor/close_all"), CLOSE_ALL);
	context_menu->add_shortcut(ED_GET_SHORTCUT("script_editor/close_other_tabs"), CLOSE_OTHER_TABS);
	context_menu->add_separator();
	if (se) {
		Ref<Script> scr = se->get_edited_resource();
		if (scr != nullptr) {
			context_menu->add_shortcut(ED_GET_SHORTCUT("script_editor/reload_script_soft"), FILE_TOOL_RELOAD_SOFT);
			if (!scr.is_null() && scr->is_tool()) {
				context_menu->add_shortcut(ED_GET_SHORTCUT("script_editor/run_file"), FILE_RUN);
				context_menu->add_separator();
			}
		}
		context_menu->add_shortcut(ED_GET_SHORTCUT("script_editor/copy_path"), FILE_COPY_PATH);
		context_menu->add_shortcut(ED_GET_SHORTCUT("script_editor/show_in_file_system"), SHOW_IN_FILE_SYSTEM);
		context_menu->add_separator();
	}

	context_menu->add_shortcut(ED_GET_SHORTCUT("script_editor/window_move_up"), WINDOW_MOVE_UP);
	context_menu->add_shortcut(ED_GET_SHORTCUT("script_editor/window_move_down"), WINDOW_MOVE_DOWN);
	context_menu->add_shortcut(ED_GET_SHORTCUT("script_editor/window_sort"), WINDOW_SORT);
	context_menu->add_shortcut(ED_GET_SHORTCUT("script_editor/toggle_scripts_panel"), TOGGLE_SCRIPTS_PANEL);

	context_menu->set_position(get_screen_position() + get_local_mouse_position());
	context_menu->reset_size();
	context_menu->popup();
}

void ScriptEditor::set_window_layout(Ref<ConfigFile> p_layout) {
	if (!bool(EDITOR_DEF("text_editor/behavior/files/restore_scripts_on_load", true))) {
		return;
	}

	if (!p_layout->has_section_key("ScriptEditor", "open_scripts") && !p_layout->has_section_key("ScriptEditor", "open_help")) {
		return;
	}

	Array scripts = p_layout->get_value("ScriptEditor", "open_scripts");
	Array helps;
	if (p_layout->has_section_key("ScriptEditor", "open_help")) {
		helps = p_layout->get_value("ScriptEditor", "open_help");
	}

	restoring_layout = true;

	Set<String> loaded_scripts;
	List<String> extensions;
	ResourceLoader::get_recognized_extensions_for_type("Script", &extensions);

	for (int i = 0; i < scripts.size(); i++) {
		String path = scripts[i];

		Dictionary script_info = scripts[i];
		if (!script_info.is_empty()) {
			path = script_info["path"];
		}

		if (!FileAccess::exists(path)) {
			if (script_editor_cache->has_section(path)) {
				script_editor_cache->erase_section(path);
			}
			continue;
		}
		loaded_scripts.insert(path);

		if (extensions.find(path.get_extension())) {
			Ref<Script> scr = ResourceLoader::load(path);
			if (!scr.is_valid()) {
				continue;
			}
			if (!edit(scr, false)) {
				continue;
			}
		} else {
			Error error;
			Ref<TextFile> text_file = _load_text_file(path, &error);
			if (error != OK || !text_file.is_valid()) {
				continue;
			}
			if (!edit(text_file, false)) {
				continue;
			}
		}

		if (!script_info.is_empty()) {
			ScriptEditorBase *se = Object::cast_to<ScriptEditorBase>(tab_container->get_child(tab_container->get_tab_count() - 1));
			if (se) {
				se->set_edit_state(script_info["state"]);
			}
		}
	}

	for (int i = 0; i < helps.size(); i++) {
		String path = helps[i];
		if (path.is_empty()) { // invalid, skip
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

	// Remove any deleted editors that have been removed between launches.
	// and if a Script, register breakpoints with the debugger.
	List<String> cached_editors;
	script_editor_cache->get_sections(&cached_editors);
	for (const String &E : cached_editors) {
		if (loaded_scripts.has(E)) {
			continue;
		}

		if (!FileAccess::exists(E)) {
			script_editor_cache->erase_section(E);
			continue;
		}

		Array breakpoints = _get_cached_breakpoints_for_script(E);
		for (int i = 0; i < breakpoints.size(); i++) {
			EditorDebuggerNode::get_singleton()->set_breakpoint(E, (int)breakpoints[i] + 1, true);
		}
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
			String path = se->get_edited_resource()->get_path();
			if (!path.is_resource_file()) {
				continue;
			}

			_save_editor_state(se);
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

	// Save the cache.
	script_editor_cache->save(EditorSettings::get_singleton()->get_project_settings_dir().plus_file("script_editor_cache.cfg"));
}

void ScriptEditor::_help_class_open(const String &p_class) {
	if (p_class.is_empty()) {
		return;
	}

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
	eh->connect("go_to_help", callable_mp(this, &ScriptEditor::_help_class_goto));
	_add_recent_script(p_class);
	_sort_list_on_update = true;
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
	eh->connect("go_to_help", callable_mp(this, &ScriptEditor::_help_class_goto));
	_add_recent_script(eh->get_class());
	_sort_list_on_update = true;
	_update_script_names();
	_save_layout();
}

void ScriptEditor::update_doc(const String &p_name) {
	ERR_FAIL_COND(!EditorHelp::get_doc_data()->has_doc(p_name));

	for (int i = 0; i < tab_container->get_child_count(); i++) {
		EditorHelp *eh = Object::cast_to<EditorHelp>(tab_container->get_child(i));
		if (eh && eh->get_class() == p_name) {
			eh->update_doc();
			return;
		}
	}
}

void ScriptEditor::_update_selected_editor_menu() {
	for (int i = 0; i < tab_container->get_child_count(); i++) {
		bool current = tab_container->get_current_tab() == i;

		ScriptEditorBase *se = Object::cast_to<ScriptEditorBase>(tab_container->get_child(i));
		if (se && se->get_edit_menu()) {
			if (current) {
				se->get_edit_menu()->show();
			} else {
				se->get_edit_menu()->hide();
			}
		}
	}

	EditorHelp *eh = Object::cast_to<EditorHelp>(tab_container->get_current_tab_control());
	script_search_menu->get_popup()->clear();
	if (eh) {
		script_search_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/find", TTR("Find..."), KeyModifierMask::CMD | Key::F), HELP_SEARCH_FIND);
		script_search_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/find_next", TTR("Find Next"), Key::F3), HELP_SEARCH_FIND_NEXT);
		script_search_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/find_previous", TTR("Find Previous"), KeyModifierMask::SHIFT | Key::F3), HELP_SEARCH_FIND_PREVIOUS);
		script_search_menu->get_popup()->add_separator();
		script_search_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/find_in_files", TTR("Find in Files"), KeyModifierMask::CMD | KeyModifierMask::SHIFT | Key::F), SEARCH_IN_FILES);
		script_search_menu->show();
	} else {
		if (tab_container->get_child_count() == 0) {
			script_search_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/find_in_files", TTR("Find in Files"), KeyModifierMask::CMD | KeyModifierMask::SHIFT | Key::F), SEARCH_IN_FILES);
			script_search_menu->show();
		} else {
			script_search_menu->hide();
		}
	}
}

void ScriptEditor::_update_history_pos(int p_new_pos) {
	Node *n = tab_container->get_current_tab_control();

	if (Object::cast_to<ScriptEditorBase>(n)) {
		history.write[history_pos].state = Object::cast_to<ScriptEditorBase>(n)->get_edit_state();
	}
	if (Object::cast_to<EditorHelp>(n)) {
		history.write[history_pos].state = Object::cast_to<EditorHelp>(n)->get_scroll();
	}

	history_pos = p_new_pos;
	tab_container->set_current_tab(history[history_pos].control->get_index());

	n = history[history_pos].control;

	if (Object::cast_to<ScriptEditorBase>(n)) {
		Object::cast_to<ScriptEditorBase>(n)->set_edit_state(history[history_pos].state);
		Object::cast_to<ScriptEditorBase>(n)->ensure_focus();

		Ref<Script> script = Object::cast_to<ScriptEditorBase>(n)->get_edited_resource();
		if (script != nullptr) {
			notify_script_changed(script);
		}
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

Vector<Ref<Script>> ScriptEditor::get_open_scripts() const {
	Vector<Ref<Script>> out_scripts = Vector<Ref<Script>>();

	for (int i = 0; i < tab_container->get_child_count(); i++) {
		ScriptEditorBase *se = Object::cast_to<ScriptEditorBase>(tab_container->get_child(i));
		if (!se) {
			continue;
		}

		Ref<Script> script = se->get_edited_resource();
		if (script != nullptr) {
			out_scripts.push_back(script);
		}
	}

	return out_scripts;
}

Array ScriptEditor::_get_open_script_editors() const {
	Array script_editors;
	for (int i = 0; i < tab_container->get_child_count(); i++) {
		ScriptEditorBase *se = Object::cast_to<ScriptEditorBase>(tab_container->get_child(i));
		if (!se) {
			continue;
		}
		script_editors.push_back(se);
	}
	return script_editors;
}

void ScriptEditor::set_scene_root_script(Ref<Script> p_script) {
	// Don't open dominant script if using an external editor.
	bool use_external_editor =
			EditorSettings::get_singleton()->get("text_editor/external/use_external_editor") ||
			(p_script.is_valid() && p_script->get_language()->overrides_external_editor());
	use_external_editor = use_external_editor && !(p_script.is_valid() && p_script->is_built_in()); // Ignore external editor for built-in scripts.
	const bool open_dominant = EditorSettings::get_singleton()->get("text_editor/behavior/files/open_dominant_script_on_scene_change");

	if (open_dominant && !use_external_editor && p_script.is_valid()) {
		edit(p_script);
	}
}

bool ScriptEditor::script_goto_method(Ref<Script> p_script, const String &p_method) {
	int line = p_script->get_member_line(p_method);

	if (line == -1) {
		return false;
	}

	return edit(p_script, line, 0);
}

void ScriptEditor::set_live_auto_reload_running_scripts(bool p_enabled) {
	auto_reload_running_scripts = p_enabled;
}

void ScriptEditor::_help_search(String p_text) {
	help_search_dialog->popup_dialog(p_text);
}

void ScriptEditor::_open_script_request(const String &p_path) {
	Ref<Script> script = ResourceLoader::load(p_path);
	if (script.is_valid()) {
		script_editor->edit(script, false);
		return;
	}

	Error err;
	Ref<TextFile> text_file = script_editor->_load_text_file(p_path, &err);
	if (text_file.is_valid()) {
		script_editor->edit(text_file, false);
		return;
	}
}

void ScriptEditor::register_syntax_highlighter(const Ref<EditorSyntaxHighlighter> &p_syntax_highlighter) {
	ERR_FAIL_COND(p_syntax_highlighter.is_null());

	if (syntax_highlighters.find(p_syntax_highlighter) == -1) {
		syntax_highlighters.push_back(p_syntax_highlighter);
	}
}

void ScriptEditor::unregister_syntax_highlighter(const Ref<EditorSyntaxHighlighter> &p_syntax_highlighter) {
	ERR_FAIL_COND(p_syntax_highlighter.is_null());

	syntax_highlighters.erase(p_syntax_highlighter);
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

void ScriptEditor::_on_find_in_files_requested(String text) {
	find_in_files_dialog->set_find_in_files_mode(FindInFilesDialog::SEARCH_MODE);
	find_in_files_dialog->set_search_text(text);
	find_in_files_dialog->popup_centered();
}

void ScriptEditor::_on_replace_in_files_requested(String text) {
	find_in_files_dialog->set_find_in_files_mode(FindInFilesDialog::REPLACE_MODE);
	find_in_files_dialog->set_search_text(text);
	find_in_files_dialog->set_replace_text("");
	find_in_files_dialog->popup_centered();
}

void ScriptEditor::_on_find_in_files_result_selected(String fpath, int line_number, int begin, int end) {
	if (ResourceLoader::exists(fpath)) {
		RES res = ResourceLoader::load(fpath);

		if (fpath.get_extension() == "gdshader") {
			ShaderEditorPlugin *shader_editor = Object::cast_to<ShaderEditorPlugin>(EditorNode::get_singleton()->get_editor_data().get_editor("Shader"));
			shader_editor->edit(res.ptr());
			shader_editor->make_visible(true);
			shader_editor->get_shader_editor()->goto_line_selection(line_number - 1, begin, end);
			return;
		} else if (fpath.get_extension() == "tscn") {
			editor->load_scene(fpath);
			return;
		} else {
			Ref<Script> script = res;
			if (script.is_valid()) {
				edit(script);

				ScriptTextEditor *ste = Object::cast_to<ScriptTextEditor>(_get_current_editor());
				if (ste) {
					ste->goto_line_selection(line_number - 1, begin, end);
				}
				return;
			}
		}
	}

	// If the file is not a valid resource/script, load it as a text file.
	Error err;
	Ref<TextFile> text_file = _load_text_file(fpath, &err);
	if (text_file.is_valid()) {
		edit(text_file);

		TextEditor *te = Object::cast_to<TextEditor>(_get_current_editor());
		if (te) {
			te->goto_line_selection(line_number - 1, begin, end);
		}
	}
}

void ScriptEditor::_start_find_in_files(bool with_replace) {
	FindInFiles *f = find_in_files->get_finder();

	f->set_search_text(find_in_files_dialog->get_search_text());
	f->set_match_case(find_in_files_dialog->is_match_case());
	f->set_whole_words(find_in_files_dialog->is_whole_words());
	f->set_folder(find_in_files_dialog->get_folder());
	f->set_filter(find_in_files_dialog->get_filter());

	find_in_files->set_with_replace(with_replace);
	find_in_files->set_replace_text(find_in_files_dialog->get_replace_text());
	find_in_files->start_search();

	editor->make_bottom_panel_item_visible(find_in_files);
}

void ScriptEditor::_on_find_in_files_modified_files(PackedStringArray paths) {
	_test_script_times_on_disk();
	_update_modified_scripts_for_external_editor();
}

void ScriptEditor::_filter_scripts_text_changed(const String &p_newtext) {
	_update_script_names();
}

void ScriptEditor::_filter_methods_text_changed(const String &p_newtext) {
	_update_members_overview();
}

void ScriptEditor::_bind_methods() {
	ClassDB::bind_method("_close_docs_tab", &ScriptEditor::_close_docs_tab);
	ClassDB::bind_method("_close_all_tabs", &ScriptEditor::_close_all_tabs);
	ClassDB::bind_method("_close_other_tabs", &ScriptEditor::_close_other_tabs);
	ClassDB::bind_method("_goto_script_line2", &ScriptEditor::_goto_script_line2);
	ClassDB::bind_method("_copy_script_path", &ScriptEditor::_copy_script_path);

	ClassDB::bind_method("_get_debug_tooltip", &ScriptEditor::_get_debug_tooltip);
	ClassDB::bind_method("_update_script_connections", &ScriptEditor::_update_script_connections);
	ClassDB::bind_method("_help_class_open", &ScriptEditor::_help_class_open);
	ClassDB::bind_method("_live_auto_reload_running_scripts", &ScriptEditor::_live_auto_reload_running_scripts);
	ClassDB::bind_method("_update_members_overview", &ScriptEditor::_update_members_overview);
	ClassDB::bind_method("_update_recent_scripts", &ScriptEditor::_update_recent_scripts);

	ClassDB::bind_method("get_current_editor", &ScriptEditor::_get_current_editor);
	ClassDB::bind_method("get_open_script_editors", &ScriptEditor::_get_open_script_editors);

	ClassDB::bind_method(D_METHOD("register_syntax_highlighter", "syntax_highlighter"), &ScriptEditor::register_syntax_highlighter);
	ClassDB::bind_method(D_METHOD("unregister_syntax_highlighter", "syntax_highlighter"), &ScriptEditor::unregister_syntax_highlighter);

	ClassDB::bind_method(D_METHOD("_get_drag_data_fw", "point", "from"), &ScriptEditor::get_drag_data_fw);
	ClassDB::bind_method(D_METHOD("_can_drop_data_fw", "point", "data", "from"), &ScriptEditor::can_drop_data_fw);
	ClassDB::bind_method(D_METHOD("_drop_data_fw", "point", "data", "from"), &ScriptEditor::drop_data_fw);

	ClassDB::bind_method(D_METHOD("goto_line", "line_number"), &ScriptEditor::_goto_script_line2);
	ClassDB::bind_method(D_METHOD("get_current_script"), &ScriptEditor::_get_current_script);
	ClassDB::bind_method(D_METHOD("get_open_scripts"), &ScriptEditor::_get_open_scripts);
	ClassDB::bind_method(D_METHOD("open_script_create_dialog", "base_name", "base_path"), &ScriptEditor::open_script_create_dialog);

	ADD_SIGNAL(MethodInfo("editor_script_changed", PropertyInfo(Variant::OBJECT, "script", PROPERTY_HINT_RESOURCE_TYPE, "Script")));
	ADD_SIGNAL(MethodInfo("script_close", PropertyInfo(Variant::OBJECT, "script", PROPERTY_HINT_RESOURCE_TYPE, "Script")));
}

ScriptEditor::ScriptEditor(EditorNode *p_editor) {
	current_theme = "";

	script_editor_cache.instantiate();
	script_editor_cache->load(EditorSettings::get_singleton()->get_project_settings_dir().plus_file("script_editor_cache.cfg"));

	completion_cache = memnew(EditorScriptCodeCompletionCache);
	restoring_layout = false;
	waiting_update_names = false;
	pending_auto_reload = false;
	auto_reload_running_scripts = true;
	members_overview_enabled = EditorSettings::get_singleton()->get("text_editor/script_list/show_members_overview");
	help_overview_enabled = EditorSettings::get_singleton()->get("text_editor/help/show_help_index");
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

	scripts_vbox = memnew(VBoxContainer);
	scripts_vbox->set_v_size_flags(SIZE_EXPAND_FILL);
	list_split->add_child(scripts_vbox);

	filter_scripts = memnew(LineEdit);
	filter_scripts->set_placeholder(TTR("Filter scripts"));
	filter_scripts->set_clear_button_enabled(true);
	filter_scripts->connect("text_changed", callable_mp(this, &ScriptEditor::_filter_scripts_text_changed));
	scripts_vbox->add_child(filter_scripts);

	script_list = memnew(ItemList);
	scripts_vbox->add_child(script_list);
	script_list->set_custom_minimum_size(Size2(150, 60) * EDSCALE); //need to give a bit of limit to avoid it from disappearing
	script_list->set_v_size_flags(SIZE_EXPAND_FILL);
	script_split->set_split_offset(70 * EDSCALE);
	_sort_list_on_update = true;
	script_list->connect("gui_input", callable_mp(this, &ScriptEditor::_script_list_gui_input), varray(), CONNECT_DEFERRED);
	script_list->set_allow_rmb_select(true);
	script_list->set_drag_forwarding(this);

	context_menu = memnew(PopupMenu);
	add_child(context_menu);
	context_menu->connect("id_pressed", callable_mp(this, &ScriptEditor::_menu_option));

	overview_vbox = memnew(VBoxContainer);
	overview_vbox->set_custom_minimum_size(Size2(0, 90));
	overview_vbox->set_v_size_flags(SIZE_EXPAND_FILL);

	list_split->add_child(overview_vbox);
	buttons_hbox = memnew(HBoxContainer);
	overview_vbox->add_child(buttons_hbox);

	filename = memnew(Label);
	filename->set_clip_text(true);
	filename->set_h_size_flags(SIZE_EXPAND_FILL);
	filename->add_theme_style_override("normal", EditorNode::get_singleton()->get_gui_base()->get_theme_stylebox(SNAME("normal"), SNAME("LineEdit")));
	buttons_hbox->add_child(filename);

	members_overview_alphabeta_sort_button = memnew(Button);
	members_overview_alphabeta_sort_button->set_flat(true);
	members_overview_alphabeta_sort_button->set_tooltip(TTR("Toggle alphabetical sorting of the method list."));
	members_overview_alphabeta_sort_button->set_toggle_mode(true);
	members_overview_alphabeta_sort_button->set_pressed(EditorSettings::get_singleton()->get("text_editor/script_list/sort_members_outline_alphabetically"));
	members_overview_alphabeta_sort_button->connect("toggled", callable_mp(this, &ScriptEditor::_toggle_members_overview_alpha_sort));

	buttons_hbox->add_child(members_overview_alphabeta_sort_button);

	filter_methods = memnew(LineEdit);
	filter_methods->set_placeholder(TTR("Filter methods"));
	filter_methods->set_clear_button_enabled(true);
	filter_methods->connect("text_changed", callable_mp(this, &ScriptEditor::_filter_methods_text_changed));
	overview_vbox->add_child(filter_methods);

	members_overview = memnew(ItemList);
	overview_vbox->add_child(members_overview);

	members_overview->set_allow_reselect(true);
	members_overview->set_custom_minimum_size(Size2(0, 60) * EDSCALE); //need to give a bit of limit to avoid it from disappearing
	members_overview->set_v_size_flags(SIZE_EXPAND_FILL);
	members_overview->set_allow_rmb_select(true);

	help_overview = memnew(ItemList);
	overview_vbox->add_child(help_overview);
	help_overview->set_allow_reselect(true);
	help_overview->set_custom_minimum_size(Size2(0, 60) * EDSCALE); //need to give a bit of limit to avoid it from disappearing
	help_overview->set_v_size_flags(SIZE_EXPAND_FILL);

	VBoxContainer *code_editor_container = memnew(VBoxContainer);
	script_split->add_child(code_editor_container);

	tab_container = memnew(TabContainer);
	tab_container->set_tabs_visible(false);
	tab_container->set_custom_minimum_size(Size2(200, 0) * EDSCALE);
	code_editor_container->add_child(tab_container);
	tab_container->set_h_size_flags(SIZE_EXPAND_FILL);
	tab_container->set_v_size_flags(SIZE_EXPAND_FILL);

	find_replace_bar = memnew(FindReplaceBar);
	code_editor_container->add_child(find_replace_bar);
	find_replace_bar->hide();

	ED_SHORTCUT("script_editor/window_sort", TTR("Sort"));
	ED_SHORTCUT("script_editor/window_move_up", TTR("Move Up"), KeyModifierMask::SHIFT | KeyModifierMask::ALT | Key::UP);
	ED_SHORTCUT("script_editor/window_move_down", TTR("Move Down"), KeyModifierMask::SHIFT | KeyModifierMask::ALT | Key::DOWN);
	// FIXME: These should be `Key::GREATER` and `Key::LESS` but those don't work.
	ED_SHORTCUT("script_editor/next_script", TTR("Next Script"), KeyModifierMask::CMD | KeyModifierMask::SHIFT | Key::PERIOD);
	ED_SHORTCUT("script_editor/prev_script", TTR("Previous Script"), KeyModifierMask::CMD | KeyModifierMask::SHIFT | Key::COMMA);
	set_process_input(true);
	set_process_unhandled_input(true);

	file_menu = memnew(MenuButton);
	file_menu->set_text(TTR("File"));
	file_menu->set_switch_on_hover(true);
	file_menu->set_shortcut_context(this);
	menu_hb->add_child(file_menu);

	file_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/new", TTR("New Script...")), FILE_NEW);
	file_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/new_textfile", TTR("New Text File...")), FILE_NEW_TEXTFILE);
	file_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/open", TTR("Open...")), FILE_OPEN);
	file_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/reopen_closed_script", TTR("Reopen Closed Script"), KeyModifierMask::CMD | KeyModifierMask::SHIFT | Key::T), FILE_REOPEN_CLOSED);
	file_menu->get_popup()->add_submenu_item(TTR("Open Recent"), "RecentScripts", FILE_OPEN_RECENT);

	recent_scripts = memnew(PopupMenu);
	recent_scripts->set_name("RecentScripts");
	file_menu->get_popup()->add_child(recent_scripts);
	recent_scripts->connect("id_pressed", callable_mp(this, &ScriptEditor::_open_recent_script));
	_update_recent_scripts();

	file_menu->get_popup()->add_separator();
	file_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/save", TTR("Save"), KeyModifierMask::ALT | KeyModifierMask::CMD | Key::S), FILE_SAVE);
	file_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/save_as", TTR("Save As...")), FILE_SAVE_AS);
	file_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/save_all", TTR("Save All"), KeyModifierMask::SHIFT | KeyModifierMask::ALT | Key::S), FILE_SAVE_ALL);
	file_menu->get_popup()->add_separator();
	file_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/reload_script_soft", TTR("Soft Reload Script"), KeyModifierMask::CMD | KeyModifierMask::ALT | Key::R), FILE_TOOL_RELOAD_SOFT);
	file_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/copy_path", TTR("Copy Script Path")), FILE_COPY_PATH);
	file_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/show_in_file_system", TTR("Show in FileSystem")), SHOW_IN_FILE_SYSTEM);
	file_menu->get_popup()->add_separator();

	file_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/history_previous", TTR("History Previous"), KeyModifierMask::ALT | Key::LEFT), WINDOW_PREV);
	file_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/history_next", TTR("History Next"), KeyModifierMask::ALT | Key::RIGHT), WINDOW_NEXT);
	file_menu->get_popup()->add_separator();

	file_menu->get_popup()->add_submenu_item(TTR("Theme"), "Theme", FILE_THEME);

	theme_submenu = memnew(PopupMenu);
	theme_submenu->set_name("Theme");
	file_menu->get_popup()->add_child(theme_submenu);
	theme_submenu->connect("id_pressed", callable_mp(this, &ScriptEditor::_theme_option));
	theme_submenu->add_shortcut(ED_SHORTCUT("script_editor/import_theme", TTR("Import Theme...")), THEME_IMPORT);
	theme_submenu->add_shortcut(ED_SHORTCUT("script_editor/reload_theme", TTR("Reload Theme")), THEME_RELOAD);

	theme_submenu->add_separator();
	theme_submenu->add_shortcut(ED_SHORTCUT("script_editor/save_theme", TTR("Save Theme")), THEME_SAVE);
	theme_submenu->add_shortcut(ED_SHORTCUT("script_editor/save_theme_as", TTR("Save Theme As...")), THEME_SAVE_AS);

	file_menu->get_popup()->add_separator();
	file_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/close_file", TTR("Close"), KeyModifierMask::CMD | Key::W), FILE_CLOSE);
	file_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/close_all", TTR("Close All")), CLOSE_ALL);
	file_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/close_other_tabs", TTR("Close Other Tabs")), CLOSE_OTHER_TABS);
	file_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/close_docs", TTR("Close Docs")), CLOSE_DOCS);

	file_menu->get_popup()->add_separator();
	file_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/run_file", TTR("Run"), KeyModifierMask::CMD | KeyModifierMask::SHIFT | Key::X), FILE_RUN);

	file_menu->get_popup()->add_separator();
	file_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/toggle_scripts_panel", TTR("Toggle Scripts Panel"), KeyModifierMask::CMD | Key::BACKSLASH), TOGGLE_SCRIPTS_PANEL);
	file_menu->get_popup()->connect("id_pressed", callable_mp(this, &ScriptEditor::_menu_option));

	script_search_menu = memnew(MenuButton);
	script_search_menu->set_text(TTR("Search"));
	script_search_menu->set_switch_on_hover(true);
	script_search_menu->set_shortcut_context(this);
	script_search_menu->get_popup()->connect("id_pressed", callable_mp(this, &ScriptEditor::_menu_option));
	menu_hb->add_child(script_search_menu);

	MenuButton *debug_menu = memnew(MenuButton);
	menu_hb->add_child(debug_menu);
	debug_menu->hide(); // Handled by EditorDebuggerNode below.

	EditorDebuggerNode *debugger = EditorDebuggerNode::get_singleton();
	debugger->set_script_debug_button(debug_menu);
	debugger->connect("goto_script_line", callable_mp(this, &ScriptEditor::_goto_script_line));
	debugger->connect("set_execution", callable_mp(this, &ScriptEditor::_set_execution));
	debugger->connect("clear_execution", callable_mp(this, &ScriptEditor::_clear_execution));
	debugger->connect("breaked", callable_mp(this, &ScriptEditor::_breaked));
	debugger->get_default_debugger()->connect("set_breakpoint", callable_mp(this, &ScriptEditor::_set_breakpoint));
	debugger->get_default_debugger()->connect("clear_breakpoints", callable_mp(this, &ScriptEditor::_clear_breakpoints));

	menu_hb->add_spacer();

	script_icon = memnew(TextureRect);
	menu_hb->add_child(script_icon);
	script_name_label = memnew(Label);
	menu_hb->add_child(script_name_label);

	script_icon->hide();
	script_name_label->hide();

	menu_hb->add_spacer();

	site_search = memnew(Button);
	site_search->set_flat(true);
	site_search->set_text(TTR("Online Docs"));
	site_search->connect("pressed", callable_mp(this, &ScriptEditor::_menu_option), varray(SEARCH_WEBSITE));
	menu_hb->add_child(site_search);
	site_search->set_tooltip(TTR("Open Godot online documentation."));

	help_search = memnew(Button);
	help_search->set_flat(true);
	help_search->set_text(TTR("Search Help"));
	help_search->connect("pressed", callable_mp(this, &ScriptEditor::_menu_option), varray(SEARCH_HELP));
	menu_hb->add_child(help_search);
	help_search->set_tooltip(TTR("Search the reference documentation."));

	menu_hb->add_child(memnew(VSeparator));

	script_back = memnew(Button);
	script_back->set_flat(true);
	script_back->connect("pressed", callable_mp(this, &ScriptEditor::_history_back));
	menu_hb->add_child(script_back);
	script_back->set_disabled(true);
	script_back->set_tooltip(TTR("Go to previous edited document."));

	script_forward = memnew(Button);
	script_forward->set_flat(true);
	script_forward->connect("pressed", callable_mp(this, &ScriptEditor::_history_forward));
	menu_hb->add_child(script_forward);
	script_forward->set_disabled(true);
	script_forward->set_tooltip(TTR("Go to next edited document."));

	tab_container->connect("tab_changed", callable_mp(this, &ScriptEditor::_tab_changed));

	erase_tab_confirm = memnew(ConfirmationDialog);
	erase_tab_confirm->get_ok_button()->set_text(TTR("Save"));
	erase_tab_confirm->add_button(TTR("Discard"), DisplayServer::get_singleton()->get_swap_cancel_ok(), "discard");
	erase_tab_confirm->connect("confirmed", callable_mp(this, &ScriptEditor::_close_current_tab), varray(true));
	erase_tab_confirm->connect("custom_action", callable_mp(this, &ScriptEditor::_close_discard_current_tab));
	add_child(erase_tab_confirm);

	script_create_dialog = memnew(ScriptCreateDialog);
	script_create_dialog->set_title(TTR("Create Script"));
	add_child(script_create_dialog);
	script_create_dialog->connect("script_created", callable_mp(this, &ScriptEditor::_script_created));

	file_dialog_option = -1;
	file_dialog = memnew(EditorFileDialog);
	add_child(file_dialog);
	file_dialog->connect("file_selected", callable_mp(this, &ScriptEditor::_file_dialog_action));

	error_dialog = memnew(AcceptDialog);
	add_child(error_dialog);

	disk_changed = memnew(ConfirmationDialog);
	{
		VBoxContainer *vbc = memnew(VBoxContainer);
		disk_changed->add_child(vbc);

		Label *dl = memnew(Label);
		dl->set_text(TTR("The following files are newer on disk.\nWhat action should be taken?:"));
		vbc->add_child(dl);

		disk_changed_list = memnew(Tree);
		vbc->add_child(disk_changed_list);
		disk_changed_list->set_v_size_flags(SIZE_EXPAND_FILL);

		disk_changed->connect("confirmed", callable_mp(this, &ScriptEditor::_reload_scripts));
		disk_changed->get_ok_button()->set_text(TTR("Reload"));

		disk_changed->add_button(TTR("Resave"), !DisplayServer::get_singleton()->get_swap_cancel_ok(), "resave");
		disk_changed->connect("custom_action", callable_mp(this, &ScriptEditor::_resave_scripts));
	}

	add_child(disk_changed);

	script_editor = this;

	autosave_timer = memnew(Timer);
	autosave_timer->set_one_shot(false);
	autosave_timer->connect(SceneStringNames::get_singleton()->tree_entered, callable_mp(this, &ScriptEditor::_update_autosave_timer));
	autosave_timer->connect("timeout", callable_mp(this, &ScriptEditor::_autosave_scripts));
	add_child(autosave_timer);

	grab_focus_block = false;

	help_search_dialog = memnew(EditorHelpSearch);
	add_child(help_search_dialog);
	help_search_dialog->connect("go_to_help", callable_mp(this, &ScriptEditor::_help_class_goto));

	find_in_files_dialog = memnew(FindInFilesDialog);
	find_in_files_dialog->connect(FindInFilesDialog::SIGNAL_FIND_REQUESTED, callable_mp(this, &ScriptEditor::_start_find_in_files), varray(false));
	find_in_files_dialog->connect(FindInFilesDialog::SIGNAL_REPLACE_REQUESTED, callable_mp(this, &ScriptEditor::_start_find_in_files), varray(true));
	add_child(find_in_files_dialog);
	find_in_files = memnew(FindInFilesPanel);
	find_in_files_button = editor->add_bottom_panel_item(TTR("Search Results"), find_in_files);
	find_in_files->set_custom_minimum_size(Size2(0, 200) * EDSCALE);
	find_in_files->connect(FindInFilesPanel::SIGNAL_RESULT_SELECTED, callable_mp(this, &ScriptEditor::_on_find_in_files_result_selected));
	find_in_files->connect(FindInFilesPanel::SIGNAL_FILES_MODIFIED, callable_mp(this, &ScriptEditor::_on_find_in_files_modified_files));
	find_in_files->hide();
	find_in_files_button->hide();

	history_pos = -1;

	edit_pass = 0;
	trim_trailing_whitespace_on_save = EditorSettings::get_singleton()->get("text_editor/behavior/files/trim_trailing_whitespace_on_save");
	convert_indent_on_save = EditorSettings::get_singleton()->get("text_editor/behavior/files/convert_indent_on_save");
	use_space_indentation = EditorSettings::get_singleton()->get("text_editor/behavior/indent/type");

	ScriptServer::edit_request_func = _open_script_request;

	add_theme_style_override("panel", editor->get_gui_base()->get_theme_stylebox(SNAME("ScriptEditorPanel"), SNAME("EditorStyles")));
	tab_container->add_theme_style_override("panel", editor->get_gui_base()->get_theme_stylebox(SNAME("ScriptEditor"), SNAME("EditorStyles")));
}

ScriptEditor::~ScriptEditor() {
	memdelete(completion_cache);
}

void ScriptEditorPlugin::edit(Object *p_object) {
	if (Object::cast_to<Script>(p_object)) {
		Script *p_script = Object::cast_to<Script>(p_object);
		String res_path = p_script->get_path().get_slice("::", 0);

		if (p_script->is_built_in()) {
			if (ResourceLoader::get_resource_type(res_path) == "PackedScene") {
				if (!EditorNode::get_singleton()->is_scene_open(res_path)) {
					EditorNode::get_singleton()->load_scene(res_path);
				}
			} else {
				EditorNode::get_singleton()->load_resource(res_path);
			}
		}
		script_editor->edit(p_script);
	} else if (Object::cast_to<TextFile>(p_object)) {
		script_editor->edit(Object::cast_to<TextFile>(p_object));
	}
}

bool ScriptEditorPlugin::handles(Object *p_object) const {
	if (Object::cast_to<TextFile>(p_object)) {
		return true;
	}

	if (Object::cast_to<Script>(p_object)) {
		return true;
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
	script_editor->get_breakpoints(p_breakpoints);
}

void ScriptEditorPlugin::edited_scene_changed() {
	script_editor->edited_scene_changed();
}

ScriptEditorPlugin::ScriptEditorPlugin(EditorNode *p_node) {
	editor = p_node;
	script_editor = memnew(ScriptEditor(p_node));
	editor->get_main_control()->add_child(script_editor);
	script_editor->set_v_size_flags(Control::SIZE_EXPAND_FILL);

	script_editor->hide();

	EDITOR_DEF("text_editor/behavior/files/auto_reload_scripts_on_external_change", true);
	ScriptServer::set_reload_scripts_on_save(EDITOR_DEF("text_editor/behavior/files/auto_reload_and_parse_scripts_on_save", true));
	EDITOR_DEF("text_editor/behavior/files/open_dominant_script_on_scene_change", true);
	EDITOR_DEF("text_editor/external/use_external_editor", false);
	EDITOR_DEF("text_editor/external/exec_path", "");
	EDITOR_DEF("text_editor/script_list/script_temperature_enabled", true);
	EDITOR_DEF("text_editor/script_list/script_temperature_history_size", 15);
	EDITOR_DEF("text_editor/script_list/group_help_pages", true);
	EditorSettings::get_singleton()->add_property_hint(PropertyInfo(Variant::INT, "text_editor/script_list/sort_scripts_by", PROPERTY_HINT_ENUM, "Name,Path,None"));
	EDITOR_DEF("text_editor/script_list/sort_scripts_by", 0);
	EditorSettings::get_singleton()->add_property_hint(PropertyInfo(Variant::INT, "text_editor/script_list/list_script_names_as", PROPERTY_HINT_ENUM, "Name,Parent Directory And Name,Full Path"));
	EDITOR_DEF("text_editor/script_list/list_script_names_as", 0);
	EditorSettings::get_singleton()->add_property_hint(PropertyInfo(Variant::STRING, "text_editor/external/exec_path", PROPERTY_HINT_GLOBAL_FILE));
	EDITOR_DEF("text_editor/external/exec_flags", "{file}");
	EditorSettings::get_singleton()->add_property_hint(PropertyInfo(Variant::STRING, "text_editor/external/exec_flags", PROPERTY_HINT_PLACEHOLDER_TEXT, "Call flags with placeholders: {project}, {file}, {col}, {line}."));

	ED_SHORTCUT("script_editor/reopen_closed_script", TTR("Reopen Closed Script"), KeyModifierMask::CMD | KeyModifierMask::SHIFT | Key::T);
	ED_SHORTCUT("script_editor/clear_recent", TTR("Clear Recent Scripts"));
}

ScriptEditorPlugin::~ScriptEditorPlugin() {
}
