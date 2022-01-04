/*************************************************************************/
/*  editor_debugger_node.cpp                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "editor_debugger_node.h"

#include "editor/debugger/editor_debugger_tree.h"
#include "editor/debugger/script_editor_debugger.h"
#include "editor/editor_log.h"
#include "editor/editor_node.h"
#include "editor/plugins/editor_debugger_plugin.h"
#include "editor/plugins/script_editor_plugin.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/tab_container.h"

template <typename Func>
void _for_all(TabContainer *p_node, const Func &p_func) {
	for (int i = 0; i < p_node->get_tab_count(); i++) {
		ScriptEditorDebugger *dbg = Object::cast_to<ScriptEditorDebugger>(p_node->get_tab_control(i));
		ERR_FAIL_COND(!dbg);
		p_func(dbg);
	}
}

EditorDebuggerNode *EditorDebuggerNode::singleton = nullptr;

EditorDebuggerNode::EditorDebuggerNode() {
	if (!singleton) {
		singleton = this;
	}

	add_theme_constant_override("margin_left", -EditorNode::get_singleton()->get_gui_base()->get_theme_stylebox(SNAME("BottomPanelDebuggerOverride"), SNAME("EditorStyles"))->get_margin(SIDE_LEFT));
	add_theme_constant_override("margin_right", -EditorNode::get_singleton()->get_gui_base()->get_theme_stylebox(SNAME("BottomPanelDebuggerOverride"), SNAME("EditorStyles"))->get_margin(SIDE_RIGHT));

	tabs = memnew(TabContainer);
	tabs->set_tab_alignment(TabContainer::ALIGNMENT_LEFT);
	tabs->set_tabs_visible(false);
	tabs->connect("tab_changed", callable_mp(this, &EditorDebuggerNode::_debugger_changed));
	add_child(tabs);

	Ref<StyleBoxEmpty> empty;
	empty.instantiate();
	tabs->add_theme_style_override("panel", empty);

	auto_switch_remote_scene_tree = EDITOR_DEF("debugger/auto_switch_to_remote_scene_tree", false);
	_add_debugger();

	// Remote scene tree
	remote_scene_tree = memnew(EditorDebuggerTree);
	remote_scene_tree->connect("object_selected", callable_mp(this, &EditorDebuggerNode::_remote_object_requested));
	remote_scene_tree->connect("save_node", callable_mp(this, &EditorDebuggerNode::_save_node_requested));
	EditorNode::get_singleton()->get_scene_tree_dock()->add_remote_tree_editor(remote_scene_tree);
	EditorNode::get_singleton()->get_scene_tree_dock()->connect("remote_tree_selected", callable_mp(this, &EditorDebuggerNode::request_remote_tree));

	remote_scene_tree_timeout = EDITOR_DEF("debugger/remote_scene_tree_refresh_interval", 1.0);
	inspect_edited_object_timeout = EDITOR_DEF("debugger/remote_inspect_refresh_interval", 0.2);

	EditorNode *editor = EditorNode::get_singleton();
	editor->get_undo_redo()->set_method_notify_callback(_method_changeds, this);
	editor->get_undo_redo()->set_property_notify_callback(_property_changeds, this);
	editor->get_pause_button()->connect("pressed", callable_mp(this, &EditorDebuggerNode::_paused));
}

ScriptEditorDebugger *EditorDebuggerNode::_add_debugger() {
	ScriptEditorDebugger *node = memnew(ScriptEditorDebugger(EditorNode::get_singleton()));

	int id = tabs->get_tab_count();
	node->connect("stop_requested", callable_mp(this, &EditorDebuggerNode::_debugger_wants_stop), varray(id));
	node->connect("stopped", callable_mp(this, &EditorDebuggerNode::_debugger_stopped), varray(id));
	node->connect("stack_frame_selected", callable_mp(this, &EditorDebuggerNode::_stack_frame_selected), varray(id));
	node->connect("error_selected", callable_mp(this, &EditorDebuggerNode::_error_selected), varray(id));
	node->connect("clear_execution", callable_mp(this, &EditorDebuggerNode::_clear_execution));
	node->connect("breaked", callable_mp(this, &EditorDebuggerNode::_breaked), varray(id));
	node->connect("remote_tree_updated", callable_mp(this, &EditorDebuggerNode::_remote_tree_updated), varray(id));
	node->connect("remote_object_updated", callable_mp(this, &EditorDebuggerNode::_remote_object_updated), varray(id));
	node->connect("remote_object_property_updated", callable_mp(this, &EditorDebuggerNode::_remote_object_property_updated), varray(id));
	node->connect("remote_object_requested", callable_mp(this, &EditorDebuggerNode::_remote_object_requested), varray(id));

	if (tabs->get_tab_count() > 0) {
		get_debugger(0)->clear_style();
	}

	tabs->add_child(node);

	node->set_name("Session " + itos(tabs->get_tab_count()));
	if (tabs->get_tab_count() > 1) {
		node->clear_style();
		tabs->set_tabs_visible(true);
		tabs->add_theme_style_override("panel", EditorNode::get_singleton()->get_gui_base()->get_theme_stylebox(SNAME("DebuggerPanel"), SNAME("EditorStyles")));
	}

	if (!debugger_plugins.is_empty()) {
		for (Set<Ref<Script>>::Element *i = debugger_plugins.front(); i; i = i->next()) {
			node->add_debugger_plugin(i->get());
		}
	}

	return node;
}

void EditorDebuggerNode::_stack_frame_selected(int p_debugger) {
	const ScriptEditorDebugger *dbg = get_debugger(p_debugger);
	ERR_FAIL_COND(!dbg);
	if (dbg != get_current_debugger()) {
		return;
	}
	_text_editor_stack_goto(dbg);
}

void EditorDebuggerNode::_error_selected(const String &p_file, int p_line, int p_debugger) {
	Ref<Script> s = ResourceLoader::load(p_file);
	emit_signal(SNAME("goto_script_line"), s, p_line - 1);
}

void EditorDebuggerNode::_text_editor_stack_goto(const ScriptEditorDebugger *p_debugger) {
	const String file = p_debugger->get_stack_script_file();
	if (file.is_empty()) {
		return;
	}
	stack_script = ResourceLoader::load(file);
	const int line = p_debugger->get_stack_script_line() - 1;
	emit_signal(SNAME("goto_script_line"), stack_script, line);
	emit_signal(SNAME("set_execution"), stack_script, line);
	stack_script.unref(); // Why?!?
}

void EditorDebuggerNode::_bind_methods() {
	// LiveDebug.
	ClassDB::bind_method("live_debug_create_node", &EditorDebuggerNode::live_debug_create_node);
	ClassDB::bind_method("live_debug_instance_node", &EditorDebuggerNode::live_debug_instance_node);
	ClassDB::bind_method("live_debug_remove_node", &EditorDebuggerNode::live_debug_remove_node);
	ClassDB::bind_method("live_debug_remove_and_keep_node", &EditorDebuggerNode::live_debug_remove_and_keep_node);
	ClassDB::bind_method("live_debug_restore_node", &EditorDebuggerNode::live_debug_restore_node);
	ClassDB::bind_method("live_debug_duplicate_node", &EditorDebuggerNode::live_debug_duplicate_node);
	ClassDB::bind_method("live_debug_reparent_node", &EditorDebuggerNode::live_debug_reparent_node);

	ADD_SIGNAL(MethodInfo("goto_script_line"));
	ADD_SIGNAL(MethodInfo("set_execution", PropertyInfo("script"), PropertyInfo(Variant::INT, "line")));
	ADD_SIGNAL(MethodInfo("clear_execution", PropertyInfo("script")));
	ADD_SIGNAL(MethodInfo("breaked", PropertyInfo(Variant::BOOL, "reallydid"), PropertyInfo(Variant::BOOL, "can_debug")));
	ADD_SIGNAL(MethodInfo("breakpoint_toggled", PropertyInfo(Variant::STRING, "path"), PropertyInfo(Variant::INT, "line"), PropertyInfo(Variant::BOOL, "enabled")));
}

EditorDebuggerRemoteObject *EditorDebuggerNode::get_inspected_remote_object() {
	return Object::cast_to<EditorDebuggerRemoteObject>(ObjectDB::get_instance(EditorNode::get_singleton()->get_editor_history()->get_current()));
}

ScriptEditorDebugger *EditorDebuggerNode::get_debugger(int p_id) const {
	return Object::cast_to<ScriptEditorDebugger>(tabs->get_tab_control(p_id));
}

ScriptEditorDebugger *EditorDebuggerNode::get_current_debugger() const {
	return Object::cast_to<ScriptEditorDebugger>(tabs->get_tab_control(tabs->get_current_tab()));
}

ScriptEditorDebugger *EditorDebuggerNode::get_default_debugger() const {
	return Object::cast_to<ScriptEditorDebugger>(tabs->get_tab_control(0));
}

String EditorDebuggerNode::get_server_uri() const {
	ERR_FAIL_COND_V(server.is_null(), "");
	return server->get_uri();
}

Error EditorDebuggerNode::start(const String &p_uri) {
	stop();
	ERR_FAIL_COND_V(p_uri.find("://") < 0, ERR_INVALID_PARAMETER);
	if (EDITOR_GET("run/output/always_open_output_on_play")) {
		EditorNode::get_singleton()->make_bottom_panel_item_visible(EditorNode::get_log());
	} else {
		EditorNode::get_singleton()->make_bottom_panel_item_visible(this);
	}
	server = Ref<EditorDebuggerServer>(EditorDebuggerServer::create(p_uri.substr(0, p_uri.find("://") + 3)));
	const Error err = server->start(p_uri);
	if (err != OK) {
		return err;
	}
	set_process(true);
	EditorNode::get_log()->add_message("--- Debugging process started ---", EditorLog::MSG_TYPE_EDITOR);
	return OK;
}

void EditorDebuggerNode::stop() {
	if (server.is_valid()) {
		server->stop();
		EditorNode::get_log()->add_message("--- Debugging process stopped ---", EditorLog::MSG_TYPE_EDITOR);
		server.unref();
	}
	// Also close all debugging sessions.
	_for_all(tabs, [&](ScriptEditorDebugger *dbg) {
		if (dbg->is_session_active()) {
			dbg->_stop_and_notify();
		}
	});
	_break_state_changed();
	if (hide_on_stop) {
		if (is_visible_in_tree()) {
			EditorNode::get_singleton()->hide_bottom_panel();
		}
	}
	breakpoints.clear();
	set_process(false);
}

void EditorDebuggerNode::_notification(int p_what) {
	switch (p_what) {
		case EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED: {
			if (tabs->get_tab_count() > 1) {
				add_theme_constant_override("margin_left", -EditorNode::get_singleton()->get_gui_base()->get_theme_stylebox(SNAME("BottomPanelDebuggerOverride"), SNAME("EditorStyles"))->get_margin(SIDE_LEFT));
				add_theme_constant_override("margin_right", -EditorNode::get_singleton()->get_gui_base()->get_theme_stylebox(SNAME("BottomPanelDebuggerOverride"), SNAME("EditorStyles"))->get_margin(SIDE_RIGHT));

				tabs->add_theme_style_override("panel", EditorNode::get_singleton()->get_gui_base()->get_theme_stylebox(SNAME("DebuggerPanel"), SNAME("EditorStyles")));
			}
		} break;
		case NOTIFICATION_READY: {
			_update_debug_options();
		} break;
		default:
			break;
	}

	if (p_what != NOTIFICATION_PROCESS || !server.is_valid()) {
		return;
	}

	if (!server.is_valid() || !server->is_active()) {
		stop();
		return;
	}
	server->poll();

	// Errors and warnings
	int error_count = 0;
	int warning_count = 0;
	_for_all(tabs, [&](ScriptEditorDebugger *dbg) {
		error_count += dbg->get_error_count();
		warning_count += dbg->get_warning_count();
	});

	if (error_count != last_error_count || warning_count != last_warning_count) {
		_for_all(tabs, [&](ScriptEditorDebugger *dbg) {
			dbg->update_tabs();
		});

		if (error_count == 0 && warning_count == 0) {
			debugger_button->set_text(TTR("Debugger"));
			debugger_button->remove_theme_color_override("font_color");
			debugger_button->set_icon(Ref<Texture2D>());
		} else {
			debugger_button->set_text(TTR("Debugger") + " (" + itos(error_count + warning_count) + ")");
			if (error_count >= 1 && warning_count >= 1) {
				debugger_button->set_icon(get_theme_icon(SNAME("ErrorWarning"), SNAME("EditorIcons")));
				// Use error color to represent the highest level of severity reported.
				debugger_button->add_theme_color_override("font_color", get_theme_color(SNAME("error_color"), SNAME("Editor")));
			} else if (error_count >= 1) {
				debugger_button->set_icon(get_theme_icon(SNAME("Error"), SNAME("EditorIcons")));
				debugger_button->add_theme_color_override("font_color", get_theme_color(SNAME("error_color"), SNAME("Editor")));
			} else {
				debugger_button->set_icon(get_theme_icon(SNAME("Warning"), SNAME("EditorIcons")));
				debugger_button->add_theme_color_override("font_color", get_theme_color(SNAME("warning_color"), SNAME("Editor")));
			}
		}
		last_error_count = error_count;
		last_warning_count = warning_count;
	}

	// Remote scene tree update
	remote_scene_tree_timeout -= get_process_delta_time();
	if (remote_scene_tree_timeout < 0) {
		remote_scene_tree_timeout = EditorSettings::get_singleton()->get("debugger/remote_scene_tree_refresh_interval");
		if (remote_scene_tree->is_visible_in_tree()) {
			get_current_debugger()->request_remote_tree();
		}
	}

	// Remote inspector update
	inspect_edited_object_timeout -= get_process_delta_time();
	if (inspect_edited_object_timeout < 0) {
		inspect_edited_object_timeout = EditorSettings::get_singleton()->get("debugger/remote_inspect_refresh_interval");
		if (EditorDebuggerRemoteObject *obj = get_inspected_remote_object()) {
			get_current_debugger()->request_remote_object(obj->remote_object_id);
		}
	}

	// Take connections.
	if (server->is_connection_available()) {
		ScriptEditorDebugger *debugger = nullptr;
		_for_all(tabs, [&](ScriptEditorDebugger *dbg) {
			if (debugger || dbg->is_session_active()) {
				return;
			}
			debugger = dbg;
		});
		if (debugger == nullptr) {
			if (tabs->get_tab_count() <= 4) { // Max 4 debugging sessions active.
				debugger = _add_debugger();
			} else {
				// We already have too many sessions, disconnecting new clients to prevent them from hanging.
				server->take_connection()->close();
				return; // Can't add, stop here.
			}
		}

		EditorNode::get_singleton()->get_pause_button()->set_disabled(false);
		// Switch to remote tree view if so desired.
		auto_switch_remote_scene_tree = (bool)EditorSettings::get_singleton()->get("debugger/auto_switch_to_remote_scene_tree");
		if (auto_switch_remote_scene_tree) {
			EditorNode::get_singleton()->get_scene_tree_dock()->show_remote_tree();
		}
		// Good to go.
		EditorNode::get_singleton()->get_scene_tree_dock()->show_tab_buttons();
		debugger->set_editor_remote_tree(remote_scene_tree);
		debugger->start(server->take_connection());
		// Send breakpoints.
		for (const KeyValue<Breakpoint, bool> &E : breakpoints) {
			const Breakpoint &bp = E.key;
			debugger->set_breakpoint(bp.source, bp.line, E.value);
		} // Will arrive too late, how does the regular run work?

		debugger->update_live_edit_root();
	}
}

void EditorDebuggerNode::_debugger_stopped(int p_id) {
	ScriptEditorDebugger *dbg = get_debugger(p_id);
	ERR_FAIL_COND(!dbg);

	bool found = false;
	_for_all(tabs, [&](ScriptEditorDebugger *p_debugger) {
		if (p_debugger->is_session_active()) {
			found = true;
		}
	});
	if (!found) {
		EditorNode::get_singleton()->get_pause_button()->set_pressed(false);
		EditorNode::get_singleton()->get_pause_button()->set_disabled(true);
		EditorNode::get_singleton()->get_scene_tree_dock()->hide_remote_tree();
		EditorNode::get_singleton()->get_scene_tree_dock()->hide_tab_buttons();
		EditorNode::get_singleton()->notify_all_debug_sessions_exited();
	}
}

void EditorDebuggerNode::_debugger_wants_stop(int p_id) {
	// Ask editor to kill PID.
	int pid = get_debugger(p_id)->get_remote_pid();
	if (pid) {
		EditorNode::get_singleton()->call_deferred(SNAME("stop_child_process"), pid);
	}
}

void EditorDebuggerNode::_debugger_changed(int p_tab) {
	if (get_inspected_remote_object()) {
		// Clear inspected object, you can only inspect objects in selected debugger.
		// Hopefully, in the future, we will have one inspector per debugger.
		EditorNode::get_singleton()->push_item(nullptr);
	}
	if (remote_scene_tree->is_visible_in_tree()) {
		get_current_debugger()->request_remote_tree();
	}
	if (get_current_debugger()->is_breaked()) {
		_text_editor_stack_goto(get_current_debugger());
	}
}

void EditorDebuggerNode::set_script_debug_button(MenuButton *p_button) {
	script_menu = p_button;
	script_menu->set_text(TTR("Debug"));
	script_menu->set_switch_on_hover(true);
	PopupMenu *p = script_menu->get_popup();
	p->add_shortcut(ED_GET_SHORTCUT("debugger/step_into"), DEBUG_STEP);
	p->add_shortcut(ED_GET_SHORTCUT("debugger/step_over"), DEBUG_NEXT);
	p->add_separator();
	p->add_shortcut(ED_GET_SHORTCUT("debugger/break"), DEBUG_BREAK);
	p->add_shortcut(ED_GET_SHORTCUT("debugger/continue"), DEBUG_CONTINUE);
	p->add_separator();
	p->add_check_shortcut(ED_GET_SHORTCUT("debugger/keep_debugger_open"), DEBUG_KEEP_DEBUGGER_OPEN);
	p->add_check_shortcut(ED_GET_SHORTCUT("debugger/debug_with_external_editor"), DEBUG_WITH_EXTERNAL_EDITOR);
	p->connect("id_pressed", callable_mp(this, &EditorDebuggerNode::_menu_option));

	_break_state_changed();
	script_menu->show();
}

void EditorDebuggerNode::_break_state_changed() {
	const bool breaked = get_current_debugger()->is_breaked();
	const bool can_debug = get_current_debugger()->is_debuggable();
	if (breaked) { // Show debugger.
		EditorNode::get_singleton()->make_bottom_panel_item_visible(this);
	}

	// Update script menu.
	if (!script_menu) {
		return;
	}
	PopupMenu *p = script_menu->get_popup();
	p->set_item_disabled(p->get_item_index(DEBUG_NEXT), !(breaked && can_debug));
	p->set_item_disabled(p->get_item_index(DEBUG_STEP), !(breaked && can_debug));
	p->set_item_disabled(p->get_item_index(DEBUG_BREAK), breaked);
	p->set_item_disabled(p->get_item_index(DEBUG_CONTINUE), !breaked);
}

void EditorDebuggerNode::_menu_option(int p_id) {
	switch (p_id) {
		case DEBUG_NEXT: {
			debug_next();
		} break;
		case DEBUG_STEP: {
			debug_step();
		} break;
		case DEBUG_BREAK: {
			debug_break();
		} break;
		case DEBUG_CONTINUE: {
			debug_continue();
		} break;
		case DEBUG_KEEP_DEBUGGER_OPEN: {
			bool ischecked = script_menu->get_popup()->is_item_checked(script_menu->get_popup()->get_item_index(DEBUG_KEEP_DEBUGGER_OPEN));
			hide_on_stop = ischecked;
			script_menu->get_popup()->set_item_checked(script_menu->get_popup()->get_item_index(DEBUG_KEEP_DEBUGGER_OPEN), !ischecked);
			EditorSettings::get_singleton()->set_project_metadata("debug_options", "keep_debugger_open", !ischecked);
		} break;
		case DEBUG_WITH_EXTERNAL_EDITOR: {
			bool ischecked = script_menu->get_popup()->is_item_checked(script_menu->get_popup()->get_item_index(DEBUG_WITH_EXTERNAL_EDITOR));
			debug_with_external_editor = !ischecked;
			script_menu->get_popup()->set_item_checked(script_menu->get_popup()->get_item_index(DEBUG_WITH_EXTERNAL_EDITOR), !ischecked);
			EditorSettings::get_singleton()->set_project_metadata("debug_options", "debug_with_external_editor", !ischecked);
		} break;
	}
}

void EditorDebuggerNode::_update_debug_options() {
	bool keep_debugger_open = EditorSettings::get_singleton()->get_project_metadata("debug_options", "keep_debugger_open", false);
	bool debug_with_external_editor = EditorSettings::get_singleton()->get_project_metadata("debug_options", "debug_with_external_editor", false);

	if (keep_debugger_open) {
		_menu_option(DEBUG_KEEP_DEBUGGER_OPEN);
	}
	if (debug_with_external_editor) {
		_menu_option(DEBUG_WITH_EXTERNAL_EDITOR);
	}
}

void EditorDebuggerNode::_paused() {
	const bool paused = EditorNode::get_singleton()->get_pause_button()->is_pressed();
	_for_all(tabs, [&](ScriptEditorDebugger *dbg) {
		if (paused && !dbg->is_breaked()) {
			dbg->debug_break();
		} else if (!paused && dbg->is_breaked()) {
			dbg->debug_continue();
		}
	});
}

void EditorDebuggerNode::_breaked(bool p_breaked, bool p_can_debug, String p_message, bool p_has_stackdump, int p_debugger) {
	if (get_current_debugger() != get_debugger(p_debugger)) {
		if (!p_breaked) {
			return;
		}
		tabs->set_current_tab(p_debugger);
	}
	_break_state_changed();
	EditorNode::get_singleton()->get_pause_button()->set_pressed(p_breaked);
	emit_signal(SNAME("breaked"), p_breaked, p_can_debug);
}

bool EditorDebuggerNode::is_skip_breakpoints() const {
	return get_default_debugger()->is_skip_breakpoints();
}

void EditorDebuggerNode::set_breakpoint(const String &p_path, int p_line, bool p_enabled) {
	breakpoints[Breakpoint(p_path, p_line)] = p_enabled;
	_for_all(tabs, [&](ScriptEditorDebugger *dbg) {
		dbg->set_breakpoint(p_path, p_line, p_enabled);
	});

	emit_signal("breakpoint_toggled", p_path, p_line, p_enabled);
}

void EditorDebuggerNode::set_breakpoints(const String &p_path, Array p_lines) {
	for (int i = 0; i < p_lines.size(); i++) {
		set_breakpoint(p_path, p_lines[i], true);
	}

	for (const KeyValue<Breakpoint, bool> &E : breakpoints) {
		Breakpoint b = E.key;
		if (b.source == p_path && !p_lines.has(b.line)) {
			set_breakpoint(p_path, b.line, false);
		}
	}
}

void EditorDebuggerNode::reload_scripts() {
	_for_all(tabs, [&](ScriptEditorDebugger *dbg) {
		dbg->reload_scripts();
	});
}

void EditorDebuggerNode::debug_next() {
	get_default_debugger()->debug_next();
}

void EditorDebuggerNode::debug_step() {
	get_default_debugger()->debug_step();
}

void EditorDebuggerNode::debug_break() {
	get_default_debugger()->debug_break();
}

void EditorDebuggerNode::debug_continue() {
	get_default_debugger()->debug_continue();
}

String EditorDebuggerNode::get_var_value(const String &p_var) const {
	return get_default_debugger()->get_var_value(p_var);
}

// LiveEdit/Inspector
void EditorDebuggerNode::request_remote_tree() {
	get_current_debugger()->request_remote_tree();
}

void EditorDebuggerNode::_remote_tree_updated(int p_debugger) {
	if (p_debugger != tabs->get_current_tab()) {
		return;
	}
	remote_scene_tree->clear();
	remote_scene_tree->update_scene_tree(get_current_debugger()->get_remote_tree(), p_debugger);
}

void EditorDebuggerNode::_remote_object_updated(ObjectID p_id, int p_debugger) {
	if (p_debugger != tabs->get_current_tab()) {
		return;
	}
	if (EditorDebuggerRemoteObject *obj = get_inspected_remote_object()) {
		if (obj->remote_object_id == p_id) {
			return; // Already being edited
		}
	}

	EditorNode::get_singleton()->push_item(get_current_debugger()->get_remote_object(p_id));
}

void EditorDebuggerNode::_remote_object_property_updated(ObjectID p_id, const String &p_property, int p_debugger) {
	if (p_debugger != tabs->get_current_tab()) {
		return;
	}
	if (EditorDebuggerRemoteObject *obj = get_inspected_remote_object()) {
		if (obj->remote_object_id != p_id) {
			return;
		}
		EditorNode::get_singleton()->get_inspector()->update_property(p_property);
	}
}

void EditorDebuggerNode::_remote_object_requested(ObjectID p_id, int p_debugger) {
	if (p_debugger != tabs->get_current_tab()) {
		return;
	}
	inspect_edited_object_timeout = 0.7; // Temporarily disable timeout to avoid multiple requests.
	get_current_debugger()->request_remote_object(p_id);
}

void EditorDebuggerNode::_save_node_requested(ObjectID p_id, const String &p_file, int p_debugger) {
	if (p_debugger != tabs->get_current_tab()) {
		return;
	}
	get_current_debugger()->save_node(p_id, p_file);
}

// Remote inspector/edit.
void EditorDebuggerNode::_method_changeds(void *p_ud, Object *p_base, const StringName &p_name, VARIANT_ARG_DECLARE) {
	if (!singleton) {
		return;
	}
	_for_all(singleton->tabs, [&](ScriptEditorDebugger *dbg) {
		dbg->_method_changed(p_base, p_name, VARIANT_ARG_PASS);
	});
}

void EditorDebuggerNode::_property_changeds(void *p_ud, Object *p_base, const StringName &p_property, const Variant &p_value) {
	if (!singleton) {
		return;
	}
	_for_all(singleton->tabs, [&](ScriptEditorDebugger *dbg) {
		dbg->_property_changed(p_base, p_property, p_value);
	});
}

// LiveDebug
void EditorDebuggerNode::set_live_debugging(bool p_enabled) {
	_for_all(tabs, [&](ScriptEditorDebugger *dbg) {
		dbg->set_live_debugging(p_enabled);
	});
}

void EditorDebuggerNode::update_live_edit_root() {
	_for_all(tabs, [&](ScriptEditorDebugger *dbg) {
		dbg->update_live_edit_root();
	});
}

void EditorDebuggerNode::live_debug_create_node(const NodePath &p_parent, const String &p_type, const String &p_name) {
	_for_all(tabs, [&](ScriptEditorDebugger *dbg) {
		dbg->live_debug_create_node(p_parent, p_type, p_name);
	});
}

void EditorDebuggerNode::live_debug_instance_node(const NodePath &p_parent, const String &p_path, const String &p_name) {
	_for_all(tabs, [&](ScriptEditorDebugger *dbg) {
		dbg->live_debug_instance_node(p_parent, p_path, p_name);
	});
}

void EditorDebuggerNode::live_debug_remove_node(const NodePath &p_at) {
	_for_all(tabs, [&](ScriptEditorDebugger *dbg) {
		dbg->live_debug_remove_node(p_at);
	});
}

void EditorDebuggerNode::live_debug_remove_and_keep_node(const NodePath &p_at, ObjectID p_keep_id) {
	_for_all(tabs, [&](ScriptEditorDebugger *dbg) {
		dbg->live_debug_remove_and_keep_node(p_at, p_keep_id);
	});
}

void EditorDebuggerNode::live_debug_restore_node(ObjectID p_id, const NodePath &p_at, int p_at_pos) {
	_for_all(tabs, [&](ScriptEditorDebugger *dbg) {
		dbg->live_debug_restore_node(p_id, p_at, p_at_pos);
	});
}

void EditorDebuggerNode::live_debug_duplicate_node(const NodePath &p_at, const String &p_new_name) {
	_for_all(tabs, [&](ScriptEditorDebugger *dbg) {
		dbg->live_debug_duplicate_node(p_at, p_new_name);
	});
}

void EditorDebuggerNode::live_debug_reparent_node(const NodePath &p_at, const NodePath &p_new_place, const String &p_new_name, int p_at_pos) {
	_for_all(tabs, [&](ScriptEditorDebugger *dbg) {
		dbg->live_debug_reparent_node(p_at, p_new_place, p_new_name, p_at_pos);
	});
}

void EditorDebuggerNode::set_camera_override(CameraOverride p_override) {
	_for_all(tabs, [&](ScriptEditorDebugger *dbg) {
		dbg->set_camera_override(p_override);
	});
	camera_override = p_override;
}

EditorDebuggerNode::CameraOverride EditorDebuggerNode::get_camera_override() {
	return camera_override;
}

void EditorDebuggerNode::add_debugger_plugin(const Ref<Script> &p_script) {
	ERR_FAIL_COND_MSG(debugger_plugins.has(p_script), "Debugger plugin already exists.");
	ERR_FAIL_COND_MSG(p_script.is_null(), "Debugger plugin script is null");
	ERR_FAIL_COND_MSG(String(p_script->get_instance_base_type()) == "", "Debugger plugin script has error.");
	ERR_FAIL_COND_MSG(String(p_script->get_instance_base_type()) != "EditorDebuggerPlugin", "Base type of debugger plugin is not 'EditorDebuggerPlugin'.");
	ERR_FAIL_COND_MSG(!p_script->is_tool(), "Debugger plugin script is not in tool mode.");
	debugger_plugins.insert(p_script);
	for (int i = 0; get_debugger(i); i++) {
		get_debugger(i)->add_debugger_plugin(p_script);
	}
}

void EditorDebuggerNode::remove_debugger_plugin(const Ref<Script> &p_script) {
	ERR_FAIL_COND_MSG(!debugger_plugins.has(p_script), "Debugger plugin doesn't exists.");
	debugger_plugins.erase(p_script);
	for (int i = 0; get_debugger(i); i++) {
		get_debugger(i)->remove_debugger_plugin(p_script);
	}
}
