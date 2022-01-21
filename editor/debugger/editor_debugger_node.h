/*************************************************************************/
/*  editor_debugger_node.h                                               */
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

#ifndef EDITOR_DEBUGGER_NODE_H
#define EDITOR_DEBUGGER_NODE_H

#include "editor/debugger/editor_debugger_server.h"
#include "scene/gui/margin_container.h"

class Button;
class DebugAdapterParser;
class EditorDebuggerTree;
class EditorDebuggerRemoteObject;
class MenuButton;
class ScriptEditorDebugger;
class TabContainer;

class EditorDebuggerNode : public MarginContainer {
	GDCLASS(EditorDebuggerNode, MarginContainer);

public:
	enum CameraOverride {
		OVERRIDE_NONE,
		OVERRIDE_2D,
		OVERRIDE_3D_1, // 3D Viewport 1
		OVERRIDE_3D_2, // 3D Viewport 2
		OVERRIDE_3D_3, // 3D Viewport 3
		OVERRIDE_3D_4 // 3D Viewport 4
	};

private:
	enum Options {
		DEBUG_NEXT,
		DEBUG_STEP,
		DEBUG_BREAK,
		DEBUG_CONTINUE,
		DEBUG_KEEP_DEBUGGER_OPEN,
		DEBUG_WITH_EXTERNAL_EDITOR,
	};

	class Breakpoint {
	public:
		String source;
		int line = 0;

		bool operator<(const Breakpoint &p_b) const {
			if (line == p_b.line) {
				return source < p_b.source;
			}
			return line < p_b.line;
		}

		Breakpoint() {}

		Breakpoint(const String &p_source, int p_line) {
			line = p_line;
			source = p_source;
		}
	};

	Ref<EditorDebuggerServer> server;
	TabContainer *tabs = nullptr;
	Button *debugger_button = nullptr;
	MenuButton *script_menu = nullptr;

	Ref<Script> stack_script; // Why?!?

	int last_error_count = 0;
	int last_warning_count = 0;

	float inspect_edited_object_timeout = 0;
	EditorDebuggerTree *remote_scene_tree = nullptr;
	float remote_scene_tree_timeout = 0.0;
	bool auto_switch_remote_scene_tree = false;
	bool debug_with_external_editor = false;
	bool hide_on_stop = true;
	CameraOverride camera_override = OVERRIDE_NONE;
	Map<Breakpoint, bool> breakpoints;

	Set<Ref<Script>> debugger_plugins;

	ScriptEditorDebugger *_add_debugger();
	EditorDebuggerRemoteObject *get_inspected_remote_object();

	friend class DebuggerEditorPlugin;
	friend class DebugAdapterParser;
	static EditorDebuggerNode *singleton;
	EditorDebuggerNode();

protected:
	void _debugger_stopped(int p_id);
	void _debugger_wants_stop(int p_id);
	void _debugger_changed(int p_tab);
	void _remote_tree_updated(int p_debugger);
	void _remote_object_updated(ObjectID p_id, int p_debugger);
	void _remote_object_property_updated(ObjectID p_id, const String &p_property, int p_debugger);
	void _remote_object_requested(ObjectID p_id, int p_debugger);
	void _save_node_requested(ObjectID p_id, const String &p_file, int p_debugger);

	void _clear_execution(REF p_script) {
		emit_signal(SNAME("clear_execution"), p_script);
	}

	void _text_editor_stack_goto(const ScriptEditorDebugger *p_debugger);
	void _stack_frame_selected(int p_debugger);
	void _error_selected(const String &p_file, int p_line, int p_debugger);
	void _breaked(bool p_breaked, bool p_can_debug, String p_message, bool p_has_stackdump, int p_debugger);
	void _paused();
	void _break_state_changed();
	void _menu_option(int p_id);
	void _update_debug_options();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	static EditorDebuggerNode *get_singleton() { return singleton; }

	ScriptEditorDebugger *get_current_debugger() const;
	ScriptEditorDebugger *get_default_debugger() const;
	ScriptEditorDebugger *get_debugger(int p_debugger) const;

	void debug_next();
	void debug_step();
	void debug_break();
	void debug_continue();

	void set_script_debug_button(MenuButton *p_button);

	void set_tool_button(Button *p_button) {
		debugger_button = p_button;
	}

	String get_var_value(const String &p_var) const;
	Ref<Script> get_dump_stack_script() const { return stack_script; } // Why do we need this?

	bool get_debug_with_external_editor() { return debug_with_external_editor; }

	bool is_skip_breakpoints() const;
	void set_breakpoint(const String &p_path, int p_line, bool p_enabled);
	void set_breakpoints(const String &p_path, Array p_lines);
	void reload_scripts();

	// Remote inspector/edit.
	void request_remote_tree();
	static void _method_changeds(void *p_ud, Object *p_base, const StringName &p_name, VARIANT_ARG_DECLARE);
	static void _property_changeds(void *p_ud, Object *p_base, const StringName &p_property, const Variant &p_value);

	// LiveDebug
	void set_live_debugging(bool p_enabled);
	void update_live_edit_root();
	void live_debug_create_node(const NodePath &p_parent, const String &p_type, const String &p_name);
	void live_debug_instance_node(const NodePath &p_parent, const String &p_path, const String &p_name);
	void live_debug_remove_node(const NodePath &p_at);
	void live_debug_remove_and_keep_node(const NodePath &p_at, ObjectID p_keep_id);
	void live_debug_restore_node(ObjectID p_id, const NodePath &p_at, int p_at_pos);
	void live_debug_duplicate_node(const NodePath &p_at, const String &p_new_name);
	void live_debug_reparent_node(const NodePath &p_at, const NodePath &p_new_place, const String &p_new_name, int p_at_pos);

	void set_camera_override(CameraOverride p_override);
	CameraOverride get_camera_override();

	String get_server_uri() const;

	Error start(const String &p_uri = "tcp://");
	void stop();

	void add_debugger_plugin(const Ref<Script> &p_script);
	void remove_debugger_plugin(const Ref<Script> &p_script);
};
#endif // EDITOR_DEBUGGER_NODE_H
