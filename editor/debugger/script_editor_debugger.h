/*************************************************************************/
/*  script_editor_debugger.h                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef SCRIPT_EDITOR_DEBUGGER_H
#define SCRIPT_EDITOR_DEBUGGER_H

#include "core/os/os.h"
#include "editor/debugger/editor_debugger_inspector.h"
#include "editor/debugger/editor_debugger_node.h"
#include "editor/debugger/editor_debugger_server.h"
#include "editor/editor_file_dialog.h"
#include "scene/gui/button.h"
#include "scene/gui/margin_container.h"

class Tree;
class EditorNode;
class LineEdit;
class TabContainer;
class RichTextLabel;
class TextureButton;
class AcceptDialog;
class TreeItem;
class HSplitContainer;
class ItemList;
class EditorProfiler;
class EditorVisualProfiler;
class EditorNetworkProfiler;
class EditorPerformanceProfiler;
class SceneDebuggerTree;
class EditorDebuggerPlugin;

class ScriptEditorDebugger : public MarginContainer {
	GDCLASS(ScriptEditorDebugger, MarginContainer);

	friend class EditorDebuggerNode;

private:
	enum MessageType {
		MESSAGE_ERROR,
		MESSAGE_WARNING,
		MESSAGE_SUCCESS,
	};

	enum ProfilerType {
		PROFILER_NETWORK,
		PROFILER_VISUAL,
		PROFILER_SCRIPTS_SERVERS
	};

	AcceptDialog *msgdialog;

	LineEdit *clicked_ctrl;
	LineEdit *clicked_ctrl_type;
	LineEdit *live_edit_root;
	Button *le_set;
	Button *le_clear;
	Button *export_csv;

	VBoxContainer *errors_tab;
	Tree *error_tree;
	Button *clearbutton;
	PopupMenu *item_menu;

	EditorFileDialog *file_dialog;
	enum FileDialogPurpose {
		SAVE_MONITORS_CSV,
		SAVE_VRAM_CSV,
	};
	FileDialogPurpose file_dialog_purpose;

	int error_count;
	int warning_count;

	bool skip_breakpoints_value = false;
	Ref<Script> stack_script;

	TabContainer *tabs;

	Label *reason;

	Button *skip_breakpoints;
	Button *copy;
	Button *step;
	Button *next;
	Button *dobreak;
	Button *docontinue;
	// Reference to "Remote" tab in scene tree. Needed by _live_edit_set and buttons state.
	// Each debugger should have it's tree in the future I guess.
	const Tree *editor_remote_tree = nullptr;

	Map<int, String> profiler_signature;

	Tree *vmem_tree;
	Button *vmem_refresh;
	Button *vmem_export;
	LineEdit *vmem_total;

	Tree *stack_dump;
	EditorDebuggerInspector *inspector;
	SceneDebuggerTree *scene_tree;

	Ref<RemoteDebuggerPeer> peer;

	HashMap<NodePath, int> node_path_cache;
	int last_path_id;
	Map<String, int> res_path_cache;

	EditorProfiler *profiler;
	EditorVisualProfiler *visual_profiler;
	EditorNetworkProfiler *network_profiler;
	EditorPerformanceProfiler *performance_profiler;

	EditorNode *editor;

	OS::ProcessID remote_pid = 0;
	bool breaked = false;
	bool can_debug = false;

	bool live_debug;

	EditorDebuggerNode::CameraOverride camera_override;

	Map<Ref<Script>, EditorDebuggerPlugin *> debugger_plugins;

	Map<StringName, Callable> captures;

	void _stack_dump_frame_selected();

	void _file_selected(const String &p_file);
	void _parse_message(const String &p_msg, const Array &p_data);
	void _set_reason_text(const String &p_reason, MessageType p_type);
	void _update_buttons_state();
	void _remote_object_selected(ObjectID p_object);
	void _remote_object_edited(ObjectID, const String &p_prop, const Variant &p_value);
	void _remote_object_property_updated(ObjectID p_id, const String &p_property);

	void _video_mem_request();
	void _video_mem_export();

	int _get_node_path_cache(const NodePath &p_path);

	int _get_res_path_cache(const String &p_path);

	void _live_edit_set();
	void _live_edit_clear();

	void _method_changed(Object *p_base, const StringName &p_name, VARIANT_ARG_DECLARE);
	void _property_changed(Object *p_base, const StringName &p_property, const Variant &p_value);

	void _error_activated();
	void _error_selected();

	void _expand_errors_list();
	void _collapse_errors_list();

	void _profiler_activate(bool p_enable, int p_profiler);
	void _profiler_seeked();

	void _clear_errors_list();

	void _error_tree_item_rmb_selected(const Vector2 &p_pos);
	void _item_menu_id_pressed(int p_option);
	void _tab_changed(int p_tab);

	void _put_msg(String p_message, Array p_data);
	void _export_csv();

	void _clear_execution();
	void _stop_and_notify();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void request_remote_object(ObjectID p_obj_id);
	void update_remote_object(ObjectID p_obj_id, const String &p_prop, const Variant &p_value);
	Object *get_remote_object(ObjectID p_id);

	// Needed by _live_edit_set, buttons state.
	void set_editor_remote_tree(const Tree *p_tree) { editor_remote_tree = p_tree; }

	void request_remote_tree();
	const SceneDebuggerTree *get_remote_tree();

	void start(Ref<RemoteDebuggerPeer> p_peer);
	void stop();

	void debug_skip_breakpoints();
	void debug_copy();

	void debug_next();
	void debug_step();
	void debug_break();
	void debug_continue();
	bool is_breaked() const { return breaked; }
	bool is_debuggable() const { return can_debug; }
	bool is_session_active() { return peer.is_valid() && peer->is_peer_connected(); };
	int get_remote_pid() const { return remote_pid; }

	int get_error_count() const { return error_count; }
	int get_warning_count() const { return warning_count; }
	String get_stack_script_file() const;
	int get_stack_script_line() const;
	int get_stack_script_frame() const;

	void update_tabs();
	void clear_style();
	String get_var_value(const String &p_var) const;

	void save_node(ObjectID p_id, const String &p_file);
	void set_live_debugging(bool p_enable);

	void live_debug_create_node(const NodePath &p_parent, const String &p_type, const String &p_name);
	void live_debug_instance_node(const NodePath &p_parent, const String &p_path, const String &p_name);
	void live_debug_remove_node(const NodePath &p_at);
	void live_debug_remove_and_keep_node(const NodePath &p_at, ObjectID p_keep_id);
	void live_debug_restore_node(ObjectID p_id, const NodePath &p_at, int p_at_pos);
	void live_debug_duplicate_node(const NodePath &p_at, const String &p_new_name);
	void live_debug_reparent_node(const NodePath &p_at, const NodePath &p_new_place, const String &p_new_name, int p_at_pos);

	EditorDebuggerNode::CameraOverride get_camera_override() const;
	void set_camera_override(EditorDebuggerNode::CameraOverride p_override);

	void set_breakpoint(const String &p_path, int p_line, bool p_enabled);

	void update_live_edit_root();

	void reload_scripts();

	bool is_skip_breakpoints();

	virtual Size2 get_minimum_size() const override;

	void add_debugger_plugin(const Ref<Script> &p_script);
	void remove_debugger_plugin(const Ref<Script> &p_script);

	void send_message(const String &p_message, const Array &p_args);

	void register_message_capture(const StringName &p_name, const Callable &p_callable);
	void unregister_message_capture(const StringName &p_name);
	bool has_capture(const StringName &p_name);

	ScriptEditorDebugger(EditorNode *p_editor = nullptr);
	~ScriptEditorDebugger();
};

#endif // SCRIPT_EDITOR_DEBUGGER_H
