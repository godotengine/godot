/*************************************************************************/
/*  script_editor_debugger.h                                             */
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

#ifndef SCRIPT_EDITOR_DEBUGGER_H
#define SCRIPT_EDITOR_DEBUGGER_H

#include "core/io/packet_peer.h"
#include "core/io/tcp_server.h"
#include "editor/editor_inspector.h"
#include "editor/property_editor.h"
#include "scene/3d/camera.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"

class Tree;
class EditorNode;
class ScriptEditorDebuggerVariables;
class LineEdit;
class TabContainer;
class RichTextLabel;
class TextureButton;
class AcceptDialog;
class TreeItem;
class HSplitContainer;
class ItemList;
class EditorProfiler;
class EditorNetworkProfiler;

class ScriptEditorDebuggerInspectedObject;

class ScriptEditorDebugger : public MarginContainer {
	GDCLASS(ScriptEditorDebugger, MarginContainer);

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
	enum MessageType {
		MESSAGE_ERROR,
		MESSAGE_WARNING,
		MESSAGE_SUCCESS,
	};

	enum ItemMenu {
		ITEM_MENU_COPY_ERROR,
		ITEM_MENU_SAVE_REMOTE_NODE,
		ITEM_MENU_COPY_NODE_PATH,
		ITEM_MENU_OPEN_SOURCE,
	};

	AcceptDialog *msgdialog;

	Button *debugger_button;

	LineEdit *clicked_ctrl;
	LineEdit *clicked_ctrl_type;
	LineEdit *live_edit_root;
	Button *le_set;
	Button *le_clear;
	Button *export_csv;

	bool updating_scene_tree;
	float inspect_scene_tree_timeout;
	float inspect_edited_object_timeout;
	bool auto_switch_remote_scene_tree;
	ObjectID inspected_object_id;
	String last_filter;
	ScriptEditorDebuggerVariables *variables;
	Map<ObjectID, ScriptEditorDebuggerInspectedObject *> remote_objects;
	Set<RES> remote_dependencies;
	Set<ObjectID> unfold_cache;

	VBoxContainer *errors_tab;
	Tree *error_tree;
	Tree *inspect_scene_tree;
	Button *clearbutton;
	PopupMenu *item_menu;

	EditorFileDialog *file_dialog;
	enum FileDialogMode {
		SAVE_MONITORS_CSV,
		SAVE_VRAM_CSV,
		SAVE_NODE,
	};
	FileDialogMode file_dialog_mode;

	int error_count;
	int warning_count;
	int last_error_count;
	int last_warning_count;

	bool hide_on_stop;
	int remote_port;
	bool enable_external_editor;

	bool skip_breakpoints_value = false;
	Ref<Script> stack_script;

	TabContainer *tabs;

	Label *reason;

	Button *skip_breakpoints;
	Button *copy;
	Button *step;
	Button *next;
	Button *back;
	Button *forward;
	Button *dobreak;
	Button *docontinue;

	List<Vector<float>> perf_history;
	Vector<float> perf_max;
	Vector<TreeItem *> perf_items;

	Map<int, String> profiler_signature;

	Tree *perf_monitors;
	Control *perf_draw;
	Label *info_message;

	Tree *vmem_tree;
	Button *vmem_refresh;
	Button *vmem_export;
	LineEdit *vmem_total;

	Tree *stack_dump;
	LineEdit *search;
	EditorInspector *inspector;

	Ref<TCP_Server> server;
	Ref<StreamPeerTCP> connection;
	Ref<PacketPeerStream> ppeer;

	String message_type;
	Array message;
	int pending_in_queue;

	HashMap<NodePath, int> node_path_cache;
	int last_path_id;
	Map<String, int> res_path_cache;

	EditorProfiler *profiler;
	EditorNetworkProfiler *network_profiler;

	EditorNode *editor;

	bool breaked;

	bool live_debug;

	CameraOverride camera_override;

	void _performance_draw();
	void _performance_select();
	void _stack_dump_frame_selected();
	void _output_clear();

	void _scene_tree_folded(Object *obj);
	void _scene_tree_selected();
	void _scene_tree_rmb_selected(const Vector2 &p_position);
	void _file_selected(const String &p_file);
	void _scene_tree_request();
	void _parse_message(const String &p_msg, const Array &p_data);
	void _set_reason_text(const String &p_reason, MessageType p_type);
	void _scene_tree_property_select_object(ObjectID p_object);
	void _scene_tree_property_value_edited(const String &p_prop, const Variant &p_value);
	int _update_scene_tree(TreeItem *parent, const Array &nodes, int current_index);

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

	void _profiler_activate(bool p_enable);
	void _profiler_seeked();

	void _network_profiler_activate(bool p_enable);

	void _paused();

	void _set_remote_object(ObjectID p_id, ScriptEditorDebuggerInspectedObject *p_obj);
	void _clear_remote_objects();
	void _clear_errors_list();

	void _error_tree_item_rmb_selected(const Vector2 &p_pos);
	void _item_menu_id_pressed(int p_option);
	void _tab_changed(int p_tab);

	void _export_csv();

	void _clear_execution();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void start(int p_port = -1, const IP_Address &p_bind_address = IP_Address("*"));
	void pause();
	void unpause();
	void stop();

	void debug_skip_breakpoints();
	void debug_copy();

	void debug_next();
	void debug_step();
	void debug_break();
	void debug_continue();

	String get_var_value(const String &p_var) const;

	void set_live_debugging(bool p_enable);

	static void _method_changeds(void *p_ud, Object *p_base, const StringName &p_name, VARIANT_ARG_DECLARE);
	static void _property_changeds(void *p_ud, Object *p_base, const StringName &p_property, const Variant &p_value);

	void live_debug_create_node(const NodePath &p_parent, const String &p_type, const String &p_name);
	void live_debug_instance_node(const NodePath &p_parent, const String &p_path, const String &p_name);
	void live_debug_remove_node(const NodePath &p_at);
	void live_debug_remove_and_keep_node(const NodePath &p_at, ObjectID p_keep_id);
	void live_debug_restore_node(ObjectID p_id, const NodePath &p_at, int p_at_pos);
	void live_debug_duplicate_node(const NodePath &p_at, const String &p_new_name);
	void live_debug_reparent_node(const NodePath &p_at, const NodePath &p_new_place, const String &p_new_name, int p_at_pos);

	CameraOverride get_camera_override() const;
	void set_camera_override(CameraOverride p_override);

	void set_breakpoint(const String &p_path, int p_line, bool p_enabled);

	void update_live_edit_root();

	void set_hide_on_stop(bool p_hide);

	bool get_debug_with_external_editor() const;
	String get_connection_string() const;
	void set_debug_with_external_editor(bool p_enabled);

	Ref<Script> get_dump_stack_script() const;

	void set_tool_button(Button *p_tb) { debugger_button = p_tb; }

	void reload_scripts();

	bool is_skip_breakpoints();

	virtual Size2 get_minimum_size() const;
	ScriptEditorDebugger(EditorNode *p_editor = nullptr);
	~ScriptEditorDebugger();
};

#endif // SCRIPT_EDITOR_DEBUGGER_H
