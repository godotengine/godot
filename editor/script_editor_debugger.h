/*************************************************************************/
/*  script_editor_debugger.h                                             */
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
#ifndef SCRIPT_EDITOR_DEBUGGER_H
#define SCRIPT_EDITOR_DEBUGGER_H

#include "core/io/packet_peer.h"
#include "core/io/tcp_server.h"
#include "property_editor.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"

class Tree;
class PropertyEditor;
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

class ScriptEditorDebuggerInspectedObject;

class ScriptEditorDebugger : public Control {

	GDCLASS(ScriptEditorDebugger, Control);

	AcceptDialog *msgdialog;

	Button *debugger_button;

	LineEdit *clicked_ctrl;
	LineEdit *clicked_ctrl_type;
	LineEdit *live_edit_root;
	Button *le_set;
	Button *le_clear;

	Tree *inspect_scene_tree;
	HSplitContainer *inspect_info;
	PropertyEditor *inspect_properties;
	float inspect_scene_tree_timeout;
	float inspect_edited_object_timeout;
	ObjectID inspected_object_id;
	ScriptEditorDebuggerInspectedObject *inspected_object;
	bool updating_scene_tree;
	Set<ObjectID> unfold_cache;

	HSplitContainer *error_split;
	ItemList *error_list;
	ItemList *error_stack;

	int error_count;
	int last_error_count;

	bool hide_on_stop;

	TabContainer *tabs;

	LineEdit *reason;
	ScriptEditorDebuggerVariables *variables;

	Button *step;
	Button *next;
	Button *back;
	Button *forward;
	Button *dobreak;
	Button *docontinue;

	List<Vector<float> > perf_history;
	Vector<float> perf_max;
	Vector<TreeItem *> perf_items;

	Map<int, String> profiler_signature;

	Tree *perf_monitors;
	Control *perf_draw;

	Tree *vmem_tree;
	Button *vmem_refresh;
	LineEdit *vmem_total;

	Tree *stack_dump;
	PropertyEditor *inspector;

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

	EditorNode *editor;

	bool breaked;

	bool live_debug;

	void _performance_draw();
	void _performance_select(Object *, int, bool);
	void _stack_dump_frame_selected();
	void _output_clear();

	void _scene_tree_folded(Object *obj);
	void _scene_tree_selected();
	void _scene_tree_request();
	void _parse_message(const String &p_msg, const Array &p_data);
	void _scene_tree_property_select_object(ObjectID p_object);
	void _scene_tree_property_value_edited(const String &p_prop, const Variant &p_value);

	void _video_mem_request();

	int _get_node_path_cache(const NodePath &p_path);

	int _get_res_path_cache(const String &p_path);

	void _live_edit_set();
	void _live_edit_clear();

	void _method_changed(Object *p_base, const StringName &p_name, VARIANT_ARG_DECLARE);
	void _property_changed(Object *p_base, const StringName &p_property, const Variant &p_value);

	static void _method_changeds(void *p_ud, Object *p_base, const StringName &p_name, VARIANT_ARG_DECLARE);
	static void _property_changeds(void *p_ud, Object *p_base, const StringName &p_property, const Variant &p_value);

	void _error_selected(int p_idx);
	void _error_stack_selected(int p_idx);

	void _profiler_activate(bool p_enable);
	void _profiler_seeked();

	void _paused();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void start();
	void pause();
	void unpause();
	void stop();

	void debug_next();
	void debug_step();
	void debug_break();
	void debug_continue();

	String get_var_value(const String &p_var) const;

	void set_live_debugging(bool p_enable);

	void live_debug_create_node(const NodePath &p_parent, const String &p_type, const String &p_name);
	void live_debug_instance_node(const NodePath &p_parent, const String &p_path, const String &p_name);
	void live_debug_remove_node(const NodePath &p_at);
	void live_debug_remove_and_keep_node(const NodePath &p_at, ObjectID p_keep_id);
	void live_debug_restore_node(ObjectID p_id, const NodePath &p_at, int p_at_pos);
	void live_debug_duplicate_node(const NodePath &p_at, const String &p_new_name);
	void live_debug_reparent_node(const NodePath &p_at, const NodePath &p_new_place, const String &p_new_name, int p_at_pos);

	void set_breakpoint(const String &p_path, int p_line, bool p_enabled);

	void update_live_edit_root();

	void set_hide_on_stop(bool p_hide);

	void set_tool_button(Button *p_tb) { debugger_button = p_tb; }

	void reload_scripts();

	virtual Size2 get_minimum_size() const;
	ScriptEditorDebugger(EditorNode *p_editor = NULL);
	~ScriptEditorDebugger();
};

#endif // SCRIPT_EDITOR_DEBUGGER_H
