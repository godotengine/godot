/*************************************************************************/
/*  node.h                                                               */
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

#ifndef NODE_H
#define NODE_H

#include "core/class_db.h"
#include "core/map.h"
#include "core/node_path.h"
#include "core/object.h"
#include "core/project_settings.h"
#include "core/script_language.h"
#include "scene/main/scene_tree.h"

class Viewport;
class SceneState;
class Node : public Object {
	GDCLASS(Node, Object);
	OBJ_CATEGORY("Nodes");

public:
	enum PauseMode {

		PAUSE_MODE_INHERIT,
		PAUSE_MODE_STOP,
		PAUSE_MODE_PROCESS
	};

	enum DuplicateFlags {

		DUPLICATE_SIGNALS = 1,
		DUPLICATE_GROUPS = 2,
		DUPLICATE_SCRIPTS = 4,
		DUPLICATE_USE_INSTANCING = 8,
#ifdef TOOLS_ENABLED
		DUPLICATE_FROM_EDITOR = 16,
#endif
	};

	enum NameCasing {
		NAME_CASING_PASCAL_CASE,
		NAME_CASING_CAMEL_CASE,
		NAME_CASING_SNAKE_CASE
	};

	struct Comparator {
		bool operator()(const Node *p_a, const Node *p_b) const { return p_b->is_greater_than(p_a); }
	};

	struct ComparatorWithPriority {
		bool operator()(const Node *p_a, const Node *p_b) const { return p_b->data.process_priority == p_a->data.process_priority ? p_b->is_greater_than(p_a) : p_b->data.process_priority > p_a->data.process_priority; }
	};

	static int orphan_node_count;

private:
	struct GroupData {
		bool persistent;
		SceneTree::Group *group;
		GroupData() { persistent = false; }
	};

	struct Data {
		String filename;
		Ref<SceneState> instance_state;
		Ref<SceneState> inherited_state;

		Node *parent;
		Node *owner;
		Vector<Node *> children; // list of children
		int pos;
		int depth;
		int blocked; // safeguard that throws an error when attempting to modify the tree in a harmful way while being traversed.
		StringName name;
		SceneTree *tree;
		bool inside_tree;
		bool ready_notified; //this is a small hack, so if a node is added during _ready() to the tree, it correctly gets the _ready() notification
		bool ready_first;
#ifdef TOOLS_ENABLED
		NodePath import_path; //path used when imported, used by scene editors to keep tracking
#endif

		Viewport *viewport;

		Map<StringName, GroupData> grouped;
		List<Node *>::Element *OW; // owned element
		List<Node *> owned;

		PauseMode pause_mode;
		Node *pause_owner;

		int network_master;
		Map<StringName, MultiplayerAPI::RPCMode> rpc_methods;
		Map<StringName, MultiplayerAPI::RPCMode> rpc_properties;

		// variables used to properly sort the node when processing, ignored otherwise
		//should move all the stuff below to bits
		bool physics_process;
		bool idle_process;
		int process_priority;

		bool physics_process_internal;
		bool idle_process_internal;

		bool input;
		bool unhandled_input;
		bool unhandled_key_input;

		bool parent_owned;
		bool in_constructor;
		bool use_placeholder;

		bool display_folded;
		bool editable_instance;

		mutable NodePath *path_cache;

	} data;

	Ref<MultiplayerAPI> multiplayer;

	void _print_tree_pretty(const String &prefix, const bool last);
	void _print_tree(const Node *p_node);

	Node *_get_child_by_name(const StringName &p_name) const;

	void _replace_connections_target(Node *p_new_target);

	void _validate_child_name(Node *p_child, bool p_force_human_readable = false);
	void _generate_serial_child_name(const Node *p_child, StringName &name) const;

	void _propagate_reverse_notification(int p_notification);
	void _propagate_deferred_notification(int p_notification, bool p_reverse);
	void _propagate_enter_tree();
	void _propagate_ready();
	void _propagate_exit_tree();
	void _propagate_after_exit_tree();
	void _propagate_validate_owner();
	void _print_stray_nodes();
	void _propagate_pause_owner(Node *p_owner);
	Array _get_node_and_resource(const NodePath &p_path);

	void _duplicate_signals(const Node *p_original, Node *p_copy) const;
	void _duplicate_and_reown(Node *p_new_parent, const Map<Node *, Node *> &p_reown_map) const;
	Node *_duplicate(int p_flags, Map<const Node *, Node *> *r_duplimap = nullptr) const;

	Array _get_children() const;
	Array _get_groups() const;

	Variant _rpc_bind(const Variant **p_args, int p_argcount, Variant::CallError &r_error);
	Variant _rpc_unreliable_bind(const Variant **p_args, int p_argcount, Variant::CallError &r_error);
	Variant _rpc_id_bind(const Variant **p_args, int p_argcount, Variant::CallError &r_error);
	Variant _rpc_unreliable_id_bind(const Variant **p_args, int p_argcount, Variant::CallError &r_error);

	friend class SceneTree;

	void _set_tree(SceneTree *p_tree);

protected:
	void _block() { data.blocked++; }
	void _unblock() { data.blocked--; }

	void _notification(int p_notification);

	virtual void add_child_notify(Node *p_child);
	virtual void remove_child_notify(Node *p_child);
	virtual void move_child_notify(Node *p_child);

	void _propagate_replace_owner(Node *p_owner, Node *p_by_owner);

	static void _bind_methods();
	static String _get_name_num_separator();

	friend class SceneState;

	void _add_child_nocheck(Node *p_child, const StringName &p_name);
	void _set_owner_nocheck(Node *p_owner);
	void _set_name_nocheck(const StringName &p_name);

public:
	enum {

		// you can make your own, but don't use the same numbers as other notifications in other nodes
		NOTIFICATION_ENTER_TREE = 10,
		NOTIFICATION_EXIT_TREE = 11,
		NOTIFICATION_MOVED_IN_PARENT = 12,
		NOTIFICATION_READY = 13,
		NOTIFICATION_PAUSED = 14,
		NOTIFICATION_UNPAUSED = 15,
		NOTIFICATION_PHYSICS_PROCESS = 16,
		NOTIFICATION_PROCESS = 17,
		NOTIFICATION_PARENTED = 18,
		NOTIFICATION_UNPARENTED = 19,
		NOTIFICATION_INSTANCED = 20,
		NOTIFICATION_DRAG_BEGIN = 21,
		NOTIFICATION_DRAG_END = 22,
		NOTIFICATION_PATH_CHANGED = 23,
		//NOTIFICATION_TRANSLATION_CHANGED = 24, moved below
		NOTIFICATION_INTERNAL_PROCESS = 25,
		NOTIFICATION_INTERNAL_PHYSICS_PROCESS = 26,
		NOTIFICATION_POST_ENTER_TREE = 27,
		//keep these linked to node
		NOTIFICATION_WM_MOUSE_ENTER = MainLoop::NOTIFICATION_WM_MOUSE_ENTER,
		NOTIFICATION_WM_MOUSE_EXIT = MainLoop::NOTIFICATION_WM_MOUSE_EXIT,
		NOTIFICATION_WM_FOCUS_IN = MainLoop::NOTIFICATION_WM_FOCUS_IN,
		NOTIFICATION_WM_FOCUS_OUT = MainLoop::NOTIFICATION_WM_FOCUS_OUT,
		NOTIFICATION_WM_QUIT_REQUEST = MainLoop::NOTIFICATION_WM_QUIT_REQUEST,
		NOTIFICATION_WM_GO_BACK_REQUEST = MainLoop::NOTIFICATION_WM_GO_BACK_REQUEST,
		NOTIFICATION_WM_UNFOCUS_REQUEST = MainLoop::NOTIFICATION_WM_UNFOCUS_REQUEST,
		NOTIFICATION_OS_MEMORY_WARNING = MainLoop::NOTIFICATION_OS_MEMORY_WARNING,
		NOTIFICATION_TRANSLATION_CHANGED = MainLoop::NOTIFICATION_TRANSLATION_CHANGED,
		NOTIFICATION_WM_ABOUT = MainLoop::NOTIFICATION_WM_ABOUT,
		NOTIFICATION_CRASH = MainLoop::NOTIFICATION_CRASH,
		NOTIFICATION_OS_IME_UPDATE = MainLoop::NOTIFICATION_OS_IME_UPDATE,
		NOTIFICATION_APP_RESUMED = MainLoop::NOTIFICATION_APP_RESUMED,
		NOTIFICATION_APP_PAUSED = MainLoop::NOTIFICATION_APP_PAUSED

	};

	/* NODE/TREE */

	StringName get_name() const;
	void set_name(const String &p_name);

	void add_child(Node *p_child, bool p_legible_unique_name = false);
	void add_child_below_node(Node *p_node, Node *p_child, bool p_legible_unique_name = false);
	void remove_child(Node *p_child);

	int get_child_count() const;
	Node *get_child(int p_index) const;
	bool has_node(const NodePath &p_path) const;
	Node *get_node(const NodePath &p_path) const;
	Node *get_node_or_null(const NodePath &p_path) const;
	Node *find_node(const String &p_mask, bool p_recursive = true, bool p_owned = true) const;
	bool has_node_and_resource(const NodePath &p_path) const;
	Node *get_node_and_resource(const NodePath &p_path, RES &r_res, Vector<StringName> &r_leftover_subpath, bool p_last_is_property = true) const;

	Node *get_parent() const;
	Node *find_parent(const String &p_mask) const;

	_FORCE_INLINE_ SceneTree *get_tree() const {
		ERR_FAIL_COND_V(!data.tree, nullptr);
		return data.tree;
	}

	_FORCE_INLINE_ bool is_inside_tree() const { return data.inside_tree; }

	bool is_a_parent_of(const Node *p_node) const;
	bool is_greater_than(const Node *p_node) const;

	NodePath get_path() const;
	NodePath get_path_to(const Node *p_node) const;
	Node *find_common_parent_with(const Node *p_node) const;

	void add_to_group(const StringName &p_identifier, bool p_persistent = false);
	void remove_from_group(const StringName &p_identifier);
	bool is_in_group(const StringName &p_identifier) const;

	struct GroupInfo {
		StringName name;
		bool persistent;
	};

	void get_groups(List<GroupInfo> *p_groups) const;
	int get_persistent_group_count() const;

	void move_child(Node *p_child, int p_pos);
	void raise();

	void set_owner(Node *p_owner);
	Node *get_owner() const;
	void get_owned_by(Node *p_by, List<Node *> *p_owned);

	void remove_and_skip();
	int get_index() const;

	void print_tree();
	void print_tree_pretty();

	void set_filename(const String &p_filename);
	String get_filename() const;

	void set_editor_description(const String &p_editor_description);
	String get_editor_description() const;

	void set_editable_instance(Node *p_node, bool p_editable);
	bool is_editable_instance(const Node *p_node) const;
	Node *get_deepest_editable_node(Node *start_node) const;

	virtual String to_string();

	/* NOTIFICATIONS */

	void propagate_notification(int p_notification);

	void propagate_call(const StringName &p_method, const Array &p_args = Array(), const bool p_parent_first = false);

	/* PROCESSING */
	void set_physics_process(bool p_process);
	float get_physics_process_delta_time() const;
	bool is_physics_processing() const;

	void set_process(bool p_idle_process);
	float get_process_delta_time() const;
	bool is_processing() const;

	void set_physics_process_internal(bool p_process_internal);
	bool is_physics_processing_internal() const;

	void set_process_internal(bool p_idle_process_internal);
	bool is_processing_internal() const;

	void set_process_priority(int p_priority);
	int get_process_priority() const;

	void set_process_input(bool p_enable);
	bool is_processing_input() const;

	void set_process_unhandled_input(bool p_enable);
	bool is_processing_unhandled_input() const;

	void set_process_unhandled_key_input(bool p_enable);
	bool is_processing_unhandled_key_input() const;

	int get_position_in_parent() const;

	Node *duplicate(int p_flags = DUPLICATE_GROUPS | DUPLICATE_SIGNALS | DUPLICATE_SCRIPTS) const;
	Node *duplicate_and_reown(const Map<Node *, Node *> &p_reown_map) const;
#ifdef TOOLS_ENABLED
	Node *duplicate_from_editor(Map<const Node *, Node *> &r_duplimap) const;
	Node *duplicate_from_editor(Map<const Node *, Node *> &r_duplimap, const Map<RES, RES> &p_resource_remap) const;
	void remap_node_resources(Node *p_node, const Map<RES, RES> &p_resource_remap) const;
	void remap_nested_resources(RES p_resource, const Map<RES, RES> &p_resource_remap) const;
#endif

	// used by editors, to save what has changed only
	void set_scene_instance_state(const Ref<SceneState> &p_state);
	Ref<SceneState> get_scene_instance_state() const;

	void set_scene_inherited_state(const Ref<SceneState> &p_state);
	Ref<SceneState> get_scene_inherited_state() const;

	void set_scene_instance_load_placeholder(bool p_enable);
	bool get_scene_instance_load_placeholder() const;

	static Vector<Variant> make_binds(VARIANT_ARG_LIST);

	void replace_by(Node *p_node, bool p_keep_data = false);

	void set_pause_mode(PauseMode p_mode);
	PauseMode get_pause_mode() const;
	bool can_process() const;
	bool can_process_notification(int p_what) const;

	void request_ready();

	static void print_stray_nodes();

#ifdef TOOLS_ENABLED
	String validate_child_name(Node *p_child);
#endif

	void queue_delete();

	//hacks for speed
	static void set_human_readable_collision_renaming(bool p_enabled);
	static void init_node_hrcr();

	void force_parent_owned() { data.parent_owned = true; } //hack to avoid duplicate nodes

	void set_import_path(const NodePath &p_import_path); //path used when imported, used by scene editors to keep tracking
	NodePath get_import_path() const;

	bool is_owned_by_parent() const;

	void get_argument_options(const StringName &p_function, int p_idx, List<String> *r_options) const;

	void clear_internal_tree_resource_paths();

	_FORCE_INLINE_ Viewport *get_viewport() const { return data.viewport; }

	virtual String get_configuration_warning() const;

	void update_configuration_warning();

	void set_display_folded(bool p_folded);
	bool is_displayed_folded() const;
	/* NETWORK */

	void set_network_master(int p_peer_id, bool p_recursive = true);
	int get_network_master() const;
	bool is_network_master() const;

	void rpc_config(const StringName &p_method, MultiplayerAPI::RPCMode p_mode); // config a local method for RPC
	void rset_config(const StringName &p_property, MultiplayerAPI::RPCMode p_mode); // config a local property for RPC

	void rpc(const StringName &p_method, VARIANT_ARG_LIST); //rpc call, honors RPCMode
	void rpc_unreliable(const StringName &p_method, VARIANT_ARG_LIST); //rpc call, honors RPCMode
	void rpc_id(int p_peer_id, const StringName &p_method, VARIANT_ARG_LIST); //rpc call, honors RPCMode
	void rpc_unreliable_id(int p_peer_id, const StringName &p_method, VARIANT_ARG_LIST); //rpc call, honors RPCMode

	void rset(const StringName &p_property, const Variant &p_value); //remote set call, honors RPCMode
	void rset_unreliable(const StringName &p_property, const Variant &p_value); //remote set call, honors RPCMode
	void rset_id(int p_peer_id, const StringName &p_property, const Variant &p_value); //remote set call, honors RPCMode
	void rset_unreliable_id(int p_peer_id, const StringName &p_property, const Variant &p_value); //remote set call, honors RPCMode

	void rpcp(int p_peer_id, bool p_unreliable, const StringName &p_method, const Variant **p_arg, int p_argcount);
	void rsetp(int p_peer_id, bool p_unreliable, const StringName &p_property, const Variant &p_value);

	Ref<MultiplayerAPI> get_multiplayer() const;
	Ref<MultiplayerAPI> get_custom_multiplayer() const;
	void set_custom_multiplayer(Ref<MultiplayerAPI> p_multiplayer);
	const Map<StringName, MultiplayerAPI::RPCMode>::Element *get_node_rpc_mode(const StringName &p_method);
	const Map<StringName, MultiplayerAPI::RPCMode>::Element *get_node_rset_mode(const StringName &p_property);

	Node();
	~Node();
};

VARIANT_ENUM_CAST(Node::DuplicateFlags);

typedef Set<Node *, Node::Comparator> NodeSet;

#endif
