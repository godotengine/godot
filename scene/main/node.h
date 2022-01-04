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

#include "core/config/project_settings.h"
#include "core/object/class_db.h"
#include "core/object/script_language.h"
#include "core/string/node_path.h"
#include "core/templates/map.h"
#include "core/variant/typed_array.h"
#include "scene/main/scene_tree.h"

class Viewport;
class SceneState;
class Tween;
class PropertyTweener;

class Node : public Object {
	GDCLASS(Node, Object);
	OBJ_CATEGORY("Nodes");

public:
	enum ProcessMode {
		PROCESS_MODE_INHERIT, // same as parent node
		PROCESS_MODE_PAUSABLE, // process only if not paused
		PROCESS_MODE_WHEN_PAUSED, // process only if paused
		PROCESS_MODE_ALWAYS, // process always
		PROCESS_MODE_DISABLED, // never process
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

	enum InternalMode {
		INTERNAL_MODE_DISABLED,
		INTERNAL_MODE_FRONT,
		INTERNAL_MODE_BACK,
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
		bool persistent = false;
		SceneTree::Group *group = nullptr;
	};

	struct Data {
		String scene_file_path;
		Ref<SceneState> instance_state;
		Ref<SceneState> inherited_state;

		Node *parent = nullptr;
		Node *owner = nullptr;
		Vector<Node *> children;
		int internal_children_front = 0;
		int internal_children_back = 0;
		int pos = -1;
		int depth = -1;
		int blocked = 0; // Safeguard that throws an error when attempting to modify the tree in a harmful way while being traversed.
		StringName name;
		SceneTree *tree = nullptr;
		bool inside_tree = false;
		bool ready_notified = false; // This is a small hack, so if a node is added during _ready() to the tree, it correctly gets the _ready() notification.
		bool ready_first = true;
#ifdef TOOLS_ENABLED
		NodePath import_path; // Path used when imported, used by scene editors to keep tracking.
#endif
		String editor_description;

		Viewport *viewport = nullptr;

		Map<StringName, GroupData> grouped;
		List<Node *>::Element *OW = nullptr; // Owned element.
		List<Node *> owned;

		ProcessMode process_mode = PROCESS_MODE_INHERIT;
		Node *process_owner = nullptr;

		int multiplayer_authority = 1; // Server by default.
		Vector<Multiplayer::RPCConfig> rpc_methods;

		// Variables used to properly sort the node when processing, ignored otherwise.
		// TODO: Should move all the stuff below to bits.
		bool physics_process = false;
		bool process = false;
		int process_priority = 0;

		bool physics_process_internal = false;
		bool process_internal = false;

		bool input = false;
		bool unhandled_input = false;
		bool unhandled_key_input = false;

		bool parent_owned = false;
		bool in_constructor = true;
		bool use_placeholder = false;

		bool display_folded = false;
		bool editable_instance = false;

		mutable NodePath *path_cache = nullptr;

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
	void _propagate_process_owner(Node *p_owner, int p_pause_notification, int p_enabled_notification);
	Array _get_node_and_resource(const NodePath &p_path);

	void _duplicate_signals(const Node *p_original, Node *p_copy) const;
	Node *_duplicate(int p_flags, Map<const Node *, Node *> *r_duplimap = nullptr) const;

	TypedArray<Node> _get_children(bool p_include_internal = true) const;
	Array _get_groups() const;

	Variant _rpc_bind(const Variant **p_args, int p_argcount, Callable::CallError &r_error);
	Variant _rpc_id_bind(const Variant **p_args, int p_argcount, Callable::CallError &r_error);

	_FORCE_INLINE_ bool _is_internal_front() const { return data.parent && data.pos < data.parent->data.internal_children_front; }
	_FORCE_INLINE_ bool _is_internal_back() const { return data.parent && data.pos >= data.parent->data.children.size() - data.parent->data.internal_children_back; }

	friend class SceneTree;

	void _set_tree(SceneTree *p_tree);
	void _propagate_pause_notification(bool p_enable);

	_FORCE_INLINE_ bool _can_process(bool p_paused) const;
	_FORCE_INLINE_ bool _is_enabled() const;

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
	friend class MultiplayerReplicator;

	void _add_child_nocheck(Node *p_child, const StringName &p_name);
	void _set_owner_nocheck(Node *p_owner);
	void _set_name_nocheck(const StringName &p_name);

	//call from SceneTree
	void _call_input(const Ref<InputEvent> &p_event);
	void _call_unhandled_input(const Ref<InputEvent> &p_event);
	void _call_unhandled_key_input(const Ref<InputEvent> &p_event);

protected:
	virtual void input(const Ref<InputEvent> &p_event);
	virtual void unhandled_input(const Ref<InputEvent> &p_event);
	virtual void unhandled_key_input(const Ref<InputEvent> &p_key_event);

	GDVIRTUAL1(_process, double)
	GDVIRTUAL1(_physics_process, double)
	GDVIRTUAL0(_enter_tree)
	GDVIRTUAL0(_exit_tree)
	GDVIRTUAL0(_ready)
	GDVIRTUAL0RC(Vector<String>, _get_configuration_warnings)

	GDVIRTUAL1(_input, Ref<InputEvent>)
	GDVIRTUAL1(_unhandled_input, Ref<InputEvent>)
	GDVIRTUAL1(_unhandled_key_input, Ref<InputEvent>)

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
		NOTIFICATION_PATH_RENAMED = 23,
		//NOTIFICATION_TRANSLATION_CHANGED = 24, moved below
		NOTIFICATION_INTERNAL_PROCESS = 25,
		NOTIFICATION_INTERNAL_PHYSICS_PROCESS = 26,
		NOTIFICATION_POST_ENTER_TREE = 27,
		NOTIFICATION_DISABLED = 28,
		NOTIFICATION_ENABLED = 29,
		//keep these linked to node

		NOTIFICATION_WM_MOUSE_ENTER = 1002,
		NOTIFICATION_WM_MOUSE_EXIT = 1003,
		NOTIFICATION_WM_WINDOW_FOCUS_IN = 1004,
		NOTIFICATION_WM_WINDOW_FOCUS_OUT = 1005,
		NOTIFICATION_WM_CLOSE_REQUEST = 1006,
		NOTIFICATION_WM_GO_BACK_REQUEST = 1007,
		NOTIFICATION_WM_SIZE_CHANGED = 1008,
		NOTIFICATION_WM_DPI_CHANGE = 1009,

		NOTIFICATION_OS_MEMORY_WARNING = MainLoop::NOTIFICATION_OS_MEMORY_WARNING,
		NOTIFICATION_TRANSLATION_CHANGED = MainLoop::NOTIFICATION_TRANSLATION_CHANGED,
		NOTIFICATION_WM_ABOUT = MainLoop::NOTIFICATION_WM_ABOUT,
		NOTIFICATION_CRASH = MainLoop::NOTIFICATION_CRASH,
		NOTIFICATION_OS_IME_UPDATE = MainLoop::NOTIFICATION_OS_IME_UPDATE,
		NOTIFICATION_APPLICATION_RESUMED = MainLoop::NOTIFICATION_APPLICATION_RESUMED,
		NOTIFICATION_APPLICATION_PAUSED = MainLoop::NOTIFICATION_APPLICATION_PAUSED,
		NOTIFICATION_APPLICATION_FOCUS_IN = MainLoop::NOTIFICATION_APPLICATION_FOCUS_IN,
		NOTIFICATION_APPLICATION_FOCUS_OUT = MainLoop::NOTIFICATION_APPLICATION_FOCUS_OUT,
		NOTIFICATION_TEXT_SERVER_CHANGED = MainLoop::NOTIFICATION_TEXT_SERVER_CHANGED,

		// Editor specific node notifications
		NOTIFICATION_EDITOR_PRE_SAVE = 9001,
		NOTIFICATION_EDITOR_POST_SAVE = 9002,
	};

	/* NODE/TREE */

	StringName get_name() const;
	void set_name(const String &p_name);

	void add_child(Node *p_child, bool p_legible_unique_name = false, InternalMode p_internal = INTERNAL_MODE_DISABLED);
	void add_sibling(Node *p_sibling, bool p_legible_unique_name = false);
	void remove_child(Node *p_child);

	int get_child_count(bool p_include_internal = true) const;
	Node *get_child(int p_index, bool p_include_internal = true) const;
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

	bool is_ancestor_of(const Node *p_node) const;
	bool is_greater_than(const Node *p_node) const;

	NodePath get_path() const;
	NodePath get_path_to(const Node *p_node) const;
	Node *find_common_parent_with(const Node *p_node) const;

	void add_to_group(const StringName &p_identifier, bool p_persistent = false);
	void remove_from_group(const StringName &p_identifier);
	bool is_in_group(const StringName &p_identifier) const;

	struct GroupInfo {
		StringName name;
		bool persistent = false;
	};

	void get_groups(List<GroupInfo> *p_groups) const;
	int get_persistent_group_count() const;

	void move_child(Node *p_child, int p_pos);
	void _move_child(Node *p_child, int p_pos, bool p_ignore_end = false);
	void raise();

	void set_owner(Node *p_owner);
	Node *get_owner() const;
	void get_owned_by(Node *p_by, List<Node *> *p_owned);

	void remove_and_skip();
	int get_index(bool p_include_internal = true) const;

	Ref<Tween> create_tween();

	void print_tree();
	void print_tree_pretty();

	void set_scene_file_path(const String &p_scene_file_path);
	String get_scene_file_path() const;

	void set_editor_description(const String &p_editor_description);
	String get_editor_description() const;

	void set_editable_instance(Node *p_node, bool p_editable);
	bool is_editable_instance(const Node *p_node) const;
	Node *get_deepest_editable_node(Node *p_start_node) const;

#ifdef TOOLS_ENABLED
	void set_property_pinned(const String &p_property, bool p_pinned);
	bool is_property_pinned(const StringName &p_property) const;
	virtual StringName get_property_store_alias(const StringName &p_property) const;
#endif
	void get_storable_properties(Set<StringName> &r_storable_properties) const;

	virtual String to_string() override;

	/* NOTIFICATIONS */

	void propagate_notification(int p_notification);

	void propagate_call(const StringName &p_method, const Array &p_args = Array(), const bool p_parent_first = false);

	/* PROCESSING */
	void set_physics_process(bool p_process);
	double get_physics_process_delta_time() const;
	bool is_physics_processing() const;

	void set_process(bool p_process);
	double get_process_delta_time() const;
	bool is_processing() const;

	void set_physics_process_internal(bool p_process_internal);
	bool is_physics_processing_internal() const;

	void set_process_internal(bool p_process_internal);
	bool is_processing_internal() const;

	void set_process_priority(int p_priority);
	int get_process_priority() const;

	void set_process_input(bool p_enable);
	bool is_processing_input() const;

	void set_process_unhandled_input(bool p_enable);
	bool is_processing_unhandled_input() const;

	void set_process_unhandled_key_input(bool p_enable);
	bool is_processing_unhandled_key_input() const;

	Node *duplicate(int p_flags = DUPLICATE_GROUPS | DUPLICATE_SIGNALS | DUPLICATE_SCRIPTS) const;
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

	void set_process_mode(ProcessMode p_mode);
	ProcessMode get_process_mode() const;
	bool can_process() const;
	bool can_process_notification(int p_what) const;
	bool is_enabled() const;

	void request_ready();

	static void print_stray_nodes();

#ifdef TOOLS_ENABLED
	String validate_child_name(Node *p_child);
#endif

	void queue_delete();

	//hacks for speed
	static void init_node_hrcr();

	void force_parent_owned() { data.parent_owned = true; } //hack to avoid duplicate nodes

	void set_import_path(const NodePath &p_import_path); //path used when imported, used by scene editors to keep tracking
	NodePath get_import_path() const;

	bool is_owned_by_parent() const;

	void get_argument_options(const StringName &p_function, int p_idx, List<String> *r_options) const override;

	void clear_internal_tree_resource_paths();

	_FORCE_INLINE_ Viewport *get_viewport() const { return data.viewport; }

	virtual TypedArray<String> get_configuration_warnings() const;
	String get_configuration_warnings_as_string() const;

	void update_configuration_warnings();

	void set_display_folded(bool p_folded);
	bool is_displayed_folded() const;
	/* NETWORK */

	void set_multiplayer_authority(int p_peer_id, bool p_recursive = true);
	int get_multiplayer_authority() const;
	bool is_multiplayer_authority() const;

	uint16_t rpc_config(const StringName &p_method, Multiplayer::RPCMode p_rpc_mode, bool p_call_local = false, Multiplayer::TransferMode p_transfer_mode = Multiplayer::TRANSFER_MODE_RELIABLE, int p_channel = 0); // config a local method for RPC
	Vector<Multiplayer::RPCConfig> get_node_rpc_methods() const;

	void rpc(const StringName &p_method, VARIANT_ARG_LIST); // RPC, honors RPCMode, TransferMode, channel
	void rpc_id(int p_peer_id, const StringName &p_method, VARIANT_ARG_LIST); // RPC to specific peer(s), honors RPCMode, TransferMode, channel
	void rpcp(int p_peer_id, const StringName &p_method, const Variant **p_arg, int p_argcount);

	Ref<MultiplayerAPI> get_multiplayer() const;
	Ref<MultiplayerAPI> get_custom_multiplayer() const;
	void set_custom_multiplayer(Ref<MultiplayerAPI> p_multiplayer);

	Node();
	~Node();
};

VARIANT_ENUM_CAST(Node::DuplicateFlags);

typedef Set<Node *, Node::Comparator> NodeSet;

#endif
