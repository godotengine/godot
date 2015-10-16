/*************************************************************************/
/*  node.h                                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2015 Juan Linietsky, Ariel Manzur.                 */
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

#include "object.h"
#include "path_db.h"
#include "map.h"
#include "object_type_db.h"
#include "script_language.h"
#include "scene/main/scene_main_loop.h"


class Viewport;
class SceneState;
class Node : public Object {

	OBJ_TYPE( Node, Object );
	OBJ_CATEGORY("Nodes");
	
public:

	enum PauseMode {

		PAUSE_MODE_INHERIT,
		PAUSE_MODE_STOP,
		PAUSE_MODE_PROCESS
	};


	struct Comparator {

		bool operator()(const Node* p_a, const Node* p_b) const { return p_b->is_greater_than(p_a); }
	};

private:	
	
	struct GroupData {
		
		bool persistent;	
		GroupData() { persistent=false; }
	};
		
	struct Data {
	
		String filename;
		Ref<SceneState> instance_state;
		Ref<SceneState> inherited_state;

		HashMap<NodePath,int> editable_instances;

		Node *parent;
		Node *owner;
		Vector<Node*> children;	// list of children
		int pos;
		int depth;
		int blocked; // safeguard that throws an error when attempting to modify the tree in a harmful way while being traversed.
		StringName name;
		SceneTree *tree;
		bool inside_tree;
#ifdef TOOLS_ENABLED
		NodePath import_path; //path used when imported, used by scene editors to keep tracking
#endif

		Viewport *viewport;

				
		HashMap< StringName, GroupData,StringNameHasher>  grouped;
		List<Node*>::Element *OW; // owned element
		List<Node*> owned;
		
		PauseMode pause_mode;
		Node *pause_owner;
		// variables used to properly sort the node when processing, ignored otherwise
		//should move all the stuff below to bits
		bool fixed_process;
		bool idle_process;

		bool input;
		bool unhandled_input;
		bool unhandled_key_input;

		bool parent_owned;
		bool in_constructor;
		bool use_placeholder;


	} data;
	

	void _print_tree(const Node *p_node);	
	
	virtual bool _use_builtin_script() const { return true; }
	Node *_get_node(const NodePath& p_path) const;
	Node *_get_child_by_name(const StringName& p_name) const;



	void _validate_child_name(Node *p_name);

	void _propagate_reverse_notification(int p_notification);	
	void _propagate_deferred_notification(int p_notification, bool p_reverse);
	void _propagate_enter_tree();
	void _propagate_ready();
	void _propagate_exit_tree();
	void _propagate_validate_owner();
	void _print_stray_nodes();
	void _propagate_pause_owner(Node*p_owner);
	Array _get_node_and_resource(const NodePath& p_path);

	void _duplicate_signals(const Node* p_original,Node* p_copy) const;
	void _duplicate_and_reown(Node* p_new_parent, const Map<Node*,Node*>& p_reown_map) const;
	Array _get_children() const;
	Array _get_groups() const;

friend class SceneTree;

	void _set_tree(SceneTree *p_tree);
protected:

	void _block() { data.blocked++; }
	void _unblock()  { data.blocked--; }

	void _notification(int p_notification);	
	
	virtual void add_child_notify(Node *p_child);
	virtual void remove_child_notify(Node *p_child);
	virtual void move_child_notify(Node *p_child);
	//void remove_and_delete_child(Node *p_child);
	
	void _propagate_replace_owner(Node *p_owner,Node* p_by_owner); 
	
	static void _bind_methods();

friend class SceneState;

	void _add_child_nocheck(Node* p_child,const StringName& p_name);
	void _set_owner_nocheck(Node* p_owner);
	void _set_name_nocheck(const StringName& p_name);

public:

	enum {
		// you can make your own, but don't use the same numbers as other notifications in other nodes
		NOTIFICATION_ENTER_TREE=10,
		NOTIFICATION_EXIT_TREE =11,
		NOTIFICATION_MOVED_IN_PARENT =12,
		NOTIFICATION_READY=13,
		//NOTIFICATION_PARENT_DECONFIGURED =15, - it's confusing, it's going away
		NOTIFICATION_PAUSED=14,
		NOTIFICATION_UNPAUSED=15,
		NOTIFICATION_FIXED_PROCESS = 16,
		NOTIFICATION_PROCESS = 17,
		NOTIFICATION_PARENTED=18,
		NOTIFICATION_UNPARENTED=19,
		NOTIFICATION_INSTANCED=20,
	};
			
	/* NODE/TREE */			
			
	StringName get_name() const;
	void set_name(const String& p_name);
	
	void add_child(Node *p_child);
	void remove_child(Node *p_child);
	
	int get_child_count() const;
	Node *get_child(int p_index) const;
	bool has_node(const NodePath& p_path) const;
	Node *get_node(const NodePath& p_path) const;
	Node* find_node(const String& p_mask,bool p_recursive=true,bool p_owned=true) const;
	bool has_node_and_resource(const NodePath& p_path) const;
	Node *get_node_and_resource(const NodePath& p_path,RES& r_res) const;
	
	Node *get_parent() const;
	_FORCE_INLINE_ SceneTree *get_tree() const { ERR_FAIL_COND_V( !data.tree, NULL ); return data.tree; }

	_FORCE_INLINE_ bool is_inside_tree() const { return data.inside_tree; }
	
	bool is_a_parent_of(const Node *p_node) const;
	bool is_greater_than(const Node *p_node) const;
	
	NodePath get_path() const;
	NodePath get_path_to(const Node *p_node) const;
	
	void add_to_group(const StringName& p_identifier,bool p_persistent=false);
	void remove_from_group(const StringName& p_identifier);
	bool is_in_group(const StringName& p_identifier) const;
	
	struct GroupInfo {
	
		StringName name;
		bool persistent;
	};
	
	void get_groups(List<GroupInfo> *p_groups) const;
	
	void move_child(Node *p_child,int p_pos);
	void raise();
	
	void set_owner(Node *p_owner);
	Node *get_owner() const;
	void get_owned_by(Node *p_by,List<Node*> *p_owned);

	
	void remove_and_skip();
	int get_index() const;
	
	void print_tree();
	
	void set_filename(const String& p_filename);
	String get_filename() const;

	void set_editable_instance(Node* p_node,bool p_editable);
	bool is_editable_instance(Node* p_node) const;


	/* NOTIFICATIONS */
	
	void propagate_notification(int p_notification);
	
	/* PROCESSING */
	void set_fixed_process(bool p_process);
	float get_fixed_process_delta_time() const;
	bool is_fixed_processing() const;

	void set_process(bool p_process);
	float get_process_delta_time() const;
	bool is_processing() const;


	void set_process_input(bool p_enable);
	bool is_processing_input() const;

	void set_process_unhandled_input(bool p_enable);
	bool is_processing_unhandled_input() const;

	void set_process_unhandled_key_input(bool p_enable);
	bool is_processing_unhandled_key_input() const;

	int get_position_in_parent() const;

	Node *duplicate(bool p_use_instancing=false) const;
	Node *duplicate_and_reown(const Map<Node*,Node*>& p_reown_map) const;

	//Node *clone_tree() const;

	// used by editors, to save what has changed only
	void set_scene_instance_state(const Ref<SceneState>& p_state);
	Ref<SceneState> get_scene_instance_state() const;

	void set_scene_inherited_state(const Ref<SceneState>& p_state);
	Ref<SceneState> get_scene_inherited_state() const;

	void set_scene_instance_load_placeholder(bool p_enable);
	bool get_scene_instance_load_placeholder() const;

	static Vector<Variant> make_binds(VARIANT_ARG_LIST);

	void replace_by(Node* p_node,bool p_keep_data=false);

	void set_pause_mode(PauseMode p_mode);
	PauseMode get_pause_mode() const;
	bool can_process() const;

	static void print_stray_nodes();

	String validate_child_name(const String& p_name) const;

	void queue_delete();

//shitty hacks for speed
	static void set_human_readable_collision_renaming(bool p_enabled);
	static void init_node_hrcr();

	void force_parent_owned() { data.parent_owned=true; } //hack to avoid duplicate nodes

#ifdef TOOLS_ENABLED
	void set_import_path(const NodePath& p_import_path); //path used when imported, used by scene editors to keep tracking
	NodePath get_import_path() const;
#endif

	void get_argument_options(const StringName& p_function,int p_idx,List<String>*r_options) const;

	void clear_internal_tree_resource_paths();

	_FORCE_INLINE_ Viewport *get_viewport() const { return data.viewport; }

	/* CANVAS */

	Node();
	~Node();
};


typedef Set<Node*,Node::Comparator> NodeSet;


#endif
