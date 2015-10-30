/*************************************************************************/
/*  node.cpp                                                             */
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
#include "node.h"
#include "print_string.h"
#include "message_queue.h"
#include "scene/scene_string_names.h"
#include "scene/resources/packed_scene.h"
#include "io/resource_loader.h"
#include "viewport.h"

VARIANT_ENUM_CAST(Node::PauseMode);




void Node::_notification(int p_notification) {
	
	switch(p_notification) {

		case NOTIFICATION_PROCESS: {

			if (get_script_instance()) {

				Variant time=get_process_delta_time();
				const Variant*ptr[1]={&time};
				Variant::CallError err;
				get_script_instance()->call_multilevel(SceneStringNames::get_singleton()->_process,ptr,1);
			}
		} break;
		case NOTIFICATION_FIXED_PROCESS: {

			if (get_script_instance()) {

				Variant time=get_fixed_process_delta_time();
				const Variant*ptr[1]={&time};
				Variant::CallError err;
				get_script_instance()->call_multilevel(SceneStringNames::get_singleton()->_fixed_process,ptr,1);
			}

		} break;
		case NOTIFICATION_ENTER_TREE: {

			if (data.pause_mode==PAUSE_MODE_INHERIT) {

				if (data.parent)
					data.pause_owner=data.parent->data.pause_owner;
				else
					data.pause_owner=NULL;
			} else {
				data.pause_owner=this;
			}

			if (data.input)
				add_to_group("_vp_input"+itos(get_viewport()->get_instance_ID()));
			if (data.unhandled_input)
				add_to_group("_vp_unhandled_input"+itos(get_viewport()->get_instance_ID()));
			if (data.unhandled_key_input)
				add_to_group("_vp_unhandled_key_input"+itos(get_viewport()->get_instance_ID()));

			get_tree()->node_count++;

		} break;
		case NOTIFICATION_EXIT_TREE: {

			get_tree()->node_count--;
			if (data.input)
				remove_from_group("_vp_input"+itos(get_viewport()->get_instance_ID()));
			if (data.unhandled_input)
				remove_from_group("_vp_unhandled_input"+itos(get_viewport()->get_instance_ID()));
			if (data.unhandled_key_input)
				remove_from_group("_vp_unhandled_key_input"+itos(get_viewport()->get_instance_ID()));

		} break;
		case NOTIFICATION_READY: {

			if (get_script_instance()) {

				Variant::CallError err;
				get_script_instance()->call_multilevel_reversed(SceneStringNames::get_singleton()->_ready,NULL,0);
			}
			//emit_signal(SceneStringNames::get_singleton()->enter_tree);

		} break;
		case NOTIFICATION_POSTINITIALIZE: {
			data.in_constructor=false;
		} break;
		case NOTIFICATION_PREDELETE: {

			set_owner(NULL);
			
			while ( data.owned.size() ) {
			
				data.owned.front()->get()->set_owner(NULL);	
			}

			if (data.parent) {
				
				data.parent->remove_child(this);
			}

			// kill children as cleanly as possible
			while( data.children.size() ) {
				
				Node *child = data.children[0];		
				remove_child(child);
				memdelete( child );
			}

		} break;
	}
}


void Node::_propagate_ready() {

	data.blocked++;
	for (int i=0;i<data.children.size();i++) {

		data.children[i]->_propagate_ready();
	}
	data.blocked--;
	notification(NOTIFICATION_READY);

}


void Node::_propagate_enter_tree() {
	// this needs to happen to all childs before any enter_tree

	if (data.parent) {
		data.tree=data.parent->data.tree;
		data.depth=data.parent->data.depth+1;
	} else {

		data.depth=1;
	}
	
	data.viewport = cast_to<Viewport>();
	if (!data.viewport)
		data.viewport = data.parent->data.viewport;

	data.inside_tree=true;

	const StringName *K=NULL;

	while ((K=data.grouped.next(K))) {

		data.tree->add_to_group(*K,this);
	}


	notification(NOTIFICATION_ENTER_TREE);

	if (get_script_instance()) {

		Variant::CallError err;
		get_script_instance()->call_multilevel_reversed(SceneStringNames::get_singleton()->_enter_tree,NULL,0);
	}

	emit_signal(SceneStringNames::get_singleton()->enter_tree);


	data.blocked++;
	//block while adding children

	for (int i=0;i<data.children.size();i++) {
		
		if (!data.children[i]->is_inside_tree()) // could have been added in enter_tree
			data.children[i]->_propagate_enter_tree();
	}	

	data.blocked--;

#ifdef DEBUG_ENABLED

	if (ScriptDebugger::get_singleton() && data.filename!=String()) {
		//used for live edit
		data.tree->live_scene_edit_cache[data.filename].insert(this);
	}
#endif
	// enter groups
}



void Node::_propagate_exit_tree() {

	//block while removing children

#ifdef DEBUG_ENABLED

	if (ScriptDebugger::get_singleton() && data.filename!=String()) {
		//used for live edit
		Map<String,Set<Node*> >::Element *E=data.tree->live_scene_edit_cache.find(data.filename);
		if (E) {
			E->get().erase(this);
			if (E->get().size()==0) {
				data.tree->live_scene_edit_cache.erase(E);
			}
		}

		Map<Node*,Map<ObjectID,Node*> >::Element *F=data.tree->live_edit_remove_list.find(this);
		if (F) {
			for (Map<ObjectID,Node*>::Element*G=F->get().front();G;G=G->next()) {

				memdelete(G->get());
			}
			data.tree->live_edit_remove_list.erase(F);
		}
	}
#endif
	data.blocked++;

	for (int i=data.children.size()-1;i>=0;i--) {

		data.children[i]->_propagate_exit_tree();
	}

	data.blocked--;

	if (get_script_instance()) {

		Variant::CallError err;
		get_script_instance()->call_multilevel(SceneStringNames::get_singleton()->_exit_tree,NULL,0);
	}
	emit_signal(SceneStringNames::get_singleton()->exit_tree);

	notification(NOTIFICATION_EXIT_TREE,true);
	if (data.tree)
		data.tree->node_removed(this);

	// exit groups
	const StringName *K=NULL;

	while ((K=data.grouped.next(K))) {

		data.tree->remove_from_group(*K,this);
	}

	data.viewport = NULL;

	if (data.tree)
		data.tree->tree_changed();

	data.inside_tree=false;
	data.tree=NULL;
	data.depth=-1;

}





void Node::move_child(Node *p_child,int p_pos) {	
	
	ERR_FAIL_NULL(p_child);
	ERR_EXPLAIN("Invalid new child position: "+itos(p_pos));
	ERR_FAIL_INDEX( p_pos, data.children.size()+1 );
	ERR_EXPLAIN("child is not a child of this node.");
	ERR_FAIL_COND(p_child->data.parent!=this);
	ERR_FAIL_COND(data.blocked>0);
	
	data.children.remove( p_child->data.pos );
	data.children.insert( p_pos, p_child );

	if (data.tree) {
		data.tree->tree_changed();
	}

	data.blocked++;
	//new pos first
	for (int i=0;i<data.children.size();i++) {
		
		data.children[i]->data.pos=i;
	}
	// notification second
	move_child_notify(p_child);
	for (int i=0;i<data.children.size();i++) {
		data.children[i]->notification( NOTIFICATION_MOVED_IN_PARENT );

	}

	data.blocked--;

}

void Node::raise() {
	
	if (!data.parent)
		return;
	
	data.parent->move_child(this,data.parent->data.children.size()-1);

}

void Node::add_child_notify(Node *p_child) {
	
	// to be used when not wanted	
}

/*
void Node::remove_and_delete_child(Node *p_child) {

	ERR_FAIL_NULL( p_child );
	ERR_FAIL_COND( p_child->get_parent()!=this );

	remove_child(p_child);
	memdelete(p_child);

}
*/

void Node::remove_child_notify(Node *p_child) {

	// to be used when not wanted	
}

void Node::move_child_notify(Node *p_child) {

	// to be used when not wanted	
}

void Node::set_fixed_process(bool p_process) {
	
	if (data.fixed_process==p_process)
		return;

	data.fixed_process=p_process;
	
	if (data.fixed_process)
		add_to_group("fixed_process",false);
	else
		remove_from_group("fixed_process");
				
	data.fixed_process=p_process;
	_change_notify("fixed_process");
}

void Node::set_pause_mode(PauseMode p_mode) {

	if (data.pause_mode==p_mode)
		return;

	bool prev_inherits=data.pause_mode==PAUSE_MODE_INHERIT;
	data.pause_mode=p_mode;
	if (!is_inside_tree())
		return; //pointless
	if ((data.pause_mode==PAUSE_MODE_INHERIT) == prev_inherits)
		return; ///nothing changed

	Node *owner=NULL;

	if (data.pause_mode==PAUSE_MODE_INHERIT) {

		if (data.parent)
			owner=data.parent->data.pause_owner;
	} else {
		owner=this;
	}

	_propagate_pause_owner(owner);



}

Node::PauseMode Node::get_pause_mode() const {

	return data.pause_mode;
}

void Node::_propagate_pause_owner(Node*p_owner) {

	if (data.pause_mode!=PAUSE_MODE_INHERIT)
		return;
	data.pause_owner=p_owner;
	for(int i=0;i<data.children.size();i++) {

		data.children[i]->_propagate_pause_owner(p_owner);
	}
}

bool Node::can_process() const {

	ERR_FAIL_COND_V( !is_inside_tree(), false );

	if (get_tree()->is_paused()) {

		if (data.pause_mode==PAUSE_MODE_STOP)
			return false;
		if (data.pause_mode==PAUSE_MODE_PROCESS)
			return true;
		if (data.pause_mode==PAUSE_MODE_INHERIT) {

			if (!data.pause_owner)
				return false; //clearly no pause owner by default

			if (data.pause_owner->data.pause_mode==PAUSE_MODE_PROCESS)
				return true;

			if (data.pause_owner->data.pause_mode==PAUSE_MODE_STOP)
				return false;
		}

	}

	return true;
}


float Node::get_fixed_process_delta_time() const {
	
	if (data.tree)
		return data.tree->get_fixed_process_time();
	else
		return 0;
}

void Node::set_process(bool p_idle_process) {

	if (data.idle_process==p_idle_process)
		return;

	data.idle_process=p_idle_process;

	if (data.idle_process)
		add_to_group("idle_process",false);
	else
		remove_from_group("idle_process");

	data.idle_process=p_idle_process;
	_change_notify("idle_process");
}

float Node::get_process_delta_time() const {

	if (data.tree)
		return data.tree->get_idle_process_time();
	else
		return 0;
}

bool Node::is_fixed_processing() const {
	
	return data.fixed_process;
}

bool Node::is_processing() const {

	return data.idle_process;
}


void Node::set_process_input(bool p_enable) {

	if (p_enable==data.input)
		return;

	data.input=p_enable;
	if (!is_inside_tree())
		return;

	if (p_enable)
		add_to_group("_vp_input"+itos(get_viewport()->get_instance_ID()));
	else
		remove_from_group("_vp_input"+itos(get_viewport()->get_instance_ID()));


}

bool Node::is_processing_input() const {
	return data.input;
}

void Node::set_process_unhandled_input(bool p_enable) {

	if (p_enable==data.unhandled_input)
		return;
	data.unhandled_input=p_enable;
	if (!is_inside_tree())
		return;

	if (p_enable)
		add_to_group("_vp_unhandled_input"+itos(get_viewport()->get_instance_ID()));
	else
		remove_from_group("_vp_unhandled_input"+itos(get_viewport()->get_instance_ID()));
}


bool Node::is_processing_unhandled_input() const {
	return data.unhandled_input;
}


void Node::set_process_unhandled_key_input(bool p_enable) {

	if (p_enable==data.unhandled_key_input)
		return;
	data.unhandled_key_input=p_enable;
	if (!is_inside_tree())
		return;

	if (p_enable)
		add_to_group("_vp_unhandled_key_input"+itos(get_viewport()->get_instance_ID()));
	else
		remove_from_group("_vp_unhandled_key_input"+itos(get_viewport()->get_instance_ID()));
}


bool Node::is_processing_unhandled_key_input() const {
	return data.unhandled_key_input;
}


StringName Node::get_name() const {
	
	return data.name;
}

void Node::_set_name_nocheck(const StringName& p_name) {

	data.name=p_name;

}

void Node::set_name(const String& p_name) {
	
	String name=p_name.replace(":","").replace("/","").replace("@","");

	ERR_FAIL_COND(name=="");
	data.name=name;
	
	if (data.parent) {
		
		data.parent->_validate_child_name(this);
	}

	if (is_inside_tree()) {

		emit_signal("renamed");
		get_tree()->tree_changed();
	}
}

static bool node_hrcr=false;
static SafeRefCount node_hrcr_count;

void Node::init_node_hrcr() {
	node_hrcr_count.init(1);
}

void Node::set_human_readable_collision_renaming(bool p_enabled) {

	node_hrcr=p_enabled;
}



String Node::validate_child_name(const String& p_name) const {

	//this approach to autoset node names is human readable but very slow
	//it's turned on while running in the editor

	String basename = p_name;

	if (basename==String()) {

		return String();
	}

	int val=1;

	for(;;) {

		String attempted = val > 1 ? (basename + " " +itos(val) ) : basename;

		bool found=false;

		for (int i=0;i<data.children.size();i++) {

			//if (data.children[i]==p_child)
			//	continue;
			if (data.children[i]->get_name() == attempted) {
				found=true;
				break;
			}

		}

		if (found) {

			val++;
			continue;
		}

		return attempted;
		break;
	}

	return basename;

}

void Node::_validate_child_name(Node *p_child) {

	/* Make sure the name is unique */

	if (node_hrcr) {

		//this approach to autoset node names is human readable but very slow
		//it's turned on while running in the editor

		String basename = p_child->data.name;

		if (basename=="") {

			basename = p_child->get_type();
		}

		int val=1;

		for(;;) {

			String attempted = val > 1 ? (basename + " " +itos(val) ) : basename;

			bool found=false;

			for (int i=0;i<data.children.size();i++) {

				if (data.children[i]==p_child)
					continue;
				if (data.children[i]->get_name() == attempted) {
					found=true;
					break;
				}

			}

			if (found) {

				val++;
				continue;
			}

			p_child->data.name=attempted;
			break;
		}
	} else {

		//this approach to autoset node names is fast but not as readable
		//it's the default and reserves the '@' character for unique names.

		bool unique=true;

		if (p_child->data.name==StringName() || p_child->data.name.operator String()[0]=='@') {
			//new unique name must be assigned
			unique=false;
		} else {
			//check if exists
			Node **childs=data.children.ptr();
			int cc = data.children.size();

			for(int i=0;i<cc;i++) {
				if (childs[i]==p_child)
					continue;
				if (childs[i]->data.name==p_child->data.name) {
					unique=false;
					break;
				}
			}
		}

		if (!unique) {

			node_hrcr_count.ref();
#ifdef DEBUG_ENABLED
			String name = "@"+String(p_child->get_type_name())+itos(node_hrcr_count.get());
#else
			String name = "@"+itos(node_hrcr_count.get());
#endif
			p_child->data.name=name;
		}
	}
}

void Node::_add_child_nocheck(Node* p_child,const StringName& p_name) {
	//add a child node quickly, without name validation

	p_child->data.name=p_name;
	p_child->data.pos=data.children.size();
	data.children.push_back( p_child );
	p_child->data.parent=this;
	p_child->notification(NOTIFICATION_PARENTED);

	if (data.tree) {
		p_child->_set_tree(data.tree);
	}

	/* Notify */
	//recognize childs created in this node constructor
	p_child->data.parent_owned=data.in_constructor;
	add_child_notify(p_child);


}


void Node::add_child(Node *p_child) {

	ERR_FAIL_NULL(p_child);
	/* Fail if node has a parent */
	ERR_EXPLAIN("Can't add child "+p_child->get_name()+" to itself.")
	ERR_FAIL_COND( p_child==this ); // adding to itself!
	ERR_EXPLAIN("Can't add child, already has a parent");
	ERR_FAIL_COND( p_child->data.parent );
	ERR_EXPLAIN("Can't add child while a notification is happening");
	ERR_FAIL_COND( data.blocked > 0 );
		
	/* Validate name */
	_validate_child_name(p_child);

	_add_child_nocheck(p_child,p_child->data.name);
	
}

void Node::_propagate_validate_owner() {

	if (data.owner) {

		bool found=false;
		Node *parent = data.parent;

		while(parent) {


			if (parent==data.owner) {

				found=true;
				break;
			}

			parent=parent->data.parent;
		}


		if (!found) {


			data.owner->data.owned.erase(data.OW);
			data.owner=NULL;
		}

	}


	for(int i=0;i<data.children.size();i++) {


		data.children[i]->_propagate_validate_owner();
	}
}

void Node::remove_child(Node *p_child) {

	ERR_FAIL_NULL(p_child);
	ERR_FAIL_COND( data.blocked > 0 );
	
	int idx=-1;
	for (int i=0;i<data.children.size();i++) {
	
		if (data.children[i]==p_child) {
			
			idx=i;
			break;
		}
	}
	
	ERR_FAIL_COND( idx==-1 );

	
	//if (data.scene) { does not matter
		
		p_child->_set_tree(NULL);
	//}
	
	remove_child_notify(p_child); 
	p_child->notification(NOTIFICATION_UNPARENTED);
		
	data.children.remove(idx);
	
	for (int i=idx;i<data.children.size();i++) {
		
		data.children[i]->data.pos=i;
	}
	
	p_child->data.parent=NULL;
	p_child->data.pos=-1;



	// validate owner
	p_child->_propagate_validate_owner();
		
}

int Node::get_child_count() const {
	
	return data.children.size();	
}
Node *Node::get_child(int p_index) const {
	
	ERR_FAIL_INDEX_V( p_index, data.children.size(), NULL );
	
	return data.children[p_index];
}


Node *Node::_get_child_by_name(const StringName& p_name) const {

	int cc=data.children.size();
	Node* const* cd=data.children.ptr();

	for(int i=0;i<cc;i++){
		if (cd[i]->data.name==p_name)
			return cd[i];
	}

	return NULL;
}

Node *Node::_get_node(const NodePath& p_path) const {

	ERR_FAIL_COND_V( !data.inside_tree && p_path.is_absolute(), NULL );
	
	Node *current=NULL;	
	Node *root=NULL;
		
	if (!p_path.is_absolute()) {
		current=const_cast<Node*>(this); //start from this
	} else {

		root=const_cast<Node*>(this);;
		while (root->data.parent)
			root=root->data.parent; //start from root
	}
	

	for(int i=0;i<p_path.get_name_count();i++) {
			
		
		StringName name = p_path.get_name(i);		
		Node *next = NULL;
		
		if (name==SceneStringNames::get_singleton()->dot) { // .
		
			next=current;
			
		} else if (name==SceneStringNames::get_singleton()->doubledot) { // ..
			
			if (current==NULL || !current->data.parent)
				return NULL;
			
			next=current->data.parent;
		} else if (current==NULL) {
			
			if (name==root->get_name())
				next=root;
			
		} else {
				
			next=NULL;
			
			for(int j=0;j<current->data.children.size();j++) {
			
				Node *child = current->data.children[j];

				if ( child->data.name == name ) {
				
					next = child;
					break;
				}			
			}
			if (next == NULL) {
				return NULL;
			};
		}
		current=next;
	}
		
	return current;
}

Node *Node::get_node(const NodePath& p_path) const {

	Node *node = _get_node(p_path);
	if (!node) {
		ERR_EXPLAIN("Node not found: "+p_path);
		ERR_FAIL_COND_V(!node,NULL);
	}
	return node;
}

bool Node::has_node(const NodePath& p_path) const {

	return _get_node(p_path)!=NULL;
}


Node* Node::find_node(const String& p_mask,bool p_recursive,bool p_owned) const {

	Node * const*cptr = data.children.ptr();
	int ccount = data.children.size();
	for(int i=0;i<ccount;i++) {
		if (p_owned && !cptr[i]->data.owner)
			continue;
		if (cptr[i]->data.name.operator String().match(p_mask))
			return cptr[i];

		if (!p_recursive)
			continue;

		Node* ret = cptr[i]->find_node(p_mask,true,p_owned);
		if (ret)
			return ret;
	}
	return NULL;

}

Node *Node::get_parent() const {

	return data.parent;
}


bool Node::is_a_parent_of(const Node *p_node) const {

	ERR_FAIL_NULL_V(p_node,false);
	Node *p=p_node->data.parent;
	while(p) {
		
		if (p==this)
			return true;
		p=p->data.parent;
	}
	
	return false;
}

bool Node::is_greater_than(const Node *p_node) const {

	ERR_FAIL_NULL_V(p_node,false);
	ERR_FAIL_COND_V( !data.inside_tree, false );
	ERR_FAIL_COND_V( !p_node->data.inside_tree, false );
	
	ERR_FAIL_COND_V( data.depth<0, false);
	ERR_FAIL_COND_V( p_node->data.depth<0, false);
#ifdef NO_ALLOCA
	
	Vector<int> this_stack;
	Vector<int> that_stack;
	this_stack.resize(data.depth);
	that_stack.resize(p_node->data.depth);

#else
	
	int *this_stack=(int*)alloca(sizeof(int)*data.depth);
	int *that_stack=(int*)alloca(sizeof(int)*p_node->data.depth);
	
#endif
	
	const Node *n = this;
	
	int idx=data.depth-1;
	while(n) {
		ERR_FAIL_INDEX_V(idx, data.depth,false);
		this_stack[idx--]=n->data.pos;
		n=n->data.parent;
	}
	ERR_FAIL_COND_V(idx!=-1,false);
	n = p_node;
	idx=p_node->data.depth-1;
	while(n) {
		ERR_FAIL_INDEX_V(idx, p_node->data.depth,false);
		that_stack[idx--]=n->data.pos;
		
		n=n->data.parent;
	}
	ERR_FAIL_COND_V(idx!=-1,false);
	idx=0;
	
	bool res;
	while(true) {
	
		// using -2 since out-of-tree or nonroot nodes have -1
		int this_idx = (idx >= data.depth)? -2 : this_stack[idx];
		int that_idx = (idx >= p_node->data.depth)? -2 : that_stack[idx];

		if (this_idx > that_idx) {
			res=true;
			break;
		} else if (this_idx < that_idx) {
			res=false;
			break;
		} else if (this_idx == -2 ) {
			res=false; // equal
			break;
		}
		idx++;
	}
		
	return res;
}

void Node::get_owned_by(Node *p_by,List<Node*> *p_owned) {

	if (data.owner==p_by)
		p_owned->push_back(this);

	for (int i=0;i<get_child_count();i++)
		get_child(i)->get_owned_by(p_by,p_owned);

}


void Node::_set_owner_nocheck(Node* p_owner) {

	ERR_FAIL_COND(data.owner);
	data.owner=p_owner;
	data.owner->data.owned.push_back( this );
	data.OW = data.owner->data.owned.back();
}

void Node::set_owner(Node *p_owner) {
	
	if (data.owner) {
		
		data.owner->data.owned.erase( data.OW );
		data.OW=NULL;
		data.owner=NULL;
	}
	
	ERR_FAIL_COND(p_owner==this);
	
	if (!p_owner)
		return;
		
	Node *check=this->get_parent();
	bool owner_valid=false;
	
	while(check) {
		
		if (check==p_owner) {
			owner_valid=true;
			break;
		}
		
		check=check->data.parent;
	}
		
	ERR_FAIL_COND(!owner_valid);

	_set_owner_nocheck(p_owner);
}
Node *Node::get_owner() const {
	
	return data.owner;	
}

NodePath Node::get_path_to(const Node *p_node) const {

	ERR_FAIL_NULL_V(p_node,NodePath());

	if (this==p_node)
		return NodePath(".");
	
	Set<const Node*> visited;
	
	const Node *n=this;
	
	while(n) {
	
		visited.insert(n);
		n=n->data.parent;
	}

	const Node *common_parent=p_node;
	
	while(common_parent) {
	
		if (visited.has(common_parent))
			break;
		common_parent=common_parent->data.parent;
	}
	
	ERR_FAIL_COND_V(!common_parent,NodePath()); //nodes not in the same tree
	
	visited.clear();
	
	Vector<StringName> path;
	
	n=p_node;
	
	while(n!=common_parent) {
	
		path.push_back( n->get_name() );
		n=n->data.parent;
	}
	
	n=this;
	StringName up=String("..");	
	
	while(n!=common_parent) {
	
		path.push_back( up );
		n=n->data.parent;
	}
	
	path.invert();
	
	return NodePath(path,false);	
}

NodePath Node::get_path() const {
	
	ERR_FAIL_COND_V(!is_inside_tree(),NodePath());
	const Node *n = this;
	
	Vector<StringName> path;
	
	while(n) {
		path.push_back(n->get_name());
		n=n->data.parent;
	}	
	
	path.invert();
	
	return NodePath( path, true );
}

bool Node::is_in_group(const StringName& p_identifier) const {

	return data.grouped.has(p_identifier);
}

void Node::add_to_group(const StringName& p_identifier,bool p_persistent) {
	
	ERR_FAIL_COND(!p_identifier.operator String().length());
	
	if (data.grouped.has(p_identifier))
		return;
	
	GroupData gd;
	
	if (data.tree)
		data.tree->add_to_group(p_identifier,this);

	gd.persistent=p_persistent;		
		
	data.grouped[p_identifier]=gd;

}

void Node::remove_from_group(const StringName& p_identifier) {
	

	ERR_FAIL_COND(!data.grouped.has(p_identifier) );
	
	GroupData *g=data.grouped.getptr(p_identifier);
	
	ERR_FAIL_COND(!g);
	
	if (data.tree)
		data.tree->remove_from_group(p_identifier,this);

	data.grouped.erase(p_identifier);	

}

Array Node::_get_groups() const {

	Array groups;
	List<GroupInfo> gi;
	get_groups(&gi);
	for (List<GroupInfo>::Element *E=gi.front();E;E=E->next()) {
		groups.push_back(E->get().name);
	}

	return groups;
}

void Node::get_groups(List<GroupInfo> *p_groups) const {

	const StringName *K=NULL;
	
	while ((K=data.grouped.next(K))) {
		
		GroupInfo gi;
		gi.name=*K;
		gi.persistent=data.grouped[*K].persistent;
		p_groups->push_back(gi);
	}
	
}


void Node::_print_tree(const Node *p_node) {

	print_line(String(p_node->get_path_to(this)));
	for (int i=0;i<data.children.size();i++)
		data.children[i]->_print_tree(p_node);
}

void Node::print_tree() {

	_print_tree(this);
}


void Node::_propagate_reverse_notification(int p_notification) {
		
	data.blocked++;		
	for (int i=data.children.size()-1;i>=0;i--) {
		
		data.children[i]->_propagate_reverse_notification(p_notification);	
	}
	
	notification(p_notification,true);
	data.blocked--;
}

void Node::_propagate_deferred_notification(int p_notification, bool p_reverse) {

	ERR_FAIL_COND(!is_inside_tree());

	data.blocked++;

	if (!p_reverse)
		MessageQueue::get_singleton()->push_notification(this,p_notification);

	for (int i=0;i<data.children.size();i++) {

		data.children[i]->_propagate_deferred_notification(p_notification,p_reverse);
	}

	if (p_reverse)
		MessageQueue::get_singleton()->push_notification(this,p_notification);

	data.blocked--;
}

void Node::propagate_notification(int p_notification) {
	
	data.blocked++;		
	notification(p_notification);
	
	for (int i=0;i<data.children.size();i++) {
		
		data.children[i]->propagate_notification(p_notification);	
	}
	data.blocked--;		
}


void Node::_propagate_replace_owner(Node *p_owner,Node* p_by_owner) {	
	if (get_owner()==p_owner)
		set_owner(p_by_owner);
	
	data.blocked++;
	for (int i=0;i<data.children.size();i++) 
		data.children[i]->_propagate_replace_owner(p_owner,p_by_owner);
	data.blocked--;		
}

int Node::get_index() const {
	
	return data.pos;
}
void Node::remove_and_skip() {
	
	ERR_FAIL_COND(!data.parent);
		
	Node *new_owner=get_owner();
	
	List<Node*> children;
	
	while(true) {
	
		bool clear=true;
		for (int i=0;i<data.children.size();i++) {
			if (!data.children[i]->get_owner())
				continue;
				
			remove_child(data.children[i]);
			data.children[i]->_propagate_replace_owner(this,NULL);
			children.push_back(data.children[i]);	
			clear=false;
			break;
		}
		
		if (clear)
			break;
	}
				
	while(!children.empty()) {
		
		Node *c=children.front()->get();
		data.parent->add_child(c);
		c->_propagate_replace_owner(NULL,new_owner);		
		children.pop_front();
	}
	
	data.parent->remove_child(this);
}

void Node::set_filename(const String& p_filename) {
	
	data.filename=p_filename;
}
String Node::get_filename() const {
	
	return data.filename;
}

void Node::set_editable_instance(Node* p_node,bool p_editable) {

	ERR_FAIL_NULL(p_node);
	ERR_FAIL_COND(!is_a_parent_of(p_node));
	NodePath p = get_path_to(p_node);
	if (!p_editable)
		data.editable_instances.erase(p);
	else
		data.editable_instances[p]=true;

}

bool Node::is_editable_instance(Node *p_node) const {

	if (!p_node)
		return false; //easier, null is never editable :)
	ERR_FAIL_COND_V(!is_a_parent_of(p_node),false);
	NodePath p = get_path_to(p_node);
	return data.editable_instances.has(p);
}


#if 0

void Node::generate_instance_state() {

	List<PropertyInfo> properties;
	get_property_list(&properties);

	data.instance_state.clear();

	for( List<PropertyInfo>::Element *E=properties.front();E;E=E->next() ) {

		PropertyInfo &pi=E->get();
		if ((pi.usage&PROPERTY_USAGE_NO_INSTANCE_STATE) || !(pi.usage&PROPERTY_USAGE_EDITOR) || !(pi.usage&PROPERTY_USAGE_STORAGE))
			continue;

		data.instance_state[pi.name]=get(pi.name);
	}

	List<GroupInfo> groups;
	get_groups(&groups);
	for(List<GroupInfo>::Element *E=groups.front();E;E=E->next()) {

		if (!E->get().persistent)
			continue;
		data.instance_groups.push_back(E->get().name);
	}

	List<MethodInfo> signal_list;

	get_signal_list(&signal_list);

	for(List<MethodInfo>::Element *E=signal_list.front();E;E=E->next()) {

		StringName name = E->get().name;
		List<Connection> connections;
		get_signal_connection_list(name,&connections);

		for(List<Connection>::Element *F=connections.front();F;F=F->next()) {

			if (F->get().flags&CONNECT_PERSIST)
				data.instance_connections.push_back(F->get());
		}

	}
}

Dictionary Node::get_instance_state() const {

	return data.instance_state;
}

#endif

void Node::set_scene_instance_state(const Ref<SceneState>& p_state) {

	data.instance_state=p_state;
}

Ref<SceneState> Node::get_scene_instance_state() const{

	return data.instance_state;
}

void Node::set_scene_inherited_state(const Ref<SceneState>& p_state) {

	data.inherited_state=p_state;
}

Ref<SceneState> Node::get_scene_inherited_state() const{

	return data.inherited_state;
}

void Node::set_scene_instance_load_placeholder(bool p_enable) {

	data.use_placeholder=p_enable;
}

bool Node::get_scene_instance_load_placeholder() const{

	return data.use_placeholder;
}

int Node::get_position_in_parent() const {

	return data.pos;
}



Node *Node::duplicate(bool p_use_instancing) const {


	Node *node=NULL;

	bool instanced=false;

	if (p_use_instancing && get_filename()!=String()) {

		Ref<PackedScene> res = ResourceLoader::load(get_filename());
		ERR_FAIL_COND_V(res.is_null(),NULL);
		node=res->instance();
		ERR_FAIL_COND_V(!node,NULL);

		instanced=true;

	} else {

		Object *obj = ObjectTypeDB::instance(get_type());
		ERR_FAIL_COND_V(!obj,NULL);
		node = obj->cast_to<Node>();
		if (!node)
			memdelete(obj);
		ERR_FAIL_COND_V(!node,NULL);
	}


	if (get_filename()!="") { //an instance
		node->set_filename(get_filename());
	}

	List<PropertyInfo> plist;

	get_property_list(&plist);

	for(List<PropertyInfo>::Element *E=plist.front();E;E=E->next()) {

		if (!(E->get().usage&PROPERTY_USAGE_STORAGE))
			continue;
		String name = E->get().name;
		node->set( name, get(name) );

	}

	node->set_name(get_name());

	for(int i=0;i<get_child_count();i++) {

		if (get_child(i)->data.parent_owned)
			continue;
		if (instanced && get_child(i)->data.owner==this)
			continue; //part of instance

		Node *dup = get_child(i)->duplicate(p_use_instancing);
		if (!dup) {

			memdelete(node);
			return NULL;
		}

		node->add_child(dup);
	}

	return node;
}


void Node::_duplicate_and_reown(Node* p_new_parent, const Map<Node*,Node*>& p_reown_map) const {

	if (get_owner()!=get_parent()->get_owner())
		return;

	Node *node=NULL;

	if (get_filename()!="") {

		Ref<PackedScene> res = ResourceLoader::load(get_filename());
		ERR_FAIL_COND(res.is_null());
		node=res->instance();
		ERR_FAIL_COND(!node);
	} else {

		Object *obj = ObjectTypeDB::instance(get_type());
		if (!obj) {
			print_line("could not duplicate: "+String(get_type()));
		}
		ERR_FAIL_COND(!obj);
		node = obj->cast_to<Node>();
		if (!node)
			memdelete(obj);
	}


	List<PropertyInfo> plist;

	get_property_list(&plist);

	for(List<PropertyInfo>::Element *E=plist.front();E;E=E->next()) {

		if (!(E->get().usage&PROPERTY_USAGE_STORAGE))
			continue;
		String name = E->get().name;
		node->set( name, get(name) );

	}

	node->set_name(get_name());
	p_new_parent->add_child(node);

	Node *owner=get_owner();

	if (p_reown_map.has(owner))
		owner=p_reown_map[owner];


	if (owner) {
		NodePath p = get_path_to(owner);
		if (owner!=this) {
			Node *new_owner = node->get_node(p);
			if (new_owner) {
				node->set_owner(new_owner);
			}
		}
	}

	for(int i=0;i<get_child_count();i++) {

		get_child(i)->_duplicate_and_reown(node,p_reown_map);
	}

}


void Node::_duplicate_signals(const Node* p_original,Node* p_copy) const {

	if (this!=p_original && get_owner()!=p_original)
		return;

	List<Connection> conns;
	get_all_signal_connections(&conns);

	for (List<Connection>::Element *E=conns.front();E;E=E->next()) {

		if (E->get().flags&CONNECT_PERSIST) {
			//user connected
			NodePath p = p_original->get_path_to(this);
			Node *copy = p_copy->get_node(p);

			Node *target = E->get().target->cast_to<Node>();
			if (!target)
				continue;
			NodePath ptarget = p_original->get_path_to(target);
			Node *copytarget = p_copy->get_node(ptarget);

			if (copy && copytarget) {
				copy->connect(E->get().signal,copytarget,E->get().method,E->get().binds,CONNECT_PERSIST);
			}
		}
	}

	for(int i=0;i<get_child_count();i++) {
		get_child(i)->_duplicate_signals(p_original,p_copy);
	}

}


Node *Node::duplicate_and_reown(const Map<Node*,Node*>& p_reown_map) const {


	ERR_FAIL_COND_V(get_filename()!="",NULL);

	Node *node=NULL;

	Object *obj = ObjectTypeDB::instance(get_type());
	if (!obj) {
		print_line("could not duplicate: "+String(get_type()));
	}
	ERR_FAIL_COND_V(!obj,NULL);
	node = obj->cast_to<Node>();
	if (!node)
		memdelete(obj);
	ERR_FAIL_COND_V(!node,NULL);

	node->set_name(get_name());

	List<PropertyInfo> plist;

	get_property_list(&plist);

	for(List<PropertyInfo>::Element *E=plist.front();E;E=E->next()) {

		if (!(E->get().usage&PROPERTY_USAGE_STORAGE))
			continue;
		String name = E->get().name;
		node->set( name, get(name) );

	}


	for(int i=0;i<get_child_count();i++) {

		get_child(i)->_duplicate_and_reown(node,p_reown_map);
	}

	_duplicate_signals(this,node);
	return node;

}

static void find_owned_by(Node* p_by, Node* p_node, List<Node*> *p_owned) {


	if (p_node->get_owner()==p_by)
		p_owned->push_back(p_node);

	for(int i=0;i<p_node->get_child_count();i++) {

		find_owned_by(p_by,p_node->get_child(i),p_owned);
	}
}

struct _NodeReplaceByPair {

	String name;
	Variant value;
};

void Node::replace_by(Node* p_node,bool p_keep_data) {

	ERR_FAIL_NULL(p_node);
	ERR_FAIL_COND(p_node->data.parent);

	List<Node*> owned = data.owned;
	List<Node*> owned_by_owner;
	Node *owner = (data.owner==this)?p_node:data.owner;

	List<_NodeReplaceByPair> replace_data;

	if (p_keep_data) {

		List<PropertyInfo> plist;
		get_property_list(&plist);

		for(List<PropertyInfo>::Element *E=plist.front();E;E=E->next() ) {

			_NodeReplaceByPair rd;
			if (!(E->get().usage&PROPERTY_USAGE_STORAGE))
				continue;
			rd.name=E->get().name;
			rd.value=get(rd.name);
		}
	}

	if (data.owner) {
		for(int i=0;i<get_child_count();i++)
			find_owned_by(data.owner,get_child(i),&owned_by_owner);
	}

	Node *parent = data.parent;
	int pos_in_parent = data.pos;

	if (data.parent) {

		parent->remove_child(this);
		parent->add_child(p_node);
		parent->move_child(p_node,pos_in_parent);
	}

	while(get_child_count()) {

		Node * child = get_child(0);
		remove_child(child);
		p_node->add_child(child);
	}

	p_node->set_owner(owner);
	for(int i=0;i<owned.size();i++)
		owned[i]->set_owner(p_node);

	for(int i=0;i<owned_by_owner.size();i++)
		owned_by_owner[i]->set_owner(owner);

	p_node->set_filename(get_filename());

	for (List<_NodeReplaceByPair>::Element *E=replace_data.front();E;E=E->next()) {

		p_node->set(E->get().name,E->get().value);
	}

}

Vector<Variant> Node::make_binds(VARIANT_ARG_DECLARE) {


	Vector<Variant> ret;

	if (p_arg1.get_type()==Variant::NIL)
		return ret;
	else
		ret.push_back(p_arg1);

	if (p_arg2.get_type()==Variant::NIL)
		return ret;
	else
		ret.push_back(p_arg2);


	if (p_arg3.get_type()==Variant::NIL)
		return ret;
	else
		ret.push_back(p_arg3);

	if (p_arg4.get_type()==Variant::NIL)
		return ret;
	else
		ret.push_back(p_arg4);

	if (p_arg5.get_type()==Variant::NIL)
		return ret;
	else
		ret.push_back(p_arg5);

	return ret;
}



bool Node::has_node_and_resource(const NodePath& p_path) const {

	if (!has_node(p_path))
		return false;
	Node *node = get_node(p_path);

	if (p_path.get_subname_count()) {

		RES r;
		for(int j=0;j<p_path.get_subname_count();j++) {
			r = j==0 ? node->get(p_path.get_subname(j)) : r->get(p_path.get_subname(j));
			if (r.is_null())
				return false;
		}
	}


	return true;
}


Array Node::_get_node_and_resource(const NodePath& p_path) {

	Node *node;
	RES res;
	node = get_node_and_resource(p_path,res);
	Array result;

	if (node)
		result.push_back(node);
	else
		result.push_back(Variant());

	if (res.is_valid())
		result.push_back(res);
	else
		result.push_back(Variant());

	return result;
}

Node *Node::get_node_and_resource(const NodePath& p_path,RES& r_res) const {

	Node *node = get_node(p_path);
	r_res = RES();
	if (!node)
		return NULL;

	if (p_path.get_subname_count()) {

		for(int j=0;j<p_path.get_subname_count();j++) {
			r_res = j==0 ? node->get(p_path.get_subname(j)) : r_res->get(p_path.get_subname(j));
			ERR_FAIL_COND_V( r_res.is_null(), node );
		}
	}

	return node;
}

void Node::_set_tree(SceneTree *p_tree) {

	SceneTree *tree_changed_a=NULL;
	SceneTree *tree_changed_b=NULL;

//	ERR_FAIL_COND(p_scene && data.parent && !data.parent->data.scene); //nobug if both are null

	if (data.tree) {
		_propagate_exit_tree();

		tree_changed_a=data.tree;
	}


	data.tree=p_tree;

	if (data.tree) {


		_propagate_enter_tree();
		_propagate_ready(); //reverse_notification(NOTIFICATION_READY);

		tree_changed_b=data.tree;

	}

	if (tree_changed_a)
		tree_changed_a->tree_changed();
	if (tree_changed_b)
		tree_changed_b->tree_changed();

}


static void _Node_debug_sn(Object *p_obj) {

	Node *n = p_obj->cast_to<Node>();
	if (!n)
		return;

	if (n->is_inside_tree())
		return;

	Node *p=n;
	while(p->get_parent()) {
		p=p->get_parent();
	}

	String path;
	if (p==n)
		path=n->get_name();
	else
		path=String(p->get_name())+"/"+p->get_path_to(n);
	print_line(itos(p_obj->get_instance_ID())+"- Stray Node: "+path+" (Type: "+n->get_type()+")");

}

void Node::_print_stray_nodes() {

	print_stray_nodes();
}

void Node::print_stray_nodes() {

#ifdef DEBUG_ENABLED

	ObjectDB::debug_objects(_Node_debug_sn);
#endif
}

void Node::queue_delete() {

	ERR_FAIL_COND( !is_inside_tree() );
	get_tree()->queue_delete(this);
}

Array Node::_get_children() const {

	Array arr;
	int cc = get_child_count();
	arr.resize(cc);
	for(int i=0;i<cc;i++)
		arr[i]=get_child(i);

	return arr;
}

#ifdef TOOLS_ENABLED
void Node::set_import_path(const NodePath& p_import_path) {


	data.import_path=p_import_path;
}

NodePath Node::get_import_path() const {

	return data.import_path;
}

#endif

static void _add_nodes_to_options(const Node *p_base,const Node *p_node,List<String>*r_options) {

	if (p_node!=p_base && !p_node->get_owner())
		return;
	String n = p_base->get_path_to(p_node);
	r_options->push_back("\""+n+"\"");
	for(int i=0;i<p_node->get_child_count();i++) {
		_add_nodes_to_options(p_base,p_node->get_child(i),r_options);
	}
}

void Node::get_argument_options(const StringName& p_function,int p_idx,List<String>*r_options) const {

	String pf=p_function;
	if ((pf=="has_node" || pf=="get_node") && p_idx==0) {

		_add_nodes_to_options(this,this,r_options);
	}
	Object::get_argument_options(p_function,p_idx,r_options);
}


void Node::clear_internal_tree_resource_paths() {

	clear_internal_resource_paths();
	for(int i=0;i<data.children.size();i++) {
		data.children[i]->clear_internal_tree_resource_paths();
	}

}

void Node::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_name","name"),&Node::set_name);
	ObjectTypeDB::bind_method(_MD("get_name"),&Node::get_name);
	ObjectTypeDB::bind_method(_MD("add_child","node:Node"),&Node::add_child);
	ObjectTypeDB::bind_method(_MD("remove_child","node:Node"),&Node::remove_child);
	//ObjectTypeDB::bind_method(_MD("remove_and_delete_child","node:Node"),&Node::remove_and_delete_child);
	ObjectTypeDB::bind_method(_MD("get_child_count"),&Node::get_child_count);
	ObjectTypeDB::bind_method(_MD("get_children"),&Node::_get_children);
	ObjectTypeDB::bind_method(_MD("get_child:Node","idx"),&Node::get_child);
	ObjectTypeDB::bind_method(_MD("has_node","path"),&Node::has_node);
	ObjectTypeDB::bind_method(_MD("get_node:Node","path"),&Node::get_node);
	ObjectTypeDB::bind_method(_MD("get_parent:Parent"),&Node::get_parent);
	ObjectTypeDB::bind_method(_MD("find_node:Node","mask","recursive","owned"),&Node::find_node,DEFVAL(true),DEFVAL(true));
	ObjectTypeDB::bind_method(_MD("has_node_and_resource","path"),&Node::has_node_and_resource);
	ObjectTypeDB::bind_method(_MD("get_node_and_resource","path"),&Node::_get_node_and_resource);

	ObjectTypeDB::bind_method(_MD("is_inside_tree"),&Node::is_inside_tree);
	ObjectTypeDB::bind_method(_MD("is_a_parent_of","node:Node"),&Node::is_a_parent_of);
	ObjectTypeDB::bind_method(_MD("is_greater_than","node:Node"),&Node::is_greater_than);
	ObjectTypeDB::bind_method(_MD("get_path"),&Node::get_path);
	ObjectTypeDB::bind_method(_MD("get_path_to","node:Node"),&Node::get_path_to);
	ObjectTypeDB::bind_method(_MD("add_to_group","group"),&Node::add_to_group,DEFVAL(false));
	ObjectTypeDB::bind_method(_MD("remove_from_group","group"),&Node::remove_from_group);
	ObjectTypeDB::bind_method(_MD("is_in_group","group"),&Node::is_in_group);
	ObjectTypeDB::bind_method(_MD("move_child","child_node:Node","to_pos"),&Node::move_child);
	ObjectTypeDB::bind_method(_MD("get_groups"),&Node::_get_groups);
	ObjectTypeDB::bind_method(_MD("raise"),&Node::raise);
	ObjectTypeDB::bind_method(_MD("set_owner","owner:Node"),&Node::set_owner);
	ObjectTypeDB::bind_method(_MD("get_owner:Node"),&Node::get_owner);
	ObjectTypeDB::bind_method(_MD("remove_and_skip"),&Node::remove_and_skip);
	ObjectTypeDB::bind_method(_MD("get_index"),&Node::get_index);
	ObjectTypeDB::bind_method(_MD("print_tree"),&Node::print_tree);
	ObjectTypeDB::bind_method(_MD("set_filename","filename"),&Node::set_filename);
	ObjectTypeDB::bind_method(_MD("get_filename"),&Node::get_filename);
	ObjectTypeDB::bind_method(_MD("propagate_notification","what"),&Node::propagate_notification);
	ObjectTypeDB::bind_method(_MD("set_fixed_process","enable"),&Node::set_fixed_process);
	ObjectTypeDB::bind_method(_MD("get_fixed_process_delta_time"),&Node::get_fixed_process_delta_time);
	ObjectTypeDB::bind_method(_MD("is_fixed_processing"),&Node::is_fixed_processing);
	ObjectTypeDB::bind_method(_MD("set_process","enable"),&Node::set_process);
	ObjectTypeDB::bind_method(_MD("get_process_delta_time"),&Node::get_process_delta_time);
	ObjectTypeDB::bind_method(_MD("is_processing"),&Node::is_processing);
	ObjectTypeDB::bind_method(_MD("set_process_input","enable"),&Node::set_process_input);
	ObjectTypeDB::bind_method(_MD("is_processing_input"),&Node::is_processing_input);
	ObjectTypeDB::bind_method(_MD("set_process_unhandled_input","enable"),&Node::set_process_unhandled_input);
	ObjectTypeDB::bind_method(_MD("is_processing_unhandled_input"),&Node::is_processing_unhandled_input);
	ObjectTypeDB::bind_method(_MD("set_process_unhandled_key_input","enable"),&Node::set_process_unhandled_key_input);
	ObjectTypeDB::bind_method(_MD("is_processing_unhandled_key_input"),&Node::is_processing_unhandled_key_input);
	ObjectTypeDB::bind_method(_MD("set_pause_mode","mode"),&Node::set_pause_mode);
	ObjectTypeDB::bind_method(_MD("get_pause_mode"),&Node::get_pause_mode);
	ObjectTypeDB::bind_method(_MD("can_process"),&Node::can_process);
	ObjectTypeDB::bind_method(_MD("print_stray_nodes"),&Node::_print_stray_nodes);
	ObjectTypeDB::bind_method(_MD("get_position_in_parent"),&Node::get_position_in_parent);

	ObjectTypeDB::bind_method(_MD("get_tree:SceneTree"),&Node::get_tree);

	ObjectTypeDB::bind_method(_MD("duplicate:Node","use_instancing"),&Node::duplicate,DEFVAL(false));
	ObjectTypeDB::bind_method(_MD("replace_by","node:Node","keep_data"),&Node::replace_by,DEFVAL(false));

	ObjectTypeDB::bind_method(_MD("get_viewport"),&Node::get_viewport);

	ObjectTypeDB::bind_method(_MD("queue_free"),&Node::queue_delete);
#ifdef TOOLS_ENABLED
	ObjectTypeDB::bind_method(_MD("_set_import_path","import_path"),&Node::set_import_path);
	ObjectTypeDB::bind_method(_MD("_get_import_path"),&Node::get_import_path);
	ADD_PROPERTYNZ( PropertyInfo(Variant::NODE_PATH,"_import_path",PROPERTY_HINT_NONE,"",PROPERTY_USAGE_NOEDITOR),_SCS("_set_import_path"),_SCS("_get_import_path"));

#endif

	BIND_CONSTANT( NOTIFICATION_ENTER_TREE );
	BIND_CONSTANT( NOTIFICATION_EXIT_TREE );
	BIND_CONSTANT( NOTIFICATION_MOVED_IN_PARENT );
	//BIND_CONSTANT( NOTIFICATION_PARENT_DECONFIGURED );
	BIND_CONSTANT( NOTIFICATION_READY );
	BIND_CONSTANT( NOTIFICATION_FIXED_PROCESS );
	BIND_CONSTANT( NOTIFICATION_PROCESS );
	BIND_CONSTANT( NOTIFICATION_PARENTED );
	BIND_CONSTANT( NOTIFICATION_UNPARENTED );
	BIND_CONSTANT( NOTIFICATION_PAUSED );
	BIND_CONSTANT( NOTIFICATION_UNPAUSED );


	BIND_CONSTANT( PAUSE_MODE_INHERIT );
	BIND_CONSTANT( PAUSE_MODE_STOP );
	BIND_CONSTANT( PAUSE_MODE_PROCESS );

	ADD_SIGNAL( MethodInfo("renamed") );
	ADD_SIGNAL( MethodInfo("enter_tree") );
	ADD_SIGNAL( MethodInfo("exit_tree") );

//	ADD_PROPERTYNZ( PropertyInfo( Variant::BOOL, "process/process" ),_SCS("set_process"),_SCS("is_processing") );
//	ADD_PROPERTYNZ( PropertyInfo( Variant::BOOL, "process/fixed_process" ), _SCS("set_fixed_process"),_SCS("is_fixed_processing") );
	//ADD_PROPERTYNZ( PropertyInfo( Variant::BOOL, "process/input" ), _SCS("set_process_input"),_SCS("is_processing_input" ) );
	//ADD_PROPERTYNZ( PropertyInfo( Variant::BOOL, "process/unhandled_input" ), _SCS("set_process_unhandled_input"),_SCS("is_processing_unhandled_input" ) );
	ADD_PROPERTYNZ( PropertyInfo( Variant::INT, "process/pause_mode",PROPERTY_HINT_ENUM,"Inherit,Stop,Process" ), _SCS("set_pause_mode"),_SCS("get_pause_mode" ) );

	BIND_VMETHOD( MethodInfo("_process",PropertyInfo(Variant::REAL,"delta")) );
	BIND_VMETHOD( MethodInfo("_fixed_process",PropertyInfo(Variant::REAL,"delta")) );
	BIND_VMETHOD( MethodInfo("_enter_tree") );
	BIND_VMETHOD( MethodInfo("_exit_tree") );
	BIND_VMETHOD( MethodInfo("_ready") );
	BIND_VMETHOD( MethodInfo("_input",PropertyInfo(Variant::INPUT_EVENT,"event")) );
	BIND_VMETHOD( MethodInfo("_unhandled_input",PropertyInfo(Variant::INPUT_EVENT,"event")) );
	BIND_VMETHOD( MethodInfo("_unhandled_key_input",PropertyInfo(Variant::INPUT_EVENT,"key_event")) );

	//ObjectTypeDB::bind_method(_MD("get_child",&Node::get_child,PH("index")));
	//ObjectTypeDB::bind_method(_MD("get_node",&Node::get_node,PH("path")));
}


Node::Node() {
	
	data.pos=-1;
	data.depth=-1;
	data.blocked=0;
	data.parent=NULL;
	data.tree=NULL;
	data.fixed_process=false;
	data.idle_process=false;
	data.inside_tree=false;

	data.owner=NULL;
	data.OW=NULL;
	data.input=false;
	data.unhandled_input=false;
	data.unhandled_key_input=false;
	data.pause_mode=PAUSE_MODE_INHERIT;
	data.pause_owner=NULL;
	data.parent_owned=false;
	data.in_constructor=true;
	data.viewport=NULL;
	data.use_placeholder=false;
}

Node::~Node() {
	

	data.grouped.clear();
	data.owned.clear();
	data.children.clear();
	
	ERR_FAIL_COND(data.parent);
	ERR_FAIL_COND(data.children.size());


}


////////////////////////////////
