/*************************************************************************/
/*  editor_data.cpp                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
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
#include "editor_data.h"
#include "globals.h"
#include "editor_settings.h"
#include "os/dir_access.h"
#include "io/resource_loader.h"

void EditorHistory::_cleanup_history() {

	for(int i=0;i<history.size();i++) {

		bool fail=false;

		for(int j=0;j<history[i].path.size();j++) {
			if (!history[i].path[j].res.is_null())
				continue;

			if (ObjectDB::get_instance(history[i].path[j].object))
				continue; //has isntance, try next

			if (j<=history[i].level) {
				//before or equal level, complete fail
				fail=true;
			} else {
				//after level, clip
				history[i].path.resize(j);
			}

			break;
		}

		if (fail) {
			history.remove(i);
			i--;
		}
	}

	if (current>=history.size())
		current=history.size()-1;
}

void EditorHistory::_add_object(ObjectID p_object,const String& p_property,int p_level_change) {

	Object *obj = ObjectDB::get_instance(p_object);
	ERR_FAIL_COND(!obj);
	Resource *r = obj->cast_to<Resource>();
	Obj o;
	if (r)
		o.res=RES(r);
	o.object=p_object;
	o.property=p_property;

	History h;

	bool has_prev = current>=0 && current<history.size();

	if (has_prev) {
		history.resize(current+1); //clip history to next
	}

	if (p_property!="" && has_prev) {
		//add a sub property
		History &pr = history[current];
		h=pr;
		h.path.resize(h.level+1);
		h.path.push_back(o);
		h.level++;
	} else if (p_level_change!=-1 && has_prev) {
		//add a sub property
		History &pr = history[current];
		h=pr;
		ERR_FAIL_INDEX(p_level_change,h.path.size());
		h.level=p_level_change;
	} else {
		//add a new node
		h.path.push_back(o);
		h.level=0;
	}

	history.push_back(h);
	current++;
}

void EditorHistory::add_object(ObjectID p_object) {

	_add_object(p_object,"",-1);
}

void EditorHistory::add_object(ObjectID p_object,const String& p_subprop) {

	_add_object(p_object,p_subprop,-1);
}

void EditorHistory::add_object(ObjectID p_object,int p_relevel){

	_add_object(p_object,"",p_relevel);
}


bool EditorHistory::next() {


	_cleanup_history();

	if ((current+1)<history.size())
		current++;
	else
		return false;

	return true;
}

bool EditorHistory::previous() {


	_cleanup_history();

	if (current>0)
		current--;
	else
		return false;

	return true;
}

ObjectID EditorHistory::get_current() {


	if (current<0 || current >=history.size())
		return 0;

	History &h=history[current];
	Object *obj = ObjectDB::get_instance(h.path[h.level].object);
	if (!obj)
		return 0;

	return obj->get_instance_ID();
}

int EditorHistory::get_path_size() const {

	if (current<0 || current >=history.size())
		return 0;

	const History &h=history[current];
	return h.path.size();

}

ObjectID EditorHistory::get_path_object(int p_index) const {

	if (current<0 || current >=history.size())
		return 0;

	const History &h=history[current];

	ERR_FAIL_INDEX_V(p_index,h.path.size(),0);

	Object *obj = ObjectDB::get_instance(h.path[p_index].object);
	if (!obj)
		return 0;

	return obj->get_instance_ID();
}

String EditorHistory::get_path_property(int p_index) const {

	if (current<0 || current >=history.size())
		return "";

	const History &h=history[current];

	ERR_FAIL_INDEX_V(p_index,h.path.size(),"");

	return h.path[p_index].property;

}

void EditorHistory::clear() {

	history.clear();
	current=-1;
}

EditorHistory::EditorHistory() {

	current=-1;
}

EditorPlugin* EditorData::get_editor(Object *p_object) {

	for (int i=0;i<editor_plugins.size();i++) {

		if (editor_plugins[i]->has_main_screen() && editor_plugins[i]->handles(p_object))
			return editor_plugins[i];
	}

	return NULL;
}

EditorPlugin* EditorData::get_subeditor(Object *p_object) {

	for (int i=0;i<editor_plugins.size();i++) {

		if (!editor_plugins[i]->has_main_screen() && editor_plugins[i]->handles(p_object))
			return editor_plugins[i];
	}

	return NULL;
}

EditorPlugin* EditorData::get_editor(String p_name) {
	
	for(int i=0;i<editor_plugins.size();i++) {
		
		if (editor_plugins[i]->get_name()==p_name)
			return editor_plugins[i];
	}
	
	return NULL;
}

void EditorData::copy_object_params(Object *p_object) {

	clipboard.clear();

	List<PropertyInfo> pinfo;
	p_object->get_property_list(&pinfo);

	for( List<PropertyInfo>::Element *E=pinfo.front();E;E=E->next()) {

		if (!(E->get().usage&PROPERTY_USAGE_EDITOR))
			continue;

		PropertyData pd;
		pd.name=E->get().name;
		pd.value=p_object->get(pd.name);
		clipboard.push_back(pd);
	}
}

void EditorData::get_editor_breakpoints(List<String> *p_breakpoints) {


	for(int i=0;i<editor_plugins.size();i++) {

		editor_plugins[i]->get_breakpoints(p_breakpoints);
	}


}

Dictionary EditorData::get_editor_states() const {

	Dictionary metadata;
	for(int i=0;i<editor_plugins.size();i++) {

		Dictionary state=editor_plugins[i]->get_state();
		if (state.empty())
			continue;
		metadata[editor_plugins[i]->get_name()]=state;
	}

	return metadata;

}

void EditorData::set_editor_states(const Dictionary& p_states) {

	List<Variant> keys;
	p_states.get_key_list(&keys);

	List<Variant>::Element *E=keys.front();
	for(;E;E=E->next()) {

		String name = E->get();
		int idx=-1;
		for(int i=0;i<editor_plugins.size();i++) {

			if (editor_plugins[i]->get_name()==name) {
				idx=i;
				break;
			}
		}

		if (idx==-1)
			continue;
		editor_plugins[idx]->set_state(p_states[name]);
	}

}

void EditorData::clear_editor_states() {

	for(int i=0;i<editor_plugins.size();i++) {

		editor_plugins[i]->clear();
	}

}

void EditorData::save_editor_external_data() {

	for(int i=0;i<editor_plugins.size();i++) {

		editor_plugins[i]->save_external_data();
	}
}

void EditorData::apply_changes_in_editors() {

	for(int i=0;i<editor_plugins.size();i++) {

		editor_plugins[i]->apply_changes();
	}

}

void EditorData::save_editor_global_states() {

	for(int i=0;i<editor_plugins.size();i++) {

		editor_plugins[i]->save_global_state();
	}
}

void EditorData::restore_editor_global_states(){

	for(int i=0;i<editor_plugins.size();i++) {

		editor_plugins[i]->restore_global_state();
	}

}


void EditorData::paste_object_params(Object *p_object) {


	for( List<PropertyData>::Element *E=clipboard.front();E;E=E->next()) {

		p_object->set( E->get().name, E->get().value);
	}

}


UndoRedo &EditorData::get_undo_redo() {

	return undo_redo;
}

void EditorData::remove_editor_plugin(EditorPlugin *p_plugin) {

	p_plugin->undo_redo=NULL;
	editor_plugins.erase(p_plugin);

}

void EditorData::add_editor_plugin(EditorPlugin *p_plugin) {

	p_plugin->undo_redo=&undo_redo;
	editor_plugins.push_back(p_plugin);
}



void EditorData::add_custom_type(const String& p_type, const String& p_inherits,const Ref<Script>& p_script,const Ref<Texture>& p_icon ) {

	ERR_FAIL_COND(p_script.is_null());
	CustomType ct;
	ct.name=p_type;
	ct.icon=p_icon;
	ct.script=p_script;
	if (!custom_types.has(p_inherits)) {
		custom_types[p_inherits]=Vector<CustomType>();
	}

	custom_types[p_inherits].push_back(ct);
}

void EditorData::remove_custom_type(const String& p_type){


	for (Map<String,Vector<CustomType> >::Element *E=custom_types.front();E;E=E->next()) {

		for(int i=0;i<E->get().size();i++) {
			if (E->get()[i].name==p_type) {
				E->get().remove(i);
				if (E->get().empty()) {
					custom_types.erase(E->key());
				}
				return;
			}
		}
	}

}


EditorData::EditorData() {

//	load_imported_scenes_from_globals();
}

///////////
void EditorSelection::_node_removed(Node *p_node) {

	if (!selection.has(p_node))
		return;

	Object *meta = selection[p_node];
	if (meta)
		memdelete(meta);
	selection.erase(p_node);
	changed=true;
	nl_changed=true;
}

void EditorSelection::add_node(Node *p_node) {

	if (selection.has(p_node))
		return;

	changed=true;
	nl_changed=true;
	Object *meta=NULL;
	for(List<Object*>::Element *E=editor_plugins.front();E;E=E->next()) {

		meta = E->get()->call("_get_editor_data",p_node);
		if (meta) {
			break;
		}

	}
	selection[p_node]=meta;

	p_node->connect("exit_tree",this,"_node_removed",varray(p_node),CONNECT_ONESHOT);

	//emit_signal("selection_changed");
}

void EditorSelection::remove_node(Node *p_node) {

	if (!selection.has(p_node))
		return;

	changed=true;
	nl_changed=true;
	Object *meta = selection[p_node];
	if (meta)
		memdelete(meta);
	selection.erase(p_node);
	p_node->disconnect("exit_tree",this,"_node_removed");
	//emit_signal("selection_changed");
}
bool EditorSelection::is_selected(Node * p_node) const {

	return selection.has(p_node);
}



void EditorSelection::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("_node_removed"),&EditorSelection::_node_removed);
	ObjectTypeDB::bind_method(_MD("clear"),&EditorSelection::clear);
	ObjectTypeDB::bind_method(_MD("add_node"),&EditorSelection::add_node);
	ADD_SIGNAL( MethodInfo("selection_changed") );

}

void EditorSelection::add_editor_plugin(Object *p_object) {

	editor_plugins.push_back(p_object);
}

void EditorSelection::_update_nl() {

	if (!nl_changed)
		return;

	selected_node_list.clear();

	for (Map<Node*,Object*>::Element *E=selection.front();E;E=E->next()) {


		Node *parent = E->key();
		parent=parent->get_parent();
		bool skip=false;
		while (parent) {
			if (selection.has(parent)) {
				skip=true;
				break;
			}
			parent=parent->get_parent();
		}

		if (skip)
			continue;
		selected_node_list.push_back(E->key());
	}

	nl_changed=true;
}

void EditorSelection::update() {

	_update_nl();

	if (!changed)
		return;
	emit_signal("selection_changed");
	changed=false;


}

List<Node*>& EditorSelection::get_selected_node_list() {

	if (changed)
		update();
	else
		_update_nl();
	return selected_node_list;
}




void EditorSelection::clear() {

	while( !selection.empty() ) {

		remove_node(selection.front()->key());
	}


	changed=true;
	nl_changed=true;

}
EditorSelection::EditorSelection() {

	changed=false;
	nl_changed=false;

}

EditorSelection::~EditorSelection() {

	clear();
}
