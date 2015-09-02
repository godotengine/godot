#include "multi_node_edit.h"
#include "editor_node.h"

bool MultiNodeEdit::_set(const StringName& p_name, const Variant& p_value){

	Node *es = EditorNode::get_singleton()->get_edited_scene();
	if (!es)
		return false;

	UndoRedo *ur=EditorNode::get_singleton()->get_undo_redo();

	ur->create_action("MultiNode Set "+String(p_name));
	for (const List<NodePath>::Element *E=nodes.front();E;E=E->next()) {

		if (!es->has_node(E->get()))
			continue;

		Node*n=es->get_node(E->get());
		if (!n)
			continue;

		ur->add_do_property(n,p_name,p_value);
		ur->add_undo_property(n,p_name,n->get(p_name));

	}

	ur->commit_action();
	return true;
}

bool MultiNodeEdit::_get(const StringName& p_name,Variant &r_ret) const {

	Node *es = EditorNode::get_singleton()->get_edited_scene();
	if (!es)
		return false;

	for (const List<NodePath>::Element *E=nodes.front();E;E=E->next()) {

		if (!es->has_node(E->get()))
			continue;

		const Node*n=es->get_node(E->get());
		if (!n)
			continue;

		bool found;
		r_ret=n->get(p_name,&found);
		if (found)
			return true;

	}

	return false;
}

void MultiNodeEdit::_get_property_list( List<PropertyInfo> *p_list) const{

	HashMap<String,PLData> usage;

	Node *es = EditorNode::get_singleton()->get_edited_scene();
	if (!es)
		return;

	int nc=0;

	List<PLData*> datas;

	for (const List<NodePath>::Element *E=nodes.front();E;E=E->next()) {

		if (!es->has_node(E->get()))
			continue;

		Node*n=es->get_node(E->get());
		if (!n)
			continue;

		List<PropertyInfo> plist;
		n->get_property_list(&plist,true);

		for(List<PropertyInfo>::Element *F=plist.front();F;F=F->next()) {

			if (!usage.has(F->get().name)) {
				PLData pld;
				pld.uses=0;
				pld.info=F->get();
				usage[F->get().name]=pld;
				datas.push_back(usage.getptr(F->get().name));
			}

			usage[F->get().name].uses++;
		}

		nc++;
	}

	for (List<PLData*>::Element *E=datas.front();E;E=E->next()) {

		if (nc==E->get()->uses) {
			p_list->push_back(E->get()->info);
		}
	}


}

void MultiNodeEdit::clear_nodes() {

	nodes.clear();
}

void MultiNodeEdit::add_node(const NodePath& p_node){

	nodes.push_back(p_node);
}

MultiNodeEdit::MultiNodeEdit()
{
}
