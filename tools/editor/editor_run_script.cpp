#include "editor_run_script.h"
#include "editor_node.h"






void EditorScript::add_root_node(Node *p_node) {

	if (!editor) {
		EditorNode::add_io_error("EditorScript::add_root_node : Write your logic in the _run() method.");
		return;
	}

	if (editor->get_edited_scene()) {
		EditorNode::add_io_error("EditorScript::add_root_node : There is an edited scene already.");
		return;
	}

//	editor->set_edited_scene(p_node);
}

Node *EditorScript::get_scene() {

	if (!editor) {
		EditorNode::add_io_error("EditorScript::get_scene : Write your logic in the _run() method.");
		return NULL;
	}

	return editor->get_edited_scene();
}

void EditorScript::_run() {

	Ref<Script> s = get_script();
	ERR_FAIL_COND(!s.is_valid());
	if (!get_script_instance()) {
		EditorNode::add_io_error("Couldn't instance script:\n "+s->get_path()+"\nDid you forget the 'tool' keyword?");
		return;

	}

	Variant::CallError ce;
	ce.error=Variant::CallError::CALL_OK;
	get_script_instance()->call("_run",NULL,0,ce);
	if (ce.error!=Variant::CallError::CALL_OK) {

		EditorNode::add_io_error("Couldn't run script:\n "+s->get_path()+"\nDid you forget the '_run' method?");
	}
}

void EditorScript::set_editor(EditorNode *p_editor) {

	editor=p_editor;
}


void EditorScript::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("add_root_node","node"),&EditorScript::add_root_node);
	ObjectTypeDB::bind_method(_MD("get_scene"),&EditorScript::get_scene);
	BIND_VMETHOD( MethodInfo("_run") );


}

EditorScript::EditorScript() {

	editor=NULL;
}

