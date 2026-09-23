/**************************************************************************/
/*  slime_ai_scene_command_executor.cpp                                  */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietzky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "slime_ai_scene_command_executor.h"

#include "editor/editor_data.h"
#include "editor/editor_node.h"
#include "editor/editor_undo_redo_manager.h"
#include "editor/slime_ai/slime_ai_scene_commands.h"
#include "scene/2d/node_2d.h"
#include "scene/3d/node_3d.h"

namespace SlimeAI {

class ActionWriter {
	EditorUndoRedoManager *editor = nullptr;
	UndoRedo *test = nullptr;

public:
	ActionWriter(UndoRedo *p_test, const String &p_name) {
		editor = EditorNode::get_singleton() ? EditorUndoRedoManager::get_singleton() : nullptr;
		test = p_test;
		if (editor) {
			editor->create_action_for_history(p_name, EditorNode::get_editor_data().get_current_edited_scene_history_id());
		} else {
			test->create_action(p_name);
		}
	}

	template <typename... Args>
	void add_do_method(Object *p_object, const StringName &p_name, Args... p_args) {
		if (editor) {
			editor->add_do_method(p_object, p_name, p_args...);
		} else {
			test->add_do_method(Callable(p_object, p_name).bind(p_args...));
		}
	}

	template <typename... Args>
	void add_undo_method(Object *p_object, const StringName &p_name, Args... p_args) {
		if (editor) {
			editor->add_undo_method(p_object, p_name, p_args...);
		} else {
			test->add_undo_method(Callable(p_object, p_name).bind(p_args...));
		}
	}

	void add_do_property(Object *p_object, const StringName &p_name, const Variant &p_value) {
		if (editor) {
			editor->add_do_property(p_object, p_name, p_value);
		} else {
			test->add_do_property(p_object, p_name, p_value);
		}
	}

	void add_undo_property(Object *p_object, const StringName &p_name, const Variant &p_value) {
		if (editor) {
			editor->add_undo_property(p_object, p_name, p_value);
		} else {
			test->add_undo_property(p_object, p_name, p_value);
		}
	}

	void add_do_reference(Object *p_object) {
		if (editor) {
			editor->add_do_reference(p_object);
		} else {
			test->add_do_reference(p_object);
		}
	}

	void add_undo_reference(Object *p_object) {
		if (editor) {
			editor->add_undo_reference(p_object);
		} else {
			test->add_undo_reference(p_object);
		}
	}

	void commit() {
		if (editor) {
			editor->commit_action();
		} else {
			test->commit_action();
		}
	}
};

static Variant _typed_native(const Dictionary &p_value) {
	const String type = p_value["type"];
	if (type == "bool") {
		return p_value["value"];
	}
	const Array values = p_value["value"];
	if (type == "Vector2") {
		return Vector2(double(values[0]), double(values[1]));
	}
	return Vector3(double(values[0]), double(values[1]), double(values[2]));
}

static void _restore_subtree_owners(ActionWriter &p_action, Node *p_root, Node *p_target) {
	p_action.add_undo_method(p_target, "set_owner", p_root);
	for (int i = 0; i < p_target->get_child_count(false); i++) {
		_restore_subtree_owners(p_action, p_root, p_target->get_child(i, false));
	}
}

static void _clear_subtree_owners(ActionWriter &p_action, Node *p_target) {
	for (int i = 0; i < p_target->get_child_count(false); i++) {
		_clear_subtree_owners(p_action, p_target->get_child(i, false));
	}
	p_action.add_do_method(p_target, "set_owner", (Object *)nullptr);
}

bool execute_scene_operation(Node *p_root, const Dictionary &p_operation, const String &p_operation_id, UndoRedo *p_test_undo, Node *&r_created) {
	r_created = nullptr;
	Node *target = operation_target(p_root, p_operation);
	if (!target) {
		return false;
	}
	const String kind = p_operation["op"];
	if (kind == "create_child") {
		Node *child = String(p_operation["class_name"]) == "Node2D" ? static_cast<Node *>(memnew(Node2D)) : static_cast<Node *>(memnew(Node3D));
		child->set_name(p_operation["name"]);
		child->set_meta("slime_ai_operation_id", p_operation_id);
		const Dictionary properties = p_operation["properties"];
		child->set("position", _typed_native(properties["position"]));
		ActionWriter action(p_test_undo, "Slime AI: Create Node");
		action.add_do_method(target, "add_child", child, true);
		action.add_do_method(child, "set_owner", p_root);
		action.add_do_reference(child);
		action.add_undo_method(child, "set_owner", (Object *)nullptr);
		action.add_undo_method(target, "remove_child", child);
		action.commit();
		r_created = child;
		return true;
	}
	if (kind == "set_property") {
		const StringName property = StringName(p_operation["property"]);
		const Variant before = target->get(property);
		const Variant after = _typed_native(p_operation["value"]);
		ActionWriter action(p_test_undo, "Slime AI: Set Native Property");
		action.add_do_property(target, property, after);
		action.add_undo_property(target, property, before);
		action.commit();
		return true;
	}
	if (kind == "rename_node") {
		const StringName before = target->get_name();
		const StringName after = StringName(p_operation["name"]);
		ActionWriter action(p_test_undo, "Slime AI: Rename Node");
		action.add_do_method(target, "set_name", after);
		action.add_undo_method(target, "set_name", before);
		action.commit();
		return true;
	}
	if (kind == "remove_node") {
		Node *parent = target->get_parent();
		const int index = target->get_index(false);
		ActionWriter action(p_test_undo, "Slime AI: Remove Node");
		_clear_subtree_owners(action, target);
		action.add_do_method(parent, "remove_child", target);
		action.add_undo_method(parent, "add_child", target, true);
		action.add_undo_method(parent, "move_child", target, index);
		_restore_subtree_owners(action, p_root, target);
		action.add_undo_reference(target);
		action.commit();
		return true;
	}
	return false;
}

} // namespace SlimeAI
