/*************************************************************************/
/*  multi_node_edit.cpp                                                  */
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

#include "multi_node_edit.h"

#include "core/math/math_fieldwise.h"
#include "editor_node.h"

bool MultiNodeEdit::_set(const StringName &p_name, const Variant &p_value) {
	return _set_impl(p_name, p_value, "");
}

bool MultiNodeEdit::_set_impl(const StringName &p_name, const Variant &p_value, const String &p_field) {
	Node *es = EditorNode::get_singleton()->get_edited_scene();
	if (!es) {
		return false;
	}

	String name = p_name;

	if (name == "scripts") { // script set is intercepted at object level (check Variant Object::get() ) ,so use a different name
		name = "script";
	}

	Node *node_path_target = nullptr;
	if (p_value.get_type() == Variant::NODE_PATH && p_value != NodePath()) {
		node_path_target = es->get_node(p_value);
	}

	UndoRedo *ur = EditorNode::get_undo_redo();

	ur->create_action(TTR("MultiNode Set") + " " + String(name), UndoRedo::MERGE_ENDS);
	for (const List<NodePath>::Element *E = nodes.front(); E; E = E->next()) {
		if (!es->has_node(E->get())) {
			continue;
		}

		Node *n = es->get_node(E->get());
		if (!n) {
			continue;
		}

		if (p_value.get_type() == Variant::NODE_PATH) {
			NodePath path;
			if (node_path_target) {
				path = n->get_path_to(node_path_target);
			}
			ur->add_do_property(n, name, path);
		} else {
			Variant new_value;
			if (p_field == "") {
				// whole value
				new_value = p_value;
			} else {
				// only one field
				new_value = fieldwise_assign(n->get(name), p_value, p_field);
			}
			ur->add_do_property(n, name, new_value);
		}

		ur->add_undo_property(n, name, n->get(name));
	}
	ur->add_do_method(EditorNode::get_singleton()->get_inspector(), "refresh");
	ur->add_undo_method(EditorNode::get_singleton()->get_inspector(), "refresh");

	ur->commit_action();
	return true;
}

bool MultiNodeEdit::_get(const StringName &p_name, Variant &r_ret) const {
	Node *es = EditorNode::get_singleton()->get_edited_scene();
	if (!es) {
		return false;
	}

	String name = p_name;
	if (name == "scripts") { // script set is intercepted at object level (check Variant Object::get() ) ,so use a different name
		name = "script";
	}

	for (const List<NodePath>::Element *E = nodes.front(); E; E = E->next()) {
		if (!es->has_node(E->get())) {
			continue;
		}

		const Node *n = es->get_node(E->get());
		if (!n) {
			continue;
		}

		bool found;
		r_ret = n->get(name, &found);
		if (found) {
			return true;
		}
	}

	return false;
}

void MultiNodeEdit::_get_property_list(List<PropertyInfo> *p_list) const {
	HashMap<String, PLData> usage;

	Node *es = EditorNode::get_singleton()->get_edited_scene();
	if (!es) {
		return;
	}

	int nc = 0;

	List<PLData *> data_list;

	for (const List<NodePath>::Element *E = nodes.front(); E; E = E->next()) {
		if (!es->has_node(E->get())) {
			continue;
		}

		Node *n = es->get_node(E->get());
		if (!n) {
			continue;
		}

		List<PropertyInfo> plist;
		n->get_property_list(&plist, true);

		for (List<PropertyInfo>::Element *F = plist.front(); F; F = F->next()) {
			if (F->get().name == "script") {
				continue; //added later manually, since this is intercepted before being set (check Variant Object::get() )
			}
			if (!usage.has(F->get().name)) {
				PLData pld;
				pld.uses = 0;
				pld.info = F->get();
				usage[F->get().name] = pld;
				data_list.push_back(usage.getptr(F->get().name));
			}

			// Make sure only properties with the same exact PropertyInfo data will appear
			if (usage[F->get().name].info == F->get()) {
				usage[F->get().name].uses++;
			}
		}

		nc++;
	}

	for (List<PLData *>::Element *E = data_list.front(); E; E = E->next()) {
		if (nc == E->get()->uses) {
			p_list->push_back(E->get()->info);
		}
	}

	p_list->push_back(PropertyInfo(Variant::OBJECT, "scripts", PROPERTY_HINT_RESOURCE_TYPE, "Script"));
}

void MultiNodeEdit::clear_nodes() {
	nodes.clear();
}

void MultiNodeEdit::add_node(const NodePath &p_node) {
	nodes.push_back(p_node);
}

int MultiNodeEdit::get_node_count() const {
	return nodes.size();
}

NodePath MultiNodeEdit::get_node(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, nodes.size(), NodePath());
	return nodes[p_index];
}

StringName MultiNodeEdit::get_edited_class_name() const {
	Node *es = EditorNode::get_singleton()->get_edited_scene();
	if (!es) {
		return StringName("Node");
	}

	// Get the class name of the first node.
	StringName class_name;
	for (const List<NodePath>::Element *E = nodes.front(); E; E = E->next()) {
		Node *node = es->get_node_or_null(E->get());
		if (!node) {
			continue;
		}

		class_name = node->get_class_name();
		break;
	}

	if (class_name == StringName()) {
		return StringName("Node");
	}

	bool check_again = true;
	while (check_again) {
		check_again = false;

		if (class_name == StringName("Node") || class_name == StringName()) {
			// All nodes inherit from Node, so no need to continue checking.
			return StringName("Node");
		}

		// Check that all nodes inherit from class_name.
		for (const List<NodePath>::Element *E = nodes.front(); E; E = E->next()) {
			Node *node = es->get_node_or_null(E->get());
			if (!node) {
				continue;
			}

			const StringName node_class_name = node->get_class_name();
			if (class_name == node_class_name || ClassDB::is_parent_class(node_class_name, class_name)) {
				// class_name is the same or a parent of the node's class.
				continue;
			}

			// class_name is not a parent of the node's class, so check again with the parent class.
			class_name = ClassDB::get_parent_class(class_name);
			check_again = true;
			break;
		}
	}

	return class_name;
}

void MultiNodeEdit::set_property_field(const StringName &p_property, const Variant &p_value, const String &p_field) {
	_set_impl(p_property, p_value, p_field);
}

MultiNodeEdit::MultiNodeEdit() {
}
