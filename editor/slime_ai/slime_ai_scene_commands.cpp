/**************************************************************************/
/*  slime_ai_scene_commands.cpp                                           */
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

#include "slime_ai_scene_commands.h"

#include "editor/slime_ai/slime_ai_scene_inspector.h"
#include "scene/2d/node_2d.h"
#include "scene/3d/node_3d.h"

#include <cmath>

namespace SlimeAI {

static bool _only_keys(const Dictionary &p_value, const Vector<String> &p_keys) {
	if (p_value.size() != p_keys.size()) {
		return false;
	}
	for (const String &key : p_keys) {
		if (!p_value.has(key)) {
			return false;
		}
	}
	return true;
}

static bool _built_in_target(Node *p_root, Node *p_node, bool p_allow_root) {
	if (!p_node || (!p_allow_root && p_node == p_root) || (p_node != p_root && p_node->get_owner() != p_root) || p_node->get_script().get_type() != Variant::NIL || (p_node != p_root && p_node->is_instance())) {
		return false;
	}
	return p_node->get_class() == "Node2D" || p_node->get_class() == "Node3D";
}

static bool _valid_name(Node *p_parent, const String &p_name, Node *p_current = nullptr) {
	if (p_name.is_empty() || p_name.utf8().length() > 64 || p_name.begins_with("@") || p_name.contains("/") || p_name.contains(":") || p_name == "." || p_name == "..") {
		return false;
	}
	Node *sibling = p_parent->get_node_or_null(NodePath(p_name));
	return !sibling || sibling == p_current;
}

static bool _valid_typed_value(const Dictionary &p_value, const String &p_property, const String &p_class) {
	if (!_only_keys(p_value, { "type", "value" }) || p_value["type"].get_type() != Variant::STRING) {
		return false;
	}
	if (p_property == "visible") {
		return String(p_value["type"]) == "bool" && p_value["value"].get_type() == Variant::BOOL;
	}
	if (p_property != "position" || p_value["value"].get_type() != Variant::ARRAY) {
		return false;
	}
	const Array values = p_value["value"];
	const int dimension = p_class == "Node2D" ? 2 : 3;
	if (String(p_value["type"]) != (dimension == 2 ? "Vector2" : "Vector3") || values.size() != dimension) {
		return false;
	}
	for (int i = 0; i < values.size(); i++) {
		const Variant component = values[i];
		if ((component.get_type() != Variant::INT && component.get_type() != Variant::FLOAT) || !std::isfinite(double(component))) {
			return false;
		}
	}
	return true;
}

static bool _owned_subtree(Node *p_root, Node *p_node) {
	if (!_built_in_target(p_root, p_node, false)) {
		return false;
	}
	for (int i = 0; i < p_node->get_child_count(false); i++) {
		if (!_owned_subtree(p_root, p_node->get_child(i, false))) {
			return false;
		}
	}
	return true;
}

Node *operation_target(Node *p_root, const Dictionary &p_operation) {
	if (!p_root) {
		return nullptr;
	}
	if (p_operation.has("parent_ref")) {
		return SceneInspector::resolve(p_root, p_operation["parent_ref"]);
	}
	if (p_operation.has("node_ref")) {
		return SceneInspector::resolve(p_root, p_operation["node_ref"]);
	}
	return nullptr;
}

bool validate_scene_operation(const Dictionary &p_proposal, Node *p_root, Dictionary &r_operation, String &r_error_code) {
	if (!_only_keys(p_proposal, { "scene_ref", "base_revision", "operations" }) || p_proposal["scene_ref"].get_type() != Variant::STRING || p_proposal["base_revision"].get_type() != Variant::STRING || p_proposal["operations"].get_type() != Variant::ARRAY) {
		r_error_code = "UNSUPPORTED_SCHEMA";
		return false;
	}
	const Array operations = p_proposal["operations"];
	if (operations.size() != 1 || operations[0].get_type() != Variant::DICTIONARY) {
		r_error_code = "UNSUPPORTED_OPERATION";
		return false;
	}
	r_operation = operations[0];
	if (!r_operation.has("op") || r_operation["op"].get_type() != Variant::STRING) {
		r_error_code = "UNSUPPORTED_SCHEMA";
		return false;
	}
	const String kind = r_operation["op"];
	if (kind == "create_child") {
		if (!_only_keys(r_operation, { "op", "parent_ref", "class_name", "name", "properties" }) || r_operation["parent_ref"].get_type() != Variant::STRING || r_operation["class_name"].get_type() != Variant::STRING || r_operation["name"].get_type() != Variant::STRING || r_operation["properties"].get_type() != Variant::DICTIONARY) {
			r_error_code = "UNSUPPORTED_SCHEMA";
			return false;
		}
		Node *parent = operation_target(p_root, r_operation);
		if (!parent) {
			r_error_code = "STALE_REFERENCE";
			return false;
		}
		if (!_built_in_target(p_root, parent, true)) {
			r_error_code = "READ_ONLY_RESOURCE";
			return false;
		}
		const String class_name = r_operation["class_name"];
		if ((class_name != "Node2D" && class_name != "Node3D") || parent->get_class() != class_name) {
			r_error_code = "UNSUPPORTED_OPERATION";
			return false;
		}
		if (!_valid_name(parent, r_operation["name"])) {
			r_error_code = "REVISION_CONFLICT";
			return false;
		}
		Dictionary properties = r_operation["properties"];
		if (!_only_keys(properties, { "position" }) || properties["position"].get_type() != Variant::DICTIONARY || !_valid_typed_value(properties["position"], "position", class_name)) {
			r_error_code = "UNSUPPORTED_SCHEMA";
			return false;
		}
		return true;
	}
	if (kind == "set_property") {
		if (!_only_keys(r_operation, { "op", "node_ref", "property", "value" }) || r_operation["node_ref"].get_type() != Variant::STRING || r_operation["property"].get_type() != Variant::STRING || r_operation["value"].get_type() != Variant::DICTIONARY) {
			r_error_code = "UNSUPPORTED_SCHEMA";
			return false;
		}
		Node *target = operation_target(p_root, r_operation);
		if (!target) {
			r_error_code = "STALE_REFERENCE";
			return false;
		}
		if (!_built_in_target(p_root, target, true)) {
			r_error_code = "READ_ONLY_RESOURCE";
			return false;
		}
		if (!_valid_typed_value(r_operation["value"], r_operation["property"], target->get_class())) {
			r_error_code = "UNSUPPORTED_OPERATION";
			return false;
		}
		return true;
	}
	if (kind == "rename_node" || kind == "remove_node") {
		const Vector<String> keys = kind == "rename_node" ? Vector<String>({ "op", "node_ref", "name" }) : Vector<String>({ "op", "node_ref" });
		if (!_only_keys(r_operation, keys) || r_operation["node_ref"].get_type() != Variant::STRING || (kind == "rename_node" && r_operation["name"].get_type() != Variant::STRING)) {
			r_error_code = "UNSUPPORTED_SCHEMA";
			return false;
		}
		Node *target = operation_target(p_root, r_operation);
		if (!target) {
			r_error_code = "STALE_REFERENCE";
			return false;
		}
		if (!_owned_subtree(p_root, target)) {
			r_error_code = "READ_ONLY_RESOURCE";
			return false;
		}
		if (kind == "rename_node" && !_valid_name(target->get_parent(), r_operation["name"], target)) {
			r_error_code = "REVISION_CONFLICT";
			return false;
		}
		return true;
	}
	r_error_code = "UNSUPPORTED_OPERATION";
	return false;
}

static Dictionary _typed_current(Node *p_target, const String &p_property) {
	Dictionary value;
	if (p_property == "visible") {
		value["type"] = "bool";
		value["value"] = p_target->get("visible");
	} else if (Node2D *node_2d = Object::cast_to<Node2D>(p_target)) {
		const Vector2 position = node_2d->get_position();
		value["type"] = "Vector2";
		Array coordinates;
		coordinates.push_back(position.x);
		coordinates.push_back(position.y);
		value["value"] = coordinates;
	} else {
		const Vector3 position = Object::cast_to<Node3D>(p_target)->get_position();
		value["type"] = "Vector3";
		Array coordinates;
		coordinates.push_back(position.x);
		coordinates.push_back(position.y);
		coordinates.push_back(position.z);
		value["value"] = coordinates;
	}
	return value;
}

static void _describe_subtree(Node *p_root, Node *p_node, Array &r_nodes) {
	Dictionary item;
	item["ref"] = SceneInspector::object_ref(p_root, p_node);
	item["path"] = String(p_root->get_path_to(p_node));
	item["class_name"] = p_node->get_class();
	item["name"] = String(p_node->get_name());
	item["position"] = _typed_current(p_node, "position");
	r_nodes.push_back(item);
	for (int i = 0; i < p_node->get_child_count(false); i++) {
		_describe_subtree(p_root, p_node->get_child(i, false), r_nodes);
	}
}

Dictionary describe_scene_operation(Node *p_root, const Dictionary &p_operation) {
	Dictionary description;
	const String kind = p_operation["op"];
	Node *target = operation_target(p_root, p_operation);
	if (kind == "create_child") {
		description["before"] = Variant();
		description["after"] = p_operation;
		description["risk"] = "Adds one scene-owned node.";
	} else if (kind == "set_property") {
		description["before"] = _typed_current(target, p_operation["property"]);
		description["after"] = p_operation["value"];
		description["risk"] = "Changes one native property.";
	} else if (kind == "rename_node") {
		description["before"] = String(target->get_name());
		description["after"] = p_operation["name"];
		description["risk"] = "Changes one node path; references may need review.";
	} else if (kind == "remove_node") {
		Array nodes;
		_describe_subtree(p_root, target, nodes);
		description["before"] = nodes;
		description["after"] = Variant();
		description["risk"] = "Removes the listed owned subtree; native undo is available this session.";
	}
	return description;
}

} // namespace SlimeAI
