/**************************************************************************/
/*  agent_actions.cpp                                                     */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "agent_actions.h"

#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/io/json.h"
#include "core/object/class_db.h"
#include "editor/editor_node.h"
#include "editor/editor_undo_redo_manager.h"
#include "editor/file_system/editor_file_system.h"
#include "scene/main/node.h"

void AgentActions::_bind_methods() {
	ClassDB::bind_method(D_METHOD("execute_action", "tool_name", "params"), &AgentActions::execute_action);
	ClassDB::bind_method(D_METHOD("get_tool_definitions_json"), &AgentActions::get_tool_definitions_json);
	ClassDB::bind_method(D_METHOD("is_path_safe", "path"), &AgentActions::is_path_safe);
}

bool AgentActions::is_path_safe(const String &p_path) const {
	// Must start with res://.
	if (!p_path.begins_with("res://")) {
		return false;
	}
	// No directory traversal.
	if (p_path.contains("..")) {
		return false;
	}
	// Not in internal .godot directory.
	if (p_path.begins_with("res://.godot/") || p_path.begins_with("res://.godot")) {
		return false;
	}
	return true;
}

String AgentActions::execute_action(const String &p_tool_name, const Dictionary &p_params) {
	if (p_tool_name == "create_script") {
		return _execute_create_script(p_params);
	} else if (p_tool_name == "edit_script") {
		return _execute_edit_script(p_params);
	} else if (p_tool_name == "read_file") {
		return _execute_read_file(p_params);
	} else if (p_tool_name == "list_files") {
		return _execute_list_files(p_params);
	} else if (p_tool_name == "delete_file") {
		return _execute_delete_file(p_params);
	} else if (p_tool_name == "add_node") {
		return _execute_add_node(p_params);
	} else if (p_tool_name == "remove_node") {
		return _execute_remove_node(p_params);
	} else if (p_tool_name == "set_property") {
		return _execute_set_property(p_params);
	}
	return "Error: Unknown tool '" + p_tool_name + "'.";
}

String AgentActions::_execute_create_script(const Dictionary &p_params) {
	String path = p_params.get("path", "");
	String content = p_params.get("content", "");

	if (path.is_empty()) {
		return "Error: 'path' parameter is required.";
	}
	if (!is_path_safe(path)) {
		return "Error: Path '" + path + "' is not safe. Must be under res:// and not in .godot/.";
	}

	Ref<FileAccess> file = FileAccess::open(path, FileAccess::WRITE);
	if (file.is_null()) {
		return "Error: Could not open '" + path + "' for writing.";
	}

	file->store_string(content);
	file->flush();
	file.unref();

	// Trigger filesystem rescan.
	EditorFileSystem *efs = EditorFileSystem::get_singleton();
	if (efs) {
		efs->scan();
	}

	return "Successfully created script at '" + path + "'.";
}

String AgentActions::_execute_edit_script(const Dictionary &p_params) {
	String path = p_params.get("path", "");
	String old_content = p_params.get("old_content", "");
	String new_content = p_params.get("new_content", "");

	if (path.is_empty()) {
		return "Error: 'path' parameter is required.";
	}
	if (!is_path_safe(path)) {
		return "Error: Path '" + path + "' is not safe.";
	}
	if (old_content.is_empty()) {
		return "Error: 'old_content' parameter is required.";
	}

	// Read existing file.
	Ref<FileAccess> read_file = FileAccess::open(path, FileAccess::READ);
	if (read_file.is_null()) {
		return "Error: Could not open '" + path + "' for reading.";
	}

	String file_content = read_file->get_as_text();
	read_file.unref();

	// Perform replacement.
	if (!file_content.contains(old_content)) {
		return "Error: Could not find the specified old_content in '" + path + "'.";
	}

	String updated_content = file_content.replace(old_content, new_content);

	// Write back.
	Ref<FileAccess> write_file = FileAccess::open(path, FileAccess::WRITE);
	if (write_file.is_null()) {
		return "Error: Could not open '" + path + "' for writing.";
	}

	write_file->store_string(updated_content);
	write_file->flush();
	write_file.unref();

	// Trigger filesystem rescan.
	EditorFileSystem *efs = EditorFileSystem::get_singleton();
	if (efs) {
		efs->scan();
	}

	return "Successfully edited '" + path + "'.";
}

String AgentActions::_execute_read_file(const Dictionary &p_params) {
	String path = p_params.get("path", "");

	if (path.is_empty()) {
		return "Error: 'path' parameter is required.";
	}
	if (!is_path_safe(path)) {
		return "Error: Path '" + path + "' is not safe.";
	}

	Ref<FileAccess> file = FileAccess::open(path, FileAccess::READ);
	if (file.is_null()) {
		return "Error: Could not open '" + path + "' for reading.";
	}

	String content = file->get_as_text();

	// Cap read size.
	const int MAX_READ_SIZE = 10000;
	if (content.length() > MAX_READ_SIZE) {
		content = content.substr(0, MAX_READ_SIZE) + "\n... (truncated, file is " + itos(content.length()) + " characters)";
	}

	return content;
}

String AgentActions::_execute_list_files(const Dictionary &p_params) {
	String directory = p_params.get("directory", "res://");

	if (!is_path_safe(directory)) {
		return "Error: Directory '" + directory + "' is not safe.";
	}

	Ref<DirAccess> dir = DirAccess::open(directory);
	if (dir.is_null()) {
		return "Error: Could not open directory '" + directory + "'.";
	}

	String result;
	int count = 0;
	const int MAX_FILES = 200;

	// List directories first.
	dir->list_dir_begin();
	String item = dir->get_next();
	while (!item.is_empty() && count < MAX_FILES) {
		if (item != "." && item != ".." && !item.begins_with(".godot")) {
			if (dir->current_is_dir()) {
				result += "  [DIR] " + item + "/\n";
				count++;
			}
		}
		item = dir->get_next();
	}
	dir->list_dir_end();

	// List files.
	dir->list_dir_begin();
	item = dir->get_next();
	while (!item.is_empty() && count < MAX_FILES) {
		if (item != "." && item != ".." && !item.begins_with(".godot")) {
			if (!dir->current_is_dir()) {
				result += "  " + item + "\n";
				count++;
			}
		}
		item = dir->get_next();
	}
	dir->list_dir_end();

	if (result.is_empty()) {
		return "(Empty directory)";
	}

	return directory + ":\n" + result;
}

String AgentActions::_execute_delete_file(const Dictionary &p_params) {
	String path = p_params.get("path", "");

	if (path.is_empty()) {
		return "Error: 'path' parameter is required.";
	}
	if (!is_path_safe(path)) {
		return "Error: Path '" + path + "' is not safe.";
	}

	Ref<DirAccess> dir = DirAccess::open("res://");
	if (dir.is_null()) {
		return "Error: Could not access project directory.";
	}

	Error err = dir->remove(path);
	if (err != OK) {
		return "Error: Could not delete '" + path + "'. Error code: " + itos(err);
	}

	// Trigger filesystem rescan.
	EditorFileSystem *efs = EditorFileSystem::get_singleton();
	if (efs) {
		efs->scan();
	}

	return "Successfully deleted '" + path + "'.";
}

String AgentActions::_execute_add_node(const Dictionary &p_params) {
	String type = p_params.get("type", "");
	String name = p_params.get("name", "");
	String parent_path = p_params.get("parent", ".");

	if (type.is_empty()) {
		return "Error: 'type' parameter is required.";
	}
	if (name.is_empty()) {
		return "Error: 'name' parameter is required.";
	}

	// Verify the class exists and is a Node type.
	if (!ClassDB::class_exists(StringName(type))) {
		return "Error: Class '" + type + "' does not exist.";
	}
	if (!ClassDB::is_parent_class(StringName(type), "Node")) {
		return "Error: Class '" + type + "' is not a Node type.";
	}

	Node *edited_scene = EditorNode::get_singleton()->get_edited_scene();
	if (!edited_scene) {
		return "Error: No scene is currently open.";
	}

	// Find parent node.
	Node *parent = nullptr;
	if (parent_path == "." || parent_path.is_empty()) {
		parent = edited_scene;
	} else {
		parent = edited_scene->get_node_or_null(NodePath(parent_path));
	}

	if (!parent) {
		return "Error: Parent node '" + parent_path + "' not found.";
	}

	// Create the node.
	Object *obj = ClassDB::instantiate(StringName(type));
	Node *new_node = Object::cast_to<Node>(obj);
	if (!new_node) {
		if (obj) {
			memdelete(obj);
		}
		return "Error: Failed to instantiate '" + type + "'.";
	}

	new_node->set_name(name);

	// Use undo/redo manager.
	EditorUndoRedoManager *ur = EditorUndoRedoManager::get_singleton();
	ur->create_action("Agent: Add Node '" + name + "'");
	ur->add_do_method(parent, "add_child", new_node, true);
	ur->add_do_method(new_node, "set_owner", edited_scene);
	ur->add_undo_method(parent, "remove_child", new_node);
	ur->add_do_reference(new_node);
	ur->commit_action();

	return "Successfully added " + type + " node '" + name + "' to '" + String(parent->get_path()) + "'.";
}

String AgentActions::_execute_remove_node(const Dictionary &p_params) {
	String node_path = p_params.get("node_path", "");

	if (node_path.is_empty()) {
		return "Error: 'node_path' parameter is required.";
	}

	Node *edited_scene = EditorNode::get_singleton()->get_edited_scene();
	if (!edited_scene) {
		return "Error: No scene is currently open.";
	}

	Node *target = edited_scene->get_node_or_null(NodePath(node_path));
	if (!target) {
		return "Error: Node '" + node_path + "' not found.";
	}

	if (target == edited_scene) {
		return "Error: Cannot remove the scene root node.";
	}

	Node *parent = target->get_parent();
	if (!parent) {
		return "Error: Node has no parent.";
	}

	EditorUndoRedoManager *ur = EditorUndoRedoManager::get_singleton();
	ur->create_action("Agent: Remove Node '" + target->get_name() + "'");
	ur->add_do_method(parent, "remove_child", target);
	ur->add_undo_method(parent, "add_child", target, true);
	ur->add_undo_method(target, "set_owner", edited_scene);
	ur->add_undo_reference(target);
	ur->commit_action();

	return "Successfully removed node '" + node_path + "'.";
}

String AgentActions::_execute_set_property(const Dictionary &p_params) {
	String node_path = p_params.get("node_path", "");
	String property = p_params.get("property", "");
	Variant value = p_params.get("value", Variant());

	if (node_path.is_empty()) {
		return "Error: 'node_path' parameter is required.";
	}
	if (property.is_empty()) {
		return "Error: 'property' parameter is required.";
	}

	Node *edited_scene = EditorNode::get_singleton()->get_edited_scene();
	if (!edited_scene) {
		return "Error: No scene is currently open.";
	}

	Node *target = edited_scene->get_node_or_null(NodePath(node_path));
	if (!target) {
		return "Error: Node '" + node_path + "' not found.";
	}

	// Verify property exists.
	bool found = false;
	List<PropertyInfo> properties;
	target->get_property_list(&properties);
	for (const PropertyInfo &pi : properties) {
		if (pi.name == property) {
			found = true;
			break;
		}
	}

	if (!found) {
		return "Error: Property '" + property + "' not found on node '" + node_path + "'.";
	}

	Variant old_value = target->get(StringName(property));

	EditorUndoRedoManager *ur = EditorUndoRedoManager::get_singleton();
	ur->create_action("Agent: Set Property '" + property + "'");
	ur->add_do_method(target, "set", property, value);
	ur->add_undo_method(target, "set", property, old_value);
	ur->commit_action();

	return "Successfully set '" + property + "' on '" + node_path + "'.";
}

String AgentActions::get_tool_definitions_json() const {
	// Return tool definitions in a format suitable for injection into the system prompt.
	Array tools;

	{
		Dictionary tool;
		tool["name"] = "create_script";
		tool["description"] = "Create a new script file at the specified path with the given content.";
		tool["parameters"] = "path (String, required), content (String, required)";
		tools.push_back(tool);
	}
	{
		Dictionary tool;
		tool["name"] = "edit_script";
		tool["description"] = "Edit an existing script by replacing old_content with new_content.";
		tool["parameters"] = "path (String, required), old_content (String, required), new_content (String, required)";
		tools.push_back(tool);
	}
	{
		Dictionary tool;
		tool["name"] = "read_file";
		tool["description"] = "Read the contents of a file.";
		tool["parameters"] = "path (String, required)";
		tools.push_back(tool);
	}
	{
		Dictionary tool;
		tool["name"] = "list_files";
		tool["description"] = "List files in a directory.";
		tool["parameters"] = "directory (String, optional, default: res://)";
		tools.push_back(tool);
	}
	{
		Dictionary tool;
		tool["name"] = "delete_file";
		tool["description"] = "Delete a file.";
		tool["parameters"] = "path (String, required)";
		tools.push_back(tool);
	}
	{
		Dictionary tool;
		tool["name"] = "add_node";
		tool["description"] = "Add a new node to the current scene.";
		tool["parameters"] = "type (String, required), name (String, required), parent (String, optional, default: scene root)";
		tools.push_back(tool);
	}
	{
		Dictionary tool;
		tool["name"] = "remove_node";
		tool["description"] = "Remove a node from the current scene.";
		tool["parameters"] = "node_path (String, required)";
		tools.push_back(tool);
	}
	{
		Dictionary tool;
		tool["name"] = "set_property";
		tool["description"] = "Set a property on a node in the current scene.";
		tool["parameters"] = "node_path (String, required), property (String, required), value (Variant, required)";
		tools.push_back(tool);
	}

	return JSON::stringify(tools, "\t");
}

AgentActions::AgentActions() {
	singleton = this;
}

AgentActions::~AgentActions() {
	if (singleton == this) {
		singleton = nullptr;
	}
}
