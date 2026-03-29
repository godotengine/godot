/**************************************************************************/
/*  agent_context.cpp                                                     */
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

#include "agent_context.h"

#include "editor/editor_data.h"
#include "editor/editor_node.h"
#include "editor/file_system/editor_file_system.h"
#include "editor/script/script_editor_plugin.h"
#include "scene/main/node.h"

void AgentContext::_bind_methods() {
	ClassDB::bind_method(D_METHOD("build_context_string"), &AgentContext::build_context_string);
	ClassDB::bind_method(D_METHOD("mark_dirty"), &AgentContext::mark_dirty);
	ClassDB::bind_method(D_METHOD("get_scene_tree_context"), &AgentContext::get_scene_tree_context);
	ClassDB::bind_method(D_METHOD("get_filesystem_context"), &AgentContext::get_filesystem_context);
	ClassDB::bind_method(D_METHOD("get_selected_node_context"), &AgentContext::get_selected_node_context);
	ClassDB::bind_method(D_METHOD("get_open_script_context"), &AgentContext::get_open_script_context);
}

void AgentContext::_on_scene_changed() {
	context_dirty = true;
}

void AgentContext::_on_selection_changed() {
	context_dirty = true;
}

void AgentContext::_on_filesystem_changed() {
	context_dirty = true;
}

void AgentContext::connect_editor_signals() {
	EditorNode *editor = EditorNode::get_singleton();
	if (editor && !editor->is_connected("scene_changed", callable_mp(this, &AgentContext::_on_scene_changed))) {
		editor->connect("scene_changed", callable_mp(this, &AgentContext::_on_scene_changed), CONNECT_DEFERRED);
	}

	EditorSelection *selection = editor ? editor->get_editor_selection() : nullptr;
	if (selection && !selection->is_connected("selection_changed", callable_mp(this, &AgentContext::_on_selection_changed))) {
		selection->connect("selection_changed", callable_mp(this, &AgentContext::_on_selection_changed), CONNECT_DEFERRED);
	}

	EditorFileSystem *fs = EditorFileSystem::get_singleton();
	if (fs && !fs->is_connected("filesystem_changed", callable_mp(this, &AgentContext::_on_filesystem_changed))) {
		fs->connect("filesystem_changed", callable_mp(this, &AgentContext::_on_filesystem_changed), CONNECT_DEFERRED);
	}
}

void AgentContext::mark_dirty() {
	context_dirty = true;
}

String AgentContext::_gather_scene_tree() {
	Node *edited_scene = EditorNode::get_singleton()->get_edited_scene();
	if (!edited_scene) {
		return "(No scene open)";
	}
	return _gather_scene_tree_recursive(edited_scene, 0);
}

String AgentContext::_gather_scene_tree_recursive(Node *p_node, int p_depth) {
	if (!p_node) {
		return String();
	}

	// Limit depth to avoid enormous outputs.
	if (p_depth > 10) {
		return String();
	}

	String result;
	String indent;
	for (int i = 0; i < p_depth; i++) {
		indent += "  ";
	}

	result += indent + p_node->get_name() + " (" + p_node->get_class() + ")";

	// Show script if attached.
	Ref<Script> script = p_node->get_script();
	if (script.is_valid() && !script->get_path().is_empty()) {
		result += " [" + script->get_path() + "]";
	}
	result += "\n";

	for (int i = 0; i < p_node->get_child_count(); i++) {
		result += _gather_scene_tree_recursive(p_node->get_child(i), p_depth + 1);
	}

	return result;
}

String AgentContext::_gather_filesystem(const String &p_path, int p_depth) {
	EditorFileSystem *efs = EditorFileSystem::get_singleton();
	if (!efs) {
		return "(Filesystem not available)";
	}

	EditorFileSystemDirectory *root = efs->get_filesystem();
	if (!root) {
		return "(Filesystem not scanned)";
	}

	return _gather_filesystem_recursive(root, 0);
}

String AgentContext::_gather_filesystem_recursive(EditorFileSystemDirectory *p_dir, int p_depth) {
	if (!p_dir || p_depth > 4) {
		return String();
	}

	String result;
	String indent;
	for (int i = 0; i < p_depth; i++) {
		indent += "  ";
	}

	result += indent + p_dir->get_name() + "/\n";

	// List files.
	for (int i = 0; i < p_dir->get_file_count(); i++) {
		result += indent + "  " + p_dir->get_file(i) + "\n";
	}

	// List subdirectories.
	for (int i = 0; i < p_dir->get_subdir_count(); i++) {
		result += _gather_filesystem_recursive(p_dir->get_subdir(i), p_depth + 1);
	}

	// Cap total output size.
	const int MAX_FS_CHARS = 4000;
	if (result.length() > MAX_FS_CHARS) {
		result = result.substr(0, MAX_FS_CHARS) + "\n... (truncated)\n";
	}

	return result;
}

String AgentContext::_gather_selected_node() {
	EditorNode *editor = EditorNode::get_singleton();
	if (!editor) {
		return "(No editor)";
	}

	EditorSelection *selection = editor->get_editor_selection();
	if (!selection) {
		return "(No selection)";
	}

	List<Node *> selected = selection->get_full_selected_node_list();
	if (selected.is_empty()) {
		return "(No node selected)";
	}

	String result;
	for (const Node *node : selected) {
		result += "Name: " + node->get_name() + "\n";
		result += "Type: " + node->get_class() + "\n";
		result += "Path: " + String(node->get_path()) + "\n";

		// Show script.
		Ref<Script> script = node->get_script();
		if (script.is_valid() && !script->get_path().is_empty()) {
			result += "Script: " + script->get_path() + "\n";
		}

		// Show key properties.
		List<PropertyInfo> properties;
		const_cast<Node *>(node)->get_property_list(&properties);
		int prop_count = 0;
		for (const PropertyInfo &pi : properties) {
			if (pi.usage & PROPERTY_USAGE_EDITOR && !(pi.usage & PROPERTY_USAGE_INTERNAL)) {
				// Only show a manageable number of properties.
				if (prop_count >= 20) {
					result += "  ... (more properties)\n";
					break;
				}
				Variant val = const_cast<Node *>(node)->get(pi.name);
				result += "  " + pi.name + " = " + String(val) + "\n";
				prop_count++;
			}
		}
		result += "\n";
	}

	return result;
}

String AgentContext::_gather_open_script() {
	ScriptEditor *script_editor = ScriptEditor::get_singleton();
	if (!script_editor) {
		return "(Script editor not available)";
	}

	Vector<Ref<Script>> open_scripts = script_editor->get_open_scripts();
	if (open_scripts.is_empty()) {
		return "(No script open)";
	}

	// Use the first open script as the "current" one.
	Ref<Script> current_script = open_scripts[0];
	if (current_script.is_null()) {
		return "(No script open)";
	}

	String result;
	result += "Path: " + current_script->get_path() + "\n";
	if (current_script->get_language()) {
		result += "Language: " + current_script->get_language()->get_name() + "\n";
	}
	result += "---\n";
	result += current_script->get_source_code();

	// Cap script content.
	const int MAX_SCRIPT_CHARS = 8000;
	if (result.length() > MAX_SCRIPT_CHARS) {
		result = result.substr(0, MAX_SCRIPT_CHARS) + "\n... (truncated)\n";
	}

	return result;
}

String AgentContext::get_scene_tree_context() {
	return _gather_scene_tree();
}

String AgentContext::get_filesystem_context() {
	return _gather_filesystem();
}

String AgentContext::get_selected_node_context() {
	return _gather_selected_node();
}

String AgentContext::get_open_script_context() {
	return _gather_open_script();
}

String AgentContext::build_context_string() {
	String context;

	context += "## Current Scene\n";
	cached_scene_tree = _gather_scene_tree();
	context += cached_scene_tree + "\n";

	context += "## Selected Node\n";
	cached_selected_node = _gather_selected_node();
	context += cached_selected_node + "\n";

	context += "## Open Script\n";
	cached_open_script = _gather_open_script();
	context += cached_open_script + "\n";

	context += "## Project Files\n";
	cached_filesystem = _gather_filesystem();
	context += cached_filesystem + "\n";

	context_dirty = false;

	// Apply total context budget.
	const int MAX_CONTEXT_CHARS = 16000;
	if (context.length() > MAX_CONTEXT_CHARS) {
		context = context.substr(0, MAX_CONTEXT_CHARS) + "\n... (context truncated)\n";
	}

	return context;
}

AgentContext::AgentContext() {
	singleton = this;
}

AgentContext::~AgentContext() {
	if (singleton == this) {
		singleton = nullptr;
	}
}
