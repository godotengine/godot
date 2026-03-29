/**************************************************************************/
/*  agent_context.h                                                       */
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

#pragma once

#include "core/object/object.h"
#include "core/string/ustring.h"

class Node;
class EditorFileSystemDirectory;

class AgentContext : public Object {
	GDCLASS(AgentContext, Object);

	static inline AgentContext *singleton = nullptr;

	// Cached context.
	String cached_scene_tree;
	String cached_filesystem;
	String cached_selected_node;
	String cached_open_script;
	bool context_dirty = true;

	// Internal.
	String _gather_scene_tree();
	String _gather_scene_tree_recursive(Node *p_node, int p_depth);
	String _gather_filesystem(const String &p_path = "res://", int p_depth = 0);
	String _gather_filesystem_recursive(EditorFileSystemDirectory *p_dir, int p_depth);
	String _gather_selected_node();
	String _gather_open_script();

	void _on_scene_changed();
	void _on_selection_changed();
	void _on_filesystem_changed();

protected:
	static void _bind_methods();

public:
	static AgentContext *get_singleton() { return singleton; }

	String build_context_string();
	void mark_dirty();
	void connect_editor_signals();

	String get_scene_tree_context();
	String get_filesystem_context();
	String get_selected_node_context();
	String get_open_script_context();

	AgentContext();
	~AgentContext();
};
