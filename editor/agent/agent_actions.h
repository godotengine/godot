/**************************************************************************/
/*  agent_actions.h                                                       */
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
#include "core/variant/dictionary.h"

class AgentActions : public Object {
	GDCLASS(AgentActions, Object);

	static inline AgentActions *singleton = nullptr;

	// Action execution.
	String _execute_create_script(const Dictionary &p_params);
	String _execute_edit_script(const Dictionary &p_params);
	String _execute_read_file(const Dictionary &p_params);
	String _execute_list_files(const Dictionary &p_params);
	String _execute_delete_file(const Dictionary &p_params);
	String _execute_add_node(const Dictionary &p_params);
	String _execute_remove_node(const Dictionary &p_params);
	String _execute_set_property(const Dictionary &p_params);

protected:
	static void _bind_methods();

public:
	static AgentActions *get_singleton() { return singleton; }

	// Execute a tool call from the LLM.
	String execute_action(const String &p_tool_name, const Dictionary &p_params);

	// Get available tool definitions (for system prompt).
	String get_tool_definitions_json() const;

	// Validation.
	bool is_path_safe(const String &p_path) const;

	AgentActions();
	~AgentActions();
};
