/**************************************************************************/
/*  vertex_ai_assistant.h                                                */
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
#include "core/object/ref_counted.h"
#include "core/variant/callable.h"
#include "core/templates/hash_map.h"
#include "vertex_ai_command.h"
#include "vertex_ai_context.h"

// Vertex AI assistant architecture. Maintains a registry of named commands,
// gathers context (project structure, scripts, scenes, shaders, assets,
// errors, logs, profiler/perf results) and dispatches commands through a
// pluggable backend Callable. Destructive commands require an explicit
// confirmation token; [method request] returns a pending action that must be
// confirmed via [method confirm] before execution. No network calls are made
// by the engine; a real LLM is wired in by setting the backend Callable.
class VertexAIAssistant : public Object {
	GDCLASS(VertexAIAssistant, Object)

private:
	HashMap<String, Ref<VertexAICommand>> commands;
	Callable backend;
	uint64_t pending_token = 0;
	String pending_command;
	Dictionary pending_args;
	Dictionary error_log;

	Ref<VertexAIContext> _build_context() const;

protected:
	static void _bind_methods();

public:
	void register_command(const Ref<VertexAICommand> &p_command);
	void unregister_command(const String &p_name);
	bool has_command(const String &p_name) const;
	Ref<VertexAICommand> get_command(const String &p_name) const;
	Array list_commands() const;

	void set_backend(const Callable &p_backend) { backend = p_backend; }
	Callable get_backend() const { return backend; }

	// Context feeders the AI reasons over.
	Ref<VertexAIContext> gather_context(const String &p_root = "res://") const;
	void push_error_entry(const String &p_key, const Variant &p_value);
	void clear_error_log();
	Dictionary get_error_log() const { return error_log; }

	// Returns a Dictionary with keys: "needs_confirmation" (bool), "token" (int),
	// "result" (Variant) if executed immediately, "message" (String).
	Dictionary request(const String &p_command, const Dictionary &p_args = Dictionary());
	Dictionary confirm(uint64_t p_token);

	// Built-in convenience wrappers (registered as commands during init).
	Dictionary create_player(const Dictionary &p_args) const;
	Dictionary create_scene(const Dictionary &p_args) const;
	Dictionary fix_error(const Dictionary &p_args) const;
	Dictionary optimize_project(const Dictionary &p_args) const;
	Dictionary reduce_memory_usage(const Dictionary &p_args) const;
	Dictionary explain_node(const Dictionary &p_args) const;
	Dictionary create_animation(const Dictionary &p_args) const;
	Dictionary create_ui(const Dictionary &p_args) const;
	Dictionary optimize_shader(const Dictionary &p_args) const;

	VertexAIAssistant();
	~VertexAIAssistant();
};
