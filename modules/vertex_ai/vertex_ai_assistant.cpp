/**************************************************************************/
/*  vertex_ai_assistant.cpp                                              */
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

#include "vertex_ai_assistant.h"

#include "core/config/engine.h"
#include "core/object/class_db.h"
#include "core/os/os.h"
#include "core/string/print_string.h"

Ref<VertexAIContext> VertexAIAssistant::_build_context() const {
	Ref<VertexAIContext> ctx = memnew(VertexAIContext);
	ctx->build_from_project();
	Dictionary errs = error_log;
	ctx->set_section("errors", errs);
	return ctx;
}

void VertexAIAssistant::register_command(const Ref<VertexAICommand> &p_command) {
	if (p_command.is_null() || p_command->get_name().is_empty()) {
		return;
	}
	commands[p_command->get_name()] = p_command;
}

void VertexAIAssistant::unregister_command(const String &p_name) {
	commands.erase(p_name);
}

bool VertexAIAssistant::has_command(const String &p_name) const {
	return commands.has(p_name);
}

Ref<VertexAICommand> VertexAIAssistant::get_command(const String &p_name) const {
	if (commands.has(p_name)) {
		return commands[p_name];
	}
	return Ref<VertexAICommand>();
}

Array VertexAIAssistant::list_commands() const {
	Array out;
	for (const KeyValue<String, Ref<VertexAICommand>> &E : commands) {
		out.push_back(E.value->to_dictionary());
	}
	return out;
}

Ref<VertexAIContext> VertexAIAssistant::gather_context(const String &p_root) const {
	Ref<VertexAIContext> ctx = memnew(VertexAIContext);
	ctx->build_from_project(p_root);
	ctx->set_section("errors", error_log);
	return ctx;
}

void VertexAIAssistant::push_error_entry(const String &p_key, const Variant &p_value) {
	error_log[p_key] = p_value;
}

void VertexAIAssistant::clear_error_log() {
	error_log.clear();
}

Dictionary VertexAIAssistant::request(const String &p_command, const Dictionary &p_args) {
	Dictionary response;
	if (!commands.has(p_command)) {
		response["needs_confirmation"] = false;
		response["message"] = vformat("Unknown command: '%s'", p_command);
		return response;
	}
	Ref<VertexAICommand> cmd = commands[p_command];

	// If a backend Callable is configured, it can interpret the request. It is
	// not required for built-in handlers.
	Dictionary enriched_args = p_args;
	enriched_args["context"] = _build_context()->to_dictionary();
	enriched_args["command"] = p_command;
	if (backend.is_valid()) {
		Variant backend_result = backend.callv(Array{ Variant(enriched_args) });
		response["backend_result"] = backend_result;
	}

	if (cmd->get_is_destructive()) {
		pending_token = (uint64_t)OS::get_singleton()->get_unix_time() + 1; // Simple unique token.
		pending_command = p_command;
		pending_args = enriched_args;
		response["needs_confirmation"] = true;
		response["token"] = (uint64_t)pending_token;
		response["message"] = vformat("Command '%s' is destructive and requires confirmation.", p_command);
		return response;
	}

	// Non-destructive: execute the handler immediately.
	Callable handler = cmd->get_handler();
	if (handler.is_valid()) {
		Variant result = handler.callv(Array{ Variant(enriched_args) });
		response["needs_confirmation"] = false;
		response["result"] = result;
		response["message"] = "ok";
	} else {
		response["needs_confirmation"] = false;
		response["message"] = "Command has no handler.";
	}
	return response;
}

Dictionary VertexAIAssistant::confirm(uint64_t p_token) {
	Dictionary response;
	if (p_token == 0 || p_token != pending_token) {
		response["executed"] = false;
		response["message"] = "Invalid or expired confirmation token.";
		return response;
	}
	if (!commands.has(pending_command)) {
		response["executed"] = false;
		response["message"] = "Pending command no longer registered.";
		pending_token = 0;
		return response;
	}
	Ref<VertexAICommand> cmd = commands[pending_command];
	Callable handler = cmd->get_handler();
	Variant result;
	if (handler.is_valid()) {
		result = handler.callv(Array{ Variant(pending_args) });
	}
	response["executed"] = true;
	response["result"] = result;
	response["command"] = pending_command;
	pending_token = 0;
	pending_command = "";
	pending_args.clear();
	return response;
}

// Built-in handlers return structured "instructions" describing what to create
// or change rather than mutating the filesystem directly. This keeps the engine
// safe and lets an editor/LLM backend act on the plan.
Dictionary VertexAIAssistant::create_player(const Dictionary &p_args) const {
	Dictionary plan;
	plan["action"] = "create_player";
	plan["scene"] = "res://player.tscn";
	plan["nodes"] = Array{ Variant("CharacterBody2D"), Variant("Sprite2D"), Variant("CollisionShape2D"), Variant("Camera2D") };
	plan["script"] = "res://player.gd";
	return plan;
}

Dictionary VertexAIAssistant::create_scene(const Dictionary &p_args) const {
	Dictionary plan;
	plan["action"] = "create_scene";
	plan["path"] = p_args.has("path") ? p_args["path"] : String("res://new_scene.tscn");
	plan["root_type"] = p_args.has("root_type") ? p_args["root_type"] : String("Node2D");
	return plan;
}

Dictionary VertexAIAssistant::fix_error(const Dictionary &p_args) const {
	Dictionary plan;
	plan["action"] = "fix_error";
	plan["error"] = p_args.has("error") ? p_args["error"] : String("unspecified");
	Dictionary errs = error_log;
	plan["errors"] = errs;
	return plan;
}

Dictionary VertexAIAssistant::optimize_project(const Dictionary &p_args) const {
	Dictionary plan;
	plan["action"] = "optimize_project";
	// Delegate to the Vertex optimizer singleton when present (looked up by name).
	Object *opt = Engine::get_singleton()->get_singleton_object("Vertex/Optimizer");
	if (opt) {
		plan["report"] = opt->call("analyze_project", String("res://"));
	}
	return plan;
}

Dictionary VertexAIAssistant::reduce_memory_usage(const Dictionary &p_args) const {
	Dictionary plan;
	plan["action"] = "reduce_memory_usage";
	Object *opt = Engine::get_singleton()->get_singleton_object("Vertex/Optimizer");
	if (opt) {
		Dictionary metrics = opt->call("get_runtime_metrics");
		plan["metrics"] = metrics;
	}
	plan["steps"] = Array{ Variant("free_orphan_nodes"), Variant("reduce_texture_sizes"), Variant("stream_large_assets"), Variant("lower_texture_budget") };
	return plan;
}

Dictionary VertexAIAssistant::explain_node(const Dictionary &p_args) const {
	Dictionary plan;
	plan["action"] = "explain_node";
	plan["node_path"] = p_args.has("node_path") ? p_args["node_path"] : String("");
	return plan;
}

Dictionary VertexAIAssistant::create_animation(const Dictionary &p_args) const {
	Dictionary plan;
	plan["action"] = "create_animation";
	plan["type"] = p_args.has("type") ? p_args["type"] : String("AnimationPlayer");
	plan["tracks"] = Array{ Variant("position"), Variant("rotation"), Variant("scale") };
	return plan;
}

Dictionary VertexAIAssistant::create_ui(const Dictionary &p_args) const {
	Dictionary plan;
	plan["action"] = "create_ui";
	plan["controls"] = Array{ Variant("Control"), Variant("VBoxContainer"), Variant("Label"), Variant("Button") };
	plan["scene"] = "res://ui.tscn";
	return plan;
}

Dictionary VertexAIAssistant::optimize_shader(const Dictionary &p_args) const {
	Dictionary plan;
	plan["action"] = "optimize_shader";
	plan["shader"] = p_args.has("shader") ? p_args["shader"] : String("");
	plan["steps"] = Array{ Variant("remove_unused_varyings"), Variant("reduce_texture_samples"), Variant("simplify_math"), Variant("move_to_vertex") };
	return plan;
}

static Ref<VertexAICommand> _make_cmd(const String &p_name, const String &p_desc, bool p_destructive, Object *p_obj, const char *p_method) {
	Ref<VertexAICommand> cmd = memnew(VertexAICommand);
	cmd->set_name(p_name);
	cmd->set_description(p_desc);
	cmd->set_is_destructive(p_destructive);
	cmd->set_handler(Callable(p_obj, p_method));
	return cmd;
}

VertexAIAssistant::VertexAIAssistant() {
	register_command(_make_cmd("create_player", "Plan a player character scene", false, this, "create_player"));
	register_command(_make_cmd("create_scene", "Plan a new scene", false, this, "create_scene"));
	register_command(_make_cmd("fix_error", "Propose a fix for an error", false, this, "fix_error"));
	register_command(_make_cmd("optimize_project", "Run the Vertex optimizer and return a report", false, this, "optimize_project"));
	register_command(_make_cmd("reduce_memory_usage", "Propose memory-reduction steps", false, this, "reduce_memory_usage"));
	register_command(_make_cmd("explain_node", "Explain a node in the scene tree", false, this, "explain_node"));
	register_command(_make_cmd("create_animation", "Plan an animation", false, this, "create_animation"));
	register_command(_make_cmd("create_ui", "Plan a UI layout", false, this, "create_ui"));
	register_command(_make_cmd("optimize_shader", "Propose shader optimizations", false, this, "optimize_shader"));
}

VertexAIAssistant::~VertexAIAssistant() {}

void VertexAIAssistant::_bind_methods() {
	ClassDB::bind_method(D_METHOD("register_command", "command"), &VertexAIAssistant::register_command);
	ClassDB::bind_method(D_METHOD("unregister_command", "name"), &VertexAIAssistant::unregister_command);
	ClassDB::bind_method(D_METHOD("has_command", "name"), &VertexAIAssistant::has_command);
	ClassDB::bind_method(D_METHOD("get_command", "name"), &VertexAIAssistant::get_command);
	ClassDB::bind_method(D_METHOD("list_commands"), &VertexAIAssistant::list_commands);

	ClassDB::bind_method(D_METHOD("set_backend", "backend"), &VertexAIAssistant::set_backend);
	ClassDB::bind_method(D_METHOD("get_backend"), &VertexAIAssistant::get_backend);
	ADD_PROPERTY(PropertyInfo(Variant::CALLABLE, "backend"), "set_backend", "get_backend");

	ClassDB::bind_method(D_METHOD("gather_context", "root"), &VertexAIAssistant::gather_context, DEFVAL("res://"));
	ClassDB::bind_method(D_METHOD("push_error_entry", "key", "value"), &VertexAIAssistant::push_error_entry);
	ClassDB::bind_method(D_METHOD("clear_error_log"), &VertexAIAssistant::clear_error_log);
	ClassDB::bind_method(D_METHOD("get_error_log"), &VertexAIAssistant::get_error_log);

	ClassDB::bind_method(D_METHOD("request", "command", "args"), &VertexAIAssistant::request, DEFVAL(Dictionary()));
	ClassDB::bind_method(D_METHOD("confirm", "token"), &VertexAIAssistant::confirm);

	ClassDB::bind_method(D_METHOD("create_player", "args"), &VertexAIAssistant::create_player, DEFVAL(Dictionary()));
	ClassDB::bind_method(D_METHOD("create_scene", "args"), &VertexAIAssistant::create_scene, DEFVAL(Dictionary()));
	ClassDB::bind_method(D_METHOD("fix_error", "args"), &VertexAIAssistant::fix_error, DEFVAL(Dictionary()));
	ClassDB::bind_method(D_METHOD("optimize_project", "args"), &VertexAIAssistant::optimize_project, DEFVAL(Dictionary()));
	ClassDB::bind_method(D_METHOD("reduce_memory_usage", "args"), &VertexAIAssistant::reduce_memory_usage, DEFVAL(Dictionary()));
	ClassDB::bind_method(D_METHOD("explain_node", "args"), &VertexAIAssistant::explain_node, DEFVAL(Dictionary()));
	ClassDB::bind_method(D_METHOD("create_animation", "args"), &VertexAIAssistant::create_animation, DEFVAL(Dictionary()));
	ClassDB::bind_method(D_METHOD("create_ui", "args"), &VertexAIAssistant::create_ui, DEFVAL(Dictionary()));
	ClassDB::bind_method(D_METHOD("optimize_shader", "args"), &VertexAIAssistant::optimize_shader, DEFVAL(Dictionary()));

	ADD_SIGNAL(MethodInfo("command_requested", PropertyInfo(Variant::STRING, "command"), PropertyInfo(Variant::DICTIONARY, "args")));
	ADD_SIGNAL(MethodInfo("confirmation_required", PropertyInfo(Variant::INT, "token"), PropertyInfo(Variant::STRING, "command")));
	ADD_SIGNAL(MethodInfo("command_executed", PropertyInfo(Variant::STRING, "command"), PropertyInfo(Variant::DICTIONARY, "result")));
}
