/**************************************************************************/
/*  agent_controller.cpp                                                  */
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

#include "agent_controller.h"

#include "editor/agent/agent_actions.h"
#include "editor/agent/agent_context.h"
#include "editor/agent/agent_llm_backend.h"
#include "editor/docks/agent_dock.h"
#include "editor/settings/editor_settings.h"

#include "core/io/file_access.h"
#include "core/os/os.h"

void AgentController::_bind_methods() {
}

AgentController::AgentController() {
	singleton = this;
	llm_backend = memnew(AgentLLMBackend);
	context = memnew(AgentContext);
	actions = memnew(AgentActions);
}

AgentController::~AgentController() {
	if (initialized) {
		shutdown();
	}
	singleton = nullptr;
}

void AgentController::initialize(AgentDock *p_dock) {
	ERR_FAIL_NULL(p_dock);
	ERR_FAIL_COND(initialized);

	dock = p_dock;

	// Connect dock signals.
	dock->connect("message_sent", callable_mp(this, &AgentController::_on_user_message));

	// Connect LLM backend signals.
	llm_backend->connect("token_received", callable_mp(this, &AgentController::_on_token_received));
	llm_backend->connect("response_completed", callable_mp(this, &AgentController::_on_response_completed));
	llm_backend->connect("request_failed", callable_mp(this, &AgentController::_on_request_failed));
	llm_backend->connect("tool_call_received", callable_mp(this, &AgentController::_on_tool_call_received));

	// Defer context signal connections until editor is fully initialized.
	callable_mp(context, &AgentContext::connect_editor_signals).call_deferred();

	// Listen for settings changes.
	EditorSettings::get_singleton()->connect("settings_changed", callable_mp(this, &AgentController::_on_settings_changed));

	// Load settings and configure backend.
	_load_settings();

	// Set system prompt.
	llm_backend->set_system_prompt(_build_system_prompt());

	initialized = true;
}

void AgentController::shutdown() {
	if (!initialized) {
		return;
	}

	// Abort any pending LLM request.
	llm_backend->abort_current_request();

	// Disconnect signals.
	if (dock && dock->is_connected("message_sent", callable_mp(this, &AgentController::_on_user_message))) {
		dock->disconnect("message_sent", callable_mp(this, &AgentController::_on_user_message));
	}

	if (llm_backend->is_connected("token_received", callable_mp(this, &AgentController::_on_token_received))) {
		llm_backend->disconnect("token_received", callable_mp(this, &AgentController::_on_token_received));
	}
	if (llm_backend->is_connected("response_completed", callable_mp(this, &AgentController::_on_response_completed))) {
		llm_backend->disconnect("response_completed", callable_mp(this, &AgentController::_on_response_completed));
	}
	if (llm_backend->is_connected("request_failed", callable_mp(this, &AgentController::_on_request_failed))) {
		llm_backend->disconnect("request_failed", callable_mp(this, &AgentController::_on_request_failed));
	}
	if (llm_backend->is_connected("tool_call_received", callable_mp(this, &AgentController::_on_tool_call_received))) {
		llm_backend->disconnect("tool_call_received", callable_mp(this, &AgentController::_on_tool_call_received));
	}

	if (EditorSettings::get_singleton() && EditorSettings::get_singleton()->is_connected("settings_changed", callable_mp(this, &AgentController::_on_settings_changed))) {
		EditorSettings::get_singleton()->disconnect("settings_changed", callable_mp(this, &AgentController::_on_settings_changed));
	}

	// Clean up owned objects.
	memdelete(llm_backend);
	llm_backend = nullptr;
	memdelete(context);
	context = nullptr;
	memdelete(actions);
	actions = nullptr;

	dock = nullptr;
	initialized = false;
}

// --- Signal handlers ---

void AgentController::_on_user_message(const String &p_message) {
	// Gather current editor context if auto_context is enabled.
	String ctx;
	bool auto_context = EDITOR_GET("agent/chat/auto_context");
	if (auto_context) {
		ctx = context->build_context_string();
	}

	// Send to LLM with context.
	llm_backend->send_message(p_message, ctx);

	// Update dock status.
	dock->set_status("Thinking...");
}

void AgentController::_on_token_received(const String &p_token) {
	dock->append_streamed_token(p_token);
}

void AgentController::_on_response_completed(const String &p_response) {
	dock->finish_streamed_response();
	dock->set_status("Idle");
}

void AgentController::_on_request_failed(const String &p_error) {
	dock->finish_streamed_response();
	dock->set_status("Error: " + p_error);
	dock->append_system_message("Request failed: " + p_error);
}

void AgentController::_on_tool_call_received(const String &p_tool_name, const Dictionary &p_params) {
	// Check if file delete is allowed when the tool is a delete action.
	if (p_tool_name == "delete_file") {
		bool allow_delete = EDITOR_GET("agent/actions/allow_file_delete");
		if (!allow_delete) {
			dock->set_status("Blocked: file deletion disabled");
			dock->append_system_message("File deletion is disabled in Agent settings.");
			// Send error result back via a follow-up message.
			llm_backend->send_message("Tool error for '" + p_tool_name + "': File deletion is disabled in editor settings.", String());
			return;
		}
	}

	// Execute the action.
	String result = actions->execute_action(p_tool_name, p_params);

	// Show result in dock.
	dock->append_system_message("Tool '" + p_tool_name + "' executed.");
	dock->set_status("Idle");

	// Send result back to LLM for continued conversation.
	llm_backend->send_message("Tool result for '" + p_tool_name + "': " + result, String());
}

// --- Settings ---

void AgentController::_load_settings() {
	String provider_str = EDITOR_GET("agent/api/provider");
	String model = EDITOR_GET("agent/api/model");

	// Map provider string to enum.
	AgentLLMBackend::Provider provider = AgentLLMBackend::PROVIDER_ANTHROPIC;
	if (provider_str == "openai") {
		provider = AgentLLMBackend::PROVIDER_OPENAI;
	} else if (provider_str == "local") {
		provider = AgentLLMBackend::PROVIDER_LOCAL;
	}

	llm_backend->set_provider(provider);
	llm_backend->set_model(model);

	// Resolve API key: env var > .env file > EditorSettings.
	String api_key;

	if (provider == AgentLLMBackend::PROVIDER_ANTHROPIC) {
		// 1. Check environment variable.
		api_key = OS::get_singleton()->get_environment("ANTHROPIC_API_KEY");

		// 2. Check .env file next to the executable.
		if (api_key.is_empty()) {
			String exe_dir = OS::get_singleton()->get_executable_path().get_base_dir();
			String env_path = exe_dir.path_join("../.env"); // bin/ -> project root.
			Ref<FileAccess> env_file = FileAccess::open(env_path, FileAccess::READ);
			if (env_file.is_valid()) {
				while (!env_file->eof_reached()) {
					String line = env_file->get_line().strip_edges();
					if (line.begins_with("ANTHROPIC_API_KEY=")) {
						api_key = line.substr(String("ANTHROPIC_API_KEY=").length());
						break;
					}
				}
			}
		}

		// 3. Fall back to EditorSettings.
		if (api_key.is_empty()) {
			api_key = EDITOR_GET("agent/api/anthropic_api_key");
		}

		llm_backend->set_api_key(api_key);
	} else if (provider == AgentLLMBackend::PROVIDER_OPENAI) {
		api_key = OS::get_singleton()->get_environment("OPENAI_API_KEY");
		if (api_key.is_empty()) {
			api_key = EDITOR_GET("agent/api/openai_api_key");
		}
		llm_backend->set_api_key(api_key);
	} else if (provider == AgentLLMBackend::PROVIDER_LOCAL) {
		String endpoint = EDITOR_GET("agent/api/local_endpoint");
		llm_backend->set_api_base_url(endpoint);
	}
}

void AgentController::_on_settings_changed() {
	if (!initialized) {
		return;
	}

	if (EditorSettings::get_singleton()->check_changed_settings_in_group("agent/")) {
		_load_settings();
		llm_backend->set_system_prompt(_build_system_prompt());
	}
}

String AgentController::_build_system_prompt() {
	String prompt;
	prompt += "You are an AI assistant integrated into the Godot game engine editor. ";
	prompt += "You help users create and modify game projects.\n\n";

	prompt += "You have access to the following tools:\n";
	prompt += actions->get_tool_definitions_json();
	prompt += "\n\n";

	prompt += "When the user asks you to create or modify something, use the appropriate tools. ";
	prompt += "Always explain what you're about to do before using tools.\n\n";

	prompt += "Current editor context will be injected with each message.\n";

	return prompt;
}
