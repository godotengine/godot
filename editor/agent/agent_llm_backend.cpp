/**************************************************************************/
/*  agent_llm_backend.cpp                                                 */
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

#include "agent_llm_backend.h"

#include "core/io/json.h"
#include "core/io/stream_peer_tls.h"
#include "core/os/os.h"

void AgentLLMBackend::_bind_methods() {
	// Signals.
	ADD_SIGNAL(MethodInfo("token_received", PropertyInfo(Variant::STRING, "token")));
	ADD_SIGNAL(MethodInfo("response_completed", PropertyInfo(Variant::STRING, "full_response")));
	ADD_SIGNAL(MethodInfo("request_failed", PropertyInfo(Variant::STRING, "error")));
	ADD_SIGNAL(MethodInfo("tool_call_received", PropertyInfo(Variant::STRING, "tool_name"), PropertyInfo(Variant::DICTIONARY, "params")));

	// Enum constants.
	BIND_ENUM_CONSTANT(PROVIDER_ANTHROPIC);
	BIND_ENUM_CONSTANT(PROVIDER_OPENAI);
	BIND_ENUM_CONSTANT(PROVIDER_LOCAL);
}

void AgentLLMBackend::set_provider(Provider p_provider) {
	current_provider = p_provider;
	switch (p_provider) {
		case PROVIDER_ANTHROPIC: {
			api_base_url = "api.anthropic.com";
			api_port = 443;
			if (model_id.is_empty()) {
				model_id = "claude-sonnet-4-20250514";
			}
		} break;
		case PROVIDER_OPENAI: {
			api_base_url = "api.openai.com";
			api_port = 443;
			if (model_id.is_empty()) {
				model_id = "gpt-4o";
			}
		} break;
		case PROVIDER_LOCAL: {
			api_base_url = "127.0.0.1";
			api_port = 8080;
			if (model_id.is_empty()) {
				model_id = "local";
			}
		} break;
	}
}

AgentLLMBackend::Provider AgentLLMBackend::get_provider() const {
	return current_provider;
}

void AgentLLMBackend::set_api_key(const String &p_key) {
	api_key = p_key;
}

String AgentLLMBackend::get_api_key() const {
	return api_key;
}

void AgentLLMBackend::set_model(const String &p_model) {
	model_id = p_model;
}

String AgentLLMBackend::get_model() const {
	return model_id;
}

void AgentLLMBackend::set_api_base_url(const String &p_url) {
	api_base_url = p_url;
}

String AgentLLMBackend::get_api_base_url() const {
	return api_base_url;
}

void AgentLLMBackend::set_system_prompt(const String &p_prompt) {
	system_prompt = p_prompt;
}

String AgentLLMBackend::get_system_prompt() const {
	return system_prompt;
}

void AgentLLMBackend::send_message(const String &p_message, const String &p_context) {
	ERR_FAIL_COND_MSG(requesting.is_set(), "A request is already in progress.");
	ERR_FAIL_COND_MSG(api_key.is_empty(), "API key is not set.");

	// Add user message to conversation history.
	{
		MutexLock lock(conversation_mutex);
		Message msg;
		msg.role = "user";
		msg.content = p_message;
		conversation_history.push_back(msg);
	}

	// Wait for any previous thread to finish.
	if (request_thread.is_started()) {
		request_thread.wait_to_finish();
	}

	requesting.set();
	abort_request.clear();
	accumulated_response.clear();

	thread_data.user_message = p_message;
	thread_data.context = p_context;
	request_thread.start(_thread_func, this);
}

void AgentLLMBackend::_thread_func(void *p_userdata) {
	AgentLLMBackend *self = static_cast<AgentLLMBackend *>(p_userdata);
	self->_thread_request();
}

void AgentLLMBackend::abort_current_request() {
	if (requesting.is_set()) {
		abort_request.set();
	}
}

void AgentLLMBackend::clear_conversation() {
	MutexLock lock(conversation_mutex);
	conversation_history.clear();
	accumulated_response.clear();
}

bool AgentLLMBackend::is_requesting() const {
	return requesting.is_set();
}

Vector<AgentLLMBackend::Message> AgentLLMBackend::get_conversation_history() const {
	return conversation_history;
}

Array AgentLLMBackend::_build_messages_array() const {
	Array messages;
	for (int i = 0; i < conversation_history.size(); i++) {
		Dictionary msg;
		msg["role"] = conversation_history[i].role;
		msg["content"] = conversation_history[i].content;
		messages.push_back(msg);
	}
	return messages;
}

Dictionary AgentLLMBackend::_build_anthropic_request(const String &p_user_message, const String &p_context) {
	Dictionary request;
	request["model"] = model_id;
	request["max_tokens"] = 4096;
	request["stream"] = true;

	// Build system prompt with context.
	String full_system = system_prompt;
	if (!p_context.is_empty()) {
		full_system += "\n\n" + p_context;
	}
	request["system"] = full_system;

	// Build messages array.
	Array messages;
	{
		MutexLock lock(conversation_mutex);
		messages = _build_messages_array();
	}
	request["messages"] = messages;

	return request;
}

Dictionary AgentLLMBackend::_build_openai_request(const String &p_user_message, const String &p_context) {
	Dictionary request;
	request["model"] = model_id;
	request["max_tokens"] = 4096;
	request["stream"] = true;

	// Build messages array with system message prepended.
	Array messages;
	{
		Dictionary sys_msg;
		sys_msg["role"] = "system";
		String full_system = system_prompt;
		if (!p_context.is_empty()) {
			full_system += "\n\n" + p_context;
		}
		sys_msg["content"] = full_system;
		messages.push_back(sys_msg);
	}
	{
		MutexLock lock(conversation_mutex);
		Array conv = _build_messages_array();
		for (int i = 0; i < conv.size(); i++) {
			messages.push_back(conv[i]);
		}
	}
	request["messages"] = messages;

	return request;
}

String AgentLLMBackend::_build_tool_definitions_json() const {
	// Stub tool definitions for future use.
	Array tools;

	{
		Dictionary tool;
		tool["name"] = "create_script";
		tool["description"] = "Create a new script file at the specified path with the given content.";
		Dictionary params;
		params["type"] = "object";
		Dictionary properties;
		{
			Dictionary path_prop;
			path_prop["type"] = "string";
			path_prop["description"] = "The res:// path for the new script file.";
			properties["path"] = path_prop;
		}
		{
			Dictionary content_prop;
			content_prop["type"] = "string";
			content_prop["description"] = "The content of the script file.";
			properties["content"] = content_prop;
		}
		params["properties"] = properties;
		Array required;
		required.push_back("path");
		required.push_back("content");
		params["required"] = required;
		tool["input_schema"] = params;
		tools.push_back(tool);
	}

	{
		Dictionary tool;
		tool["name"] = "edit_script";
		tool["description"] = "Edit an existing script file by replacing old content with new content.";
		Dictionary params;
		params["type"] = "object";
		Dictionary properties;
		{
			Dictionary path_prop;
			path_prop["type"] = "string";
			path_prop["description"] = "The res:// path of the script file to edit.";
			properties["path"] = path_prop;
		}
		{
			Dictionary old_prop;
			old_prop["type"] = "string";
			old_prop["description"] = "The text to search for and replace.";
			properties["old_content"] = old_prop;
		}
		{
			Dictionary new_prop;
			new_prop["type"] = "string";
			new_prop["description"] = "The replacement text.";
			properties["new_content"] = new_prop;
		}
		params["properties"] = properties;
		Array required;
		required.push_back("path");
		required.push_back("old_content");
		required.push_back("new_content");
		params["required"] = required;
		tool["input_schema"] = params;
		tools.push_back(tool);
	}

	{
		Dictionary tool;
		tool["name"] = "read_file";
		tool["description"] = "Read the contents of a file.";
		Dictionary params;
		params["type"] = "object";
		Dictionary properties;
		{
			Dictionary path_prop;
			path_prop["type"] = "string";
			path_prop["description"] = "The res:// path of the file to read.";
			properties["path"] = path_prop;
		}
		params["properties"] = properties;
		Array required;
		required.push_back("path");
		params["required"] = required;
		tool["input_schema"] = params;
		tools.push_back(tool);
	}

	{
		Dictionary tool;
		tool["name"] = "list_files";
		tool["description"] = "List files in a directory.";
		Dictionary params;
		params["type"] = "object";
		Dictionary properties;
		{
			Dictionary dir_prop;
			dir_prop["type"] = "string";
			dir_prop["description"] = "The res:// directory path to list. Defaults to res://.";
			properties["directory"] = dir_prop;
		}
		params["properties"] = properties;
		tool["input_schema"] = params;
		tools.push_back(tool);
	}

	{
		Dictionary tool;
		tool["name"] = "add_node";
		tool["description"] = "Add a new node to the current scene.";
		Dictionary params;
		params["type"] = "object";
		Dictionary properties;
		{
			Dictionary type_prop;
			type_prop["type"] = "string";
			type_prop["description"] = "The type of node to create (e.g., Node3D, CharacterBody3D).";
			properties["type"] = type_prop;
		}
		{
			Dictionary parent_prop;
			parent_prop["type"] = "string";
			parent_prop["description"] = "NodePath to the parent node. Use '.' for scene root.";
			properties["parent"] = parent_prop;
		}
		{
			Dictionary name_prop;
			name_prop["type"] = "string";
			name_prop["description"] = "The name for the new node.";
			properties["name"] = name_prop;
		}
		params["properties"] = properties;
		Array required;
		required.push_back("type");
		required.push_back("name");
		params["required"] = required;
		tool["input_schema"] = params;
		tools.push_back(tool);
	}

	{
		Dictionary tool;
		tool["name"] = "set_property";
		tool["description"] = "Set a property on a node in the current scene.";
		Dictionary params;
		params["type"] = "object";
		Dictionary properties;
		{
			Dictionary np_prop;
			np_prop["type"] = "string";
			np_prop["description"] = "NodePath to the target node.";
			properties["node_path"] = np_prop;
		}
		{
			Dictionary prop_prop;
			prop_prop["type"] = "string";
			prop_prop["description"] = "The property name to set.";
			properties["property"] = prop_prop;
		}
		{
			Dictionary val_prop;
			val_prop["type"] = "string";
			val_prop["description"] = "The value to set (will be parsed appropriately).";
			properties["value"] = val_prop;
		}
		params["properties"] = properties;
		Array required;
		required.push_back("node_path");
		required.push_back("property");
		required.push_back("value");
		params["required"] = required;
		tool["input_schema"] = params;
		tools.push_back(tool);
	}

	return JSON::stringify(tools, "\t");
}

void AgentLLMBackend::_parse_tool_calls(const String &p_response) {
	// Parse tool_use blocks from Anthropic response format.
	// This is a simplified parser; full implementation would handle the streaming tool_use events.
	// For now, this is a stub that will be expanded when tool use is fully integrated.
}

void AgentLLMBackend::_emit_token(const String &p_token) {
	emit_signal(SNAME("token_received"), p_token);
}

void AgentLLMBackend::_emit_response_completed(const String &p_response) {
	emit_signal(SNAME("response_completed"), p_response);
}

void AgentLLMBackend::_emit_request_failed(const String &p_error) {
	emit_signal(SNAME("request_failed"), p_error);
}

void AgentLLMBackend::_emit_tool_call(const String &p_tool_name, const Dictionary &p_params) {
	emit_signal(SNAME("tool_call_received"), p_tool_name, p_params);
}

void AgentLLMBackend::_thread_request() {
	String user_message = thread_data.user_message;
	String context = thread_data.context;

	// Determine endpoint and headers based on provider.
	String host = api_base_url;
	int port = api_port;
	String request_path;
	Vector<String> headers;
	Dictionary request_body;

	switch (current_provider) {
		case PROVIDER_ANTHROPIC: {
			request_path = "/v1/messages";
			headers.push_back("Content-Type: application/json");
			headers.push_back("x-api-key: " + api_key);
			headers.push_back("anthropic-version: 2023-06-01");
			request_body = _build_anthropic_request(user_message, context);
		} break;
		case PROVIDER_OPENAI: {
			request_path = "/v1/chat/completions";
			headers.push_back("Content-Type: application/json");
			headers.push_back("Authorization: Bearer " + api_key);
			request_body = _build_openai_request(user_message, context);
		} break;
		case PROVIDER_LOCAL: {
			request_path = "/v1/chat/completions";
			headers.push_back("Content-Type: application/json");
			request_body = _build_openai_request(user_message, context);
		} break;
	}

	String body_text = JSON::stringify(request_body);

	// Create HTTP client and connect.
	Ref<HTTPClient> client = HTTPClient::create();
	ERR_FAIL_COND(client.is_null());

	bool use_tls = (current_provider != PROVIDER_LOCAL);
	Ref<TLSOptions> tls_options;
	if (use_tls) {
		tls_options = TLSOptions::client();
	}

	Error err = client->connect_to_host(host, port, tls_options);
	if (err != OK) {
		call_deferred(SNAME("_emit_request_failed"), "Failed to connect to " + host);
		requesting.clear();
		return;
	}

	// Wait for connection.
	while (client->get_status() == HTTPClient::STATUS_CONNECTING ||
			client->get_status() == HTTPClient::STATUS_RESOLVING) {
		client->poll();
		if (abort_request.is_set()) {
			client->close();
			requesting.clear();
			return;
		}
		OS::get_singleton()->delay_usec(10000); // 10ms.
	}

	if (client->get_status() != HTTPClient::STATUS_CONNECTED) {
		call_deferred(SNAME("_emit_request_failed"), "Connection failed with status: " + itos(client->get_status()));
		requesting.clear();
		return;
	}

	// Send request.
	CharString body_utf8 = body_text.utf8();
	err = client->request(HTTPClient::METHOD_POST, request_path, headers, (const uint8_t *)body_utf8.get_data(), body_utf8.length());
	if (err != OK) {
		call_deferred(SNAME("_emit_request_failed"), "Failed to send request.");
		requesting.clear();
		return;
	}

	// Wait for response.
	while (client->get_status() == HTTPClient::STATUS_REQUESTING) {
		client->poll();
		if (abort_request.is_set()) {
			client->close();
			requesting.clear();
			return;
		}
		OS::get_singleton()->delay_usec(10000);
	}

	if (!client->has_response()) {
		call_deferred(SNAME("_emit_request_failed"), "No response received from server.");
		requesting.clear();
		return;
	}

	int response_code = client->get_response_code();
	if (response_code != 200) {
		// Read error body.
		String error_body;
		while (client->get_status() == HTTPClient::STATUS_BODY) {
			client->poll();
			PackedByteArray chunk = client->read_response_body_chunk();
			if (chunk.size() > 0) {
				error_body += String::utf8((const char *)chunk.ptr(), chunk.size());
			}
			if (abort_request.is_set()) {
				break;
			}
		}
		call_deferred(SNAME("_emit_request_failed"), "HTTP " + itos(response_code) + ": " + error_body);
		requesting.clear();
		return;
	}

	// Process streaming response based on provider.
	if (current_provider == PROVIDER_ANTHROPIC) {
		_process_anthropic_stream(client);
	} else {
		_process_openai_stream(client);
	}

	client->close();

	// Add assistant response to conversation history.
	if (!abort_request.is_set() && !accumulated_response.is_empty()) {
		MutexLock lock(conversation_mutex);
		Message msg;
		msg.role = "assistant";
		msg.content = accumulated_response;
		conversation_history.push_back(msg);
	}

	if (!abort_request.is_set()) {
		call_deferred(SNAME("_emit_response_completed"), accumulated_response);
	}

	requesting.clear();
}

void AgentLLMBackend::_process_anthropic_stream(Ref<HTTPClient> p_client) {
	// Anthropic SSE format:
	// event: content_block_delta
	// data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"..."}}
	//
	// event: message_stop
	// data: {"type":"message_stop"}

	String buffer;

	while (p_client->get_status() == HTTPClient::STATUS_BODY) {
		if (abort_request.is_set()) {
			return;
		}

		p_client->poll();
		PackedByteArray chunk = p_client->read_response_body_chunk();
		if (chunk.size() == 0) {
			OS::get_singleton()->delay_usec(1000);
			continue;
		}

		buffer += String::utf8((const char *)chunk.ptr(), chunk.size());

		// Process complete SSE lines.
		while (buffer.contains("\n")) {
			int newline_pos = buffer.find("\n");
			String line = buffer.substr(0, newline_pos).strip_edges();
			buffer = buffer.substr(newline_pos + 1);

			if (line.begins_with("data: ")) {
				String json_str = line.substr(6);
				if (json_str == "[DONE]") {
					return;
				}

				Variant parsed = JSON::parse_string(json_str);
				if (parsed.get_type() != Variant::DICTIONARY) {
					continue;
				}

				Dictionary data = parsed;
				String type = data.get("type", "");

				if (type == "content_block_delta") {
					Dictionary delta = data.get("delta", Dictionary());
					String delta_type = delta.get("type", "");
					if (delta_type == "text_delta") {
						String text = delta.get("text", "");
						if (!text.is_empty()) {
							accumulated_response += text;
							call_deferred(SNAME("_emit_token"), text);
						}
					}
				} else if (type == "message_stop") {
					return;
				} else if (type == "error") {
					Dictionary error = data.get("error", Dictionary());
					String error_msg = error.get("message", "Unknown error");
					call_deferred(SNAME("_emit_request_failed"), error_msg);
					return;
				}
			}
		}
	}
}

void AgentLLMBackend::_process_openai_stream(Ref<HTTPClient> p_client) {
	// OpenAI SSE format:
	// data: {"choices":[{"delta":{"content":"..."}}]}
	// data: [DONE]

	String buffer;

	while (p_client->get_status() == HTTPClient::STATUS_BODY) {
		if (abort_request.is_set()) {
			return;
		}

		p_client->poll();
		PackedByteArray chunk = p_client->read_response_body_chunk();
		if (chunk.size() == 0) {
			OS::get_singleton()->delay_usec(1000);
			continue;
		}

		buffer += String::utf8((const char *)chunk.ptr(), chunk.size());

		// Process complete SSE lines.
		while (buffer.contains("\n")) {
			int newline_pos = buffer.find("\n");
			String line = buffer.substr(0, newline_pos).strip_edges();
			buffer = buffer.substr(newline_pos + 1);

			if (line.begins_with("data: ")) {
				String json_str = line.substr(6);
				if (json_str == "[DONE]") {
					return;
				}

				Variant parsed = JSON::parse_string(json_str);
				if (parsed.get_type() != Variant::DICTIONARY) {
					continue;
				}

				Dictionary data = parsed;
				Array choices = data.get("choices", Array());
				if (choices.size() > 0) {
					Dictionary choice = choices[0];
					Dictionary delta = choice.get("delta", Dictionary());
					if (delta.has("content")) {
						String text = delta.get("content", "");
						if (!text.is_empty()) {
							accumulated_response += text;
							call_deferred(SNAME("_emit_token"), text);
						}
					}
				}
			}
		}
	}
}

AgentLLMBackend::AgentLLMBackend() {
	singleton = this;

	// Set defaults.
	current_provider = PROVIDER_ANTHROPIC;
	model_id = "claude-sonnet-4-20250514";
	api_base_url = "api.anthropic.com";
	api_port = 443;

	system_prompt = "You are an AI assistant integrated into the Godot game engine editor. "
					"You help users build games by creating scripts, adding nodes, setting properties, "
					"and explaining Godot concepts. You have access to the current scene tree, selected nodes, "
					"open scripts, and project files. When asked to make changes, use the available tools to "
					"modify the project. Always explain what you are doing and why. "
					"Respond concisely and focus on practical solutions.";
}

AgentLLMBackend::~AgentLLMBackend() {
	if (requesting.is_set()) {
		abort_request.set();
	}
	if (request_thread.is_started()) {
		request_thread.wait_to_finish();
	}
	if (singleton == this) {
		singleton = nullptr;
	}
}
