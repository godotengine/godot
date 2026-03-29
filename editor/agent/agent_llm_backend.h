/**************************************************************************/
/*  agent_llm_backend.h                                                   */
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

#include "core/io/http_client.h"
#include "core/object/object.h"
#include "core/os/mutex.h"
#include "core/os/thread.h"
#include "core/string/ustring.h"
#include "core/templates/safe_refcount.h"
#include "core/variant/array.h"
#include "core/variant/dictionary.h"

class AgentLLMBackend : public Object {
	GDCLASS(AgentLLMBackend, Object);

	static inline AgentLLMBackend *singleton = nullptr;

public:
	enum Provider {
		PROVIDER_ANTHROPIC,
		PROVIDER_OPENAI,
		PROVIDER_LOCAL,
	};

	struct Message {
		String role; // "user", "assistant", "system"
		String content;
	};

private:
	Provider current_provider = PROVIDER_ANTHROPIC;
	String api_key;
	String model_id;
	String api_base_url;
	int api_port = 443;

	// Conversation.
	Mutex conversation_mutex;
	Vector<Message> conversation_history;
	String system_prompt;

	// Streaming.
	Thread request_thread;
	SafeFlag abort_request;
	SafeFlag requesting;
	String accumulated_response;

	// Thread data.
	struct RequestData {
		String user_message;
		String context;
	};
	RequestData thread_data;

	// Internal.
	static void _thread_func(void *p_userdata);
	void _thread_request();
	void _process_anthropic_stream(Ref<HTTPClient> p_client);
	void _process_openai_stream(Ref<HTTPClient> p_client);
	Dictionary _build_anthropic_request(const String &p_user_message, const String &p_context);
	Dictionary _build_openai_request(const String &p_user_message, const String &p_context);
	Array _build_messages_array() const;
	String _build_tool_definitions_json() const;
	void _parse_tool_calls(const String &p_response);

	void _emit_token(const String &p_token);
	void _emit_response_completed(const String &p_response);
	void _emit_request_failed(const String &p_error);
	void _emit_tool_call(const String &p_tool_name, const Dictionary &p_params);

protected:
	static void _bind_methods();

public:
	static AgentLLMBackend *get_singleton() { return singleton; }

	// Configuration.
	void set_provider(Provider p_provider);
	Provider get_provider() const;
	void set_api_key(const String &p_key);
	String get_api_key() const;
	void set_model(const String &p_model);
	String get_model() const;
	void set_api_base_url(const String &p_url);
	String get_api_base_url() const;
	void set_system_prompt(const String &p_prompt);
	String get_system_prompt() const;

	// Conversation.
	void send_message(const String &p_message, const String &p_context = String());
	void abort_current_request();
	void clear_conversation();

	// State.
	bool is_requesting() const;
	Vector<Message> get_conversation_history() const;

	AgentLLMBackend();
	~AgentLLMBackend();
};

VARIANT_ENUM_CAST(AgentLLMBackend::Provider);
