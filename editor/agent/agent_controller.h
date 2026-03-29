/**************************************************************************/
/*  agent_controller.h                                                    */
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

class AgentDock;
class AgentLLMBackend;
class AgentContext;
class AgentActions;

class AgentController : public Object {
	GDCLASS(AgentController, Object);

	static inline AgentController *singleton = nullptr;

	AgentDock *dock = nullptr;
	AgentLLMBackend *llm_backend = nullptr;
	AgentContext *context = nullptr;
	AgentActions *actions = nullptr;

	bool initialized = false;

	// Signal handlers.
	void _on_user_message(const String &p_message);
	void _on_token_received(const String &p_token);
	void _on_response_completed(const String &p_response);
	void _on_request_failed(const String &p_error);
	void _on_tool_call_received(const String &p_tool_name, const Dictionary &p_params);

	// Settings.
	void _load_settings();
	void _on_settings_changed();

	// System prompt.
	String _build_system_prompt();

protected:
	static void _bind_methods();

public:
	static AgentController *get_singleton() { return singleton; }

	void initialize(AgentDock *p_dock);
	void shutdown();

	AgentController();
	~AgentController();
};
