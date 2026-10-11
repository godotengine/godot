/**************************************************************************/
/*  remote_debugger_v2.h                                                  */
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
/* "Software"), to deal in the Software without restriction, including   */
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

#include "core/debugger/debug_session_v2.h"
#include "core/debugger/debug_transport_v2.h"
#include "core/debugger/debug_variable_store_v2.h"
#include "core/debugger/engine_debugger.h"
#include "core/os/mutex.h"
#include "core/templates/list.h"

class ScriptLanguage;

// Opt-in protocol for external debugger clients. The stock tcp:// debugger
// remains available to the Godot editor and unmodified DAP clients.
class RemoteDebuggerV2 : public EngineDebugger {
	struct QueuedMessage {
		String type;
		Array payload;
	};

	DebugTransportV2 transport;
	DebugSessionV2 session;
	DebugVariableStoreV2 variables;
	Mutex queue_mutex;
	List<QueuedMessage> queued_messages;
	bool stopping = false;
	bool in_break = false;
	uint64_t next_reconnect_msec = 0;
	uint64_t pending_scene_request_id = 0;
	uint64_t pending_scene_stop_id = 0;
	uint64_t pending_inspect_request_id = 0;
	uint64_t pending_inspect_stop_id = 0;
	uint64_t scene_revision = 0;

	PrintHandlerList print_handler;
	ErrorHandlerList error_handler;

	static void _print_handler(void *p_self, const String &p_text, bool p_error, bool p_rich);
	static void _error_handler(void *p_self, const char *p_function, const char *p_file, int p_line, const char *p_error, const char *p_description, bool p_editor_notify, ErrorHandlerType p_type);
	void _flush_messages();
	void _request_error(const DebugProtocolV2::Envelope &p_request, const String &p_code, const String &p_description);
	DebugSessionV2::Action _control(const DebugProtocolV2::Envelope &p_message);
	bool _poll_data(ScriptLanguage *p_language);
	void _poll_control_running();
	void _resume_without_frontend();

public:
	Error connect(const String &p_host, uint16_t p_port);
	void poll_events(bool p_is_idle) override;
	void send_message(const String &p_message, const Array &p_data) override;
	void send_error(const String &p_function, const String &p_file, int p_line, const String &p_error, const String &p_description, bool p_editor_notify, ErrorHandlerType p_type) override;
	void debug(bool p_can_continue = true, bool p_is_error_breakpoint = false) override;

	RemoteDebuggerV2();
	~RemoteDebuggerV2() override;
};
