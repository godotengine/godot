/*************************************************************************/
/*  remote_debugger.h                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#ifndef REMOTE_DEBUGGER_H
#define REMOTE_DEBUGGER_H

#include "core/array.h"
#include "core/class_db.h"
#include "core/debugger/debugger_marshalls.h"
#include "core/debugger/engine_debugger.h"
#include "core/debugger/remote_debugger_peer.h"
#include "core/string_name.h"
#include "core/ustring.h"

class RemoteDebugger : public EngineDebugger {
public:
	enum MessageType {
		MESSAGE_TYPE_LOG,
		MESSAGE_TYPE_ERROR,
	};

private:
	typedef DebuggerMarshalls::OutputError ErrorMessage;

	struct NetworkProfiler;
	struct ServersProfiler;
	struct ScriptsProfiler;
	struct VisualProfiler;
	struct PerformanceProfiler;

	NetworkProfiler *network_profiler = nullptr;
	ServersProfiler *servers_profiler = nullptr;
	VisualProfiler *visual_profiler = nullptr;
	PerformanceProfiler *performance_profiler = nullptr;

	Ref<RemoteDebuggerPeer> peer;

	struct OutputString {
		String message;
		MessageType type;
	};
	List<OutputString> output_strings;
	List<ErrorMessage> errors;

	int n_messages_dropped = 0;
	int max_errors_per_second = 0;
	int max_chars_per_second = 0;
	int max_warnings_per_second = 0;
	int n_errors_dropped = 0;
	int n_warnings_dropped = 0;
	int char_count = 0;
	int err_count = 0;
	int warn_count = 0;
	int last_reset = 0;
	bool reload_all_scripts = false;

	// Make handlers and send_message thread safe.
	Mutex mutex;
	bool flushing = false;
	Thread::ID flush_thread = 0;

	PrintHandlerList phl;
	static void _print_handler(void *p_this, const String &p_string, bool p_error);
	ErrorHandlerList eh;
	static void _err_handler(void *p_this, const char *p_func, const char *p_file, int p_line, const char *p_err, const char *p_descr, ErrorHandlerType p_type);

	ErrorMessage _create_overflow_error(const String &p_what, const String &p_descr);
	Error _put_msg(String p_message, Array p_data);

	bool is_peer_connected() { return peer->is_peer_connected(); }
	void flush_output();

	void _send_resource_usage();
	void _send_stack_vars(List<String> &p_names, List<Variant> &p_vals, int p_type);

	Error _profiler_capture(const String &p_cmd, const Array &p_data, bool &r_captured);
	Error _core_capture(const String &p_cmd, const Array &p_data, bool &r_captured);

	template <typename T>
	void _bind_profiler(const String &p_name, T *p_prof);
	Error _try_capture(const String &p_name, const Array &p_data, bool &r_captured);

public:
	// Overrides
	void poll_events(bool p_is_idle);
	void send_message(const String &p_message, const Array &p_args);
	void send_error(const String &p_func, const String &p_file, int p_line, const String &p_err, const String &p_descr, ErrorHandlerType p_type);
	void debug(bool p_can_continue = true, bool p_is_error_breakpoint = false);

	RemoteDebugger(Ref<RemoteDebuggerPeer> p_peer);
	~RemoteDebugger();
};

#endif // REMOTE_DEBUGGER_H
