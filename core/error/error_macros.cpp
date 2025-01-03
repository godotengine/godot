/**************************************************************************/
/*  error_macros.cpp                                                      */
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

#include "error_macros.h"

#include "core/io/logger.h"
#include "core/object/script_language.h"
#include "core/os/os.h"
#include "core/string/ustring.h"

// Optional physics interpolation warnings try to include the path to the relevant node.
#if defined(DEBUG_ENABLED) && defined(TOOLS_ENABLED)
#include "core/config/project_settings.h"
#include "scene/main/node.h"
#endif

static ErrorHandlerList *error_handler_list = nullptr;

void add_error_handler(ErrorHandlerList *p_handler) {
	// If p_handler is already in error_handler_list
	// we'd better remove it first then we can add it.
	// This prevent cyclic redundancy.
	remove_error_handler(p_handler);

	_global_lock();

	p_handler->next = error_handler_list;
	error_handler_list = p_handler;

	_global_unlock();
}

void remove_error_handler(const ErrorHandlerList *p_handler) {
	_global_lock();

	ErrorHandlerList *prev = nullptr;
	ErrorHandlerList *l = error_handler_list;

	while (l) {
		if (l == p_handler) {
			if (prev) {
				prev->next = l->next;
			} else {
				error_handler_list = l->next;
			}
			break;
		}
		prev = l;
		l = l->next;
	}

	_global_unlock();
}

// Errors without messages.
void _err_print_error(const char *p_function, const char *p_file, int p_line, const char *p_error, bool p_editor_notify, ErrorHandlerType p_type) {
	_err_print_error(p_function, p_file, p_line, p_error, "", p_editor_notify, p_type);
}

void _err_print_error(const char *p_function, const char *p_file, int p_line, const String &p_error, bool p_editor_notify, ErrorHandlerType p_type) {
	_err_print_error(p_function, p_file, p_line, p_error.utf8().get_data(), "", p_editor_notify, p_type);
}

// Main error printing function.
void _err_print_error(const char *p_function, const char *p_file, int p_line, const char *p_error, const char *p_message, bool p_editor_notify, ErrorHandlerType p_type) {
	if (OS::get_singleton()) {
		OS::get_singleton()->print_error(p_function, p_file, p_line, p_error, p_message, p_editor_notify, (Logger::ErrorType)p_type);
	} else {
		// Fallback if errors happen before OS init or after it's destroyed.
		const char *err_details = (p_message && *p_message) ? p_message : p_error;
		fprintf(stderr, "ERROR: %s\n   at: %s (%s:%i)\n", err_details, p_function, p_file, p_line);
	}

	_global_lock();
	ErrorHandlerList *l = error_handler_list;
	while (l) {
		l->errfunc(l->userdata, p_function, p_file, p_line, p_error, p_message, p_editor_notify, p_type);
		l = l->next;
	}

	_global_unlock();
}

// For printing errors when we may crash at any point, so we must flush ASAP a lot of lines
// but we don't want to make it noisy by printing lots of file & line info (because it's already
// been printing by a preceding _err_print_error).
void _err_print_error_asap(const String &p_error, ErrorHandlerType p_type) {
	if (OS::get_singleton()) {
		OS::get_singleton()->printerr("ERROR: %s\n", p_error.utf8().get_data());
	} else {
		// Fallback if errors happen before OS init or after it's destroyed.
		const char *err_details = p_error.utf8().get_data();
		fprintf(stderr, "ERROR: %s\n", err_details);
	}

	_global_lock();
	ErrorHandlerList *l = error_handler_list;
	while (l) {
		l->errfunc(l->userdata, "", "", 0, p_error.utf8().get_data(), "", false, p_type);
		l = l->next;
	}

	_global_unlock();
}

// Errors with message. (All combinations of p_error and p_message as String or char*.)
void _err_print_error(const char *p_function, const char *p_file, int p_line, const String &p_error, const char *p_message, bool p_editor_notify, ErrorHandlerType p_type) {
	_err_print_error(p_function, p_file, p_line, p_error.utf8().get_data(), p_message, p_editor_notify, p_type);
}

void _err_print_error(const char *p_function, const char *p_file, int p_line, const char *p_error, const String &p_message, bool p_editor_notify, ErrorHandlerType p_type) {
	_err_print_error(p_function, p_file, p_line, p_error, p_message.utf8().get_data(), p_editor_notify, p_type);
}

void _err_print_error(const char *p_function, const char *p_file, int p_line, const String &p_error, const String &p_message, bool p_editor_notify, ErrorHandlerType p_type) {
	_err_print_error(p_function, p_file, p_line, p_error.utf8().get_data(), p_message.utf8().get_data(), p_editor_notify, p_type);
}

// Index errors. (All combinations of p_message as String or char*.)
void _err_print_index_error(const char *p_function, const char *p_file, int p_line, int64_t p_index, int64_t p_size, const char *p_index_str, const char *p_size_str, const char *p_message, bool p_editor_notify, bool p_fatal) {
	String fstr(p_fatal ? "FATAL: " : "");
	String err(fstr + "Index " + p_index_str + " = " + itos(p_index) + " is out of bounds (" + p_size_str + " = " + itos(p_size) + ").");
	_err_print_error(p_function, p_file, p_line, err.utf8().get_data(), p_message, p_editor_notify, ERR_HANDLER_ERROR);
}

void _err_print_index_error(const char *p_function, const char *p_file, int p_line, int64_t p_index, int64_t p_size, const char *p_index_str, const char *p_size_str, const String &p_message, bool p_editor_notify, bool p_fatal) {
	_err_print_index_error(p_function, p_file, p_line, p_index, p_size, p_index_str, p_size_str, p_message.utf8().get_data(), p_editor_notify, p_fatal);
}

void _err_print_callstack(const String &p_error, bool p_editor_notify, ErrorHandlerType p_type) {
	// Print detailed call stack information from everywhere available. It is recommended to only
	// use this for debugging, as it has a fairly high overhead.
	String callstack;

	// Print script stack frames, if available.
	Vector<ScriptLanguage::StackInfo> si;
	for (int i = 0; i < ScriptServer::get_language_count(); i++) {
		si = ScriptServer::get_language(i)->debug_get_current_stack_info();
		if (si.size()) {
			callstack += "Callstack from " + ScriptServer::get_language(i)->get_name() + ":\n";
			for (int j = 0; j < si.size(); ++j) {
				callstack += si[i].file + ':' + itos(si[i].line) + " @ " + si[i].func + '\n';
			}
			callstack += '\n';
		}
	}

	// Print C++ call stack.
	Vector<OS::StackInfo> cpp_stack = OS::get_singleton()->get_cpp_stack_info();
	callstack += "C++ call stack:\n";
	for (int i = 0; i < cpp_stack.size(); ++i) {
		String descriptor = OS::get_singleton()->get_debug_descriptor(cpp_stack[i]);
		callstack += descriptor + " (" + cpp_stack[i].file + ":0x" + String::num_uint64(cpp_stack[i].offset, 16) + " @ " + cpp_stack[i].function + ")\n";
	}

	_err_print_error(__FUNCTION__, __FILE__, __LINE__, p_error + '\n' + callstack, p_editor_notify, p_type);
}

void _err_print_error_backtrace(const char *p_filter, const String &p_error, bool p_editor_notify, ErrorHandlerType p_type) {
	// Print script stack frame, if available.
	Vector<ScriptLanguage::StackInfo> si;
	for (int i = 0; i < ScriptServer::get_language_count(); i++) {
		si = ScriptServer::get_language(i)->debug_get_current_stack_info();
		if (si.size()) {
			_err_print_error(si[0].func.utf8(), si[0].file.utf8(), si[0].line, p_error, p_editor_notify, p_type);
			return;
		}
	}

	// If there is not a script stack frame, use the C++ stack frame.
	Vector<OS::StackInfo> cpp_stack = OS::get_singleton()->get_cpp_stack_info();

	for (int i = 1; i < cpp_stack.size(); ++i) {
		if (!cpp_stack[i].function.contains(p_filter)) {
			String descriptor = OS::get_singleton()->get_debug_descriptor(cpp_stack[i]);
			if (descriptor.is_empty()) {
				// If we can't get debug info, just print binary file name and address.
				_err_print_error(cpp_stack[i].function.utf8(), cpp_stack[i].file.utf8(), cpp_stack[i].offset, p_error, p_editor_notify, p_type);
			} else {
				// Expect debug descriptor to replace file and line info.
				_err_print_error(cpp_stack[i].function.utf8(), cpp_stack[i].file.utf8(), cpp_stack[i].offset, "", descriptor + ": " + p_error, p_editor_notify, p_type);
			}
			return;
		}
	}

	// If there is no usable stack frame (this should basically never happen), fall back to using the current stack frame.
	_err_print_error(__FUNCTION__, __FILE__, __LINE__, p_error, p_editor_notify, p_type);
}

void _err_flush_stdout() {
	fflush(stdout);
}

// Prevent error spam by limiting the warnings to a certain frequency.
void _physics_interpolation_warning(const char *p_function, const char *p_file, int p_line, ObjectID p_id, const char *p_warn_string) {
#if defined(DEBUG_ENABLED) && defined(TOOLS_ENABLED)
	const uint32_t warn_max = 2048;
	const uint32_t warn_timeout_seconds = 15;

	static uint32_t warn_count = warn_max;
	static uint32_t warn_timeout = warn_timeout_seconds;

	uint32_t time_now = UINT32_MAX;

	if (warn_count) {
		warn_count--;
	}

	if (!warn_count) {
		time_now = OS::get_singleton()->get_ticks_msec() / 1000;
	}

	if ((warn_count == 0) && (time_now >= warn_timeout)) {
		warn_count = warn_max;
		warn_timeout = time_now + warn_timeout_seconds;

		if (GLOBAL_GET("debug/settings/physics_interpolation/enable_warnings")) {
			// UINT64_MAX means unused.
			if (p_id.operator uint64_t() == UINT64_MAX) {
				_err_print_error(p_function, p_file, p_line, "[Physics interpolation] " + String(p_warn_string) + " (possibly benign).", false, ERR_HANDLER_WARNING);
			} else {
				String node_name;
				if (p_id.is_valid()) {
					Node *node = Object::cast_to<Node>(ObjectDB::get_instance(p_id));
					if (node && node->is_inside_tree()) {
						node_name = "\"" + String(node->get_path()) + "\"";
					} else {
						node_name = "\"unknown\"";
					}
				}

				_err_print_error(p_function, p_file, p_line, "[Physics interpolation] " + String(p_warn_string) + ": " + node_name + " (possibly benign).", false, ERR_HANDLER_WARNING);
			}
		}
	}
#endif
}
