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

void _err_print_error_backtrace(const char *filter, const String &p_error, bool p_editor_notify, ErrorHandlerType p_type) {
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
		if (!cpp_stack[i].function.contains(filter)) {
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
