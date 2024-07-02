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
#include "core/os/os.h"
#include "core/string/ustring.h"

#if defined(MACOS_ENABLED) || defined(X11_ENABLED)
#include <execinfo.h>
#elif defined(WINDOWS_ENABLED)
#include <windows.h>
// dbghelp.h must be included after windows.h
#include <dbghelp.h>
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

void _err_flush_stdout() {
	fflush(stdout);
}

void print_stack_trace(int p_skip_called, int p_skip_callers) {
	// On an optimized build of Godot, stack trace printing will not work as
	// expected (the compiler can inline methods), so it is not supported.
#ifdef DEV_ENABLED
	p_skip_called += 1; // Skip the first one on the stack trace (this method, print_stack_trace)
#if defined(MACOS_ENABLED) || defined(X11_ENABLED)
	p_skip_callers += 2; // Skip the final 2 callers (main method / program entry point).
	// Getting the stack trace is easy on macOS and Linux.
	void *callstack[128];
	int frames = backtrace(callstack, 128);
	char **strs = backtrace_symbols(callstack, frames);
	// Print out the desired stack trace frames.
	for (int i = p_skip_called; i < frames - p_skip_callers; i++) {
		OS::get_singleton()->print("%s\n", strs[i]);
	}
	free(strs);
#elif defined(WINDOWS_ENABLED)
#if defined(_M_X64) || defined(_M_IX86)
	// The StackWalk method in Windows only supports x86 architectures (technically, Itanium too).
#if defined(_M_X64)
	DWORD machine = IMAGE_FILE_MACHINE_AMD64; // x86_64
#elif defined(_M_IX86)
	DWORD machine = IMAGE_FILE_MACHINE_I386; // x86_32
#endif
	HANDLE process = GetCurrentProcess();
	HANDLE thread = GetCurrentThread();
	if (SymInitialize(process, NULL, TRUE) == FALSE) {
		return;
	}
	SymSetOptions(SYMOPT_LOAD_LINES);

	CONTEXT context = {};
	context.ContextFlags = CONTEXT_FULL;
	RtlCaptureContext(&context);

	STACKFRAME frame = {};
	frame.AddrPC.Mode = AddrModeFlat;
	frame.AddrFrame.Mode = AddrModeFlat;
	frame.AddrStack.Mode = AddrModeFlat;
#if defined(_M_X64)
	frame.AddrPC.Offset = context.Rip;
	frame.AddrFrame.Offset = context.Rbp;
	frame.AddrStack.Offset = context.Rsp;
#elif defined(_M_IX86)
	frame.AddrPC.Offset = context.Eip;
	frame.AddrFrame.Offset = context.Ebp;
	frame.AddrStack.Offset = context.Esp;
#endif
	// Helper struct, only needed in this method.
	struct StackFrame {
		String symbol_name;
		String module_name;
		String file_name;
		unsigned int line;
	};
	Vector<StackFrame> frames;
	while (StackWalk(machine, process, thread, &frame, &context, NULL, SymFunctionTableAccess, SymGetModuleBase, NULL)) {
		StackFrame f = {};
		// Module name (executable name).
		int64_t moduleBase = SymGetModuleBase(process, frame.AddrPC.Offset);
		char moduelBuff[MAX_PATH];
		if (moduleBase && GetModuleFileNameA((HINSTANCE)moduleBase, moduelBuff, MAX_PATH)) {
			f.module_name = moduelBuff;
		} else {
			continue;
		}
		// Symbol name (method name).
		char symbolBuffer[sizeof(IMAGEHLP_SYMBOL) + 255];
		PIMAGEHLP_SYMBOL symbol = (PIMAGEHLP_SYMBOL)symbolBuffer;
		symbol->SizeOfStruct = (sizeof IMAGEHLP_SYMBOL) + 255;
		symbol->MaxNameLength = 254;
		if (SymGetSymFromAddr(process, frame.AddrPC.Offset, nullptr, symbol)) {
			f.symbol_name = symbol->Name;
			if (f.symbol_name == "widechar_main") {
				break;
			}
		} else {
			continue;
		}
		// File name and line number.
		IMAGEHLP_LINE line;
		line.SizeOfStruct = sizeof(IMAGEHLP_LINE);
		DWORD offset_ln = 0;
		if (SymGetLineFromAddr(process, frame.AddrPC.Offset, &offset_ln, &line)) {
			f.file_name = line.FileName;
			f.line = line.LineNumber;
		} else {
			continue;
		}
		frames.push_back(f);
	}
	SymCleanup(process);
	// Format and print out the desired stack trace frames.
	for (int i = p_skip_called; i < frames.size() - p_skip_callers; i++) {
		String s = vformat("%s %s\t%s (%s:%s)\n",
				String::num_int64(i).rpad(3),
				frames[i].module_name.get_file(),
				frames[i].symbol_name.get_file(),
				frames[i].file_name.get_file(),
				frames[i].line);
		OS::get_singleton()->print(s.utf8().get_data());
	}
#else // Not x86_64 or x86_32.
	OS::get_singleton()->print("Stack trace printing is not supported on this architecture on Windows.\n");
#endif
#else // Not Windows, macOS, or Linux.
	OS::get_singleton()->print("Stack trace printing is only supported on Windows, macOS, and Linux.\n");
#endif
#endif // DEV_ENABLED
}
