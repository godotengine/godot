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


    struct StackFrame {
        String file;
        String function;
        int32_t line;
		void to_string(String& out)
		{
			if(out.length() > 0)
			{
				out += "\n";
			}
			out += file + "(" + itos(line) + "):" + function + "";
		}
    };

    static void initialize();

    static LocalVector<StackFrame> getStackTrace(); 
	



#if defined(WINDOWS_ENABLED)

    #include <windows.h>
    #include <dbghelp.h>


        static void initialize() {

        }

        static LocalVector<StackFrame> getStackTrace() {
            LocalVector<StackFrame> stackTrace;

            HANDLE process = GetCurrentProcess();
            HANDLE thread = GetCurrentThread();

            CONTEXT context = {};
            context.ContextFlags = CONTEXT_FULL;
            RtlCaptureContext(&context);

            SymSetOptions(SYMOPT_CASE_INSENSITIVE | SYMOPT_UNDNAME | SYMOPT_LOAD_LINES | SYMOPT_LOAD_ANYTHING);
            SymInitialize(process, nullptr, TRUE);

            DWORD image;
            STACKFRAME64 stackFrame;
            ZeroMemory(&stackFrame, sizeof(STACKFRAME64));

            image = IMAGE_FILE_MACHINE_AMD64;
            stackFrame.AddrPC.Offset = context.Rip;
            stackFrame.AddrPC.Mode = AddrModeFlat;
            stackFrame.AddrFrame.Offset = context.Rsp;
            stackFrame.AddrFrame.Mode = AddrModeFlat;
            stackFrame.AddrStack.Offset = context.Rsp;
            stackFrame.AddrStack.Mode = AddrModeFlat;

			typedef SYMBOL_INFO sym_type;
			sym_type *symbol = (sym_type *) alloca(sizeof(sym_type) + 1024);
            while (true) {
                if (StackWalk64(
                        image, process, thread,
                        &stackFrame, &context, nullptr,
                        SymFunctionTableAccess64, SymGetModuleBase64, nullptr) == FALSE)
                    break;

                if (stackFrame.AddrReturn.Offset == stackFrame.AddrPC.Offset)
                    break;

				memset(symbol, '\0', sizeof(sym_type) + 1024);
                symbol->SizeOfStruct = sizeof(sym_type);
                symbol->MaxNameLen = 1024;

                DWORD64 displacementSymbol = 0;
                const char *symbolName;
                if (SymFromAddr(process, stackFrame.AddrPC.Offset, &displacementSymbol, symbol) == TRUE) {
                    symbolName = symbol->Name;
                } else {
                    symbolName = "??";
                }

                SymSetOptions(SYMOPT_LOAD_LINES);

                IMAGEHLP_LINE64 line;
                line.SizeOfStruct = sizeof(IMAGEHLP_LINE64);

                DWORD displacementLine = 0;

                int32_t lineNumber = -1;
                const char *fileName;
                if (SymGetLineFromAddr64(process, stackFrame.AddrPC.Offset, &displacementLine, &line) == TRUE) {
                    lineNumber = line.LineNumber;
                    fileName = line.FileName;
                } else {
                    lineNumber = -1;
                    fileName = "??";
                }

                stackTrace.push_back(StackFrame { fileName, symbolName, lineNumber });
            }

            SymCleanup(process);

            return stackTrace;
        }

    

#elif defined(UNIX_ENABLED) || defined(X11_ENABLED)

    #if  __has_include(<execinfo.h>)

        #include <execinfo.h>
        #include <dlfcn.h>
		#include <cxxabi.h>
		#include <stdlib.h>


            static void initialize() {

            }

            static LocalVector<StackFrame> getStackTrace() {
                LocalVector<StackFrame> result;

				void *bt_buffer[256];
                const size_t count = backtrace(bt_buffer, 256);

                Dl_info info;
                for (size_t i = 0; i < count; i += 1) {
                    dladdr(addresses[i], &info);

                    auto fileName = info.dli_fname != nullptr ? info.dli_fname : "file??";
                    auto demangledName = "??";
					if(info.dli_sname != nullptr)
					{
						demangledName = abi::__cxa_demangle(info.dli_sname, nullptr, nullptr, &status);
					}

                    result.push_back(StackFrame { std::move(fileName), demangledName, 0 });
					if (info.dli_sname != nullptr) {
						free(demangledName);
					}
                }

                return result;
            }

        

    #endif


#else


        static void initialize() { }
        static LocalVector<StackFrame> getStackTrace() { return { StackFrame { "??", "Stacktrace collecting not available!", 0 } }; }

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
	String temp;
	CharString data;
	if(p_type == ERR_HANDLER_ERROR)
	{
		temp = p_error;
        if(p_message)
        {
            temp += ": ";
            temp += p_message;
        }
		LocalVector<StackFrame> stackFrame = getStackTrace();
		for(int i = 0; i < stackFrame.size(); ++i)
		{
			stackFrame[i].to_string(temp);
		}
		data = temp.utf8();
		p_message = data.get_data();

	}
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
