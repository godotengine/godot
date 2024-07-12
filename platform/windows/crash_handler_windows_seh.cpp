/**************************************************************************/
/*  crash_handler_windows_seh.cpp                                         */
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

#include "crash_handler_windows.h"

#include "core/config/project_settings.h"
#include "core/os/os.h"
#include "core/string/print_string.h"
#include "core/version.h"
#include "main/main.h"

#ifdef CRASH_HANDLER_EXCEPTION

// Backtrace code based on: https://stackoverflow.com/questions/6205981/windows-c-stack-trace-from-a-running-app

#include <algorithm>
#include <iterator>
#include <string>
#include <vector>

#include <psapi.h>

// Some versions of imagehlp.dll lack the proper packing directives themselves
// so we need to do it.
#pragma pack(push, before_imagehlp, 8)
#include <imagehlp.h>
#pragma pack(pop, before_imagehlp)

#ifdef MINGW_ENABLED
#include <cxxabi.h>

#include "thirdparty/libbacktrace/backtrace.h"

static LPTOP_LEVEL_EXCEPTION_FILTER prev_exception_filter = nullptr;

struct CrashHandlerData {
	int64_t index = -1;
	backtrace_state *state = nullptr;
	int64_t offset = 0;
	bool success = false;
};

int symbol_callback(void *data, uintptr_t pc, const char *filename, int lineno, const char *function) {
	CrashHandlerData *ch_data = reinterpret_cast<CrashHandlerData *>(data);
	if (!function) {
		return 0;
	}

	char fname[1024];
	snprintf(fname, 1024, "%s", function);

	if (function[0] == '_') {
		int status;
		char *demangled = abi::__cxa_demangle(function, nullptr, nullptr, &status);

		if (status == 0 && demangled) {
			snprintf(fname, 1024, "%s", demangled);
		}

		if (demangled) {
			free(demangled);
		}
	}

	print_error(vformat("[%d] %s (%s:%d)", ch_data->index++, String::utf8(fname), String::utf8(filename), lineno));
	ch_data->success = true;
	return 0;
}

void error_callback(void *data, const char *msg, int errnum) {
	CrashHandlerData *ch_data = reinterpret_cast<CrashHandlerData *>(data);
	if (ch_data->index == -1) {
		print_error(vformat("Error(%d): %s", errnum, String::utf8(msg)));
	} else if (errnum == -1) {
		// No symbols, just ignore.
	} else {
		print_error(vformat("[%d] error(%d): %s", ch_data->index, errnum, String::utf8(msg)));
	}
}

int64_t get_image_base(const String &p_path) {
	Ref<FileAccess> f = FileAccess::open(p_path, FileAccess::READ);
	if (f.is_null()) {
		return 0;
	}
	{
		f->seek(0x3c);
		uint32_t pe_pos = f->get_32();

		f->seek(pe_pos);
		uint32_t magic = f->get_32();
		if (magic != 0x00004550) {
			return 0;
		}
	}
	int64_t opt_header_pos = f->get_position() + 0x14;
	f->seek(opt_header_pos);

	uint16_t opt_header_magic = f->get_16();
	if (opt_header_magic == 0x10B) {
		f->seek(opt_header_pos + 0x1C);
		return f->get_32();
	} else if (opt_header_magic == 0x20B) {
		f->seek(opt_header_pos + 0x18);
		return f->get_64();
	} else {
		return 0;
	}
}
#endif

struct module_data {
	std::string image_name;
	std::string module_name;
	void *base_address = nullptr;
	DWORD load_size;
};

class symbol {
	typedef IMAGEHLP_SYMBOL64 sym_type;
	sym_type *sym;
	static const int max_name_len = 1024;

public:
	symbol(HANDLE process, DWORD64 address) :
			sym((sym_type *)::operator new(sizeof(*sym) + max_name_len)) {
		memset(sym, '\0', sizeof(*sym) + max_name_len);
		sym->SizeOfStruct = sizeof(*sym);
		sym->MaxNameLength = max_name_len;
		DWORD64 displacement;

		SymGetSymFromAddr64(process, address, &displacement, sym);
	}

	std::string name() { return std::string(sym->Name); }
	std::string undecorated_name() {
		if (*sym->Name == '\0') {
			return {};
		}
		std::vector<char> und_name(max_name_len);
		UnDecorateSymbolName(sym->Name, &und_name[0], max_name_len, UNDNAME_COMPLETE);
		return std::string(&und_name[0], strlen(&und_name[0]));
	}
};

class get_mod_info {
	HANDLE process;

public:
	get_mod_info(HANDLE h) :
			process(h) {}

	module_data operator()(HMODULE module) {
		module_data ret;
		char temp[4096];
		MODULEINFO mi;

		GetModuleInformation(process, module, &mi, sizeof(mi));
		ret.base_address = mi.lpBaseOfDll;
		ret.load_size = mi.SizeOfImage;

		GetModuleFileNameEx(process, module, temp, sizeof(temp));
		ret.image_name = temp;
		GetModuleBaseName(process, module, temp, sizeof(temp));
		ret.module_name = temp;
		std::vector<char> img(ret.image_name.begin(), ret.image_name.end());
		std::vector<char> mod(ret.module_name.begin(), ret.module_name.end());
		SymLoadModule64(process, nullptr, &img[0], &mod[0], (DWORD64)ret.base_address, ret.load_size);
		return ret;
	}
};

LONG CrashHandlerException(EXCEPTION_POINTERS *ep) {
	HANDLE process = GetCurrentProcess();
	HANDLE hThread = GetCurrentThread();
	DWORD offset_from_symbol = 0;
	IMAGEHLP_LINE64 line = { 0 };
	std::vector<module_data> modules;
	DWORD cbNeeded;
	std::vector<HMODULE> module_handles(1);

	if (OS::get_singleton() == nullptr || OS::get_singleton()->is_disable_crash_handler() || IsDebuggerPresent()) {
		return EXCEPTION_CONTINUE_SEARCH;
	}

	String msg;
	const ProjectSettings *proj_settings = ProjectSettings::get_singleton();
	if (proj_settings) {
		msg = proj_settings->get("debug/settings/crash_handler/message");
	}

	// Tell MainLoop about the crash. This can be handled by users too in Node.
	if (OS::get_singleton()->get_main_loop()) {
		OS::get_singleton()->get_main_loop()->notification(MainLoop::NOTIFICATION_CRASH);
	}

	print_error("\n================================================================");
	print_error(vformat("%s: Program crashed with exception 0x%08X", __FUNCTION__, (unsigned int)ep->ExceptionRecord->ExceptionCode));

	// Print the engine version just before, so that people are reminded to include the version in backtrace reports.
	if (String(VERSION_HASH).is_empty()) {
		print_error(vformat("Engine version: %s", VERSION_FULL_NAME));
	} else {
		print_error(vformat("Engine version: %s (%s)", VERSION_FULL_NAME, VERSION_HASH));
	}
	print_error(vformat("Dumping the backtrace. %s", msg));

	// Load the symbols:
	if (!SymInitialize(process, nullptr, false)) {
		return EXCEPTION_CONTINUE_SEARCH;
	}

#ifdef MINGW_ENABLED
	String _execpath = OS::get_singleton()->get_executable_path();

	// Load process and image info to determine ASLR addresses offset.
	MODULEINFO mi;
	GetModuleInformation(GetCurrentProcess(), GetModuleHandle(NULL), &mi, sizeof(mi));
	int64_t image_mem_base = reinterpret_cast<int64_t>(mi.lpBaseOfDll);
	int64_t image_file_base = get_image_base(_execpath);
	int64_t image_mem_end = image_mem_base + mi.SizeOfImage;

	CrashHandlerData data;
	data.offset = image_mem_base - image_file_base;

	CharString execpath_utf8 = _execpath.utf8();
	data.state = backtrace_create_state(execpath_utf8.get_data(), 0, &error_callback, reinterpret_cast<void *>(&data));
#endif

	SymSetOptions(SymGetOptions() | SYMOPT_LOAD_LINES | SYMOPT_UNDNAME | SYMOPT_EXACT_SYMBOLS);
	EnumProcessModules(process, &module_handles[0], module_handles.size() * sizeof(HMODULE), &cbNeeded);
	module_handles.resize(cbNeeded / sizeof(HMODULE));
	EnumProcessModules(process, &module_handles[0], module_handles.size() * sizeof(HMODULE), &cbNeeded);
	std::transform(module_handles.begin(), module_handles.end(), std::back_inserter(modules), get_mod_info(process));
	void *base = modules[0].base_address;

	// Setup stuff:
	CONTEXT *context = ep->ContextRecord;
	STACKFRAME64 frame;

	frame.AddrPC.Mode = AddrModeFlat;
	frame.AddrStack.Mode = AddrModeFlat;
	frame.AddrFrame.Mode = AddrModeFlat;

#if defined(_M_X64)
	frame.AddrPC.Offset = context->Rip;
	frame.AddrStack.Offset = context->Rsp;
	frame.AddrFrame.Offset = context->Rbp;
#elif defined(_M_ARM64) || defined(_M_ARM64EC)
	frame.AddrPC.Offset = context->Pc;
	frame.AddrStack.Offset = context->Sp;
	frame.AddrFrame.Offset = context->Fp;
#elif defined(_M_ARM)
	frame.AddrPC.Offset = context->Pc;
	frame.AddrStack.Offset = context->Sp;
	frame.AddrFrame.Offset = context->R11;
#else
	frame.AddrPC.Offset = context->Eip;
	frame.AddrStack.Offset = context->Esp;
	frame.AddrFrame.Offset = context->Ebp;
#endif

	line.SizeOfStruct = sizeof(line);
	IMAGE_NT_HEADERS *h = ImageNtHeader(base);
	DWORD image_type = h->FileHeader.Machine;

	int n = 0;
	do {
		// The first call walks the first frame.
		if (!StackWalk64(image_type, process, hThread, &frame, context, nullptr, SymFunctionTableAccess64, SymGetModuleBase64, nullptr)) {
			break;
		}

		if (frame.AddrPC.Offset != 0) {
			// The return address of stack frames point to the next instruction
			// of the call instruction, so subtracting 1 usually gives a more
			// accurate line number for the call site.
			int adjustment = (n == 0) ? 0 : -1;
			uint64_t address = frame.AddrPC.Offset + adjustment;

#ifdef MINGW_ENABLED
			// For MinGW, try printing this frame with libbacktrace using
			// DWARF symbols only if the address corresponds to the main EXE.
			if (address >= image_mem_base && address < image_mem_end) {
				data.index = n;
				data.success = false;
				backtrace_pcinfo(data.state, address - data.offset, &symbol_callback, &error_callback, &data);
				if (data.success) {
					n = data.index;
					continue;
				}
			}
#endif

			std::string fnName = symbol(process, address).undecorated_name();
			if (!fnName.empty()) {
				if (SymGetLineFromAddr64(process, address, &offset_from_symbol, &line)) {
					print_error(vformat("[%d] %s (%s:%d)", n, fnName.c_str(), (char *)line.FileName, (int)line.LineNumber));
				} else {
					print_error(vformat("[%d] %s", n, fnName.c_str()));
				}
			} else {
				// Find which module owns the address and print an offset.
				bool found = false;
				for (const module_data &module : modules) {
					uint64_t module_base = reinterpret_cast<uint64_t>(module.base_address);
					uint64_t module_end = module_base + module.load_size;
					if (address >= module_base && address < module_end) {
						print_error(vformat("[%d] %s+0x%x", n, module.module_name.c_str(), address - module_base));
						found = true;
						break;
					}
				}
				if (!found) {
					print_error(vformat(
#ifdef _WIN64
							"[%d] ? 0x%016x",
#else
							"[%d] ? 0x%08x",
#endif
							n, address));
				}
			}
		} else {
			print_error(vformat("[%d] ???", n));
		}

		n++;
	} while (frame.AddrReturn.Offset != 0 && n < 256);

	print_error("-- END OF BACKTRACE --");
	print_error("================================================================");

	SymCleanup(process);

#ifdef MINGW_ENABLED
	if (prev_exception_filter) {
		return prev_exception_filter(ep);
	}
#endif
	// Pass the exception to the OS
	return EXCEPTION_CONTINUE_SEARCH;
}
#endif

CrashHandler::CrashHandler() {
	disabled = false;
}

CrashHandler::~CrashHandler() {
}

void CrashHandler::disable() {
	if (disabled) {
		return;
	}

	disabled = true;
}

void CrashHandler::initialize() {
#if defined(CRASH_HANDLER_EXCEPTION) && defined(MINGW_ENABLED)
	prev_exception_filter = SetUnhandledExceptionFilter(CrashHandlerException);
#endif
}
