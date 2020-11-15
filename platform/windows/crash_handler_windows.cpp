/*************************************************************************/
/*  crash_handler_windows.cpp                                            */
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

#include "crash_handler_windows.h"

#include "core/config/project_settings.h"
#include "core/core_bind.h"
#include "core/io/http_client.h"
#include "core/os/os.h"
#include "core/string/ustring.h"
#include "core/version.h"
#include "core/version_hash.gen.h"
#include "main/main.h"
#include "servers/rendering/rendering_server_raster.h"

#include <map>
#include <string>
#include <vector>

#include "thirdparty/crashpad/crashpad/client/crash_report_database.h"
#include "thirdparty/crashpad/crashpad/client/crashpad_client.h"
#include "thirdparty/crashpad/crashpad/client/settings.h"
#include "thirdparty/crashpad/crashpad/third_party/mini_chromium/mini_chromium/base/files/file_path.h"

#ifdef CRASH_HANDLER_EXCEPTION

// Backtrace code code based on: https://stackoverflow.com/questions/6205981/windows-c-stack-trace-from-a-running-app

#include <algorithm>
#include <iterator>
#include <string>
#include <vector>

#include <psapi.h>

#pragma comment(lib, "psapi.lib")
#pragma comment(lib, "dbghelp.lib")

// Some versions of imagehlp.dll lack the proper packing directives themselves
// so we need to do it.
#pragma pack(push, before_imagehlp, 8)
#include <imagehlp.h>
#pragma pack(pop, before_imagehlp)

struct module_data {
	std::string image_name;
	std::string module_name;
	void *base_address;
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
		if (*sym->Name == '\0')
			return "<couldn't map PC to fn name>";
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
		SymLoadModule64(process, 0, &img[0], &mod[0], (DWORD64)ret.base_address, ret.load_size);
		return ret;
	}
};

DWORD CrashHandlerException(EXCEPTION_POINTERS *ep) {
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

	fprintf(stderr, "%s: Program crashed\n", __FUNCTION__);

	if (OS::get_singleton()->get_main_loop())
		OS::get_singleton()->get_main_loop()->notification(MainLoop::NOTIFICATION_CRASH);

	// Load the symbols:
	if (!SymInitialize(process, nullptr, false))
		return EXCEPTION_CONTINUE_SEARCH;

	if (disable_crash_reporter) {
		// Load the symbols:
		if (!SymInitialize(process, NULL, false))
			return EXCEPTION_CONTINUE_SEARCH;

		SymSetOptions(SymGetOptions() | SYMOPT_LOAD_LINES | SYMOPT_UNDNAME);
		EnumProcessModules(process, &module_handles[0], module_handles.size() * sizeof(HMODULE), &cbNeeded);
		module_handles.resize(cbNeeded / sizeof(HMODULE));
		EnumProcessModules(process, &module_handles[0], module_handles.size() * sizeof(HMODULE), &cbNeeded);
		std::transform(module_handles.begin(), module_handles.end(), std::back_inserter(modules), get_mod_info(process));
		void *base = modules[0].base_address;

		// Setup stuff:
		CONTEXT *context = ep->ContextRecord;
		STACKFRAME64 frame;
		bool skip_first = false;

		frame.AddrPC.Mode = AddrModeFlat;
		frame.AddrStack.Mode = AddrModeFlat;
		frame.AddrFrame.Mode = AddrModeFlat;

#ifdef _M_X64
		frame.AddrPC.Offset = context->Rip;
		frame.AddrStack.Offset = context->Rsp;
		frame.AddrFrame.Offset = context->Rbp;
#else
		frame.AddrPC.Offset = context->Eip;
		frame.AddrStack.Offset = context->Esp;
		frame.AddrFrame.Offset = context->Ebp;

		// Skip the first one to avoid a duplicate on 32-bit mode
		skip_first = true;
#endif

		line.SizeOfStruct = sizeof(line);
		line.SizeOfStruct = sizeof(line);
		IMAGE_NT_HEADERS *h = ImageNtHeader(base);
		DWORD image_type = h->FileHeader.Machine;

		String msg;
		const ProjectSettings *proj_settings = ProjectSettings::get_singleton();
		if (proj_settings) {
			msg = proj_settings->get("debug/settings/crash_handler/message");
		}

		fprintf(stderr, "Dumping the backtrace. %s\n", msg.utf8().get_data());

		int n = 0;
		do {
			if (skip_first) {
				skip_first = false;
			} else {
				if (frame.AddrPC.Offset != 0) {
					std::string fnName = symbol(process, frame.AddrPC.Offset).undecorated_name();
					if (SymGetLineFromAddr64(process, frame.AddrPC.Offset, &offset_from_symbol, &line))
						fprintf(stderr, "[%d] %s (%s:%d)\n", n, fnName.c_str(), line.FileName, line.LineNumber);
					else
						fprintf(stderr, "[%d] %s\n", n, fnName.c_str());
				} else
					fprintf(stderr, "[%d] ???\n", n);

				n++;
			}

			if (!StackWalk64(image_type, process, hThread, &frame, context, NULL, SymFunctionTableAccess64, SymGetModuleBase64, NULL))
				break;
		} while (frame.AddrReturn.Offset != 0 && n < 256);

		String database_path = OS::get_singleton()->get_user_data_dir() + "/crashpad/db";
		_Directory dir;
		if (dir.dir_exists(database_path) == false) {
			dir.make_dir_recursive(database_path);
		}
		const char *s = ProjectSettings::get_singleton()->globalize_path(database_path).utf8().ptr();
		size_t len = MultiByteToWideChar(CP_ACP, 0, s, -1, nullptr, 0);
		wchar_t *wide_string = new wchar_t[len];
		MultiByteToWideChar(CP_ACP, 0, s, -1, wide_string, (int)len);
		base::FilePath database = base::FilePath(wide_string);
		std::unique_ptr<crashpad::CrashReportDatabase> db =
				crashpad::CrashReportDatabase::Initialize(database);

		if (MessageBoxA(NULL,
					"Would you like to submit a report about this and "
					"previous crashes to help improve future versions "
					"of Godot Engine https://godotengine.org?\n",
					"Godot Engine has crashed",
					MB_YESNO | MB_ICONERROR) != IDYES) {
			return EXCEPTION_CONTINUE_SEARCH;
		}
		if (db != nullptr && db->GetSettings() != nullptr) {
			db->GetSettings()->SetUploadsEnabled(true);
		}
		client.DumpAndCrash(ep);

		// Pass the exception to the OS
		return EXCEPTION_CONTINUE_SEARCH;
	}
	return EXCEPTION_CONTINUE_SEARCH;
}
#endif

void CrashHandler::disable() {
	if (disabled) {
		return;
	}

	disabled = true;
}

void CrashHandler::initialize() {
	if (!Engine::get_singleton()) {
		return;
	}
	bool is_editor = Engine::get_singleton()->is_editor_hint();
	if (is_editor) {
		if (ProjectSettings::get_singleton()->get("application/crashpad/disable_editor_crashpad").booleanize()) {
			return;
		}
		String handler_exe = ProjectSettings::get_singleton()->get("application/crashpad/editor_crashpad_handler");
		if (handler_exe.empty()) {
			return;
		}
		String crashpad_server = ProjectSettings::get_singleton()->get("application/crashpad/editor_crashpad_server");
		if (crashpad_server.empty()) {
			return;
		}
		initialize_crashpad(handler_exe, crashpad_server);
	} else {
		if (ProjectSettings::get_singleton()->get("application/crashpad/disable_project_crashpad").booleanize()) {
			return;
		}
		String handler_exe = ProjectSettings::get_singleton()->get("application/crashpad/project_crashpad_handler");
		if (handler_exe.empty()) {
			return;
		}
		String crashpad_server = ProjectSettings::get_singleton()->get("application/crashpad/project_crashpad_server");
		if (crashpad_server.empty()) {
			return;
		}
		initialize_crashpad(handler_exe, crashpad_server);
	}
}

void CrashHandler::initialize_crashpad(String p_crashpad_handler_path, String p_crashpad_server) {
	//Cache directory that will store crashpad information and minidumps
	String database_path = OS::get_singleton()->get_user_data_dir() + "/crashpad/db";
	_Directory dir;
	if (!dir.dir_exists(database_path)) {
		dir.make_dir_recursive(database_path);
	}
	{
		const char *string = ProjectSettings::get_singleton()->globalize_path(database_path).utf8().ptr();
		size_t len = MultiByteToWideChar(CP_ACP, 0, string, -1, nullptr, 0);
		wchar_t *wide_string = new wchar_t[len];
		MultiByteToWideChar(CP_ACP, 0, string, -1, wide_string, (int)len);
		database = base::FilePath(wide_string);
	}
	{
		// Path to the out-of-process handler executable
		const char *string = ProjectSettings::get_singleton()->globalize_path(p_crashpad_handler_path).utf8().ptr();
		size_t len = MultiByteToWideChar(CP_ACP, 0, string, -1, nullptr, 0);
		wchar_t *wide_string = new wchar_t[len];
		MultiByteToWideChar(CP_ACP, 0, string, -1, wide_string, (int)len);
		handler = base::FilePath(wide_string);
	}
	// URL used to submit minidumps to
	std::string url = p_crashpad_server.utf8().ptr();
	// Optional annotations passed via --annotations to the handler
	std::map<std::string, std::string> annotations;

	String hash = String(VERSION_HASH);
	if (hash.length()) {
		hash = "." + hash.left(7);
	}
	annotations["ver"] = String(String("v") + String(VERSION_FULL_BUILD) + hash).utf8().ptr();
	String prod = ProjectSettings::get_singleton()->get("application/config/name");
	if (!prod.empty()) {
		annotations["prod"] = prod.utf8().ptr();
	}
	String video = RenderingServerRaster::get_singleton()->get_video_adapter_vendor();
	if (!video.empty()) {
		annotations["video"] = video.utf8().ptr();
	}
	// Optional arguments to pass to the handler
	std::vector<std::string> arguments;
	arguments.push_back("--no-upload-gzip");
	// For debugging purposes, don't limit rate.
	// arguments.push_back("--no-rate-limit");
	bool success = client.StartHandler(
			handler,
			database,
			database,
			url,
			annotations,
			arguments,
			/* restartable */ true,
			/* asynchronous_start */ true);

	if (!success) {
		return;
	}
	success = client.WaitForHandlerStart(5000);
	if (!success) {
		return;
	}
	disable_crash_reporter = false;
}
