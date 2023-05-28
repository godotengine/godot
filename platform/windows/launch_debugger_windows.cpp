/**************************************************************************/
/*  launch_debugger_windows.cpp                                           */
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

#include "launch_debugger_windows.h"

#if defined(CRASH_HANDLER_EXCEPTION) && defined(DEV_ENABLED)

#include "core/os/memory.h"
#include "core/templates/local_vector.h"

#include <wchar.h>

// The logic in this file is generally based on https://github.com/dotnet/runtime/blob/afa9a930b827ca0aa5debd693a4731229838516c/src/coreclr/debug/ee/debugger.cpp#L6952 (The implementation behind System.Diagnostics.Debugger.Launch())

// To summarize:
// 1. Fetch the registry key 'HKLM\SOFTWARE\Microsoft\Windows NT\CurrentVersion\AeDebug\Debugger'.
// 2. printf format the value with the pid to debug, an event handle to set when the debugger has attached, the address of a JIT_DEBUG_INFO structure, and a bunch of zeros for forward compatibility.
// 3. Execute that command line.
// 4. Wait on the event until the debugger has attached or the started process has exited because the user has canceled.
// Except that in the user has canceled case this doesn't work and (at least vsjitdebugger) kills the debugging target before exiting.
// To avoid this termination one must either:
// - Not provide an event argument. This requires hard-coding a modified value of the registry key as it includes the parameter and passing 0 for it is invalid.
// - Being a managed process that has called Debugger.Launch. This is detected checking for the coreclr library and the value of its CLRJitAttachState export.
// - Same as above but for the .NET framework, the check works completely differently tho.
// - Lastly if an event with the name 'Local\\Microsoft_VS80_JIT_CrashReceivedEvent-{pid}', where '{pid}' is replaced with the pid of the process to debug, exists. In that case this event will also be signaled.

bool launch_debugger_core(DWORD p_pid, HANDLE p_attach_event, JIT_DEBUG_INFO *p_debug_info, PROCESS_INFORMATION &p_process_info) {
	HKEY hkey;
	DWORD buffer_len = 0;
	DWORD vtype = REG_SZ;
	LocalVector<WCHAR> template_buffer;
	LocalVector<WCHAR> formatted_buffer;

	// Retrieve the configured JIT debugger from the registry.
	if (RegOpenKeyExW(HKEY_LOCAL_MACHINE, L"SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\\AeDebug", 0, KEY_READ, &hkey) != ERROR_SUCCESS) {
		return false;
	}
	if (RegQueryValueExW(hkey, L"Debugger", NULL, &vtype, NULL, &buffer_len) != ERROR_SUCCESS || vtype != REG_SZ) {
		RegCloseKey(hkey);
		return false;
	}
	template_buffer.resize(buffer_len / sizeof(WCHAR) + 1);
	formatted_buffer.resize(buffer_len / sizeof(WCHAR) + 64);
	if (RegQueryValueExW(hkey, L"Debugger", NULL, &vtype, reinterpret_cast<LPBYTE>(template_buffer.ptr()), &buffer_len) != ERROR_SUCCESS || buffer_len <= sizeof(WCHAR)) {
		RegCloseKey(hkey);
		return false;
	}
	RegCloseKey(hkey);
	template_buffer[buffer_len / sizeof(WCHAR)] = 0;

	// Fill the placeholder arguments.
	swprintf_s(formatted_buffer.ptr(), formatted_buffer.size(), template_buffer.ptr(), p_pid, p_attach_event, p_debug_info, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

	// Execute the JIT debugger.
	memset(&p_process_info, 0, sizeof(p_process_info));
	STARTUPINFOW startupInfo;
	memset(&startupInfo, 0, sizeof(startupInfo));
	startupInfo.cb = sizeof(STARTUPINFOW);
	return CreateProcessW(NULL, formatted_buffer.ptr(), NULL, NULL, TRUE, CREATE_NEW_CONSOLE, NULL, NULL, &startupInfo, &p_process_info);
}

// Keep this in static memory so that the debugger can retrieve this at whatever time it wants to.
static JIT_DEBUG_INFO debugger_launch_debug_info;
static EXCEPTION_RECORD debugger_launch_exception_record;
static CONTEXT debugger_launch_context;

static void init_debug_info(EXCEPTION_POINTERS *ep) {
	debugger_launch_exception_record = *ep->ExceptionRecord;
	debugger_launch_context = *ep->ContextRecord;
	debugger_launch_debug_info.dwSize = sizeof(debugger_launch_debug_info);
	debugger_launch_debug_info.dwThreadID = GetCurrentThreadId();
	debugger_launch_debug_info.dwProcessorArchitecture = 0;
	debugger_launch_debug_info.dwReserved0 = 0;
	debugger_launch_debug_info.lpExceptionRecord = (ULONG64)&debugger_launch_exception_record;
	debugger_launch_debug_info.lpContextRecord = (ULONG64)&debugger_launch_context;
	debugger_launch_debug_info.lpExceptionAddress = (ULONG64)debugger_launch_exception_record.ExceptionAddress;

#if defined(__x86_64) || defined(__x86_64__) || defined(__amd64__) || defined(_M_X64)
	debugger_launch_debug_info.dwProcessorArchitecture = PROCESSOR_ARCHITECTURE_AMD64;
#elif defined(__i386) || defined(__i386__) || defined(_M_IX86)
	debugger_launch_debug_info.dwProcessorArchitecture = PROCESSOR_ARCHITECTURE_INTEL;
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(_M_ARM64EC)
	debugger_launch_debug_info.dwProcessorArchitecture = PROCESSOR_ARCHITECTURE_ARM64;
#elif defined(__arm__) || defined(_M_ARM)
	debugger_launch_debug_info.dwProcessorArchitecture = PROCESSOR_ARCHITECTURE_ARM;
#elif defined(__powerpc__)
	debugger_launch_debug_info.dwProcessorArchitecture = PROCESSOR_ARCHITECTURE_PPC;
#else
	debugger_launch_debug_info.dwProcessorArchitecture = PROCESSOR_ARCHITECTURE_UNKNOWN;
#endif
}

void launch_debugger(EXCEPTION_POINTERS *ep) {
	DWORD pid = GetCurrentProcessId();
	init_debug_info(ep);

	// Create an event that will be signaled when the debugger has attached successfully.
	SECURITY_ATTRIBUTES sa;
	memset(&sa, 0, sizeof(sa));
	sa.nLength = sizeof(sa);
	sa.bInheritHandle = TRUE;
	HANDLE event_handle = CreateEventW(&sa, TRUE, FALSE, NULL);
	ERR_FAIL_NULL(event_handle);

	// Black magic that makes the debugger not kill this process if the user cancels the attach request, see the comment at the top of the file.
	WCHAR magic_event_name[64];
	wcscpy_s(magic_event_name, L"Local\\Microsoft_VS80_JIT_CrashReceivedEvent-");
	_ultow_s(pid, &magic_event_name[44], 11ui64, 10);
	OpenEventW(2u, 0, magic_event_name);
	HANDLE magic_event_handle = CreateEventW(&sa, TRUE, FALSE, magic_event_name);

	PROCESS_INFORMATION process_info;
	if (!launch_debugger_core(pid, event_handle, &debugger_launch_debug_info, process_info)) {
		CloseHandle(event_handle);
		CloseHandle(magic_event_handle);
		return;
	}

	// Wait until the event has been signaled because the debugger has attached or the debugger process has exited because the user has canceled the process.
	HANDLE arrHandles[2];
	arrHandles[0] = event_handle;
	arrHandles[1] = process_info.hProcess;
	WaitForMultipleObjectsEx(2, arrHandles, FALSE, INFINITE, FALSE);

	CloseHandle(magic_event_handle);
	CloseHandle(event_handle);
	CloseHandle(process_info.hProcess);
	CloseHandle(process_info.hThread);
}

#endif
