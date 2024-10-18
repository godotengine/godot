/**************************************************************************/
/*  breakpad_linuxbsd_windows.cpp                                         */
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

#include "breakpad.h"

#include "core/io/dir_access.h"
#include "core/os/os.h"

#ifdef WINDOWS_ENABLED
#include <thirdparty/breakpad/src/client/windows/handler/exception_handler.h>
#include <thirdparty/breakpad/src/google_breakpad/common/minidump_format.h>
#else
#include <thirdparty/breakpad/src/client/linux/handler/exception_handler.h>
#endif

static google_breakpad::ExceptionHandler *breakpad_handler = nullptr;
static bool register_breakpad_handlers;

#ifdef WINDOWS_ENABLED
static bool dump_callback(const wchar_t *dump_path, const wchar_t *minidump_id, void *context,
		EXCEPTION_POINTERS *exinfo, MDRawAssertionInfo *assertion, bool succeeded) {
	wprintf(L"Crash dump created at: %s/%s.dmp\n", dump_path, minidump_id);

	// This, kind of duplicate print, is here as in the default Godot console window the dump created message is otherwise not visible
	fwprintf(stderr, L"Crash dump created at: %s/%s.dmp\n", dump_path, minidump_id);
#else
static bool dump_callback(const google_breakpad::MinidumpDescriptor &descriptor, void *context, bool succeeded) {
	printf("Crash dump created at: %s\n", descriptor.path());
#endif
	return succeeded;
}

static void create_breakpad_handler(const String &crash_folder) {
#ifdef WINDOWS_ENABLED
	// Automatic register to the exception handlers can be disabled when Godot crash handler listens to them
	std::wstring crash_folder_w(reinterpret_cast<const wchar_t *>(crash_folder.utf16().get_data()));

	breakpad_handler = new google_breakpad::ExceptionHandler(crash_folder_w, nullptr, dump_callback, nullptr,
			register_breakpad_handlers ? google_breakpad::ExceptionHandler::HANDLER_ALL : google_breakpad::ExceptionHandler::HANDLER_NONE);

#else
	google_breakpad::MinidumpDescriptor descriptor(crash_folder.utf8().get_data());

	breakpad_handler = new google_breakpad::ExceptionHandler(descriptor, nullptr, dump_callback, nullptr, register_breakpad_handlers, -1);
#endif
}

static String get_settings_specific_crash_folder() {
	String crash_folder = OS::get_singleton()->get_user_data_dir() + "/crashes";
	Ref<DirAccess> dir = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	if (!dir->dir_exists(crash_folder)) {
		dir->make_dir_recursive(crash_folder);
	}

	return crash_folder;
}

void initialize_breakpad(bool register_handlers) {
	if (breakpad_handler != nullptr) {
		return;
	}

	register_breakpad_handlers = register_handlers;

#ifdef WINDOWS_ENABLED
	String crash_folder;

	wchar_t tempPath[MAX_PATH + 1];

	if (GetTempPathW(MAX_PATH + 1, tempPath) > 0) {
		crash_folder = tempPath;
	} else {
		crash_folder = L"C:/temp";
	}

	create_breakpad_handler(crash_folder);

#else
	create_breakpad_handler("/tmp");
#endif
}

void disable_breakpad() {
	if (breakpad_handler == nullptr)
		return;

	delete breakpad_handler;
	breakpad_handler = nullptr;
}

void report_user_data_dir_usable() {
	if (breakpad_handler == nullptr)
		return;

	const String &crash_folder = get_settings_specific_crash_folder();

#ifdef WINDOWS_ENABLED
	breakpad_handler->set_dump_path(reinterpret_cast<const wchar_t *>(crash_folder.utf16().get_data()));
#else
	google_breakpad::MinidumpDescriptor descriptor(crash_folder.utf8().get_data());

	breakpad_handler->set_minidump_descriptor(descriptor);
#endif
}

void breakpad_handle_signal(int signal) {
	if (breakpad_handler == nullptr)
		return;

#ifndef WINDOWS_ENABLED
	// TODO: Should this use HandleSignal(int sig, siginfo_t* info, void* uc) instead?
	// would require changing to sigaction in crash_handler_x11.cpp
	breakpad_handler->SimulateSignalDelivery(signal);
#endif
}

void breakpad_handle_exception_pointers(void *exinfo) {
	if (breakpad_handler == nullptr)
		return;

#ifdef WINDOWS_ENABLED
	breakpad_handler->WriteMinidumpForException(static_cast<EXCEPTION_POINTERS *>(exinfo));
#endif
}
