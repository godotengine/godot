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
#include "core/os/os.h"

#include <debugapi.h>

#ifdef CRASH_HANDLER_EXCEPTION

DWORD CrashHandlerException(EXCEPTION_POINTERS *ep) {
	if (OS::get_singleton() == nullptr || OS::get_singleton()->is_disable_crash_handler() || IsDebuggerPresent()) {
		return EXCEPTION_CONTINUE_SEARCH;
	}

	fprintf(stderr, "%s: Program crashed\n", __FUNCTION__);

	if (OS::get_singleton()->get_main_loop()) {
		OS::get_singleton()->get_main_loop()->notification(MainLoop::NOTIFICATION_CRASH);
	}

	String msg;
	const ProjectSettings *proj_settings = ProjectSettings::get_singleton();
	if (proj_settings) {
		msg = proj_settings->get("debug/settings/crash_handler/message");
	}

	fprintf(stderr, "Dumping the backtrace. %s\n", msg.utf8().get_data());

	CONTEXT *context = ep->ContextRecord;

	LocalVector<OS::StackFrame> stack;
	OS::get_singleton()->get_stack_trace(stack, 0, 256, context);

	int frame_count = stack.size();
	for (int frame_index = 0; frame_index < frame_count; ++frame_index) {
		OS::StackFrame const &frame = stack[frame_index];

		if (frame.function.empty()) {
			fprintf(stderr, "[%d] ???\n", frame_index);
		} else if (frame.file.empty()) {
			fprintf(stderr, "[%d] %s\n", frame_index, frame.function.utf8().get_data());
		} else {
			fprintf(stderr, "[%d] %s (%s:%u)\n", frame_index, frame.function.utf8().get_data(), frame.file.utf8().get_data(), frame.line);
		}
	}

	fprintf(stderr, "-- END OF BACKTRACE --\n");

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
	if (disabled)
		return;

	disabled = true;
}

void CrashHandler::initialize() {
}
