/*************************************************************************/
/*  error_call_stack.cpp                                                 */
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

#include "error_call_stack.h"

#ifdef DEBUG_ENABLED
#include "core/config/project_settings.h"
#include "core/debugger/engine_debugger.h"
#include "core/os/os.h"

ErrorCallStack *ErrorCallStack::singleton = nullptr;

void ErrorCallStack::initialize() {
	ERR_FAIL_COND(singleton != nullptr);

	GLOBAL_DEF("debug/settings/error_handler/max_call_stack", 0);

	singleton = new ErrorCallStack();

	singleton->handler.errfunc = _err_handler;
	singleton->handler.userdata = singleton;
	add_error_handler(&singleton->handler);
}

void ErrorCallStack::finalize() {
	ERR_FAIL_COND(singleton == nullptr);

	remove_error_handler(&singleton->handler);
	singleton->handler.errfunc = nullptr;
	singleton->handler.userdata = nullptr;

	delete singleton;
	singleton = nullptr;
}

void ErrorCallStack::_err_handler(void *p_ud, const char *p_func, const char *p_file, int p_line, const char *p_err, const char *p_descr, ErrorHandlerType p_type) {
	ERR_FAIL_COND(singleton == nullptr);
	singleton->_err_handler_internal(p_ud, p_func, p_file, p_line, p_err, p_descr, p_type);
}

void ErrorCallStack::_err_handler_internal(void *p_ud, const char *p_func, const char *p_file, int p_line, const char *p_err, const char *p_descr, ErrorHandlerType p_type) {
	if (processing) {
		return;
	}

	processing = true;

	if (!OS::get_singleton()) {
		processing = false;
		return;
	}

	if (!ProjectSettings::get_singleton()) {
		processing = false;
		return;
	}

	int max_frames = GLOBAL_GET("debug/settings/error_handler/max_call_stack");
	if (max_frames < 1) {
		processing = false;
		return;
	}

	LocalVector<OS::StackFrame> stack;
	OS::get_singleton()->get_stack_trace(stack, 3, max_frames);

	int frame_count = stack.size();
	if (frame_count > 0) {
		OS::get_singleton()->printerr("Call Stack:\n");

		bool skip_frames = true;

		for (int frame_index = 0; frame_index < frame_count; ++frame_index) {
			OS::StackFrame &frame = stack[frame_index];

			// Skip frames until the error location is found to get rid of error handling calls.
			// Skipping a fixed amount of frames is not accurate due to compiler optimizations.
			if (skip_frames) {
				if (frame.file.ends_with(p_file)) {
					skip_frames = false;
				} else {
					continue;
				}
			}

			if (frame.module.empty()) {
				OS::get_singleton()->printerr("  - %s\n", frame.function.utf8().get_data());
			} else {
				OS::get_singleton()->printerr("  - %s:%s\n", frame.module.utf8().get_data(), frame.function.utf8().get_data());
			}
			if (!frame.file.empty()) {
				OS::get_singleton()->printerr("    At %s, line %d\n", frame.file.utf8().get_data(), frame.line);
			}

			if (EngineDebugger::is_active()) {
				if (frame.function.find("GDScriptFunction::call") || frame.function.find("GDScriptFunctions::call")) {
					Vector<ScriptLanguage::StackInfo> si;
					String language;

					for (int i = 0; i < ScriptServer::get_language_count(); i++) {
						ScriptLanguage *scriptLanguage = ScriptServer::get_language(i);
						si = scriptLanguage->debug_get_current_stack_info();
						if (0 < si.size()) {
							language = scriptLanguage->get_name();
							break;
						}
					}

					int num_script_frames = MIN(si.size(), frame_count - frame_index + 1);
					if (0 < num_script_frames) {
						for (int script_index = 0; script_index < num_script_frames; ++script_index) {
							ScriptLanguage::StackInfo const &script_frame = si[script_index];
							OS::get_singleton()->printerr("  - %s:%s\n", language.utf8().get_data(), script_frame.func.utf8().get_data());
							OS::get_singleton()->printerr("    At %s, line %d\n", script_frame.file.utf8().get_data(), script_frame.line);
						}
						break;
					}
				}
			}
		}
	}

	processing = false;
}
#endif
