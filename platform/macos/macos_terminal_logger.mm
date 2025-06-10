/**************************************************************************/
/*  macos_terminal_logger.mm                                              */
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

#import "macos_terminal_logger.h"

#ifdef MACOS_ENABLED

#include <os/log.h>

void MacOSTerminalLogger::log_error(const char *p_function, const char *p_file, int p_line, const char *p_code, const char *p_rationale, bool p_editor_notify, ErrorType p_type, const Vector<Ref<ScriptBacktrace>> &p_script_backtraces) {
	if (!should_log(true)) {
		return;
	}

	const char *err_details;
	if (p_rationale && p_rationale[0]) {
		err_details = p_rationale;
	} else {
		err_details = p_code;
	}

	const char *bold_color;
	const char *normal_color;
	switch (p_type) {
		case ERR_WARNING:
			bold_color = "\E[1;33m";
			normal_color = "\E[0;93m";
			break;
		case ERR_SCRIPT:
			bold_color = "\E[1;35m";
			normal_color = "\E[0;95m";
			break;
		case ERR_SHADER:
			bold_color = "\E[1;36m";
			normal_color = "\E[0;96m";
			break;
		case ERR_ERROR:
		default:
			bold_color = "\E[1;31m";
			normal_color = "\E[0;91m";
			break;
	}

	os_log_error(OS_LOG_DEFAULT,
			"%{public}s: %{public}s\nat: %{public}s (%{public}s:%i)",
			error_type_string(p_type), err_details, p_function, p_file, p_line);
	logf_error("%s%s:%s %s\n", bold_color, error_type_string(p_type), normal_color, err_details);
	logf_error("\E[0;90m%sat: %s (%s:%i)\E[0m\n", error_type_indent(p_type), p_function, p_file, p_line);

	for (const Ref<ScriptBacktrace> &backtrace : p_script_backtraces) {
		if (!backtrace->is_empty()) {
			os_log_error(OS_LOG_DEFAULT, "%{public}s", backtrace->format().utf8().get_data());
			logf_error("\E[0;90m%s\E[0m\n", backtrace->format(strlen(error_type_indent(p_type))).utf8().get_data());
		}
	}
}

#endif // MACOS_ENABLED
