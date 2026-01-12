/**************************************************************************/
/*  os_log_logger.cpp                                                     */
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

#include "os_log_logger.h"

#include "core/object/script_backtrace.h"
#include "core/string/print_string.h"

#include <cstdlib> // For malloc/free

OsLogLogger::OsLogLogger(const char *p_subsystem) {
	const char *subsystem = p_subsystem;
	if (!subsystem) {
		subsystem = "org.godotengine.godot";
		os_log_info(OS_LOG_DEFAULT, "Missing subsystem for os_log logging; using %{public}s", subsystem);
	}

	log = os_log_create(subsystem, "engine");
	error_log = os_log_create(subsystem, error_type_string(ErrorType::ERR_ERROR));
	warning_log = os_log_create(subsystem, error_type_string(ErrorType::ERR_WARNING));
	script_log = os_log_create(subsystem, error_type_string(ErrorType::ERR_SCRIPT));
	shader_log = os_log_create(subsystem, error_type_string(ErrorType::ERR_SHADER));
}

void OsLogLogger::logv(const char *p_format, va_list p_list, bool p_err) {
	constexpr int static_buf_size = 1024;
	char static_buf[static_buf_size] = { '\0' };
	char *buf = static_buf;
	va_list list_copy;
	va_copy(list_copy, p_list);
	int len = vsnprintf(buf, static_buf_size, p_format, p_list);
	if (len >= static_buf_size) {
		buf = (char *)Memory::alloc_static(len + 1);
		vsnprintf(buf, len + 1, p_format, list_copy);
	}
	va_end(list_copy);

	// Choose appropriate log type based on error flag.
	os_log_type_t log_type = p_err ? OS_LOG_TYPE_ERROR : OS_LOG_TYPE_INFO;
	os_log_with_type(log, log_type, "%{public}s", buf);

	if (len >= static_buf_size) {
		Memory::free_static(buf);
	}
}

void OsLogLogger::log_error(const char *p_function, const char *p_file, int p_line, const char *p_code, const char *p_rationale, bool p_editor_notify, ErrorType p_type, const Vector<Ref<ScriptBacktrace>> &p_script_backtraces) {
	os_log_t selected_log;
	switch (p_type) {
		case ERR_WARNING:
			selected_log = warning_log;
			break;
		case ERR_SCRIPT:
			selected_log = script_log;
			break;
		case ERR_SHADER:
			selected_log = shader_log;
			break;
		case ERR_ERROR:
		default:
			selected_log = error_log;
			break;
	}
	const char *err_details;
	if (p_rationale && *p_rationale) {
		err_details = p_rationale;
	} else {
		err_details = p_code;
	}

	// Choose log level based on error type.
	os_log_type_t log_type;
	switch (p_type) {
		case ERR_WARNING:
			log_type = OS_LOG_TYPE_DEFAULT;
			break;
		case ERR_ERROR:
		case ERR_SCRIPT:
		case ERR_SHADER:
		default:
			log_type = OS_LOG_TYPE_ERROR;
			break;
	}

	// Append script backtraces, if any.
	String back_trace;
	for (const Ref<ScriptBacktrace> &backtrace : p_script_backtraces) {
		if (backtrace.is_valid() && !backtrace->is_empty()) {
			back_trace += "\n";
			back_trace += backtrace->format(strlen(error_type_indent(p_type)));
		}
	}

	if (back_trace.is_empty()) {
		os_log_with_type(selected_log, log_type, "%{public}s:%d:%{public}s(): %{public}s %{public}s", p_file, p_line, p_function, err_details, p_code);
	} else {
		os_log_with_type(selected_log, log_type, "%{public}s:%d:%{public}s(): %{public}s %{public}s%{public}s", p_file, p_line, p_function, err_details, p_code, back_trace.utf8().ptr());
	}
}
