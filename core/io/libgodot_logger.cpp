/**************************************************************************/
/*  libgodot_logger.cpp                                                   */
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

#ifdef LIBGODOT_ENABLED

#include "libgodot_logger.h"

void LibGodotLogger::log_error(const char *p_function, const char *p_file, int p_line, const char *p_code, const char *p_rationale, bool p_editor_notify, ErrorType p_type, const Vector<Ref<ScriptBacktrace>> &p_script_backtraces) {
	if (!should_log(true)) {
		return;
	}

	const char *err_details;
	if (p_rationale && p_rationale[0]) {
		err_details = p_rationale;
	} else {
		err_details = p_code;
	}

	const char *err_type = "ERROR";

	switch (p_type) {
		case ERR_WARNING: {
			err_type = "WARNING";
		} break;
		case ERR_SCRIPT: {
			err_type = "SCRIPT ERROR";
		} break;
		case ERR_SHADER: {
			err_type = "SHADER ERROR";
		} break;
		case ERR_ERROR:
		default: {
			err_type = "ERROR";
		} break;
	}

	logf_error("%s: %s\n   at: %s (%s:%i)\n", err_type, err_details, p_function, p_file, p_line);

	for (const Ref<ScriptBacktrace> &backtrace : p_script_backtraces) {
		if (!backtrace->is_empty()) {
			logf_error("%s\n", backtrace->format(3).utf8().get_data());
		}
	}
}

void LibGodotLogger::logv(const char *p_format, va_list p_list, bool p_err) {
	if (!should_log(p_err)) {
		return;
	}

	const int static_buffer_size = 1024;
	char static_buf[static_buffer_size];
	char *buf = static_buf;
	va_list list_copy;
	va_copy(list_copy, p_list);
	int len = vsnprintf(buf, static_buffer_size, p_format, p_list);
	if (len >= static_buffer_size) {
		buf = (char *)memalloc(len + 1);
		len = vsnprintf(buf, len + 1, p_format, list_copy);
	}
	va_end(list_copy);

	String str_buf = String::utf8(buf, len);
	if (len >= static_buffer_size) {
		memfree(buf);
	}

	forward_log(str_buf, p_err);
}

void LibGodotLogger::forward_log(const String &p_msg, bool p_err) {
	if (log_func == nullptr) {
		return;
	}

	CharString cstr_buf = p_msg.utf8();
	if (cstr_buf.length() == 0) {
		return;
	}

	log_func(log_data, cstr_buf.get_data(), p_err);
}

void LibGodotLogger::set_callback_function(LogCallbackFunction p_log_func, LogCallbackData p_log_data) {
	log_func = p_log_func;
	log_data = p_log_data;
}

#endif // LIBGODOT_ENABLED
