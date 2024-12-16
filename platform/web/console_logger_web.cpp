/**************************************************************************/
/*  console_logger_web.cpp                                                */
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

#include "console_logger_web.h"

#include "godot_js.h"

void ConsoleLoggerWeb::log_error(const char *p_function, const char *p_file, int p_line, const char *p_code, const char *p_rationale, bool p_editor_notify, ErrorType p_type) {
	if (!should_log(true)) {
		return;
	}

	const char *err_details;
	if (p_rationale && *p_rationale) {
		err_details = p_rationale;
	} else {
		err_details = p_code;
	}

	const char *err_type = "ERROR";
	switch (p_type) {
		case ERR_ERROR:
			err_type = "ERROR";
			break;
		case ERR_WARNING:
			logf_warn("WARNING: %s\n   at: %s (%s:%i)\n", err_details, p_function, p_file, p_line);
			return;
		case ERR_SCRIPT:
			err_type = "SCRIPT ERROR";
			break;
		case ERR_SHADER:
			err_type = "SHADER ERROR";
			break;
		default:
			ERR_PRINT("Unknown error type");
			break;
	}

	logf_error("%s: %s\n   at: %s (%s:%i)\n", err_type, err_details, p_function, p_file, p_line);
}

void ConsoleLoggerWeb::logf_warn(const char *p_format, ...) {
	if (!should_log(false)) {
		return;
	}

	va_list argp;
	va_start(argp, p_format);

	logv(p_format, argp, PrintType::WARN);

	va_end(argp);
}

void ConsoleLoggerWeb::logv(const char *p_format, va_list p_list, bool p_err) {
	logv(p_format, p_list, p_err ? PrintType::ERROR : PrintType::LOG);
}

void ConsoleLoggerWeb::logv(const char *p_format, va_list p_list, PrintType p_type) {
	if (!should_log(p_type == PrintType::ERROR || p_type == PrintType::WARN)) {
		return;
	}

	int str_size = vsnprintf(nullptr, 0, p_format, p_list) + 1;
	char *str = new char[str_size];
	vsnprintf(str, str_size, p_format, p_list);

	switch (p_type) {
		case PrintType::ERROR:
			godot_js_os_print_error(str);
			break;
		case PrintType::LOG:
			godot_js_os_print(str);
			break;
		case PrintType::WARN:
			godot_js_os_print_warning(str);
			break;
	}

	delete[] str;
}
