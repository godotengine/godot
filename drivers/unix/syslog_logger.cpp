/**************************************************************************/
/*  syslog_logger.cpp                                                     */
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

#ifdef UNIX_ENABLED

#include "syslog_logger.h"

#include <syslog.h>

#include <cstdio>

void SyslogLogger::logr(const char *p_format, va_list p_list, bool p_err) {
	if (!should_log(p_err)) {
		return;
	}

	const int static_buf_size = 512;
	char static_buf[static_buf_size];
	char *buf = static_buf;
	va_list list_copy;
	va_copy(list_copy, p_list);
	int len = vsnprintf(buf, static_buf_size, p_format, p_list);
	if (len >= static_buf_size) {
		buf = (char *)Memory::alloc_static(len + 1);
		vsnprintf(buf, len + 1, p_format, list_copy);
	}
	va_end(list_copy);

	const String &plain = _bbcode_to_plain(String::utf8(buf));
	syslog(p_err ? LOG_ERR : LOG_INFO, "%s", plain.utf8().get_data());

	if (len >= static_buf_size) {
		Memory::free_static(buf);
	}
}

void SyslogLogger::logv(const char *p_format, va_list p_list, bool p_err) {
	if (!should_log(p_err)) {
		return;
	}

	vsyslog(p_err ? LOG_ERR : LOG_INFO, p_format, p_list);
}

void SyslogLogger::print_error(const char *p_function, const char *p_file, int p_line, const char *p_code, const char *p_rationale, ErrorType p_type) {
	if (!should_log(true)) {
		return;
	}

	const char *err_type = error_type_string(p_type);

	const char *err_details;
	if (p_rationale && *p_rationale) {
		err_details = p_rationale;
	} else {
		err_details = p_code;
	}

	syslog(p_type == ERR_WARNING ? LOG_WARNING : LOG_ERR, "**%s**: %s\n   At: %s:%i:%s() - %s", err_type, err_details, p_file, p_line, p_function, p_code);
}

SyslogLogger::~SyslogLogger() {
}

#endif
