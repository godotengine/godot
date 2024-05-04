/**************************************************************************/
/*  windows_terminal_logger.cpp                                           */
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

#include "windows_terminal_logger.h"

#ifdef WINDOWS_ENABLED

#include <stdio.h>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

void WindowsTerminalLogger::logv(const char *p_format, va_list p_list, bool p_err) {
	if (!should_log(p_err)) {
		return;
	}

	const unsigned int BUFFER_SIZE = 16384;
	char buf[BUFFER_SIZE + 1]; // +1 for the terminating character
	int len = vsnprintf(buf, BUFFER_SIZE, p_format, p_list);
	if (len <= 0) {
		return;
	}
	if ((unsigned int)len >= BUFFER_SIZE) {
		len = BUFFER_SIZE; // Output is too big, will be truncated
	}
	buf[len] = 0;

	DWORD written = 0;
	HANDLE h = p_err ? GetStdHandle(STD_ERROR_HANDLE) : GetStdHandle(STD_OUTPUT_HANDLE);
	WriteFile(h, &buf[0], len, &written, nullptr);

#ifdef DEBUG_ENABLED
	FlushFileBuffers(h);
#endif
}

void WindowsTerminalLogger::log_error(const char *p_function, const char *p_file, int p_line, const char *p_code, const char *p_rationale, bool p_editor_notify, ErrorType p_type) {
	if (!should_log(true)) {
		return;
	}

	HANDLE hCon = GetStdHandle(STD_OUTPUT_HANDLE);
	if (!hCon || hCon == INVALID_HANDLE_VALUE) {
		StdLogger::log_error(p_function, p_file, p_line, p_code, p_rationale, p_type);
	} else {
		CONSOLE_SCREEN_BUFFER_INFO sbi; //original
		GetConsoleScreenBufferInfo(hCon, &sbi);

		WORD current_bg = sbi.wAttributes & (BACKGROUND_RED | BACKGROUND_GREEN | BACKGROUND_BLUE | BACKGROUND_INTENSITY);

		uint32_t basecol = 0;
		switch (p_type) {
			case ERR_ERROR:
				basecol = FOREGROUND_RED;
				break;
			case ERR_WARNING:
				basecol = FOREGROUND_RED | FOREGROUND_GREEN;
				break;
			case ERR_SCRIPT:
				basecol = FOREGROUND_RED | FOREGROUND_BLUE;
				break;
			case ERR_SHADER:
				basecol = FOREGROUND_GREEN | FOREGROUND_BLUE;
				break;
		}

		basecol |= current_bg;

		SetConsoleTextAttribute(hCon, basecol | FOREGROUND_INTENSITY);
		switch (p_type) {
			case ERR_ERROR:
				logf_error("ERROR:");
				break;
			case ERR_WARNING:
				logf_error("WARNING:");
				break;
			case ERR_SCRIPT:
				logf_error("SCRIPT ERROR:");
				break;
			case ERR_SHADER:
				logf_error("SHADER ERROR:");
				break;
		}

		SetConsoleTextAttribute(hCon, basecol);
		if (p_rationale && p_rationale[0]) {
			logf_error(" %s\n", p_rationale);
		} else {
			logf_error(" %s\n", p_code);
		}

		// `FOREGROUND_INTENSITY` alone results in gray text.
		SetConsoleTextAttribute(hCon, FOREGROUND_INTENSITY);
		switch (p_type) {
			case ERR_ERROR:
				logf_error("   at: ");
				break;
			case ERR_WARNING:
				logf_error("     at: ");
				break;
			case ERR_SCRIPT:
				logf_error("          at: ");
				break;
			case ERR_SHADER:
				logf_error("          at: ");
				break;
		}

		if (p_rationale && p_rationale[0]) {
			logf_error("(%s:%i)\n", p_file, p_line);
		} else {
			logf_error("%s (%s:%i)\n", p_function, p_file, p_line);
		}

		SetConsoleTextAttribute(hCon, sbi.wAttributes);
	}
}

WindowsTerminalLogger::~WindowsTerminalLogger() {}

#endif // WINDOWS_ENABLED
