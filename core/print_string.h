/**************************************************************************/
/*  print_string.h                                                        */
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

#ifndef PRINT_STRING_H
#define PRINT_STRING_H

#include "core/ustring.h"

extern void (*_print_func)(String);

typedef void (*PrintHandlerFunc)(void *, const String &p_string, bool p_error);

struct Color;

class TerminalColor {
	static const char *_terminal_color_strings[];

public:
	enum Col {
		DEFAULT,

		BLACK,
		RED,
		GREEN,
		YELLOW,
		BLUE,
		MAGENTA,
		CYAN,
		WHITE,

		BRIGHT_BLACK,
		BRIGHT_RED,
		BRIGHT_GREEN,
		BRIGHT_YELLOW,
		BRIGHT_BLUE,
		BRIGHT_MAGENTA,
		BRIGHT_CYAN,
		BRIGHT_WHITE,

		BACK_BLACK,
		BACK_RED,
		BACK_GREEN,
		BACK_YELLOW,
		BACK_BLUE,
		BACK_MAGENTA,
		BACK_CYAN,
		BACK_WHITE,

		BOLD,
		ITALIC,
		UNDERLINE,

		MAX,
	};

	static const char *get(TerminalColor::Col p_col) { return _terminal_color_strings[(int)p_col]; }
	static String draw(TerminalColor::Col p_col, String p_string) { return String(get(p_col)) + p_string + get(DEFAULT); }
	static String draw_combined(TerminalColor::Col p_col_a, TerminalColor::Col p_col_b, String p_string) { return String(get(p_col_a)) + get(p_col_b) + p_string + get(DEFAULT); }

	static bool get_color(TerminalColor::Col p_col, Color &r_color);

	// Find a recognised terminal color in a string if present.
	// Return the position immediately following the color, with the enum returned in r_color.
	// Return -1 if not found.
	static int find(const String &p_string, int p_start, Col &r_color, int &r_pos_start_color);
};

struct PrintHandlerList {
	PrintHandlerFunc printfunc;
	void *userdata;

	PrintHandlerList *next;

	PrintHandlerList() {
		printfunc = nullptr;
		next = nullptr;
		userdata = nullptr;
	}
};

void add_print_handler(PrintHandlerList *p_handler);
void remove_print_handler(PrintHandlerList *p_handler);

extern bool _print_line_enabled;
extern bool _print_error_enabled;
extern void print_line(String p_string);
extern void print_error(String p_string);
extern void print_verbose(String p_string);

#endif // PRINT_STRING_H
