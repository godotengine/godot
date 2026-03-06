/**************************************************************************/
/*  print_string.cpp                                                      */
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

#include "print_string.h"

#include "core/os/os.h"

#include <stdio.h>

static PrintHandlerList *print_handler_list = nullptr;
bool _print_line_enabled = true;
bool _print_error_enabled = true;

const char *TerminalColor::_terminal_color_strings[] = {
	"\033[0m",

	"\033[30m",
	"\033[31m",
	"\033[32m",
	"\033[33m",
	"\033[34m",
	"\033[35m",
	"\033[36m",
	"\033[37m",

	"\033[90m",
	"\033[91m",
	"\033[92m",
	"\033[93m",
	"\033[94m",
	"\033[95m",
	"\033[96m",
	"\033[97m",

	"\033[40m",
	"\033[41m",
	"\033[42m",
	"\033[43m",
	"\033[44m",
	"\033[45m",
	"\033[46m",
	"\033[47m",

	"\033[1m",
	"\033[3m",
	"\033[4m"
};

bool TerminalColor::get_color(TerminalColor::Col p_col, Color &r_color) {
	const float G = 0.7f;

	switch (p_col) {
		default: {
			r_color = Color(1, 1, 1, 1);
			return false;
		} break;
		case DEFAULT:
		case BRIGHT_WHITE: {
			r_color = Color(1, 1, 1, 1);
		} break;
		case BRIGHT_BLACK: {
			r_color = Color(0.5, 0.5, 0.5, 1);
		} break;
		case BRIGHT_RED: {
			r_color = Color(1, 0, 0, 1);
		} break;
		case BRIGHT_GREEN: {
			r_color = Color(0, 1, 0, 1);
		} break;
		case BRIGHT_BLUE: {
			r_color = Color(0, 0, 1, 1);
		} break;
		case BRIGHT_YELLOW: {
			r_color = Color(1, 1, 0, 1);
		} break;
		case BRIGHT_MAGENTA: {
			r_color = Color(1, 0, 1, 1);
		} break;
		case BRIGHT_CYAN: {
			r_color = Color(0, 1, 1, 1);
		} break;
		case WHITE: {
			r_color = Color(G, G, G, 1);
		} break;
		case BLACK: {
			r_color = Color(0, 0, 0, 1);
		} break;
		case RED: {
			r_color = Color(G, 0, 0, 1);
		} break;
		case GREEN: {
			r_color = Color(0, G, 0, 1);
		} break;
		case BLUE: {
			r_color = Color(0, 0, G, 1);
		} break;
		case YELLOW: {
			r_color = Color(G, G, 0, 1);
		} break;
		case MAGENTA: {
			r_color = Color(G, 0, G, 1);
		} break;
		case CYAN: {
			r_color = Color(0, G, G, 1);
		} break;
	}

	return true;
}

int TerminalColor::find(const String &p_string, int p_start, Col &r_color, int &r_pos_start_color) {
	int found = p_string.find("\033[", p_start);
	if (found == -1) {
		return -1;
	}

	// Attempt to recognise (slow but shouldn't happen that often).
	for (int n = 0; n < (int)Col::MAX; n++) {
		if (p_string.find(_terminal_color_strings[n], found) == found) {
			r_color = (Col)n;
			r_pos_start_color = found;
			return found + strlen(_terminal_color_strings[n]);
		}
	}

	return -1;
}

void add_print_handler(PrintHandlerList *p_handler) {
	_global_lock();
	p_handler->next = print_handler_list;
	print_handler_list = p_handler;
	_global_unlock();
}

void remove_print_handler(PrintHandlerList *p_handler) {
	_global_lock();

	PrintHandlerList *prev = nullptr;
	PrintHandlerList *l = print_handler_list;

	while (l) {
		if (l == p_handler) {
			if (prev) {
				prev->next = l->next;
			} else {
				print_handler_list = l->next;
			}
			break;
		}
		prev = l;
		l = l->next;
	}
	//OS::get_singleton()->print("print handler list is %p\n",print_handler_list);

	_global_unlock();
	ERR_FAIL_COND(l == nullptr);
}

void print_line(String p_string) {
	if (!_print_line_enabled) {
		return;
	}

	OS::get_singleton()->print("%s\n", p_string.utf8().get_data());

	_global_lock();
	PrintHandlerList *l = print_handler_list;
	while (l) {
		l->printfunc(l->userdata, p_string, false);
		l = l->next;
	}

	_global_unlock();
}

void print_error(String p_string) {
	if (!_print_error_enabled) {
		return;
	}

	OS::get_singleton()->printerr("%s\n", p_string.utf8().get_data());

	_global_lock();
	PrintHandlerList *l = print_handler_list;
	while (l) {
		l->printfunc(l->userdata, p_string, true);
		l = l->next;
	}

	_global_unlock();
}

void print_verbose(String p_string) {
	if (OS::get_singleton()->is_stdout_verbose()) {
		print_line(p_string);
	}
}
