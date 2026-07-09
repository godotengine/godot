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

#include "core/core_globals.h"
#include "core/os/os.h"

#include <cstdio>

static PrintHandlerList *print_handler_list = nullptr;
static thread_local bool is_printing = false;

static void __print_fallback(const String &p_string, bool p_err, bool p_reentrance) {
	if (p_reentrance) {
		fprintf(p_err ? stderr : stdout, "While attempting to print an error, another error was printed:\n");
	}

	fprintf(p_err ? stderr : stdout, "%s\n", p_string.utf8().get_data());
}

void add_print_handler(PrintHandlerList *p_handler) {
	_global_lock();
	p_handler->next = print_handler_list;
	print_handler_list = p_handler;
	_global_unlock();
}

void remove_print_handler(const PrintHandlerList *p_handler) {
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
	ERR_FAIL_NULL(l);
}

void __print_line(const String &p_string) {
	if (!CoreGlobals::print_line_enabled) {
		return;
	}

	if (!CoreGlobals::print_ready) {
		__print_fallback(p_string, false, false);
		return;
	}

	if (is_printing) {
		__print_fallback(p_string, false, true);
		return;
	}

	is_printing = true;

	OS::get_singleton()->print("%s\n", p_string.utf8().get_data());

	_global_lock();
	PrintHandlerList *l = print_handler_list;
	while (l) {
		l->printfunc(l->userdata, p_string, false, false);
		l = l->next;
	}

	_global_unlock();

	is_printing = false;
}

void __print_line_rich(const String &p_string) {
	if (!CoreGlobals::print_line_enabled) {
		return;
	}

	if (!CoreGlobals::print_ready) {
		__print_fallback(p_string, false, false);
		return;
	}

	if (is_printing) {
		__print_fallback(p_string, false, true);
		return;
	}

	is_printing = true;

	OS::get_singleton()->print_rich("%s\n", p_string.utf8().get_data());

	_global_lock();
	PrintHandlerList *l = print_handler_list;
	while (l) {
		l->printfunc(l->userdata, p_string, false, true);
		l = l->next;
	}

	_global_unlock();

	is_printing = false;
}

void print_raw(const String &p_string) {
	if (!CoreGlobals::print_ready) {
		__print_fallback(p_string, false, false);
		return;
	}

	if (is_printing) {
		__print_fallback(p_string, true, true);
		return;
	}

	is_printing = true;

	OS::get_singleton()->print("%s", p_string.utf8().get_data());

	is_printing = false;
}

void print_error(const String &p_string) {
	if (!CoreGlobals::print_error_enabled) {
		return;
	}

	if (!CoreGlobals::print_ready) {
		__print_fallback(p_string, false, false);
		return;
	}

	if (is_printing) {
		__print_fallback(p_string, true, true);
		return;
	}

	is_printing = true;

	OS::get_singleton()->printerr("%s\n", p_string.utf8().get_data());

	_global_lock();
	PrintHandlerList *l = print_handler_list;
	while (l) {
		l->printfunc(l->userdata, p_string, true, false);
		l = l->next;
	}

	_global_unlock();

	is_printing = false;
}

bool is_print_verbose_enabled() {
	return OS::get_singleton()->is_stdout_verbose();
}

String stringify_variants(const Span<Variant> &p_vars) {
	if (p_vars.is_empty()) {
		return String();
	}
	String result = String(p_vars[0]);
	for (const Variant &v : Span(p_vars.ptr() + 1, p_vars.size() - 1)) {
		result += ' ';
		result += v.operator String();
	}
	return result;
}
