/*************************************************************************/
/*  print_string.cpp                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "print_string.h"

#include "core/os/os.h"

#include <stdio.h>

static PrintHandlerList *print_handler_list = nullptr;
bool _print_line_enabled = true;
bool _print_error_enabled = true;

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

void __print_line(String p_string) {
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

String stringify_variants(Variant p_var) {
	return p_var.operator String();
}
