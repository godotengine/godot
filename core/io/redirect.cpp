/*************************************************************************/
/*  redirect.cpp                                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "redirect.h"
#include "core/object/ref_counted.h"
#include "core/string/print_string.h"

void Redirect::_print_handler(void *p_this, const String &p_string, bool p_error) {
	Redirect *r = (Redirect *)p_this;
	r->emit_signal("print_line", p_string, p_error);
}

void Redirect::_bind_methods() {
	ADD_SIGNAL(MethodInfo("print_line", PropertyInfo(Variant::OBJECT, "scene_root", PROPERTY_HINT_RESOURCE_TYPE, "Node")));
}

Redirect *Redirect::singleton = nullptr;

Redirect *Redirect::get_singleton() {
	return singleton;
}

Redirect *Redirect::create() {
	ERR_FAIL_COND_V_MSG(singleton, nullptr, "Redirect singleton already exist.");
	return memnew(Redirect);
}

Redirect::Redirect() {
	singleton = this;

	print_handler.printfunc = _print_handler;
	print_handler.userdata = this;
	add_print_handler(&print_handler);
}

Redirect::~Redirect() {
	remove_print_handler(&print_handler);
}
