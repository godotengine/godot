/**************************************************************************/
/*  component.cpp                                                         */
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


#include "component.h"

#include "core/object/class_db.h"
#include "core/object/script_language.h"


StringName Component::get_component_class() {
	Ref<Script> s = get_script();
	while (s.is_valid()) {
		if (!s->get_global_name().is_empty()) {
			return s->get_global_name();
		} else {
			s = s->get_base_script();
		}
	}

	return get_class_name();
}

void Component::enter_tree() {
	if (GDVIRTUAL_CALL(_enter_tree)) {
		//
	}
}

void Component::exit_tree() {
	if (GDVIRTUAL_CALL(_exit_tree)) {
		//
	}
}

void Component::ready() {
	if (GDVIRTUAL_CALL(_ready)) {
		//
	}
}

void Component::process(double delta) {
	if (GDVIRTUAL_CALL(_process, delta)) {
		//
	}
}

void Component::physics_process(double delta) {
	if (GDVIRTUAL_CALL(_physics_process, delta)) {
		//
	}
}

void Component::input(const Ref<InputEvent> &p_event) {
	if (GDVIRTUAL_CALL(_input, p_event)) {
		//
	}
}

void Component::shortcut_input(const Ref<InputEvent> &p_key_event) {
	if (GDVIRTUAL_CALL(_shortcut_input, p_key_event)) {
		//
	}
}

void Component::unhandled_input(const Ref<InputEvent> &p_event) {
	if (GDVIRTUAL_CALL(_unhandled_input, p_event)) {
		//
	}
}

void Component::unhandled_key_input(const Ref<InputEvent> &p_key_event) {
	if (GDVIRTUAL_CALL(_unhandled_key_input, p_key_event)) {
		//
	}
}

void Component::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_component_class"), &Component::get_component_class);

	GDVIRTUAL_BIND(_enter_tree);
	GDVIRTUAL_BIND(_exit_tree);
	GDVIRTUAL_BIND(_ready);
	GDVIRTUAL_BIND(_process, "delta");
	GDVIRTUAL_BIND(_physics_process, "delta");

	GDVIRTUAL_BIND(_input, "event");
	GDVIRTUAL_BIND(_shortcut_input, "event");
	GDVIRTUAL_BIND(_unhandled_input, "event");
	GDVIRTUAL_BIND(_unhandled_key_input, "event");
}
