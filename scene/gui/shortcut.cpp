/*************************************************************************/
/*  shortcut.cpp                                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "shortcut.h"

#include "core/os/keyboard.h"

void Shortcut::set_shortcut(const Ref<InputEvent> &p_shortcut) {
	shortcut = p_shortcut;
	emit_changed();
}

Ref<InputEvent> Shortcut::get_shortcut() const {
	return shortcut;
}

bool Shortcut::is_shortcut(const Ref<InputEvent> &p_event) const {
	return shortcut.is_valid() && shortcut->shortcut_match(p_event);
}

String Shortcut::get_as_text() const {
	if (shortcut.is_valid()) {
		return shortcut->as_text();
	} else {
		return "None";
	}
}

bool Shortcut::is_valid() const {
	return shortcut.is_valid();
}

void Shortcut::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_shortcut", "event"), &Shortcut::set_shortcut);
	ClassDB::bind_method(D_METHOD("get_shortcut"), &Shortcut::get_shortcut);

	ClassDB::bind_method(D_METHOD("is_valid"), &Shortcut::is_valid);

	ClassDB::bind_method(D_METHOD("is_shortcut", "event"), &Shortcut::is_shortcut);
	ClassDB::bind_method(D_METHOD("get_as_text"), &Shortcut::get_as_text);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "shortcut", PROPERTY_HINT_RESOURCE_TYPE, "InputEvent"), "set_shortcut", "get_shortcut");
}

Shortcut::Shortcut() {
}
