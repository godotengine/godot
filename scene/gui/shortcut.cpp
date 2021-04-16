/*************************************************************************/
/*  shortcut.cpp                                                         */
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

#include "shortcut.h"

#include "core/os/keyboard.h"

void Shortcut::set_shortcut(const Ref<InputEvent> &p_shortcut, bool p_primary) {
	if (p_primary) {
		primary_shortcut = p_shortcut;
	} else {
		secondary_shortcut = p_shortcut;
	}
	emit_changed();
}

Ref<InputEvent> Shortcut::get_shortcut(bool p_primary) const {
	return p_primary ? primary_shortcut : secondary_shortcut;
}

bool Shortcut::is_shortcut(const Ref<InputEvent> &p_event, bool p_match_either, bool p_primary) const {
	bool primary_match = (primary_shortcut.is_valid() && primary_shortcut->shortcut_match(p_event));
	bool secondary_match = (secondary_shortcut.is_valid() && secondary_shortcut->shortcut_match(p_event));

	if (p_match_either) {
		return primary_match || secondary_match;
	} else {
		return p_primary ? primary_match : secondary_match;
	}
}

String Shortcut::get_as_text(bool p_primary) const {
	if (p_primary && primary_shortcut.is_valid()) {
		return primary_shortcut->as_text();
	} else if (!p_primary && secondary_shortcut.is_valid()) {
		return secondary_shortcut->as_text();
	} else {
		return "None";
	}
}

bool Shortcut::is_valid() const {
	return primary_shortcut.is_valid() || secondary_shortcut.is_valid();
}

Ref<InputEvent> Shortcut::_get_primary_shortcut() {
	return get_shortcut();
}

void Shortcut::_set_primary_shortcut(const Ref<InputEvent> &p_shortcut) {
	set_shortcut(p_shortcut);
}

Ref<InputEvent> Shortcut::_get_secondary_shortcut() {
	return get_shortcut(false);
}

void Shortcut::_set_secondary_shortcut(const Ref<InputEvent> &p_shortcut) {
	set_shortcut(p_shortcut, false);
}

void Shortcut::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_shortcut", "event", "primary"), &Shortcut::set_shortcut, true);
	ClassDB::bind_method(D_METHOD("get_shortcut", "primary"), &Shortcut::get_shortcut, true);

	ClassDB::bind_method(D_METHOD("is_valid"), &Shortcut::is_valid);

	ClassDB::bind_method(D_METHOD("is_shortcut", "event", "match_either", "primary"), &Shortcut::is_shortcut, true, true);
	ClassDB::bind_method(D_METHOD("get_as_text", "primary"), &Shortcut::get_as_text, true);

	ClassDB::bind_method(D_METHOD("_get_primary_shortcut"), &Shortcut::_get_primary_shortcut);
	ClassDB::bind_method(D_METHOD("_set_primary_shortcut", "event"), &Shortcut::_set_primary_shortcut);
	ClassDB::bind_method(D_METHOD("_get_secondary_shortcut"), &Shortcut::_get_secondary_shortcut);
	ClassDB::bind_method(D_METHOD("_set_secondary_shortcut", "event"), &Shortcut::_set_secondary_shortcut);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "primary_shortcut", PROPERTY_HINT_RESOURCE_TYPE, "InputEvent"), "_set_primary_shortcut", "_get_primary_shortcut");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "secondary_shortcut", PROPERTY_HINT_RESOURCE_TYPE, "InputEvent"), "_set_secondary_shortcut", "_get_secondary_shortcut");
}

Shortcut::Shortcut() {
}
