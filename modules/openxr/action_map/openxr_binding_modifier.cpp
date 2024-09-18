/**************************************************************************/
/*  openxr_binding_modifier.cpp                                           */
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

#include "openxr_binding_modifier.h"

void OpenXRBindingModifier::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_binding_modifier_type"), &OpenXRBindingModifier::get_binding_modifier_type);

	ClassDB::bind_method(D_METHOD("set_action", "action"), &OpenXRBindingModifier::set_action);
	ClassDB::bind_method(D_METHOD("get_action"), &OpenXRBindingModifier::get_action);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "action", PROPERTY_HINT_RESOURCE_TYPE, "OpenXRAction"), "set_action", "get_action");

	ClassDB::bind_method(D_METHOD("set_input_path", "input_path"), &OpenXRBindingModifier::set_input_path);
	ClassDB::bind_method(D_METHOD("get_input_path"), &OpenXRBindingModifier::get_input_path);
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "input_path"), "set_input_path", "get_input_path");

	BIND_ENUM_CONSTANT(BINDING_MODIFIER_GLOBAL);
	BIND_ENUM_CONSTANT(BINDING_MODIFIER_IO_PATH);
	BIND_ENUM_CONSTANT(BINDING_MODIFIER_IO_ACTION);
}

void OpenXRBindingModifier::set_action(const Ref<OpenXRAction> p_action) {
	action = p_action;
	emit_changed();
}

Ref<OpenXRAction> OpenXRBindingModifier::get_action() const {
	return action;
}

void OpenXRBindingModifier::set_input_path(const String &p_input_path) {
	input_path = p_input_path;
	emit_changed();
}

String OpenXRBindingModifier::get_input_path() const {
	return input_path;
}
