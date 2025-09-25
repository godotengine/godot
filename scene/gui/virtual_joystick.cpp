/**************************************************************************/
/*  virtual_joystick.cpp                                                  */
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

#include "virtual_joystick.h"

void VirtualJoystick::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_joystick_mode", "mode"), &VirtualJoystick::set_joystick_mode);
	ClassDB::bind_method(D_METHOD("get_joystick_mode"), &VirtualJoystick::get_joystick_mode);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "joystick_mode", PROPERTY_HINT_ENUM, "Fixed,Dynamic,Following"), "set_joystick_mode", "get_joystick_mode");
}

void VirtualJoystick::set_joystick_mode(JoystickMode p_mode) {
	joystick_mode = p_mode;
}

VirtualJoystick::JoystickMode VirtualJoystick::get_joystick_mode() const {
	return joystick_mode;
}

VirtualJoystick::VirtualJoystick() {
}

VirtualJoystick::~VirtualJoystick() {
}