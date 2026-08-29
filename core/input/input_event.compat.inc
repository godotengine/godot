/**************************************************************************/
/*  input_event.compat.inc                                                */
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

#ifndef DISABLE_DEPRECATED

bool InputEvent::_is_action_pressed_bind_compat_111404(const StringName &p_action, bool p_allow_echo, bool p_exact_match) const {
	return is_action_pressed(p_action, p_allow_echo, p_exact_match, PlayerID::P1);
}

bool InputEvent::_is_action_released_bind_compat_111404(const StringName &p_action, bool p_exact_match) const {
	return is_action_released(p_action, p_exact_match, PlayerID::P1);
}

float InputEvent::_get_action_strength_bind_compat_111404(const StringName &p_action, bool p_exact_match) const {
	return get_action_strength(p_action, p_exact_match, PlayerID::P1);
}

void InputEvent::_bind_compatibility_methods() {
	ClassDB::bind_compatibility_method(D_METHOD("is_action_pressed", "action", "allow_echo", "exact_match"), &InputEvent::_is_action_pressed_bind_compat_111404, DEFVAL(false), DEFVAL(false));
	ClassDB::bind_compatibility_method(D_METHOD("is_action_released", "action", "exact_match"), &InputEvent::_is_action_released_bind_compat_111404, DEFVAL(false));
	ClassDB::bind_compatibility_method(D_METHOD("get_action_strength", "action", "exact_match"), &InputEvent::_get_action_strength_bind_compat_111404, DEFVAL(false));
}

#endif // DISABLE_DEPRECATED
