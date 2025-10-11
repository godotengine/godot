/**************************************************************************/
/*  input.compat.inc                                                      */
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

bool Input::_is_action_pressed_bind_compat_111404(const StringName &p_action, bool p_exact) const {
	return is_action_pressed(p_action, p_exact, PlayerID::P1);
}

bool Input::_is_action_just_pressed_bind_compat_111404(const StringName &p_action, bool p_exact) const {
	return is_action_just_pressed(p_action, p_exact, PlayerID::P1);
}

bool Input::_is_action_just_released_bind_compat_111404(const StringName &p_action, bool p_exact) const {
	return is_action_just_released(p_action, p_exact, PlayerID::P1);
}

bool Input::_is_action_just_pressed_by_event_bind_compat_111404(const StringName &p_action, const Ref<InputEvent> &p_event, bool p_exact) const {
	return is_action_just_pressed_by_event(p_action, p_event, p_exact, PlayerID::P1);
}

bool Input::_is_action_just_released_by_event_bind_compat_111404(const StringName &p_action, const Ref<InputEvent> &p_event, bool p_exact) const {
	return is_action_just_released_by_event(p_action, p_event, p_exact, PlayerID::P1);
}

float Input::_get_action_strength_bind_compat_111404(const StringName &p_action, bool p_exact) const {
	return get_action_strength(p_action, p_exact, PlayerID::P1);
}

float Input::_get_action_raw_strength_bind_compat_111404(const StringName &p_action, bool p_exact) const {
	return get_action_raw_strength(p_action, p_exact, PlayerID::P1);
}

float Input::_get_axis_bind_compat_111404(const StringName &p_negative_action, const StringName &p_positive_action) const {
	return get_axis(p_negative_action, p_positive_action, PlayerID::P1);
}

Vector2 Input::_get_vector_bind_compat_111404(const StringName &p_negative_x, const StringName &p_positive_x, const StringName &p_negative_y, const StringName &p_positive_y, float p_deadzone) const {
	return get_vector(p_negative_x, p_positive_x, p_negative_y, p_positive_y, p_deadzone, PlayerID::P1);
}

void Input::_action_press_bind_compat_111404(const StringName &p_action, float p_strength) {
	action_press(p_action, p_strength, PlayerID::P1);
}

void Input::_action_release_bind_compat_111404(const StringName &p_action) {
	action_release(p_action, PlayerID::P1);
}

void Input::_vibrate_handheld_bind_compat_91143(int p_duration_ms) {
	vibrate_handheld(p_duration_ms, -1.0);
}

void Input::_bind_compatibility_methods() {
	ClassDB::bind_compatibility_method(D_METHOD("is_action_pressed", "action", "exact_match"), &Input::_is_action_pressed_bind_compat_111404, DEFVAL(false));
	ClassDB::bind_compatibility_method(D_METHOD("is_action_just_pressed", "action", "exact_match"), &Input::_is_action_just_pressed_bind_compat_111404, DEFVAL(false));
	ClassDB::bind_compatibility_method(D_METHOD("is_action_just_released", "action", "exact_match"), &Input::_is_action_just_released_bind_compat_111404, DEFVAL(false));
	ClassDB::bind_compatibility_method(D_METHOD("is_action_just_pressed_by_event", "action", "event", "exact_match"), &Input::_is_action_just_pressed_by_event_bind_compat_111404, DEFVAL(false));
	ClassDB::bind_compatibility_method(D_METHOD("is_action_just_released_by_event", "action", "event", "exact_match"), &Input::_is_action_just_released_by_event_bind_compat_111404, DEFVAL(false));
	ClassDB::bind_compatibility_method(D_METHOD("get_action_strength", "action", "exact_match"), &Input::_get_action_strength_bind_compat_111404, DEFVAL(false));
	ClassDB::bind_compatibility_method(D_METHOD("get_action_raw_strength", "action", "exact_match"), &Input::_get_action_raw_strength_bind_compat_111404, DEFVAL(false));
	ClassDB::bind_compatibility_method(D_METHOD("get_axis", "negative_action", "positive_action"), &Input::_get_axis_bind_compat_111404);
	ClassDB::bind_compatibility_method(D_METHOD("get_vector", "negative_x", "positive_x", "negative_y", "positive_y", "deadzone"), &Input::_get_vector_bind_compat_111404, DEFVAL(-1.0f));
	ClassDB::bind_compatibility_method(D_METHOD("action_press", "action", "strength"), &Input::_action_press_bind_compat_111404, DEFVAL(1.f));
	ClassDB::bind_compatibility_method(D_METHOD("action_release", "action"), &Input::_action_release_bind_compat_111404);
	ClassDB::bind_compatibility_method(D_METHOD("vibrate_handheld", "duration_ms"), &Input::_vibrate_handheld_bind_compat_91143, DEFVAL(500));
}

#endif // DISABLE_DEPRECATED
