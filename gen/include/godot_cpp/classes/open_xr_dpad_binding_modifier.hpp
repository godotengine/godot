/**************************************************************************/
/*  open_xr_dpad_binding_modifier.hpp                                     */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#pragma once

#include <godot_cpp/classes/open_xrip_binding_modifier.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/variant/string.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class OpenXRActionSet;
class OpenXRHapticBase;

class OpenXRDpadBindingModifier : public OpenXRIPBindingModifier {
	GDEXTENSION_CLASS(OpenXRDpadBindingModifier, OpenXRIPBindingModifier)

public:
	void set_action_set(const Ref<OpenXRActionSet> &p_action_set);
	Ref<OpenXRActionSet> get_action_set() const;
	void set_input_path(const String &p_input_path);
	String get_input_path() const;
	void set_threshold(float p_threshold);
	float get_threshold() const;
	void set_threshold_released(float p_threshold_released);
	float get_threshold_released() const;
	void set_center_region(float p_center_region);
	float get_center_region() const;
	void set_wedge_angle(float p_wedge_angle);
	float get_wedge_angle() const;
	void set_is_sticky(bool p_is_sticky);
	bool get_is_sticky() const;
	void set_on_haptic(const Ref<OpenXRHapticBase> &p_haptic);
	Ref<OpenXRHapticBase> get_on_haptic() const;
	void set_off_haptic(const Ref<OpenXRHapticBase> &p_haptic);
	Ref<OpenXRHapticBase> get_off_haptic() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		OpenXRIPBindingModifier::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

