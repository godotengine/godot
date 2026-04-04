/**************************************************************************/
/*  animation_node_state_machine_transition.hpp                           */
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

#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/resource.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/string_name.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Curve;

class AnimationNodeStateMachineTransition : public Resource {
	GDEXTENSION_CLASS(AnimationNodeStateMachineTransition, Resource)

public:
	enum SwitchMode {
		SWITCH_MODE_IMMEDIATE = 0,
		SWITCH_MODE_SYNC = 1,
		SWITCH_MODE_AT_END = 2,
	};

	enum AdvanceMode {
		ADVANCE_MODE_DISABLED = 0,
		ADVANCE_MODE_ENABLED = 1,
		ADVANCE_MODE_AUTO = 2,
	};

	void set_switch_mode(AnimationNodeStateMachineTransition::SwitchMode p_mode);
	AnimationNodeStateMachineTransition::SwitchMode get_switch_mode() const;
	void set_advance_mode(AnimationNodeStateMachineTransition::AdvanceMode p_mode);
	AnimationNodeStateMachineTransition::AdvanceMode get_advance_mode() const;
	void set_advance_condition(const StringName &p_name);
	StringName get_advance_condition() const;
	void set_xfade_time(float p_secs);
	float get_xfade_time() const;
	void set_xfade_curve(const Ref<Curve> &p_curve);
	Ref<Curve> get_xfade_curve() const;
	void set_break_loop_at_end(bool p_enable);
	bool is_loop_broken_at_end() const;
	void set_reset(bool p_reset);
	bool is_reset() const;
	void set_priority(int32_t p_priority);
	int32_t get_priority() const;
	void set_advance_expression(const String &p_text);
	String get_advance_expression() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Resource::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(AnimationNodeStateMachineTransition::SwitchMode);
VARIANT_ENUM_CAST(AnimationNodeStateMachineTransition::AdvanceMode);

