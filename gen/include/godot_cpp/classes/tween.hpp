/**************************************************************************/
/*  tween.hpp                                                             */
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
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/variant.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Callable;
class CallbackTweener;
class IntervalTweener;
class MethodTweener;
class Node;
class NodePath;
class Object;
class PropertyTweener;
class SubtweenTweener;

class Tween : public RefCounted {
	GDEXTENSION_CLASS(Tween, RefCounted)

public:
	enum TweenProcessMode {
		TWEEN_PROCESS_PHYSICS = 0,
		TWEEN_PROCESS_IDLE = 1,
	};

	enum TweenPauseMode {
		TWEEN_PAUSE_BOUND = 0,
		TWEEN_PAUSE_STOP = 1,
		TWEEN_PAUSE_PROCESS = 2,
	};

	enum TransitionType {
		TRANS_LINEAR = 0,
		TRANS_SINE = 1,
		TRANS_QUINT = 2,
		TRANS_QUART = 3,
		TRANS_QUAD = 4,
		TRANS_EXPO = 5,
		TRANS_ELASTIC = 6,
		TRANS_CUBIC = 7,
		TRANS_CIRC = 8,
		TRANS_BOUNCE = 9,
		TRANS_BACK = 10,
		TRANS_SPRING = 11,
	};

	enum EaseType {
		EASE_IN = 0,
		EASE_OUT = 1,
		EASE_IN_OUT = 2,
		EASE_OUT_IN = 3,
	};

	Ref<PropertyTweener> tween_property(Object *p_object, const NodePath &p_property, const Variant &p_final_val, double p_duration);
	Ref<IntervalTweener> tween_interval(double p_time);
	Ref<CallbackTweener> tween_callback(const Callable &p_callback);
	Ref<MethodTweener> tween_method(const Callable &p_method, const Variant &p_from, const Variant &p_to, double p_duration);
	Ref<SubtweenTweener> tween_subtween(const Ref<Tween> &p_subtween);
	bool custom_step(double p_delta);
	void stop();
	void pause();
	void play();
	void kill();
	double get_total_elapsed_time() const;
	bool is_running();
	bool is_valid();
	Ref<Tween> bind_node(Node *p_node);
	Ref<Tween> set_process_mode(Tween::TweenProcessMode p_mode);
	Ref<Tween> set_pause_mode(Tween::TweenPauseMode p_mode);
	Ref<Tween> set_ignore_time_scale(bool p_ignore = true);
	Ref<Tween> set_parallel(bool p_parallel = true);
	Ref<Tween> set_loops(int32_t p_loops = 0);
	int32_t get_loops_left() const;
	Ref<Tween> set_speed_scale(float p_speed);
	Ref<Tween> set_trans(Tween::TransitionType p_trans);
	Ref<Tween> set_ease(Tween::EaseType p_ease);
	Ref<Tween> parallel();
	Ref<Tween> chain();
	static Variant interpolate_value(const Variant &p_initial_value, const Variant &p_delta_value, double p_elapsed_time, double p_duration, Tween::TransitionType p_trans_type, Tween::EaseType p_ease_type);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		RefCounted::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(Tween::TweenProcessMode);
VARIANT_ENUM_CAST(Tween::TweenPauseMode);
VARIANT_ENUM_CAST(Tween::TransitionType);
VARIANT_ENUM_CAST(Tween::EaseType);

