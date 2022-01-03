/*************************************************************************/
/*  tween.h                                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef TWEEN_H
#define TWEEN_H

#include "core/object/ref_counted.h"

class Tween;
class Node;

class Tweener : public RefCounted {
	GDCLASS(Tweener, RefCounted);

public:
	virtual void set_tween(Ref<Tween> p_tween);
	virtual void start() = 0;
	virtual bool step(float &r_delta) = 0;
	void clear_tween();

protected:
	static void _bind_methods();
	Ref<Tween> tween;
	float elapsed_time = 0;
	bool finished = false;
};

class PropertyTweener;
class IntervalTweener;
class CallbackTweener;
class MethodTweener;

class Tween : public RefCounted {
	GDCLASS(Tween, RefCounted);

public:
	enum TweenProcessMode {
		TWEEN_PROCESS_PHYSICS,
		TWEEN_PROCESS_IDLE,
	};

	enum TweenPauseMode {
		TWEEN_PAUSE_BOUND,
		TWEEN_PAUSE_STOP,
		TWEEN_PAUSE_PROCESS,
	};

	enum TransitionType {
		TRANS_LINEAR,
		TRANS_SINE,
		TRANS_QUINT,
		TRANS_QUART,
		TRANS_QUAD,
		TRANS_EXPO,
		TRANS_ELASTIC,
		TRANS_CUBIC,
		TRANS_CIRC,
		TRANS_BOUNCE,
		TRANS_BACK,
		TRANS_MAX
	};

	enum EaseType {
		EASE_IN,
		EASE_OUT,
		EASE_IN_OUT,
		EASE_OUT_IN,
		EASE_MAX
	};

private:
	TweenProcessMode process_mode = TweenProcessMode::TWEEN_PROCESS_IDLE;
	TweenPauseMode pause_mode = TweenPauseMode::TWEEN_PAUSE_STOP;
	TransitionType default_transition = TransitionType::TRANS_LINEAR;
	EaseType default_ease = EaseType::EASE_IN_OUT;
	ObjectID bound_node;

	Vector<List<Ref<Tweener>>> tweeners;
	int current_step = -1;
	int loops = 1;
	int loops_done = 0;
	float speed_scale = 1;

	bool is_bound = false;
	bool started = false;
	bool running = true;
	bool dead = false;
	bool valid = false;
	bool default_parallel = false;
	bool parallel_enabled = false;

	typedef real_t (*interpolater)(real_t t, real_t b, real_t c, real_t d);
	static interpolater interpolaters[TRANS_MAX][EASE_MAX];

	void start_tweeners();

protected:
	static void _bind_methods();

public:
	Ref<PropertyTweener> tween_property(Object *p_target, NodePath p_property, Variant p_to, float p_duration);
	Ref<IntervalTweener> tween_interval(float p_time);
	Ref<CallbackTweener> tween_callback(Callable p_callback);
	Ref<MethodTweener> tween_method(Callable p_callback, Variant p_from, Variant p_to, float p_duration);
	void append(Ref<Tweener> p_tweener);

	bool custom_step(float p_delta);
	void stop();
	void pause();
	void play();
	void kill();

	bool is_running();
	void set_valid(bool p_valid);
	bool is_valid();
	void clear();

	Ref<Tween> bind_node(Node *p_node);
	Ref<Tween> set_process_mode(TweenProcessMode p_mode);
	TweenProcessMode get_process_mode();
	Ref<Tween> set_pause_mode(TweenPauseMode p_mode);
	TweenPauseMode get_pause_mode();

	Ref<Tween> set_parallel(bool p_parallel);
	Ref<Tween> set_loops(int p_loops);
	Ref<Tween> set_speed_scale(float p_speed);
	Ref<Tween> set_trans(TransitionType p_trans);
	TransitionType get_trans();
	Ref<Tween> set_ease(EaseType p_ease);
	EaseType get_ease();

	Ref<Tween> parallel();
	Ref<Tween> chain();

	real_t run_equation(TransitionType p_trans_type, EaseType p_ease_type, real_t t, real_t b, real_t c, real_t d);
	Variant interpolate_variant(Variant p_initial_val, Variant p_delta_val, float p_time, float p_duration, Tween::TransitionType p_trans, Tween::EaseType p_ease);
	Variant calculate_delta_value(Variant p_intial_val, Variant p_final_val);

	bool step(float p_delta);
	bool should_pause();

	Tween() {}
};

VARIANT_ENUM_CAST(Tween::TweenPauseMode);
VARIANT_ENUM_CAST(Tween::TweenProcessMode);
VARIANT_ENUM_CAST(Tween::TransitionType);
VARIANT_ENUM_CAST(Tween::EaseType);

class PropertyTweener : public Tweener {
	GDCLASS(PropertyTweener, Tweener);

public:
	Ref<PropertyTweener> from(Variant p_value);
	Ref<PropertyTweener> from_current();
	Ref<PropertyTweener> as_relative();
	Ref<PropertyTweener> set_trans(Tween::TransitionType p_trans);
	Ref<PropertyTweener> set_ease(Tween::EaseType p_ease);
	Ref<PropertyTweener> set_delay(float p_delay);

	void set_tween(Ref<Tween> p_tween) override;
	void start() override;
	bool step(float &r_delta) override;

	PropertyTweener(Object *p_target, NodePath p_property, Variant p_to, float p_duration);
	PropertyTweener();

protected:
	static void _bind_methods();

private:
	ObjectID target;
	Vector<StringName> property;
	Variant initial_val;
	Variant base_final_val;
	Variant final_val;
	Variant delta_val;

	float duration = 0;
	Tween::TransitionType trans_type = Tween::TRANS_MAX; // This is set inside set_tween();
	Tween::EaseType ease_type = Tween::EASE_MAX;

	float delay = 0;
	bool do_continue = true;
	bool relative = false;
};

class IntervalTweener : public Tweener {
	GDCLASS(IntervalTweener, Tweener);

public:
	void start() override;
	bool step(float &r_delta) override;

	IntervalTweener(float p_time);
	IntervalTweener();

private:
	float duration = 0;
};

class CallbackTweener : public Tweener {
	GDCLASS(CallbackTweener, Tweener);

public:
	Ref<CallbackTweener> set_delay(float p_delay);

	void start() override;
	bool step(float &r_delta) override;

	CallbackTweener(Callable p_callback);
	CallbackTweener();

protected:
	static void _bind_methods();

private:
	Callable callback;
	float delay = 0;
};

class MethodTweener : public Tweener {
	GDCLASS(MethodTweener, Tweener);

public:
	Ref<MethodTweener> set_trans(Tween::TransitionType p_trans);
	Ref<MethodTweener> set_ease(Tween::EaseType p_ease);
	Ref<MethodTweener> set_delay(float p_delay);

	void set_tween(Ref<Tween> p_tween) override;
	void start() override;
	bool step(float &r_delta) override;

	MethodTweener(Callable p_callback, Variant p_from, Variant p_to, float p_duration);
	MethodTweener();

protected:
	static void _bind_methods();

private:
	float duration = 0;
	float delay = 0;
	Tween::TransitionType trans_type = Tween::TRANS_MAX;
	Tween::EaseType ease_type = Tween::EASE_MAX;

	Ref<Tween> tween;
	Variant initial_val;
	Variant delta_val;
	Callable callback;
};

#endif
