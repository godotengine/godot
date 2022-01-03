/*************************************************************************/
/*  tween.cpp                                                            */
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

#include "tween.h"

#include "scene/animation/easing_equations.h"
#include "scene/main/node.h"

Tween::interpolater Tween::interpolaters[Tween::TRANS_MAX][Tween::EASE_MAX] = {
	{ &linear::in, &linear::in, &linear::in, &linear::in }, // Linear is the same for each easing.
	{ &sine::in, &sine::out, &sine::in_out, &sine::out_in },
	{ &quint::in, &quint::out, &quint::in_out, &quint::out_in },
	{ &quart::in, &quart::out, &quart::in_out, &quart::out_in },
	{ &quad::in, &quad::out, &quad::in_out, &quad::out_in },
	{ &expo::in, &expo::out, &expo::in_out, &expo::out_in },
	{ &elastic::in, &elastic::out, &elastic::in_out, &elastic::out_in },
	{ &cubic::in, &cubic::out, &cubic::in_out, &cubic::out_in },
	{ &circ::in, &circ::out, &circ::in_out, &circ::out_in },
	{ &bounce::in, &bounce::out, &bounce::in_out, &bounce::out_in },
	{ &back::in, &back::out, &back::in_out, &back::out_in },
};

void Tweener::set_tween(Ref<Tween> p_tween) {
	tween = p_tween;
}

void Tweener::clear_tween() {
	tween.unref();
}

void Tweener::_bind_methods() {
	ADD_SIGNAL(MethodInfo("finished"));
}

void Tween::start_tweeners() {
	if (tweeners.is_empty()) {
		dead = true;
		ERR_FAIL_MSG("Tween without commands, aborting.");
	}

	for (Ref<Tweener> &tweener : tweeners.write[current_step]) {
		tweener->start();
	}
}

Ref<PropertyTweener> Tween::tween_property(Object *p_target, NodePath p_property, Variant p_to, float p_duration) {
	ERR_FAIL_NULL_V(p_target, nullptr);
	ERR_FAIL_COND_V_MSG(!valid, nullptr, "Tween invalid. Either finished or created outside scene tree.");
	ERR_FAIL_COND_V_MSG(started, nullptr, "Can't append to a Tween that has started. Use stop() first.");

#ifdef DEBUG_ENABLED
	Variant::Type property_type = p_target->get_indexed(p_property.get_as_property_path().get_subnames()).get_type();
	ERR_FAIL_COND_V_MSG(property_type != p_to.get_type(), Ref<PropertyTweener>(), "Type mismatch between property and final value: " + Variant::get_type_name(property_type) + " and " + Variant::get_type_name(p_to.get_type()));
#endif

	Ref<PropertyTweener> tweener = memnew(PropertyTweener(p_target, p_property, p_to, p_duration));
	append(tweener);
	return tweener;
}

Ref<IntervalTweener> Tween::tween_interval(float p_time) {
	ERR_FAIL_COND_V_MSG(!valid, nullptr, "Tween invalid. Either finished or created outside scene tree.");
	ERR_FAIL_COND_V_MSG(started, nullptr, "Can't append to a Tween that has started. Use stop() first.");

	Ref<IntervalTweener> tweener = memnew(IntervalTweener(p_time));
	append(tweener);
	return tweener;
}

Ref<CallbackTweener> Tween::tween_callback(Callable p_callback) {
	ERR_FAIL_COND_V_MSG(!valid, nullptr, "Tween invalid. Either finished or created outside scene tree.");
	ERR_FAIL_COND_V_MSG(started, nullptr, "Can't append to a Tween that has started. Use stop() first.");

	Ref<CallbackTweener> tweener = memnew(CallbackTweener(p_callback));
	append(tweener);
	return tweener;
}

Ref<MethodTweener> Tween::tween_method(Callable p_callback, Variant p_from, Variant p_to, float p_duration) {
	ERR_FAIL_COND_V_MSG(!valid, nullptr, "Tween invalid. Either finished or created outside scene tree.");
	ERR_FAIL_COND_V_MSG(started, nullptr, "Can't append to a Tween that has started. Use stop() first.");

	Ref<MethodTweener> tweener = memnew(MethodTweener(p_callback, p_from, p_to, p_duration));
	append(tweener);
	return tweener;
}

void Tween::append(Ref<Tweener> p_tweener) {
	p_tweener->set_tween(this);

	if (parallel_enabled) {
		current_step = MAX(current_step, 0);
	} else {
		current_step++;
	}
	parallel_enabled = default_parallel;

	tweeners.resize(current_step + 1);
	tweeners.write[current_step].push_back(p_tweener);
}

void Tween::stop() {
	started = false;
	running = false;
	dead = false;
}

void Tween::pause() {
	running = false;
}

void Tween::play() {
	ERR_FAIL_COND_MSG(!valid, "Tween invalid. Either finished or created outside scene tree.");
	ERR_FAIL_COND_MSG(dead, "Can't play finished Tween, use stop() first to reset its state.");
	running = true;
}

void Tween::kill() {
	running = false; // For the sake of is_running().
	dead = true;
}

bool Tween::is_running() {
	return running;
}

void Tween::set_valid(bool p_valid) {
	valid = p_valid;
}

bool Tween::is_valid() {
	return valid;
}

void Tween::clear() {
	valid = false;

	for (List<Ref<Tweener>> &step : tweeners) {
		for (Ref<Tweener> &tweener : step) {
			tweener->clear_tween();
		}
	}
	tweeners.clear();
}

Ref<Tween> Tween::bind_node(Node *p_node) {
	ERR_FAIL_NULL_V(p_node, this);

	bound_node = p_node->get_instance_id();
	is_bound = true;
	return this;
}

Ref<Tween> Tween::set_process_mode(TweenProcessMode p_mode) {
	process_mode = p_mode;
	return this;
}

Tween::TweenProcessMode Tween::get_process_mode() {
	return process_mode;
}

Ref<Tween> Tween::set_pause_mode(TweenPauseMode p_mode) {
	pause_mode = p_mode;
	return this;
}

Tween::TweenPauseMode Tween::get_pause_mode() {
	return pause_mode;
}

Ref<Tween> Tween::set_parallel(bool p_parallel) {
	default_parallel = p_parallel;
	parallel_enabled = p_parallel;
	return this;
}

Ref<Tween> Tween::set_loops(int p_loops) {
	loops = p_loops;
	return this;
}

Ref<Tween> Tween::set_speed_scale(float p_speed) {
	speed_scale = p_speed;
	return this;
}

Ref<Tween> Tween::set_trans(TransitionType p_trans) {
	default_transition = p_trans;
	return this;
}

Tween::TransitionType Tween::get_trans() {
	return default_transition;
}

Ref<Tween> Tween::set_ease(EaseType p_ease) {
	default_ease = p_ease;
	return this;
}

Tween::EaseType Tween::get_ease() {
	return default_ease;
}

Ref<Tween> Tween::parallel() {
	parallel_enabled = true;
	return this;
}

Ref<Tween> Tween::chain() {
	parallel_enabled = false;
	return this;
}

bool Tween::custom_step(float p_delta) {
	bool r = running;
	running = true;
	bool ret = step(p_delta);
	running = running && r; // Running might turn false when Tween finished.
	return ret;
}

bool Tween::step(float p_delta) {
	ERR_FAIL_COND_V_MSG(tweeners.is_empty(), false, "Tween started, but has no Tweeners.");

	if (dead) {
		return false;
	}

	if (!running) {
		return true;
	}

	if (is_bound) {
		Object *bound_instance = ObjectDB::get_instance(bound_node);
		if (bound_instance) {
			Node *bound_node = Object::cast_to<Node>(bound_instance);
			// This can't by anything else than Node, so we can omit checking if casting succeeded.
			if (!bound_node->is_inside_tree()) {
				return true;
			}
		} else {
			return false;
		}
	}

	if (!started) {
		current_step = 0;
		loops_done = 0;
		start_tweeners();
		started = true;
	}

	float rem_delta = p_delta * speed_scale;
	bool step_active = false;

	while (rem_delta > 0 && running) {
		float step_delta = rem_delta;
		step_active = false;

		for (Ref<Tweener> &tweener : tweeners.write[current_step]) {
			// Modified inside Tweener.step().
			float temp_delta = rem_delta;
			// Turns to true if any Tweener returns true (i.e. is still not finished).
			step_active = tweener->step(temp_delta) || step_active;
			step_delta = MIN(temp_delta, step_delta);
		}

		rem_delta = step_delta;

		if (!step_active) {
			emit_signal(SNAME("step_finished"), current_step);
			current_step++;

			if (current_step == tweeners.size()) {
				loops_done++;
				if (loops_done == loops) {
					running = false;
					dead = true;
					emit_signal(SNAME("finished"));
				} else {
					emit_signal(SNAME("loop_finished"), loops_done);
					current_step = 0;
					start_tweeners();
				}
			} else {
				start_tweeners();
			}
		}
	}

	return true;
}

bool Tween::should_pause() {
	if (is_bound && pause_mode == TWEEN_PAUSE_BOUND) {
		Object *bound_instance = ObjectDB::get_instance(bound_node);
		if (bound_instance) {
			Node *bound_node = Object::cast_to<Node>(bound_instance);
			return !bound_node->can_process();
		}
	}

	return pause_mode != TWEEN_PAUSE_PROCESS;
}

real_t Tween::run_equation(TransitionType p_trans_type, EaseType p_ease_type, real_t p_time, real_t p_initial, real_t p_delta, real_t p_duration) {
	if (p_duration == 0) {
		// Special case to avoid dividing by 0 in equations.
		return p_initial + p_delta;
	}

	interpolater func = interpolaters[p_trans_type][p_ease_type];
	return func(p_time, p_initial, p_delta, p_duration);
}

Variant Tween::interpolate_variant(Variant p_initial_val, Variant p_delta_val, float p_time, float p_duration, TransitionType p_trans, EaseType p_ease) {
	ERR_FAIL_INDEX_V(p_trans, TransitionType::TRANS_MAX, Variant());
	ERR_FAIL_INDEX_V(p_ease, EaseType::EASE_MAX, Variant());

// Helper macro to run equation on sub-elements of the value (e.g. x and y of Vector2).
#define APPLY_EQUATION(element) \
	r.element = run_equation(p_trans, p_ease, p_time, i.element, d.element, p_duration);

	switch (p_initial_val.get_type()) {
		case Variant::BOOL: {
			return (run_equation(p_trans, p_ease, p_time, p_initial_val, p_delta_val, p_duration)) >= 0.5;
		}

		case Variant::INT: {
			return (int)run_equation(p_trans, p_ease, p_time, (int)p_initial_val, (int)p_delta_val, p_duration);
		}

		case Variant::FLOAT: {
			return run_equation(p_trans, p_ease, p_time, (real_t)p_initial_val, (real_t)p_delta_val, p_duration);
		}

		case Variant::VECTOR2: {
			Vector2 i = p_initial_val;
			Vector2 d = p_delta_val;
			Vector2 r;

			APPLY_EQUATION(x);
			APPLY_EQUATION(y);
			return r;
		}

		case Variant::VECTOR2I: {
			Vector2i i = p_initial_val;
			Vector2i d = p_delta_val;
			Vector2i r;

			APPLY_EQUATION(x);
			APPLY_EQUATION(y);
			return r;
		}

		case Variant::RECT2: {
			Rect2 i = p_initial_val;
			Rect2 d = p_delta_val;
			Rect2 r;

			APPLY_EQUATION(position.x);
			APPLY_EQUATION(position.y);
			APPLY_EQUATION(size.x);
			APPLY_EQUATION(size.y);
			return r;
		}

		case Variant::RECT2I: {
			Rect2i i = p_initial_val;
			Rect2i d = p_delta_val;
			Rect2i r;

			APPLY_EQUATION(position.x);
			APPLY_EQUATION(position.y);
			APPLY_EQUATION(size.x);
			APPLY_EQUATION(size.y);
			return r;
		}

		case Variant::VECTOR3: {
			Vector3 i = p_initial_val;
			Vector3 d = p_delta_val;
			Vector3 r;

			APPLY_EQUATION(x);
			APPLY_EQUATION(y);
			APPLY_EQUATION(z);
			return r;
		}

		case Variant::VECTOR3I: {
			Vector3i i = p_initial_val;
			Vector3i d = p_delta_val;
			Vector3i r;

			APPLY_EQUATION(x);
			APPLY_EQUATION(y);
			APPLY_EQUATION(z);
			return r;
		}

		case Variant::TRANSFORM2D: {
			Transform2D i = p_initial_val;
			Transform2D d = p_delta_val;
			Transform2D r;

			APPLY_EQUATION(elements[0][0]);
			APPLY_EQUATION(elements[0][1]);
			APPLY_EQUATION(elements[1][0]);
			APPLY_EQUATION(elements[1][1]);
			APPLY_EQUATION(elements[2][0]);
			APPLY_EQUATION(elements[2][1]);
			return r;
		}

		case Variant::QUATERNION: {
			Quaternion i = p_initial_val;
			Quaternion d = p_delta_val;
			Quaternion r;

			APPLY_EQUATION(x);
			APPLY_EQUATION(y);
			APPLY_EQUATION(z);
			APPLY_EQUATION(w);
			return r;
		}

		case Variant::AABB: {
			AABB i = p_initial_val;
			AABB d = p_delta_val;
			AABB r;

			APPLY_EQUATION(position.x);
			APPLY_EQUATION(position.y);
			APPLY_EQUATION(position.z);
			APPLY_EQUATION(size.x);
			APPLY_EQUATION(size.y);
			APPLY_EQUATION(size.z);
			return r;
		}

		case Variant::BASIS: {
			Basis i = p_initial_val;
			Basis d = p_delta_val;
			Basis r;

			APPLY_EQUATION(elements[0][0]);
			APPLY_EQUATION(elements[0][1]);
			APPLY_EQUATION(elements[0][2]);
			APPLY_EQUATION(elements[1][0]);
			APPLY_EQUATION(elements[1][1]);
			APPLY_EQUATION(elements[1][2]);
			APPLY_EQUATION(elements[2][0]);
			APPLY_EQUATION(elements[2][1]);
			APPLY_EQUATION(elements[2][2]);
			return r;
		}

		case Variant::TRANSFORM3D: {
			Transform3D i = p_initial_val;
			Transform3D d = p_delta_val;
			Transform3D r;

			APPLY_EQUATION(basis.elements[0][0]);
			APPLY_EQUATION(basis.elements[0][1]);
			APPLY_EQUATION(basis.elements[0][2]);
			APPLY_EQUATION(basis.elements[1][0]);
			APPLY_EQUATION(basis.elements[1][1]);
			APPLY_EQUATION(basis.elements[1][2]);
			APPLY_EQUATION(basis.elements[2][0]);
			APPLY_EQUATION(basis.elements[2][1]);
			APPLY_EQUATION(basis.elements[2][2]);
			APPLY_EQUATION(origin.x);
			APPLY_EQUATION(origin.y);
			APPLY_EQUATION(origin.z);
			return r;
		}

		case Variant::COLOR: {
			Color i = p_initial_val;
			Color d = p_delta_val;
			Color r;

			APPLY_EQUATION(r);
			APPLY_EQUATION(g);
			APPLY_EQUATION(b);
			APPLY_EQUATION(a);
			return r;
		}

		default: {
			return p_initial_val;
		}
	};
#undef APPLY_EQUATION
}

Variant Tween::calculate_delta_value(Variant p_intial_val, Variant p_final_val) {
	ERR_FAIL_COND_V_MSG(p_intial_val.get_type() != p_final_val.get_type(), p_intial_val, "Type mismatch between initial and final value: " + Variant::get_type_name(p_intial_val.get_type()) + " and " + Variant::get_type_name(p_final_val.get_type()));

	switch (p_intial_val.get_type()) {
		case Variant::BOOL: {
			return (int)p_final_val - (int)p_intial_val;
		}

		case Variant::RECT2: {
			Rect2 i = p_intial_val;
			Rect2 f = p_final_val;
			return Rect2(f.position - i.position, f.size - i.size);
		}

		case Variant::RECT2I: {
			Rect2i i = p_intial_val;
			Rect2i f = p_final_val;
			return Rect2i(f.position - i.position, f.size - i.size);
		}

		case Variant::TRANSFORM2D: {
			Transform2D i = p_intial_val;
			Transform2D f = p_final_val;
			return Transform2D(f.elements[0][0] - i.elements[0][0],
					f.elements[0][1] - i.elements[0][1],
					f.elements[1][0] - i.elements[1][0],
					f.elements[1][1] - i.elements[1][1],
					f.elements[2][0] - i.elements[2][0],
					f.elements[2][1] - i.elements[2][1]);
		}

		case Variant::AABB: {
			AABB i = p_intial_val;
			AABB f = p_final_val;
			return AABB(f.position - i.position, f.size - i.size);
		}

		case Variant::BASIS: {
			Basis i = p_intial_val;
			Basis f = p_final_val;
			return Basis(f.elements[0][0] - i.elements[0][0],
					f.elements[0][1] - i.elements[0][1],
					f.elements[0][2] - i.elements[0][2],
					f.elements[1][0] - i.elements[1][0],
					f.elements[1][1] - i.elements[1][1],
					f.elements[1][2] - i.elements[1][2],
					f.elements[2][0] - i.elements[2][0],
					f.elements[2][1] - i.elements[2][1],
					f.elements[2][2] - i.elements[2][2]);
		}

		case Variant::TRANSFORM3D: {
			Transform3D i = p_intial_val;
			Transform3D f = p_final_val;
			return Transform3D(f.basis.elements[0][0] - i.basis.elements[0][0],
					f.basis.elements[0][1] - i.basis.elements[0][1],
					f.basis.elements[0][2] - i.basis.elements[0][2],
					f.basis.elements[1][0] - i.basis.elements[1][0],
					f.basis.elements[1][1] - i.basis.elements[1][1],
					f.basis.elements[1][2] - i.basis.elements[1][2],
					f.basis.elements[2][0] - i.basis.elements[2][0],
					f.basis.elements[2][1] - i.basis.elements[2][1],
					f.basis.elements[2][2] - i.basis.elements[2][2],
					f.origin.x - i.origin.x,
					f.origin.y - i.origin.y,
					f.origin.z - i.origin.z);
		}

		default: {
			return Variant::evaluate(Variant::OP_SUBTRACT, p_final_val, p_intial_val);
		}
	};
}

void Tween::_bind_methods() {
	ClassDB::bind_method(D_METHOD("tween_property", "object", "property", "final_val", "duration"), &Tween::tween_property);
	ClassDB::bind_method(D_METHOD("tween_interval", "time"), &Tween::tween_interval);
	ClassDB::bind_method(D_METHOD("tween_callback", "callback"), &Tween::tween_callback);
	ClassDB::bind_method(D_METHOD("tween_method", "method", "from", "to", "duration"), &Tween::tween_method);

	ClassDB::bind_method(D_METHOD("custom_step", "delta"), &Tween::custom_step);
	ClassDB::bind_method(D_METHOD("stop"), &Tween::stop);
	ClassDB::bind_method(D_METHOD("pause"), &Tween::pause);
	ClassDB::bind_method(D_METHOD("play"), &Tween::play);
	ClassDB::bind_method(D_METHOD("kill"), &Tween::kill);

	ClassDB::bind_method(D_METHOD("is_running"), &Tween::is_running);
	ClassDB::bind_method(D_METHOD("is_valid"), &Tween::is_valid);
	ClassDB::bind_method(D_METHOD("bind_node", "node"), &Tween::bind_node);
	ClassDB::bind_method(D_METHOD("set_process_mode", "mode"), &Tween::set_process_mode);
	ClassDB::bind_method(D_METHOD("set_pause_mode", "mode"), &Tween::set_pause_mode);

	ClassDB::bind_method(D_METHOD("set_parallel", "parallel"), &Tween::set_parallel, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("set_loops", "loops"), &Tween::set_loops, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("set_speed_scale", "speed"), &Tween::set_speed_scale);
	ClassDB::bind_method(D_METHOD("set_trans", "trans"), &Tween::set_trans);
	ClassDB::bind_method(D_METHOD("set_ease", "ease"), &Tween::set_ease);

	ClassDB::bind_method(D_METHOD("parallel"), &Tween::parallel);
	ClassDB::bind_method(D_METHOD("chain"), &Tween::chain);

	ClassDB::bind_method(D_METHOD("interpolate_value", "initial_value", "delta_value", "elapsed_time", "duration", "trans_type", "ease_type"), &Tween::interpolate_variant);

	ADD_SIGNAL(MethodInfo("step_finished", PropertyInfo(Variant::INT, "idx")));
	ADD_SIGNAL(MethodInfo("loop_finished", PropertyInfo(Variant::INT, "loop_count")));
	ADD_SIGNAL(MethodInfo("finished"));

	BIND_ENUM_CONSTANT(TWEEN_PROCESS_PHYSICS);
	BIND_ENUM_CONSTANT(TWEEN_PROCESS_IDLE);

	BIND_ENUM_CONSTANT(TWEEN_PAUSE_BOUND);
	BIND_ENUM_CONSTANT(TWEEN_PAUSE_STOP);
	BIND_ENUM_CONSTANT(TWEEN_PAUSE_PROCESS);

	BIND_ENUM_CONSTANT(TRANS_LINEAR);
	BIND_ENUM_CONSTANT(TRANS_SINE);
	BIND_ENUM_CONSTANT(TRANS_QUINT);
	BIND_ENUM_CONSTANT(TRANS_QUART);
	BIND_ENUM_CONSTANT(TRANS_QUAD);
	BIND_ENUM_CONSTANT(TRANS_EXPO);
	BIND_ENUM_CONSTANT(TRANS_ELASTIC);
	BIND_ENUM_CONSTANT(TRANS_CUBIC);
	BIND_ENUM_CONSTANT(TRANS_CIRC);
	BIND_ENUM_CONSTANT(TRANS_BOUNCE);
	BIND_ENUM_CONSTANT(TRANS_BACK);

	BIND_ENUM_CONSTANT(EASE_IN);
	BIND_ENUM_CONSTANT(EASE_OUT);
	BIND_ENUM_CONSTANT(EASE_IN_OUT);
	BIND_ENUM_CONSTANT(EASE_OUT_IN);
}

Ref<PropertyTweener> PropertyTweener::from(Variant p_value) {
	initial_val = p_value;
	do_continue = false;
	return this;
}

Ref<PropertyTweener> PropertyTweener::from_current() {
	do_continue = false;
	return this;
}

Ref<PropertyTweener> PropertyTweener::as_relative() {
	relative = true;
	return this;
}

Ref<PropertyTweener> PropertyTweener::set_trans(Tween::TransitionType p_trans) {
	trans_type = p_trans;
	return this;
}

Ref<PropertyTweener> PropertyTweener::set_ease(Tween::EaseType p_ease) {
	ease_type = p_ease;
	return this;
}

Ref<PropertyTweener> PropertyTweener::set_delay(float p_delay) {
	delay = p_delay;
	return this;
}

void PropertyTweener::start() {
	elapsed_time = 0;
	finished = false;

	Object *target_instance = ObjectDB::get_instance(target);
	if (!target_instance) {
		WARN_PRINT("Target object freed before starting, aborting Tweener.");
		return;
	}

	if (do_continue) {
		initial_val = target_instance->get_indexed(property);
	}

	if (relative) {
		final_val = Variant::evaluate(Variant::Operator::OP_ADD, initial_val, base_final_val);
	}

	delta_val = tween->calculate_delta_value(initial_val, final_val);
}

bool PropertyTweener::step(float &r_delta) {
	if (finished) {
		// This is needed in case there's a parallel Tweener with longer duration.
		return false;
	}

	Object *target_instance = ObjectDB::get_instance(target);
	if (!target_instance) {
		return false;
	}
	elapsed_time += r_delta;

	if (elapsed_time < delay) {
		r_delta = 0;
		return true;
	}

	float time = MIN(elapsed_time - delay, duration);
	target_instance->set_indexed(property, tween->interpolate_variant(initial_val, delta_val, time, duration, trans_type, ease_type));

	if (time < duration) {
		r_delta = 0;
		return true;
	} else {
		finished = true;
		r_delta = elapsed_time - delay - duration;
		emit_signal(SNAME("finished"));
		return false;
	}
}

void PropertyTweener::set_tween(Ref<Tween> p_tween) {
	tween = p_tween;
	if (trans_type == Tween::TRANS_MAX) {
		trans_type = tween->get_trans();
	}
	if (ease_type == Tween::EASE_MAX) {
		ease_type = tween->get_ease();
	}
}

void PropertyTweener::_bind_methods() {
	ClassDB::bind_method(D_METHOD("from", "value"), &PropertyTweener::from);
	ClassDB::bind_method(D_METHOD("from_current"), &PropertyTweener::from_current);
	ClassDB::bind_method(D_METHOD("as_relative"), &PropertyTweener::as_relative);
	ClassDB::bind_method(D_METHOD("set_trans", "trans"), &PropertyTweener::set_trans);
	ClassDB::bind_method(D_METHOD("set_ease", "ease"), &PropertyTweener::set_ease);
	ClassDB::bind_method(D_METHOD("set_delay", "delay"), &PropertyTweener::set_delay);
}

PropertyTweener::PropertyTweener(Object *p_target, NodePath p_property, Variant p_to, float p_duration) {
	target = p_target->get_instance_id();
	property = p_property.get_as_property_path().get_subnames();
	initial_val = p_target->get_indexed(property);
	base_final_val = p_to;
	final_val = base_final_val;
	duration = p_duration;
}

PropertyTweener::PropertyTweener() {
	ERR_FAIL_MSG("Can't create empty PropertyTweener. Use get_tree().tween_property() or tween_property() instead.");
}

void IntervalTweener::start() {
	elapsed_time = 0;
	finished = false;
}

bool IntervalTweener::step(float &r_delta) {
	if (finished) {
		return false;
	}

	elapsed_time += r_delta;

	if (elapsed_time < duration) {
		r_delta = 0;
		return true;
	} else {
		finished = true;
		r_delta = elapsed_time - duration;
		emit_signal(SNAME("finished"));
		return false;
	}
}

IntervalTweener::IntervalTweener(float p_time) {
	duration = p_time;
}

IntervalTweener::IntervalTweener() {
	ERR_FAIL_MSG("Can't create empty IntervalTweener. Use get_tree().tween_interval() instead.");
}

Ref<CallbackTweener> CallbackTweener::set_delay(float p_delay) {
	delay = p_delay;
	return this;
}

void CallbackTweener::start() {
	elapsed_time = 0;
	finished = false;
}

bool CallbackTweener::step(float &r_delta) {
	if (finished) {
		return false;
	}

	elapsed_time += r_delta;
	if (elapsed_time >= delay) {
		Variant result;
		Callable::CallError ce;
		callback.call(nullptr, 0, result, ce);
		if (ce.error != Callable::CallError::CALL_OK) {
			ERR_FAIL_V_MSG(false, "Error calling method from CallbackTweener: " + Variant::get_call_error_text(this, callback.get_method(), nullptr, 0, ce));
		}

		finished = true;
		r_delta = elapsed_time - delay;
		emit_signal(SNAME("finished"));
		return false;
	}

	r_delta = 0;
	return true;
}

void CallbackTweener::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_delay", "delay"), &CallbackTweener::set_delay);
}

CallbackTweener::CallbackTweener(Callable p_callback) {
	callback = p_callback;
}

CallbackTweener::CallbackTweener() {
	ERR_FAIL_MSG("Can't create empty CallbackTweener. Use get_tree().tween_callback() instead.");
}

Ref<MethodTweener> MethodTweener::set_delay(float p_delay) {
	delay = p_delay;
	return this;
}

Ref<MethodTweener> MethodTweener::set_trans(Tween::TransitionType p_trans) {
	trans_type = p_trans;
	return this;
}

Ref<MethodTweener> MethodTweener::set_ease(Tween::EaseType p_ease) {
	ease_type = p_ease;
	return this;
}

void MethodTweener::start() {
	elapsed_time = 0;
	finished = false;
}

bool MethodTweener::step(float &r_delta) {
	if (finished) {
		return false;
	}

	elapsed_time += r_delta;

	if (elapsed_time < delay) {
		r_delta = 0;
		return true;
	}

	float time = MIN(elapsed_time - delay, duration);
	Variant current_val = tween->interpolate_variant(initial_val, delta_val, time, duration, trans_type, ease_type);
	const Variant **argptr = (const Variant **)alloca(sizeof(Variant *));
	argptr[0] = &current_val;

	Variant result;
	Callable::CallError ce;
	callback.call(argptr, 1, result, ce);
	if (ce.error != Callable::CallError::CALL_OK) {
		ERR_FAIL_V_MSG(false, "Error calling method from MethodTweener: " + Variant::get_call_error_text(this, callback.get_method(), argptr, 1, ce));
	}

	if (time < duration) {
		r_delta = 0;
		return true;
	} else {
		finished = true;
		r_delta = elapsed_time - delay - duration;
		emit_signal(SNAME("finished"));
		return false;
	}
}

void MethodTweener::set_tween(Ref<Tween> p_tween) {
	tween = p_tween;
	if (trans_type == Tween::TRANS_MAX) {
		trans_type = tween->get_trans();
	}
	if (ease_type == Tween::EASE_MAX) {
		ease_type = tween->get_ease();
	}
}

void MethodTweener::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_delay", "delay"), &MethodTweener::set_delay);
	ClassDB::bind_method(D_METHOD("set_trans", "trans"), &MethodTweener::set_trans);
	ClassDB::bind_method(D_METHOD("set_ease", "ease"), &MethodTweener::set_ease);
}

MethodTweener::MethodTweener(Callable p_callback, Variant p_from, Variant p_to, float p_duration) {
	callback = p_callback;
	initial_val = p_from;
	delta_val = tween->calculate_delta_value(p_from, p_to);
	duration = p_duration;
}

MethodTweener::MethodTweener() {
	ERR_FAIL_MSG("Can't create empty MethodTweener. Use get_tree().tween_method() instead.");
}
