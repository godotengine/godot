/**************************************************************************/
/*  tween.cpp                                                             */
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

#include "tween.h"

#include "core/object/class_db.h"
#include "core/variant/array.h"
#include "scene/animation/easing_equations.h"
#include "scene/main/node.h"
#include "scene/main/scene_tree.h"
#include "scene/resources/animation.h"

#define CHECK_VALID() \
	ERR_FAIL_COND_V_MSG(!valid, nullptr, "Tween invalid. Either finished or created outside scene tree."); \
	ERR_FAIL_COND_V_MSG(started, nullptr, "Can't append to a Tween that has started. Use stop() first.");

Tween::interpolater Tween::interpolaters[Tween::TRANS_MAX][Tween::EASE_MAX] = {
	{ &Linear::in, &Linear::in, &Linear::in, &Linear::in }, // Linear is the same for each easing.
	{ &Sine::in, &Sine::out, &Sine::in_out, &Sine::out_in },
	{ &Quint::in, &Quint::out, &Quint::in_out, &Quint::out_in },
	{ &Quart::in, &Quart::out, &Quart::in_out, &Quart::out_in },
	{ &Quad::in, &Quad::out, &Quad::in_out, &Quad::out_in },
	{ &Expo::in, &Expo::out, &Expo::in_out, &Expo::out_in },
	{ &Elastic::in, &Elastic::out, &Elastic::in_out, &Elastic::out_in },
	{ &Cubic::in, &Cubic::out, &Cubic::in_out, &Cubic::out_in },
	{ &Circ::in, &Circ::out, &Circ::in_out, &Circ::out_in },
	{ &Bounce::in, &Bounce::out, &Bounce::in_out, &Bounce::out_in },
	{ &Back::in, &Back::out, &Back::in_out, &Back::out_in },
	{ &Spring::in, &Spring::out, &Spring::in_out, &Spring::out_in },
};

void Tweener::set_tween(const Ref<Tween> &p_tween) {
	tween_id = p_tween->get_instance_id();
}

void Tweener::start() {
	elapsed_time = 0;
	finished = false;
}

Ref<Tween> Tweener::_get_tween() {
	return ObjectDB::get_ref<Tween>(tween_id);
}

void Tweener::_finish() {
	finished = true;
	emit_signal(SceneStringName(finished));
}

void Tweener::_bind_methods() {
	ADD_SIGNAL(MethodInfo("finished"));
}

Dictionary Tweener::snapshot_get_state() const {
	Dictionary state;
	state["elapsed_time"] = elapsed_time;
	state["finished"] = finished;
	state["supported"] = true;
	return state;
}

Error Tweener::snapshot_set_state(const Dictionary &p_state) {
	elapsed_time = p_state.get("elapsed_time", 0.0);
	finished = p_state.get("finished", false);
	return OK;
}

void Tween::_start_tweeners() {
	if (tweeners.is_empty()) {
		dead = true;
		ERR_FAIL_MSG("Tween without commands, aborting.");
	}

	for (Ref<Tweener> &tweener : tweeners[current_step]) {
		tweener->start();
	}
}

void Tween::_stop_internal(bool p_reset) {
	running = false;
	if (p_reset) {
		started = false;
		dead = false;
		total_time = 0;
	}
}

RequiredResult<PropertyTweener> Tween::tween_property(RequiredParam<const Object> rp_target, const NodePath &p_property, Variant p_to, double p_duration) {
	EXTRACT_PARAM_OR_FAIL_V(p_target, rp_target, nullptr);
	CHECK_VALID();

	Vector<StringName> property_subnames = p_property.get_as_property_path().get_subnames();
#ifdef DEBUG_ENABLED
	bool prop_valid;
	const Variant &prop_value = p_target->get_indexed(property_subnames, &prop_valid);
	ERR_FAIL_COND_V_MSG(!prop_valid, nullptr, vformat("The tweened property \"%s\" does not exist in object \"%s\".", p_property, p_target));
#else
	const Variant &prop_value = p_target->get_indexed(property_subnames);
#endif

	if (!Animation::validate_type_match(prop_value, p_to)) {
		return nullptr;
	}

	Ref<PropertyTweener> tweener;
	tweener.instantiate(p_target, property_subnames, p_to, p_duration);
	append(tweener);
	return tweener;
}

RequiredResult<IntervalTweener> Tween::tween_interval(double p_time) {
	CHECK_VALID();

	Ref<IntervalTweener> tweener;
	tweener.instantiate(p_time);
	append(tweener);
	return tweener;
}

RequiredResult<CallbackTweener> Tween::tween_callback(const Callable &p_callback) {
	CHECK_VALID();

	Ref<CallbackTweener> tweener;
	tweener.instantiate(p_callback);
	append(tweener);
	return tweener;
}

RequiredResult<MethodTweener> Tween::tween_method(const Callable &p_callback, const Variant p_from, Variant p_to, double p_duration) {
	CHECK_VALID();

	if (!Animation::validate_type_match(p_from, p_to)) {
		return nullptr;
	}

	Ref<MethodTweener> tweener;
	tweener.instantiate(p_callback, p_from, p_to, p_duration);
	append(tweener);
	return tweener;
}

RequiredResult<SubtweenTweener> Tween::tween_subtween(RequiredParam<Tween> rp_subtween) {
	CHECK_VALID();

	// Ensure that the subtween being added is not null.
	EXTRACT_PARAM_OR_FAIL_V(p_subtween, rp_subtween, nullptr);

	Ref<SubtweenTweener> tweener;
	tweener.instantiate(p_subtween);

	// Remove the tween from its parent tree, if it has one.
	// If the user created this tween without a parent tree attached,
	// then this step isn't necessary.
	if (tweener->subtween->parent_tree != nullptr) {
		tweener->subtween->parent_tree->remove_tween(tweener->subtween);
	}
	subtweens.push_back(p_subtween);
	append(tweener);
	return tweener;
}

RequiredResult<AwaitTweener> Tween::tween_await(const Signal &p_signal) {
	CHECK_VALID();

	Ref<AwaitTweener> tweener = memnew(AwaitTweener(p_signal));
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
	tweeners[current_step].push_back(p_tweener);
}

void Tween::stop() {
	_stop_internal(true);
}

void Tween::pause() {
	_stop_internal(false);
}

void Tween::play() {
	ERR_FAIL_COND_MSG(!valid, "Tween invalid. Either finished or created outside scene tree.");
	ERR_FAIL_COND_MSG(dead, "Can't play finished Tween, use stop() first to reset its state.");
	running = true;
}

void Tween::kill() {
	running = false; // For the sake of is_running().
	valid = false;
	dead = true;

	// Kill all subtweens of this tween.
	for (Ref<Tween> &st : subtweens) {
		st->kill();
	}
}

bool Tween::has_tweeners() const {
	return !tweeners.is_empty();
}

bool Tween::is_running() {
	return running;
}

bool Tween::is_valid() {
	return valid;
}

void Tween::clear() {
	valid = false;
	tweeners.clear();
}

RequiredResult<Tween> Tween::bind_node(RequiredParam<const Node> rp_node) {
	EXTRACT_PARAM_OR_FAIL_V(p_node, rp_node, this);

	bound_node = p_node->get_instance_id();
	is_bound = true;
	return this;
}

RequiredResult<Tween> Tween::set_process_mode(TweenProcessMode p_mode) {
	process_mode = p_mode;
	return this;
}

Tween::TweenProcessMode Tween::get_process_mode() const {
	return process_mode;
}

RequiredResult<Tween> Tween::set_pause_mode(TweenPauseMode p_mode) {
	pause_mode = p_mode;
	return this;
}

Tween::TweenPauseMode Tween::get_pause_mode() const {
	return pause_mode;
}

RequiredResult<Tween> Tween::set_ignore_time_scale(bool p_ignore) {
	ignore_time_scale = p_ignore;
	return this;
}

bool Tween::is_ignoring_time_scale() const {
	return ignore_time_scale;
}

RequiredResult<Tween> Tween::set_parallel(bool p_parallel) {
	default_parallel = p_parallel;
	parallel_enabled = p_parallel;
	return this;
}

RequiredResult<Tween> Tween::set_loops(int p_loops) {
	loops = p_loops;
	return this;
}

int Tween::get_loops_left() const {
	if (loops <= 0) {
		return -1; // Infinite loop.
	} else {
		return loops - loops_done;
	}
}

RequiredResult<Tween> Tween::set_speed_scale(float p_speed) {
	speed_scale = p_speed;
	return this;
}

RequiredResult<Tween> Tween::set_trans(TransitionType p_trans) {
	default_transition = p_trans;
	return this;
}

Tween::TransitionType Tween::get_trans() const {
	return default_transition;
}

RequiredResult<Tween> Tween::set_ease(EaseType p_ease) {
	default_ease = p_ease;
	return this;
}

Tween::EaseType Tween::get_ease() const {
	return default_ease;
}

RequiredResult<Tween> Tween::parallel() {
	parallel_enabled = true;
	return this;
}

RequiredResult<Tween> Tween::chain() {
	parallel_enabled = false;
	return this;
}

bool Tween::custom_step(double p_delta) {
	ERR_FAIL_COND_V_MSG(in_step, true, "Can't call custom_step() during another Tween step.");

	bool r = running;
	running = true;
	bool ret = step(p_delta);
	running = running && r; // Running might turn false when Tween finished.
	return ret;
}

bool Tween::step(double p_delta) {
	if (dead) {
		return false;
	}

	if (is_bound) {
		Node *node = get_bound_node();
		if (node) {
			if (!node->is_inside_tree()) {
				return true;
			}
		} else {
			return false;
		}
	}

	if (!running) {
		return true;
	}
	in_step = true;

	if (!started) {
		if (tweeners.is_empty()) {
			String tween_id;
			Node *node = get_bound_node();
			if (node) {
				tween_id = vformat("Tween (bound to %s)", node->is_inside_tree() ? (String)node->get_path() : (String)node->get_name());
			} else {
				tween_id = to_string();
			}
			in_step = false;
			ERR_FAIL_V_MSG(false, tween_id + ": started with no Tweeners.");
		}
		current_step = 0;
		loops_done = 0;
		total_time = 0;
		_start_tweeners();
		started = true;
	}

	double rem_delta = p_delta * speed_scale;
	bool step_active = false;
	total_time += rem_delta;

#ifdef DEBUG_ENABLED
	double initial_delta = rem_delta;
	bool potential_infinite = false;
#endif

	while (running && rem_delta > 0) {
		double step_delta = rem_delta;
		step_active = false;

		for (Ref<Tweener> &tweener : tweeners[current_step]) {
			// Modified inside Tweener.step().
			double temp_delta = rem_delta;
			// Turns to true if any Tweener returns true (i.e. is still not finished).
			step_active = tweener->step(temp_delta) || step_active;
			step_delta = MIN(temp_delta, step_delta);
		}

		rem_delta = step_delta;

		if (!step_active) {
			emit_signal(SNAME("step_finished"), current_step);
			current_step++;

			if (current_step == (int)tweeners.size()) {
				loops_done++;
				if (loops_done == loops) {
					running = false;
					dead = true;
					emit_signal(SceneStringName(finished));
					break;
				} else {
					emit_signal(SNAME("loop_finished"), loops_done);
					current_step = 0;
					_start_tweeners();
#ifdef DEBUG_ENABLED
					if (loops <= 0 && Math::is_equal_approx(rem_delta, initial_delta)) {
						if (!potential_infinite) {
							potential_infinite = true;
						} else {
							// Looped twice without using any time, this is 100% certain infinite loop.
							in_step = false;
							ERR_FAIL_V_MSG(false, "Infinite loop detected. Check set_loops() description for more info.");
						}
					}
#endif
				}
			} else {
				_start_tweeners();
			}
		}
	}
	in_step = false;
	return true;
}

bool Tween::can_process(bool p_tree_paused) const {
	if (is_bound && pause_mode == TWEEN_PAUSE_BOUND) {
		Node *node = get_bound_node();
		if (node) {
			return node->is_inside_tree() && node->can_process();
		}
	}

	return !p_tree_paused || pause_mode == TWEEN_PAUSE_PROCESS;
}

Node *Tween::get_bound_node() const {
	if (is_bound) {
		return ObjectDB::get_instance<Node>(bound_node);
	} else {
		return nullptr;
	}
}

double Tween::get_total_time() const {
	return total_time;
}

static Ref<Tweener> _create_tweener_from_snapshot(const Dictionary &p_state, Tween *p_owner, SceneTree *p_parent_tree, Error &r_error) {
	r_error = OK;

	if (!p_state.get("supported", true)) {
		r_error = ERR_UNAVAILABLE;
		return Ref<Tweener>();
	}

	String type = p_state.get("type", "");
	Ref<Tweener> tweener;

	if (type == "property") {
		Variant target_variant = p_state.get("target", Variant());
		Object *target_object = target_variant;
		if (target_object == nullptr) {
			r_error = ERR_INVALID_DATA;
			return Ref<Tweener>();
		}
		NodePath property_path = p_state.get("property", NodePath());
		Vector<StringName> property = property_path.get_as_property_path().get_subnames();
		Variant final_val = p_state.get("base_final_val", p_state.get("final_val", Variant()));
		double duration = p_state.get("duration", 0.0);

		Ref<PropertyTweener> property_tweener;
		property_tweener.instantiate(target_object, property, final_val, duration);
		tweener = property_tweener;
	} else if (type == "interval") {
		double duration = p_state.get("duration", 0.0);

		Ref<IntervalTweener> interval_tweener;
		interval_tweener.instantiate(duration);
		tweener = interval_tweener;
	} else if (type == "subtween") {
		Variant subtween_variant = p_state.get("subtween", Variant());
		if (subtween_variant.get_type() != Variant::DICTIONARY) {
			r_error = ERR_INVALID_DATA;
			return Ref<Tweener>();
		}

		Ref<Tween> subtween;
		subtween.instantiate(p_parent_tree);
		r_error = subtween->snapshot_set_state(subtween_variant);
		if (r_error != OK) {
			return Ref<Tweener>();
		}

		Ref<SubtweenTweener> subtween_tweener;
		subtween_tweener.instantiate(subtween);
		tweener = subtween_tweener;
	} else {
		r_error = ERR_INVALID_DATA;
		return Ref<Tweener>();
	}

	tweener->set_tween(p_owner);
	r_error = tweener->snapshot_set_state(p_state);
	if (r_error != OK) {
		return Ref<Tweener>();
	}

	return tweener;
}

Dictionary Tween::snapshot_get_state() const {
	Dictionary state;
	state["version"] = 1;
	state["type"] = "tween";
	state["supported"] = true;

	state["process_mode"] = process_mode;
	state["pause_mode"] = pause_mode;
	state["default_transition"] = default_transition;
	state["default_ease"] = default_ease;
	state["bound_node"] = get_bound_node();
	state["is_bound"] = is_bound;
	state["total_time"] = total_time;
	state["current_step"] = current_step;
	state["loops"] = loops;
	state["loops_done"] = loops_done;
	state["speed_scale"] = speed_scale;
	state["ignore_time_scale"] = ignore_time_scale;
	state["started"] = started;
	state["running"] = running;
	state["dead"] = dead;
	state["valid"] = valid;
	state["default_parallel"] = default_parallel;
	state["parallel_enabled"] = parallel_enabled;

	Array steps;
	for (const List<Ref<Tweener>> &step : tweeners) {
		Array step_state;
		for (const Ref<Tweener> &tweener : step) {
			Dictionary tweener_state = tweener->snapshot_get_state();
			if (!tweener_state.get("supported", true)) {
				state["supported"] = false;
			}
			step_state.push_back(tweener_state);
		}
		steps.push_back(step_state);
	}
	state["steps"] = steps;

	if (is_bound && get_bound_node() == nullptr) {
		state["supported"] = false;
		state["reason"] = "missing_bound_node";
	}

	return state;
}

Error Tween::snapshot_set_state(const Dictionary &p_state) {
	ERR_FAIL_COND_V_MSG(in_step, ERR_BUSY, "Can't restore Tween snapshot while the Tween is stepping.");
	ERR_FAIL_COND_V_MSG(!p_state.get("supported", true), ERR_UNAVAILABLE, "Tween snapshot contains unsupported Tweeners.");

	int version = p_state.get("version", 0);
	ERR_FAIL_COND_V_MSG(version != 1, ERR_INVALID_DATA, "Unsupported Tween snapshot version.");

	int process_mode_int = p_state.get("process_mode", TWEEN_PROCESS_IDLE);
	int pause_mode_int = p_state.get("pause_mode", TWEEN_PAUSE_BOUND);
	int transition_int = p_state.get("default_transition", TRANS_LINEAR);
	int ease_int = p_state.get("default_ease", EASE_IN_OUT);
	ERR_FAIL_COND_V(process_mode_int < TWEEN_PROCESS_PHYSICS || process_mode_int > TWEEN_PROCESS_IDLE, ERR_INVALID_DATA);
	ERR_FAIL_COND_V(pause_mode_int < TWEEN_PAUSE_BOUND || pause_mode_int > TWEEN_PAUSE_PROCESS, ERR_INVALID_DATA);
	ERR_FAIL_INDEX_V(transition_int, TRANS_MAX, ERR_INVALID_DATA);
	ERR_FAIL_INDEX_V(ease_int, EASE_MAX, ERR_INVALID_DATA);

	Array steps = p_state.get("steps", Array());
	LocalVector<List<Ref<Tweener>>> restored_tweeners;
	LocalVector<Ref<Tween>> restored_subtweens;
	restored_tweeners.resize(steps.size());
	for (int i = 0; i < steps.size(); i++) {
		Variant step_variant = steps[i];
		ERR_FAIL_COND_V(step_variant.get_type() != Variant::ARRAY, ERR_INVALID_DATA);
		Array step = step_variant;
		for (int j = 0; j < step.size(); j++) {
			Variant tweener_variant = step[j];
			ERR_FAIL_COND_V(tweener_variant.get_type() != Variant::DICTIONARY, ERR_INVALID_DATA);
			Dictionary tweener_state = tweener_variant;
			Error err = OK;
			Ref<Tweener> tweener = _create_tweener_from_snapshot(tweener_state, this, parent_tree, err);
			ERR_FAIL_COND_V(err != OK, err);
			restored_tweeners[i].push_back(tweener);

			Ref<SubtweenTweener> subtween_tweener = tweener;
			if (subtween_tweener.is_valid()) {
				restored_subtweens.push_back(subtween_tweener->subtween);
			}
		}
	}

	bool restored_started = p_state.get("started", false);
	bool restored_dead = p_state.get("dead", false);
	int restored_current_step = p_state.get("current_step", -1);
	if (restored_started && !restored_dead) {
		ERR_FAIL_COND_V(restored_current_step < 0 || restored_current_step >= (int)restored_tweeners.size(), ERR_INVALID_DATA);
	}

	process_mode = (TweenProcessMode)process_mode_int;
	pause_mode = (TweenPauseMode)pause_mode_int;
	default_transition = (TransitionType)transition_int;
	default_ease = (EaseType)ease_int;

	is_bound = p_state.get("is_bound", false);
	bound_node = ObjectID();
	if (is_bound) {
		Variant bound_node_variant = p_state.get("bound_node", Variant());
		Object *bound_object = bound_node_variant;
		Node *node = Object::cast_to<Node>(bound_object);
		ERR_FAIL_NULL_V_MSG(node, ERR_INVALID_DATA, "Tween snapshot bound node is missing.");
		bound_node = node->get_instance_id();
	}

	tweeners = restored_tweeners;
	subtweens = restored_subtweens;
	total_time = p_state.get("total_time", 0.0);
	current_step = restored_current_step;
	loops = p_state.get("loops", 1);
	loops_done = p_state.get("loops_done", 0);
	speed_scale = p_state.get("speed_scale", 1.0);
	ignore_time_scale = p_state.get("ignore_time_scale", false);
	started = restored_started;
	running = p_state.get("running", true);
	in_step = false;
	dead = restored_dead;
	valid = p_state.get("valid", true);
	default_parallel = p_state.get("default_parallel", false);
	parallel_enabled = p_state.get("parallel_enabled", false);
#ifdef DEBUG_ENABLED
	is_infinite = false;
#endif

	return OK;
}

real_t Tween::run_equation(TransitionType p_trans_type, EaseType p_ease_type, real_t p_time, real_t p_initial, real_t p_delta, real_t p_duration) {
	if (p_duration == 0) {
		// Special case to avoid dividing by 0 in equations.
		return p_initial + p_delta;
	}

	interpolater func = interpolaters[p_trans_type][p_ease_type];
	return func(p_time, p_initial, p_delta, p_duration);
}

Variant Tween::interpolate_variant(const Variant &p_initial_val, const Variant &p_delta_val, double p_time, double p_duration, TransitionType p_trans, EaseType p_ease) {
	ERR_FAIL_INDEX_V(p_trans, TransitionType::TRANS_MAX, Variant());
	ERR_FAIL_INDEX_V(p_ease, EaseType::EASE_MAX, Variant());

	Variant ret = Animation::add_variant(p_initial_val, p_delta_val);
	ret = Animation::interpolate_variant(p_initial_val, ret, run_equation(p_trans, p_ease, p_time, 0.0, 1.0, p_duration), p_initial_val.is_string());
	return ret;
}

String Tween::_to_string() {
	String ret = Object::_to_string();
	Node *node = get_bound_node();
	if (node) {
		ret += vformat(" (bound to %s)", node->get_name());
	}
	return ret;
}

void Tween::_bind_methods() {
	ClassDB::bind_method(D_METHOD("tween_property", "object", "property", "final_val", "duration"), &Tween::tween_property);
	ClassDB::bind_method(D_METHOD("tween_interval", "time"), &Tween::tween_interval);
	ClassDB::bind_method(D_METHOD("tween_callback", "callback"), &Tween::tween_callback);
	ClassDB::bind_method(D_METHOD("tween_method", "method", "from", "to", "duration"), &Tween::tween_method);
	ClassDB::bind_method(D_METHOD("tween_subtween", "subtween"), &Tween::tween_subtween);
	ClassDB::bind_method(D_METHOD("tween_await", "signal"), &Tween::tween_await);

	ClassDB::bind_method(D_METHOD("custom_step", "delta"), &Tween::custom_step);
	ClassDB::bind_method(D_METHOD("stop"), &Tween::stop);
	ClassDB::bind_method(D_METHOD("pause"), &Tween::pause);
	ClassDB::bind_method(D_METHOD("play"), &Tween::play);
	ClassDB::bind_method(D_METHOD("kill"), &Tween::kill);
	ClassDB::bind_method(D_METHOD("get_total_elapsed_time"), &Tween::get_total_time);
	ClassDB::bind_method(D_METHOD("snapshot_get_state"), &Tween::snapshot_get_state);
	ClassDB::bind_method(D_METHOD("snapshot_set_state", "state"), &Tween::snapshot_set_state);

	ClassDB::bind_method(D_METHOD("has_tweeners"), &Tween::has_tweeners);
	ClassDB::bind_method(D_METHOD("is_running"), &Tween::is_running);
	ClassDB::bind_method(D_METHOD("is_valid"), &Tween::is_valid);
	ClassDB::bind_method(D_METHOD("bind_node", "node"), &Tween::bind_node);
	ClassDB::bind_method(D_METHOD("set_process_mode", "mode"), &Tween::set_process_mode);
	ClassDB::bind_method(D_METHOD("set_pause_mode", "mode"), &Tween::set_pause_mode);
	ClassDB::bind_method(D_METHOD("set_ignore_time_scale", "ignore"), &Tween::set_ignore_time_scale, DEFVAL(true));

	ClassDB::bind_method(D_METHOD("set_parallel", "parallel"), &Tween::set_parallel, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("set_loops", "loops"), &Tween::set_loops, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("get_loops_left"), &Tween::get_loops_left);
	ClassDB::bind_method(D_METHOD("set_speed_scale", "speed"), &Tween::set_speed_scale);
	ClassDB::bind_method(D_METHOD("set_trans", "trans"), &Tween::set_trans);
	ClassDB::bind_method(D_METHOD("set_ease", "ease"), &Tween::set_ease);

	ClassDB::bind_method(D_METHOD("parallel"), &Tween::parallel);
	ClassDB::bind_method(D_METHOD("chain"), &Tween::chain);

	ClassDB::bind_static_method("Tween", D_METHOD("interpolate_value", "initial_value", "delta_value", "elapsed_time", "duration", "trans_type", "ease_type"), &Tween::interpolate_variant);

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
	BIND_ENUM_CONSTANT(TRANS_SPRING);

	BIND_ENUM_CONSTANT(EASE_IN);
	BIND_ENUM_CONSTANT(EASE_OUT);
	BIND_ENUM_CONSTANT(EASE_IN_OUT);
	BIND_ENUM_CONSTANT(EASE_OUT_IN);
}

Tween::Tween() {
	ERR_FAIL_MSG("Tween can't be created directly. Use create_tween() method.");
}

Tween::Tween(SceneTree *p_parent_tree) {
	parent_tree = p_parent_tree;
	valid = true;
}

double PropertyTweener::_get_custom_interpolated_value(const Variant &p_value) {
	const Variant *argptr = &p_value;

	Variant result;
	Callable::CallError ce;
	custom_method.callp(&argptr, 1, result, ce);
	if (ce.error != Callable::CallError::CALL_OK) {
		ERR_FAIL_V_MSG(false, "Error calling custom method from PropertyTweener: " + Variant::get_callable_error_text(custom_method, &argptr, 1, ce) + ".");
	} else if (result.get_type() != Variant::FLOAT) {
		ERR_FAIL_V_MSG(false, vformat("Wrong return type in PropertyTweener custom method. Expected float, got %s.", Variant::get_type_name(result.get_type())));
	}
	return result;
}

RequiredResult<PropertyTweener> PropertyTweener::from(const Variant &p_value) {
	Ref<Tween> tween = _get_tween();
	ERR_FAIL_COND_V(tween.is_null(), nullptr);

	Variant from_value = p_value;
	if (!Animation::validate_type_match(final_val, from_value)) {
		return nullptr;
	}

	initial_val = from_value;
	do_continue = false;
	return this;
}

RequiredResult<PropertyTweener> PropertyTweener::from_current() {
	do_continue = false;
	return this;
}

RequiredResult<PropertyTweener> PropertyTweener::as_relative() {
	relative = true;
	return this;
}

RequiredResult<PropertyTweener> PropertyTweener::set_trans(Tween::TransitionType p_trans) {
	trans_type = p_trans;
	return this;
}

RequiredResult<PropertyTweener> PropertyTweener::set_ease(Tween::EaseType p_ease) {
	ease_type = p_ease;
	return this;
}

RequiredResult<PropertyTweener> PropertyTweener::set_custom_interpolator(const Callable &p_method) {
	custom_method = p_method;
	return this;
}

RequiredResult<PropertyTweener> PropertyTweener::set_delay(double p_delay) {
	delay = p_delay;
	return this;
}

void PropertyTweener::start() {
	Tweener::start();

	Object *target_instance = ObjectDB::get_instance(target);
	if (!target_instance) {
		return;
	}

	if (do_continue) {
		if (Math::is_zero_approx(delay)) {
			initial_val = target_instance->get_indexed(property);
		} else {
			do_continue_delayed = true;
		}
	}

	if (relative) {
		final_val = Animation::add_variant(initial_val, base_final_val);
	}

	delta_val = Animation::subtract_variant(final_val, initial_val);
}

bool PropertyTweener::step(double &r_delta) {
	if (finished) {
		// This is needed in case there's a parallel Tweener with longer duration.
		return false;
	}

	Object *target_instance = ObjectDB::get_instance(target);
	if (!target_instance) {
		_finish();
		return false;
	}
	elapsed_time += r_delta;

	if (elapsed_time < delay) {
		r_delta = 0;
		return true;
	} else if (do_continue_delayed && !Math::is_zero_approx(delay)) {
		initial_val = target_instance->get_indexed(property);
		delta_val = Animation::subtract_variant(final_val, initial_val);
		do_continue_delayed = false;
	}

	Ref<Tween> tween = _get_tween();

	double time = MIN(elapsed_time - delay, duration);
	if (time < duration) {
		if (custom_method.is_valid()) {
			const Variant t = tween->interpolate_variant(0.0, 1.0, time, duration, trans_type, ease_type);
			double result = _get_custom_interpolated_value(t);
			target_instance->set_indexed(property, Animation::interpolate_variant(initial_val, final_val, result));
		} else {
			target_instance->set_indexed(property, tween->interpolate_variant(initial_val, delta_val, time, duration, trans_type, ease_type));
		}
		r_delta = 0;
		return true;
	} else {
		if (custom_method.is_valid()) {
			double final_t = _get_custom_interpolated_value(1.0);
			target_instance->set_indexed(property, Animation::interpolate_variant(initial_val, final_val, final_t));
		} else {
			target_instance->set_indexed(property, final_val);
		}
		r_delta = elapsed_time - delay - duration;
		_finish();
		return false;
	}
}

void PropertyTweener::set_tween(const Ref<Tween> &p_tween) {
	Tweener::set_tween(p_tween);
	if (trans_type == Tween::TRANS_MAX) {
		trans_type = p_tween->get_trans();
	}
	if (ease_type == Tween::EASE_MAX) {
		ease_type = p_tween->get_ease();
	}
}

Dictionary PropertyTweener::snapshot_get_state() const {
	Dictionary state = Tweener::snapshot_get_state();
	state["type"] = "property";

	Object *target_instance = ObjectDB::get_instance(target);
	if (target_instance == nullptr) {
		state["supported"] = false;
		state["reason"] = "missing_property_target";
		return state;
	}
	if (custom_method.is_valid()) {
		state["supported"] = false;
		state["reason"] = "custom_property_interpolator";
		return state;
	}

	state["target"] = target_instance;
	state["property"] = NodePath(Vector<StringName>(), property, false);
	state["initial_val"] = initial_val;
	state["base_final_val"] = base_final_val;
	state["final_val"] = final_val;
	state["delta_val"] = delta_val;
	state["duration"] = duration;
	state["trans_type"] = trans_type;
	state["ease_type"] = ease_type;
	state["delay"] = delay;
	state["do_continue"] = do_continue;
	state["do_continue_delayed"] = do_continue_delayed;
	state["relative"] = relative;
	return state;
}

Error PropertyTweener::snapshot_set_state(const Dictionary &p_state) {
	ERR_FAIL_COND_V_MSG(!p_state.get("supported", true), ERR_UNAVAILABLE, "PropertyTweener snapshot is unsupported.");
	Error err = Tweener::snapshot_set_state(p_state);
	ERR_FAIL_COND_V(err != OK, err);

	Variant target_variant = p_state.get("target", Variant());
	Object *target_instance = target_variant;
	ERR_FAIL_NULL_V_MSG(target_instance, ERR_INVALID_DATA, "PropertyTweener snapshot target is missing.");

	NodePath property_path = p_state.get("property", NodePath());
	target = target_instance->get_instance_id();
	property = property_path.get_as_property_path().get_subnames();
	initial_val = p_state.get("initial_val", Variant());
	base_final_val = p_state.get("base_final_val", Variant());
	final_val = p_state.get("final_val", base_final_val);
	delta_val = p_state.get("delta_val", Variant());
	duration = p_state.get("duration", 0.0);
	trans_type = (Tween::TransitionType)(int)p_state.get("trans_type", Tween::TRANS_LINEAR);
	ease_type = (Tween::EaseType)(int)p_state.get("ease_type", Tween::EASE_IN_OUT);
	delay = p_state.get("delay", 0.0);
	do_continue = p_state.get("do_continue", true);
	do_continue_delayed = p_state.get("do_continue_delayed", false);
	relative = p_state.get("relative", false);
	custom_method = Callable();
	ref_copy.unref();
	if (target_instance->is_ref_counted()) {
		ref_copy.reference_ptr(Object::cast_to<RefCounted>(target_instance));
	}
	return OK;
}

void PropertyTweener::_bind_methods() {
	ClassDB::bind_method(D_METHOD("from", "value"), &PropertyTweener::from);
	ClassDB::bind_method(D_METHOD("from_current"), &PropertyTweener::from_current);
	ClassDB::bind_method(D_METHOD("as_relative"), &PropertyTweener::as_relative);
	ClassDB::bind_method(D_METHOD("set_trans", "trans"), &PropertyTweener::set_trans);
	ClassDB::bind_method(D_METHOD("set_ease", "ease"), &PropertyTweener::set_ease);
	ClassDB::bind_method(D_METHOD("set_custom_interpolator", "interpolator_method"), &PropertyTweener::set_custom_interpolator);
	ClassDB::bind_method(D_METHOD("set_delay", "delay"), &PropertyTweener::set_delay);
}

PropertyTweener::PropertyTweener(const Object *p_target, const Vector<StringName> &p_property, const Variant &p_to, double p_duration) {
	target = p_target->get_instance_id();
	property = p_property;
	initial_val = p_target->get_indexed(property);
	base_final_val = p_to;
	final_val = base_final_val;
	duration = p_duration;

	if (p_target->is_ref_counted()) {
		ref_copy = p_target;
	}
}

PropertyTweener::PropertyTweener() {
	ERR_FAIL_MSG("PropertyTweener can't be created directly. Use the tween_property() method in Tween.");
}

bool IntervalTweener::step(double &r_delta) {
	if (finished) {
		return false;
	}

	elapsed_time += r_delta;

	if (elapsed_time < duration) {
		r_delta = 0;
		return true;
	} else {
		r_delta = elapsed_time - duration;
		_finish();
		return false;
	}
}

IntervalTweener::IntervalTweener(double p_time) {
	duration = p_time;
}

Dictionary IntervalTweener::snapshot_get_state() const {
	Dictionary state = Tweener::snapshot_get_state();
	state["type"] = "interval";
	state["duration"] = duration;
	return state;
}

Error IntervalTweener::snapshot_set_state(const Dictionary &p_state) {
	Error err = Tweener::snapshot_set_state(p_state);
	ERR_FAIL_COND_V(err != OK, err);
	duration = p_state.get("duration", 0.0);
	return OK;
}

IntervalTweener::IntervalTweener() {
	ERR_FAIL_MSG("IntervalTweener can't be created directly. Use the tween_interval() method in Tween.");
}

RequiredResult<CallbackTweener> CallbackTweener::set_delay(double p_delay) {
	delay = p_delay;
	return this;
}

bool CallbackTweener::step(double &r_delta) {
	if (finished) {
		return false;
	}

	if (!callback.is_valid()) {
		_finish();
		return false;
	}

	elapsed_time += r_delta;
	if (elapsed_time >= delay) {
		Variant result;
		Callable::CallError ce;
		callback.callp(nullptr, 0, result, ce);
		if (ce.error != Callable::CallError::CALL_OK) {
			ERR_FAIL_V_MSG(false, "Error calling method from CallbackTweener: " + Variant::get_callable_error_text(callback, nullptr, 0, ce) + ".");
		}

		r_delta = elapsed_time - delay;
		_finish();
		return false;
	}

	r_delta = 0;
	return true;
}

Dictionary CallbackTweener::snapshot_get_state() const {
	Dictionary state = Tweener::snapshot_get_state();
	state["type"] = "callback";
	state["supported"] = false;
	state["reason"] = "callback_tweener";
	state["delay"] = delay;
	return state;
}

Error CallbackTweener::snapshot_set_state(const Dictionary &p_state) {
	return ERR_UNAVAILABLE;
}

void CallbackTweener::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_delay", "delay"), &CallbackTweener::set_delay);
}

CallbackTweener::CallbackTweener(const Callable &p_callback) {
	callback = p_callback;

	Object *callback_instance = p_callback.get_object();
	if (callback_instance && callback_instance->is_ref_counted()) {
		ref_copy = callback_instance;
	}
}

CallbackTweener::CallbackTweener() {
	ERR_FAIL_MSG("CallbackTweener can't be created directly. Use the tween_callback() method in Tween.");
}

RequiredResult<MethodTweener> MethodTweener::set_delay(double p_delay) {
	delay = p_delay;
	return this;
}

RequiredResult<MethodTweener> MethodTweener::set_trans(Tween::TransitionType p_trans) {
	trans_type = p_trans;
	return this;
}

RequiredResult<MethodTweener> MethodTweener::set_ease(Tween::EaseType p_ease) {
	ease_type = p_ease;
	return this;
}

bool MethodTweener::step(double &r_delta) {
	if (finished) {
		return false;
	}

	if (!callback.is_valid()) {
		_finish();
		return false;
	}

	elapsed_time += r_delta;

	if (elapsed_time < delay) {
		r_delta = 0;
		return true;
	}

	Ref<Tween> tween = _get_tween();

	Variant current_val;
	double time = MIN(elapsed_time - delay, duration);
	if (time < duration) {
		current_val = tween->interpolate_variant(initial_val, delta_val, time, duration, trans_type, ease_type);
	} else {
		current_val = final_val;
	}
	const Variant **argptr = (const Variant **)alloca(sizeof(Variant *));
	argptr[0] = &current_val;

	Variant result;
	Callable::CallError ce;
	callback.callp(argptr, 1, result, ce);
	if (ce.error != Callable::CallError::CALL_OK) {
		ERR_FAIL_V_MSG(false, "Error calling method from MethodTweener: " + Variant::get_callable_error_text(callback, argptr, 1, ce) + ".");
	}

	if (time < duration) {
		r_delta = 0;
		return true;
	} else {
		r_delta = elapsed_time - delay - duration;
		_finish();
		return false;
	}
}

void MethodTweener::set_tween(const Ref<Tween> &p_tween) {
	Tweener::set_tween(p_tween);
	if (trans_type == Tween::TRANS_MAX) {
		trans_type = p_tween->get_trans();
	}
	if (ease_type == Tween::EASE_MAX) {
		ease_type = p_tween->get_ease();
	}
}

Dictionary MethodTweener::snapshot_get_state() const {
	Dictionary state = Tweener::snapshot_get_state();
	state["type"] = "method";
	state["supported"] = false;
	state["reason"] = "method_tweener";
	state["duration"] = duration;
	state["delay"] = delay;
	return state;
}

Error MethodTweener::snapshot_set_state(const Dictionary &p_state) {
	return ERR_UNAVAILABLE;
}

void MethodTweener::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_delay", "delay"), &MethodTweener::set_delay);
	ClassDB::bind_method(D_METHOD("set_trans", "trans"), &MethodTweener::set_trans);
	ClassDB::bind_method(D_METHOD("set_ease", "ease"), &MethodTweener::set_ease);
}

MethodTweener::MethodTweener(const Callable &p_callback, const Variant &p_from, const Variant &p_to, double p_duration) {
	callback = p_callback;
	initial_val = p_from;
	delta_val = Animation::subtract_variant(p_to, p_from);
	final_val = p_to;
	duration = p_duration;

	Object *callback_instance = p_callback.get_object();
	if (callback_instance && callback_instance->is_ref_counted()) {
		ref_copy = callback_instance;
	}
}

MethodTweener::MethodTweener() {
	ERR_FAIL_MSG("MethodTweener can't be created directly. Use the tween_method() method in Tween.");
}

void SubtweenTweener::start() {
	Tweener::start();

	// Reset the subtween.
	subtween->stop();

	// It's possible that a subtween could be killed before it is started;
	// if so, we just want to skip it entirely.
	if (subtween->is_valid()) {
		subtween->play();
	} else {
		_finish();
	}
}

bool SubtweenTweener::step(double &r_delta) {
	if (finished) {
		return false;
	}

	elapsed_time += r_delta;

	if (elapsed_time < delay) {
		r_delta = 0;
		return true;
	}

	if (!subtween->step(r_delta)) {
		r_delta = elapsed_time - delay - subtween->get_total_time();
		_finish();
		return false;
	}

	r_delta = 0;
	return true;
}

Dictionary SubtweenTweener::snapshot_get_state() const {
	Dictionary state = Tweener::snapshot_get_state();
	state["type"] = "subtween";
	if (subtween.is_null()) {
		state["supported"] = false;
		state["reason"] = "missing_subtween";
		return state;
	}

	Dictionary subtween_state = subtween->snapshot_get_state();
	if (!subtween_state.get("supported", true)) {
		state["supported"] = false;
		state["reason"] = "unsupported_subtween";
	}
	state["subtween"] = subtween_state;
	state["delay"] = delay;
	return state;
}

Error SubtweenTweener::snapshot_set_state(const Dictionary &p_state) {
	ERR_FAIL_COND_V_MSG(!p_state.get("supported", true), ERR_UNAVAILABLE, "SubtweenTweener snapshot is unsupported.");
	Error err = Tweener::snapshot_set_state(p_state);
	ERR_FAIL_COND_V(err != OK, err);
	delay = p_state.get("delay", 0.0);
	return OK;
}

RequiredResult<SubtweenTweener> SubtweenTweener::set_delay(double p_delay) {
	delay = p_delay;
	return this;
}

void SubtweenTweener::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_delay", "delay"), &SubtweenTweener::set_delay);
}

SubtweenTweener::SubtweenTweener(const Ref<Tween> &p_subtween) {
	subtween = p_subtween;
}

SubtweenTweener::SubtweenTweener() {
	ERR_FAIL_MSG("SubtweenTweener can't be created directly. Use the tween_subtween() method in Tween.");
}

Ref<AwaitTweener> AwaitTweener::set_timeout(double p_timeout) {
	timeout = p_timeout;
	return this;
}

void AwaitTweener::start() {
	Tweener::start();
	received = false;
}

bool AwaitTweener::step(double &r_delta) {
	if (finished) {
		return false;
	}

	if (!signal.get_object() || !signal.is_connected(target_callable)) { // In case the object was destroyed before emitting.
		_finish();
		return false;
	}

	elapsed_time += r_delta;

	if (timeout >= 0 && elapsed_time >= timeout) {
		_finish();
		r_delta = elapsed_time - timeout;
		return false;
	}

	r_delta = 0; // "Consume" all remaining time to prevent infinite loops.
	if (received) {
		_finish();
		return false;
	}
	return true;
}

AwaitTweener::AwaitTweener(const Signal &p_signal) {
	signal = p_signal;
	target_callable = Callable(this, SNAME("_signal_callback"));
	signal.connect(target_callable);

	Object *signal_instance = p_signal.get_object();
	if (signal_instance && signal_instance->is_ref_counted()) {
		ref_copy = signal_instance;
	}
}

AwaitTweener::AwaitTweener() {
	ERR_FAIL_MSG("AwaitTweener can't be created directly. Use the tween_await() method in Tween.");
}

void AwaitTweener::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_timeout", "timeout"), &AwaitTweener::set_timeout);
	ClassDB::bind_vararg_method(METHOD_FLAGS_DEFAULT, "_signal_callback", &AwaitTweener::_signal_received, MethodInfo("_signal_callback"));
}

void AwaitTweener::_signal_received(const Variant **p_args, int p_argcount, Callable::CallError &r_error) {
	received = true;
}
