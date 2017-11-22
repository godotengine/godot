/*************************************************************************/
/*  tween.cpp                                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "method_bind_ext.gen.inc"

void Tween::_add_pending_command(StringName p_key, const Variant &p_arg1, const Variant &p_arg2, const Variant &p_arg3, const Variant &p_arg4, const Variant &p_arg5, const Variant &p_arg6, const Variant &p_arg7, const Variant &p_arg8, const Variant &p_arg9, const Variant &p_arg10) {

	pending_commands.push_back(PendingCommand());
	PendingCommand &cmd = pending_commands.back()->get();

	cmd.key = p_key;
	int &count = cmd.args;
	if (p_arg10.get_type() != Variant::NIL)
		count = 10;
	else if (p_arg9.get_type() != Variant::NIL)
		count = 9;
	else if (p_arg8.get_type() != Variant::NIL)
		count = 8;
	else if (p_arg7.get_type() != Variant::NIL)
		count = 7;
	else if (p_arg6.get_type() != Variant::NIL)
		count = 6;
	else if (p_arg5.get_type() != Variant::NIL)
		count = 5;
	else if (p_arg4.get_type() != Variant::NIL)
		count = 4;
	else if (p_arg3.get_type() != Variant::NIL)
		count = 3;
	else if (p_arg2.get_type() != Variant::NIL)
		count = 2;
	else if (p_arg1.get_type() != Variant::NIL)
		count = 1;
	if (count > 0)
		cmd.arg[0] = p_arg1;
	if (count > 1)
		cmd.arg[1] = p_arg2;
	if (count > 2)
		cmd.arg[2] = p_arg3;
	if (count > 3)
		cmd.arg[3] = p_arg4;
	if (count > 4)
		cmd.arg[4] = p_arg5;
	if (count > 5)
		cmd.arg[5] = p_arg6;
	if (count > 6)
		cmd.arg[6] = p_arg7;
	if (count > 7)
		cmd.arg[7] = p_arg8;
	if (count > 8)
		cmd.arg[8] = p_arg9;
	if (count > 9)
		cmd.arg[9] = p_arg10;
}

void Tween::_process_pending_commands() {

	for (List<PendingCommand>::Element *E = pending_commands.front(); E; E = E->next()) {

		PendingCommand &cmd = E->get();
		Variant::CallError err;
		Variant *arg[10] = {
			&cmd.arg[0],
			&cmd.arg[1],
			&cmd.arg[2],
			&cmd.arg[3],
			&cmd.arg[4],
			&cmd.arg[5],
			&cmd.arg[6],
			&cmd.arg[7],
			&cmd.arg[8],
			&cmd.arg[9],
		};
		this->call(cmd.key, (const Variant **)arg, cmd.args, err);
	}
	pending_commands.clear();
}

bool Tween::_set(const StringName &p_name, const Variant &p_value) {

	String name = p_name;

	if (name == "playback/speed" || name == "speed") { //bw compatibility
		set_speed_scale(p_value);

	} else if (name == "playback/active") {
		set_active(p_value);

	} else if (name == "playback/repeat") {
		set_repeat(p_value);
	}
	return true;
}

bool Tween::_get(const StringName &p_name, Variant &r_ret) const {

	String name = p_name;

	if (name == "playback/speed") { //bw compatibility

		r_ret = speed_scale;
	} else if (name == "playback/active") {

		r_ret = is_active();
	} else if (name == "playback/repeat") {

		r_ret = is_repeat();
	}

	return true;
}

void Tween::_get_property_list(List<PropertyInfo> *p_list) const {

	p_list->push_back(PropertyInfo(Variant::BOOL, "playback/active", PROPERTY_HINT_NONE, ""));
	p_list->push_back(PropertyInfo(Variant::BOOL, "playback/repeat", PROPERTY_HINT_NONE, ""));
	p_list->push_back(PropertyInfo(Variant::REAL, "playback/speed", PROPERTY_HINT_RANGE, "-64,64,0.01"));
}

void Tween::_notification(int p_what) {

	switch (p_what) {

		case NOTIFICATION_ENTER_TREE: {

			if (!processing) {
				//make sure that a previous process state was not saved
				//only process if "processing" is set
				set_physics_process_internal(false);
				set_process_internal(false);
			}
		} break;
		case NOTIFICATION_READY: {

		} break;
		case NOTIFICATION_INTERNAL_PROCESS: {
			if (tween_process_mode == TWEEN_PROCESS_PHYSICS)
				break;

			if (processing)
				_tween_process(get_process_delta_time());
		} break;
		case NOTIFICATION_INTERNAL_PHYSICS_PROCESS: {

			if (tween_process_mode == TWEEN_PROCESS_IDLE)
				break;

			if (processing)
				_tween_process(get_physics_process_delta_time());
		} break;
		case NOTIFICATION_EXIT_TREE: {

			stop_all();
		} break;
	}
}

void Tween::_bind_methods() {

	ClassDB::bind_method(D_METHOD("is_active"), &Tween::is_active);
	ClassDB::bind_method(D_METHOD("set_active", "active"), &Tween::set_active);

	ClassDB::bind_method(D_METHOD("is_repeat"), &Tween::is_repeat);
	ClassDB::bind_method(D_METHOD("set_repeat", "repeat"), &Tween::set_repeat);

	ClassDB::bind_method(D_METHOD("set_speed_scale", "speed"), &Tween::set_speed_scale);
	ClassDB::bind_method(D_METHOD("get_speed_scale"), &Tween::get_speed_scale);

	ClassDB::bind_method(D_METHOD("set_tween_process_mode", "mode"), &Tween::set_tween_process_mode);
	ClassDB::bind_method(D_METHOD("get_tween_process_mode"), &Tween::get_tween_process_mode);

	ClassDB::bind_method(D_METHOD("start"), &Tween::start);
	ClassDB::bind_method(D_METHOD("reset", "object", "key"), &Tween::reset, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("reset_all"), &Tween::reset_all);
	ClassDB::bind_method(D_METHOD("stop", "object", "key"), &Tween::stop, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("stop_all"), &Tween::stop_all);
	ClassDB::bind_method(D_METHOD("resume", "object", "key"), &Tween::resume, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("resume_all"), &Tween::resume_all);
	ClassDB::bind_method(D_METHOD("remove", "object", "key"), &Tween::remove, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("_remove", "object", "key", "first_only"), &Tween::_remove);
	ClassDB::bind_method(D_METHOD("remove_all"), &Tween::remove_all);
	ClassDB::bind_method(D_METHOD("seek", "time"), &Tween::seek);
	ClassDB::bind_method(D_METHOD("tell"), &Tween::tell);
	ClassDB::bind_method(D_METHOD("get_runtime"), &Tween::get_runtime);

	ClassDB::bind_method(D_METHOD("interpolate_property", "object", "property", "initial_val", "final_val", "duration", "trans_type", "ease_type", "delay"), &Tween::interpolate_property, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("interpolate_method", "object", "method", "initial_val", "final_val", "duration", "trans_type", "ease_type", "delay"), &Tween::interpolate_method, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("interpolate_callback", "object", "duration", "callback", "arg1", "arg2", "arg3", "arg4", "arg5"), &Tween::interpolate_callback, DEFVAL(Variant()), DEFVAL(Variant()), DEFVAL(Variant()), DEFVAL(Variant()), DEFVAL(Variant()));
	ClassDB::bind_method(D_METHOD("interpolate_deferred_callback", "object", "duration", "callback", "arg1", "arg2", "arg3", "arg4", "arg5"), &Tween::interpolate_deferred_callback, DEFVAL(Variant()), DEFVAL(Variant()), DEFVAL(Variant()), DEFVAL(Variant()), DEFVAL(Variant()));
	ClassDB::bind_method(D_METHOD("follow_property", "object", "property", "initial_val", "target", "target_property", "duration", "trans_type", "ease_type", "delay"), &Tween::follow_property, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("follow_method", "object", "method", "initial_val", "target", "target_method", "duration", "trans_type", "ease_type", "delay"), &Tween::follow_method, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("targeting_property", "object", "property", "initial", "initial_val", "final_val", "duration", "trans_type", "ease_type", "delay"), &Tween::targeting_property, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("targeting_method", "object", "method", "initial", "initial_method", "final_val", "duration", "trans_type", "ease_type", "delay"), &Tween::targeting_method, DEFVAL(0));

	ADD_SIGNAL(MethodInfo("tween_started", PropertyInfo(Variant::OBJECT, "object"), PropertyInfo(Variant::STRING, "key")));
	ADD_SIGNAL(MethodInfo("tween_step", PropertyInfo(Variant::OBJECT, "object"), PropertyInfo(Variant::STRING, "key"), PropertyInfo(Variant::REAL, "elapsed"), PropertyInfo(Variant::OBJECT, "value")));
	ADD_SIGNAL(MethodInfo("tween_completed", PropertyInfo(Variant::OBJECT, "object"), PropertyInfo(Variant::STRING, "key")));

	ADD_PROPERTY(PropertyInfo(Variant::INT, "playback_process_mode", PROPERTY_HINT_ENUM, "Physics,Idle"), "set_tween_process_mode", "get_tween_process_mode");

	BIND_ENUM_CONSTANT(TWEEN_PROCESS_PHYSICS);
	BIND_ENUM_CONSTANT(TWEEN_PROCESS_IDLE);

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

Variant &Tween::_get_initial_val(InterpolateData &p_data) {

	switch (p_data.type) {
		case INTER_PROPERTY:
		case INTER_METHOD:
		case FOLLOW_PROPERTY:
		case FOLLOW_METHOD:
			return p_data.initial_val;

		case TARGETING_PROPERTY:
		case TARGETING_METHOD: {

			Object *object = ObjectDB::get_instance(p_data.target_id);
			ERR_FAIL_COND_V(object == NULL, p_data.initial_val);

			static Variant initial_val;
			if (p_data.type == TARGETING_PROPERTY) {

				bool valid = false;
				initial_val = object->get_indexed(p_data.target_key, &valid);
				ERR_FAIL_COND_V(!valid, p_data.initial_val);
			} else {

				Variant::CallError error;
				initial_val = object->call(p_data.target_key[0], NULL, 0, error);
				ERR_FAIL_COND_V(error.error != Variant::CallError::CALL_OK, p_data.initial_val);
			}
			return initial_val;
		} break;
	}
	return p_data.delta_val;
}

Variant &Tween::_get_delta_val(InterpolateData &p_data) {

	switch (p_data.type) {
		case INTER_PROPERTY:
		case INTER_METHOD:
			return p_data.delta_val;

		case FOLLOW_PROPERTY:
		case FOLLOW_METHOD: {

			Object *target = ObjectDB::get_instance(p_data.target_id);
			ERR_FAIL_COND_V(target == NULL, p_data.initial_val);

			Variant final_val;

			if (p_data.type == FOLLOW_PROPERTY) {

				bool valid = false;
				final_val = target->get_indexed(p_data.target_key, &valid);
				ERR_FAIL_COND_V(!valid, p_data.initial_val);
			} else {

				Variant::CallError error;
				final_val = target->call(p_data.target_key[0], NULL, 0, error);
				ERR_FAIL_COND_V(error.error != Variant::CallError::CALL_OK, p_data.initial_val);
			}

			// convert INT to REAL is better for interpolaters
			if (final_val.get_type() == Variant::INT) final_val = final_val.operator real_t();
			_calc_delta_val(p_data.initial_val, final_val, p_data.delta_val);
			return p_data.delta_val;
		} break;

		case TARGETING_PROPERTY:
		case TARGETING_METHOD: {

			Variant initial_val = _get_initial_val(p_data);
			// convert INT to REAL is better for interpolaters
			if (initial_val.get_type() == Variant::INT) initial_val = initial_val.operator real_t();

			//_calc_delta_val(p_data.initial_val, p_data.final_val, p_data.delta_val);
			_calc_delta_val(initial_val, p_data.final_val, p_data.delta_val);
			return p_data.delta_val;
		} break;
	}
	return p_data.initial_val;
}

Variant Tween::_run_equation(InterpolateData &p_data) {

	Variant &initial_val = _get_initial_val(p_data);
	Variant &delta_val = _get_delta_val(p_data);
	Variant result;

#define APPLY_EQUATION(element) \
	r.element = _run_equation(p_data.trans_type, p_data.ease_type, p_data.elapsed - p_data.delay, i.element, d.element, p_data.duration);

	switch (initial_val.get_type()) {

		case Variant::BOOL:
			result = (_run_equation(p_data.trans_type, p_data.ease_type, p_data.elapsed - p_data.delay, initial_val, delta_val, p_data.duration)) >= 0.5;
			break;

		case Variant::INT:
			result = (int)_run_equation(p_data.trans_type, p_data.ease_type, p_data.elapsed - p_data.delay, (int)initial_val, (int)delta_val, p_data.duration);
			break;

		case Variant::REAL:
			result = _run_equation(p_data.trans_type, p_data.ease_type, p_data.elapsed - p_data.delay, (real_t)initial_val, (real_t)delta_val, p_data.duration);
			break;

		case Variant::VECTOR2: {
			Vector2 i = initial_val;
			Vector2 d = delta_val;
			Vector2 r;

			APPLY_EQUATION(x);
			APPLY_EQUATION(y);

			result = r;
		} break;

		case Variant::VECTOR3: {
			Vector3 i = initial_val;
			Vector3 d = delta_val;
			Vector3 r;

			APPLY_EQUATION(x);
			APPLY_EQUATION(y);
			APPLY_EQUATION(z);

			result = r;
		} break;

		case Variant::BASIS: {
			Basis i = initial_val;
			Basis d = delta_val;
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

			result = r;
		} break;

		case Variant::TRANSFORM2D: {
			Transform2D i = initial_val;
			Transform2D d = delta_val;
			Transform2D r;

			APPLY_EQUATION(elements[0][0]);
			APPLY_EQUATION(elements[0][1]);
			APPLY_EQUATION(elements[1][0]);
			APPLY_EQUATION(elements[1][1]);
			APPLY_EQUATION(elements[2][0]);
			APPLY_EQUATION(elements[2][1]);

			result = r;
		} break;
		case Variant::QUAT: {
			Quat i = initial_val;
			Quat d = delta_val;
			Quat r;

			APPLY_EQUATION(x);
			APPLY_EQUATION(y);
			APPLY_EQUATION(z);
			APPLY_EQUATION(w);

			result = r;
		} break;
		case Variant::AABB: {
			AABB i = initial_val;
			AABB d = delta_val;
			AABB r;

			APPLY_EQUATION(position.x);
			APPLY_EQUATION(position.y);
			APPLY_EQUATION(position.z);
			APPLY_EQUATION(size.x);
			APPLY_EQUATION(size.y);
			APPLY_EQUATION(size.z);

			result = r;
		} break;
		case Variant::TRANSFORM: {
			Transform i = initial_val;
			Transform d = delta_val;
			Transform r;

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

			result = r;
		} break;
		case Variant::COLOR: {
			Color i = initial_val;
			Color d = delta_val;
			Color r;

			APPLY_EQUATION(r);
			APPLY_EQUATION(g);
			APPLY_EQUATION(b);
			APPLY_EQUATION(a);

			result = r;
		} break;
		default: {
			result = initial_val;
		} break;
	};
#undef APPLY_EQUATION

	return result;
}

bool Tween::_apply_tween_value(InterpolateData &p_data, Variant &value) {

	Object *object = ObjectDB::get_instance(p_data.id);
	ERR_FAIL_COND_V(object == NULL, false);

	switch (p_data.type) {

		case INTER_PROPERTY:
		case FOLLOW_PROPERTY:
		case TARGETING_PROPERTY: {
			bool valid = false;
			object->set_indexed(p_data.key, value, &valid);
			return valid;
		}

		case INTER_METHOD:
		case FOLLOW_METHOD:
		case TARGETING_METHOD: {
			Variant::CallError error;
			if (value.get_type() != Variant::NIL) {
				Variant *arg[1] = { &value };
				object->call(p_data.key[0], (const Variant **)arg, 1, error);
			} else {
				object->call(p_data.key[0], NULL, 0, error);
			}

			if (error.error == Variant::CallError::CALL_OK)
				return true;
			return false;
		}

		case INTER_CALLBACK:
			break;
	};
	return true;
}

void Tween::_tween_process(float p_delta) {

	_process_pending_commands();

	if (speed_scale == 0)
		return;
	p_delta *= speed_scale;

	pending_update++;
	// if repeat and all interpolates was finished then reset all interpolates
	if (repeat) {
		bool all_finished = true;

		for (List<InterpolateData>::Element *E = interpolates.front(); E; E = E->next()) {

			InterpolateData &data = E->get();

			if (!data.finish) {
				all_finished = false;
				break;
			}
		}

		if (all_finished)
			reset_all();
	}

	for (List<InterpolateData>::Element *E = interpolates.front(); E; E = E->next()) {

		InterpolateData &data = E->get();
		if (!data.active || data.finish)
			continue;

		Object *object = ObjectDB::get_instance(data.id);
		if (object == NULL)
			continue;

		bool prev_delaying = data.elapsed <= data.delay;
		data.elapsed += p_delta;
		if (data.elapsed < data.delay)
			continue;
		else if (prev_delaying) {

			emit_signal("tween_started", object, NodePath(Vector<StringName>(), data.key, false));
			_apply_tween_value(data, data.initial_val);
		}

		if (data.elapsed > (data.delay + data.duration)) {

			data.elapsed = data.delay + data.duration;
			data.finish = true;
		}

		switch (data.type) {
			case INTER_PROPERTY:
			case INTER_METHOD: {
				Variant result = _run_equation(data);
				emit_signal("tween_step", object, NodePath(Vector<StringName>(), data.key, false), data.elapsed, result);
				_apply_tween_value(data, result);
				if (data.finish)
					_apply_tween_value(data, data.final_val);
			} break;

			case INTER_CALLBACK:
				if (data.finish) {
					if (data.call_deferred) {

						switch (data.args) {
							case 0:
								object->call_deferred(data.key[0]);
								break;
							case 1:
								object->call_deferred(data.key[0], data.arg[0]);
								break;
							case 2:
								object->call_deferred(data.key[0], data.arg[0], data.arg[1]);
								break;
							case 3:
								object->call_deferred(data.key[0], data.arg[0], data.arg[1], data.arg[2]);
								break;
							case 4:
								object->call_deferred(data.key[0], data.arg[0], data.arg[1], data.arg[2], data.arg[3]);
								break;
							case 5:
								object->call_deferred(data.key[0], data.arg[0], data.arg[1], data.arg[2], data.arg[3], data.arg[4]);
								break;
						}
					} else {
						Variant::CallError error;
						Variant *arg[5] = {
							&data.arg[0],
							&data.arg[1],
							&data.arg[2],
							&data.arg[3],
							&data.arg[4],
						};
						object->call(data.key[0], (const Variant **)arg, data.args, error);
					}
				}
				break;
			default: {}
		}

		if (data.finish) {
			emit_signal("tween_completed", object, NodePath(Vector<StringName>(), data.key, false));
			// not repeat mode, remove completed action
			if (!repeat)
				call_deferred("_remove", object, NodePath(Vector<StringName>(), data.key, false), true);
		}
	}
	pending_update--;
}

void Tween::set_tween_process_mode(TweenProcessMode p_mode) {

	if (tween_process_mode == p_mode)
		return;

	bool pr = processing;
	if (pr)
		_set_process(false);
	tween_process_mode = p_mode;
	if (pr)
		_set_process(true);
}

Tween::TweenProcessMode Tween::get_tween_process_mode() const {

	return tween_process_mode;
}

void Tween::_set_process(bool p_process, bool p_force) {

	if (processing == p_process && !p_force)
		return;

	switch (tween_process_mode) {

		case TWEEN_PROCESS_PHYSICS: set_physics_process_internal(p_process && active); break;
		case TWEEN_PROCESS_IDLE: set_process_internal(p_process && active); break;
	}

	processing = p_process;
}

bool Tween::is_active() const {

	return active;
}

void Tween::set_active(bool p_active) {

	if (active == p_active)
		return;

	active = p_active;
	_set_process(processing, true);
}

bool Tween::is_repeat() const {

	return repeat;
}

void Tween::set_repeat(bool p_repeat) {

	repeat = p_repeat;
}

void Tween::set_speed_scale(float p_speed) {

	speed_scale = p_speed;
}

float Tween::get_speed_scale() const {

	return speed_scale;
}

bool Tween::start() {

	set_active(true);
	_set_process(true);
	return true;
}

bool Tween::reset(Object *p_object, StringName p_key) {

	pending_update++;
	for (List<InterpolateData>::Element *E = interpolates.front(); E; E = E->next()) {

		InterpolateData &data = E->get();
		Object *object = ObjectDB::get_instance(data.id);
		if (object == NULL)
			continue;

		if (object == p_object && (data.concatenated_key == p_key || p_key == "")) {

			data.elapsed = 0;
			data.finish = false;
			if (data.delay == 0)
				_apply_tween_value(data, data.initial_val);
		}
	}
	pending_update--;
	return true;
}

bool Tween::reset_all() {

	pending_update++;
	for (List<InterpolateData>::Element *E = interpolates.front(); E; E = E->next()) {

		InterpolateData &data = E->get();
		data.elapsed = 0;
		data.finish = false;
		if (data.delay == 0)
			_apply_tween_value(data, data.initial_val);
	}
	pending_update--;
	return true;
}

bool Tween::stop(Object *p_object, StringName p_key) {

	pending_update++;
	for (List<InterpolateData>::Element *E = interpolates.front(); E; E = E->next()) {

		InterpolateData &data = E->get();
		Object *object = ObjectDB::get_instance(data.id);
		if (object == NULL)
			continue;
		if (object == p_object && (data.concatenated_key == p_key || p_key == ""))
			data.active = false;
	}
	pending_update--;
	return true;
}

bool Tween::stop_all() {

	set_active(false);
	_set_process(false);

	pending_update++;
	for (List<InterpolateData>::Element *E = interpolates.front(); E; E = E->next()) {

		InterpolateData &data = E->get();
		data.active = false;
	}
	pending_update--;
	return true;
}

bool Tween::resume(Object *p_object, StringName p_key) {

	set_active(true);
	_set_process(true);

	pending_update++;
	for (List<InterpolateData>::Element *E = interpolates.front(); E; E = E->next()) {

		InterpolateData &data = E->get();
		Object *object = ObjectDB::get_instance(data.id);
		if (object == NULL)
			continue;
		if (object == p_object && (data.concatenated_key == p_key || p_key == ""))
			data.active = true;
	}
	pending_update--;
	return true;
}

bool Tween::resume_all() {

	set_active(true);
	_set_process(true);

	pending_update++;
	for (List<InterpolateData>::Element *E = interpolates.front(); E; E = E->next()) {

		InterpolateData &data = E->get();
		data.active = true;
	}
	pending_update--;
	return true;
}

bool Tween::remove(Object *p_object, StringName p_key) {
	_remove(p_object, p_key, false);
	return true;
}

void Tween::_remove(Object *p_object, StringName p_key, bool first_only) {

	if (pending_update != 0) {
		call_deferred("_remove", p_object, p_key, first_only);
		return;
	}
	List<List<InterpolateData>::Element *> for_removal;
	for (List<InterpolateData>::Element *E = interpolates.front(); E; E = E->next()) {

		InterpolateData &data = E->get();
		Object *object = ObjectDB::get_instance(data.id);
		if (object == NULL)
			continue;
		if (object == p_object && (data.concatenated_key == p_key || p_key == "")) {
			for_removal.push_back(E);
			if (first_only) {
				break;
			}
		}
	}
	for (List<List<InterpolateData>::Element *>::Element *E = for_removal.front(); E; E = E->next()) {
		interpolates.erase(E->get());
	}
}

bool Tween::remove_all() {

	if (pending_update != 0) {
		call_deferred("remove_all");
		return true;
	}
	set_active(false);
	_set_process(false);
	interpolates.clear();
	return true;
}

bool Tween::seek(real_t p_time) {

	pending_update++;
	for (List<InterpolateData>::Element *E = interpolates.front(); E; E = E->next()) {

		InterpolateData &data = E->get();

		data.elapsed = p_time;
		if (data.elapsed < data.delay) {

			data.finish = false;
			continue;
		} else if (data.elapsed >= (data.delay + data.duration)) {

			data.finish = true;
			data.elapsed = (data.delay + data.duration);
		} else {
			data.finish = false;
		}

		switch (data.type) {
			case INTER_PROPERTY:
			case INTER_METHOD:
				break;
			case INTER_CALLBACK:
				continue;
		}

		Variant result = _run_equation(data);

		_apply_tween_value(data, result);
	}
	pending_update--;
	return true;
}

real_t Tween::tell() const {

	pending_update++;
	real_t pos = 0;
	for (const List<InterpolateData>::Element *E = interpolates.front(); E; E = E->next()) {

		const InterpolateData &data = E->get();
		if (data.elapsed > pos)
			pos = data.elapsed;
	}
	pending_update--;
	return pos;
}

real_t Tween::get_runtime() const {

	pending_update++;
	real_t runtime = 0;
	for (const List<InterpolateData>::Element *E = interpolates.front(); E; E = E->next()) {

		const InterpolateData &data = E->get();
		real_t t = data.delay + data.duration;
		if (t > runtime)
			runtime = t;
	}
	pending_update--;
	return runtime;
}

bool Tween::_calc_delta_val(const Variant &p_initial_val, const Variant &p_final_val, Variant &p_delta_val) {

	const Variant &initial_val = p_initial_val;
	const Variant &final_val = p_final_val;
	Variant &delta_val = p_delta_val;

	switch (initial_val.get_type()) {

		case Variant::BOOL:
			//delta_val = p_final_val;
			delta_val = (int)p_final_val - (int)p_initial_val;
			break;

		case Variant::INT:
			delta_val = (int)final_val - (int)initial_val;
			break;

		case Variant::REAL:
			delta_val = (real_t)final_val - (real_t)initial_val;
			break;

		case Variant::VECTOR2:
			delta_val = final_val.operator Vector2() - initial_val.operator Vector2();
			break;

		case Variant::VECTOR3:
			delta_val = final_val.operator Vector3() - initial_val.operator Vector3();
			break;

		case Variant::BASIS: {
			Basis i = initial_val;
			Basis f = final_val;
			delta_val = Basis(f.elements[0][0] - i.elements[0][0],
					f.elements[0][1] - i.elements[0][1],
					f.elements[0][2] - i.elements[0][2],
					f.elements[1][0] - i.elements[1][0],
					f.elements[1][1] - i.elements[1][1],
					f.elements[1][2] - i.elements[1][2],
					f.elements[2][0] - i.elements[2][0],
					f.elements[2][1] - i.elements[2][1],
					f.elements[2][2] - i.elements[2][2]);
		} break;

		case Variant::TRANSFORM2D: {
			Transform2D i = initial_val;
			Transform2D f = final_val;
			Transform2D d = Transform2D();
			d[0][0] = f.elements[0][0] - i.elements[0][0];
			d[0][1] = f.elements[0][1] - i.elements[0][1];
			d[1][0] = f.elements[1][0] - i.elements[1][0];
			d[1][1] = f.elements[1][1] - i.elements[1][1];
			d[2][0] = f.elements[2][0] - i.elements[2][0];
			d[2][1] = f.elements[2][1] - i.elements[2][1];
			delta_val = d;
		} break;
		case Variant::QUAT:
			delta_val = final_val.operator Quat() - initial_val.operator Quat();
			break;
		case Variant::AABB: {
			AABB i = initial_val;
			AABB f = final_val;
			delta_val = AABB(f.position - i.position, f.size - i.size);
		} break;
		case Variant::TRANSFORM: {
			Transform i = initial_val;
			Transform f = final_val;
			Transform d;
			d.set(f.basis.elements[0][0] - i.basis.elements[0][0],
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

			delta_val = d;
		} break;
		case Variant::COLOR: {
			Color i = initial_val;
			Color f = final_val;
			delta_val = Color(f.r - i.r, f.g - i.g, f.b - i.b, f.a - i.a);
		} break;

		default:
			ERR_PRINT("Invalid param type, except(int/real/vector2/vector/matrix/matrix32/quat/aabb/transform/color)");
			return false;
	};
	return true;
}

bool Tween::interpolate_property(Object *p_object, NodePath p_property, Variant p_initial_val, Variant p_final_val, real_t p_duration, TransitionType p_trans_type, EaseType p_ease_type, real_t p_delay) {
	if (pending_update != 0) {
		_add_pending_command("interpolate_property", p_object, p_property, p_initial_val, p_final_val, p_duration, p_trans_type, p_ease_type, p_delay);
		return true;
	}
	p_property = p_property.get_as_property_path();

	if (p_initial_val.get_type() == Variant::NIL) p_initial_val = p_object->get_indexed(p_property.get_subnames());

	// convert INT to REAL is better for interpolaters
	if (p_initial_val.get_type() == Variant::INT) p_initial_val = p_initial_val.operator real_t();
	if (p_final_val.get_type() == Variant::INT) p_final_val = p_final_val.operator real_t();

	ERR_FAIL_COND_V(p_object == NULL, false);
	ERR_FAIL_COND_V(!ObjectDB::instance_validate(p_object), false);
	ERR_FAIL_COND_V(p_initial_val.get_type() != p_final_val.get_type(), false);
	ERR_FAIL_COND_V(p_duration <= 0, false);
	ERR_FAIL_COND_V(p_trans_type < 0 || p_trans_type >= TRANS_COUNT, false);
	ERR_FAIL_COND_V(p_ease_type < 0 || p_ease_type >= EASE_COUNT, false);
	ERR_FAIL_COND_V(p_delay < 0, false);

	bool prop_valid = false;
	p_object->get_indexed(p_property.get_subnames(), &prop_valid);
	ERR_FAIL_COND_V(!prop_valid, false);

	InterpolateData data;
	data.active = true;
	data.type = INTER_PROPERTY;
	data.finish = false;
	data.elapsed = 0;

	data.id = p_object->get_instance_id();
	data.key = p_property.get_subnames();
	data.concatenated_key = p_property.get_concatenated_subnames();
	data.initial_val = p_initial_val;
	data.final_val = p_final_val;
	data.duration = p_duration;
	data.trans_type = p_trans_type;
	data.ease_type = p_ease_type;
	data.delay = p_delay;

	if (!_calc_delta_val(data.initial_val, data.final_val, data.delta_val))
		return false;

	interpolates.push_back(data);
	return true;
}

bool Tween::interpolate_method(Object *p_object, StringName p_method, Variant p_initial_val, Variant p_final_val, real_t p_duration, TransitionType p_trans_type, EaseType p_ease_type, real_t p_delay) {
	if (pending_update != 0) {
		_add_pending_command("interpolate_method", p_object, p_method, p_initial_val, p_final_val, p_duration, p_trans_type, p_ease_type, p_delay);
		return true;
	}
	// convert INT to REAL is better for interpolaters
	if (p_initial_val.get_type() == Variant::INT) p_initial_val = p_initial_val.operator real_t();
	if (p_final_val.get_type() == Variant::INT) p_final_val = p_final_val.operator real_t();

	ERR_FAIL_COND_V(p_object == NULL, false);
	ERR_FAIL_COND_V(!ObjectDB::instance_validate(p_object), false);
	ERR_FAIL_COND_V(p_initial_val.get_type() != p_final_val.get_type(), false);
	ERR_FAIL_COND_V(p_duration <= 0, false);
	ERR_FAIL_COND_V(p_trans_type < 0 || p_trans_type >= TRANS_COUNT, false);
	ERR_FAIL_COND_V(p_ease_type < 0 || p_ease_type >= EASE_COUNT, false);
	ERR_FAIL_COND_V(p_delay < 0, false);

	ERR_EXPLAIN("Object has no method named: %s" + p_method);
	ERR_FAIL_COND_V(!p_object->has_method(p_method), false);

	InterpolateData data;
	data.active = true;
	data.type = INTER_METHOD;
	data.finish = false;
	data.elapsed = 0;

	data.id = p_object->get_instance_id();
	data.key.push_back(p_method);
	data.concatenated_key = p_method;
	data.initial_val = p_initial_val;
	data.final_val = p_final_val;
	data.duration = p_duration;
	data.trans_type = p_trans_type;
	data.ease_type = p_ease_type;
	data.delay = p_delay;

	if (!_calc_delta_val(data.initial_val, data.final_val, data.delta_val))
		return false;

	interpolates.push_back(data);
	return true;
}

bool Tween::interpolate_callback(Object *p_object, real_t p_duration, String p_callback, VARIANT_ARG_DECLARE) {

	if (pending_update != 0) {
		_add_pending_command("interpolate_callback", p_object, p_duration, p_callback, p_arg1, p_arg2, p_arg3, p_arg4, p_arg5);
		return true;
	}

	ERR_FAIL_COND_V(p_object == NULL, false);
	ERR_FAIL_COND_V(!ObjectDB::instance_validate(p_object), false);
	ERR_FAIL_COND_V(p_duration < 0, false);

	ERR_EXPLAIN("Object has no callback named: %s" + p_callback);
	ERR_FAIL_COND_V(!p_object->has_method(p_callback), false);

	InterpolateData data;
	data.active = true;
	data.type = INTER_CALLBACK;
	data.finish = false;
	data.call_deferred = false;
	data.elapsed = 0;

	data.id = p_object->get_instance_id();
	data.key.push_back(p_callback);
	data.concatenated_key = p_callback;
	data.duration = p_duration;
	data.delay = 0;

	int args = 0;
	if (p_arg5.get_type() != Variant::NIL)
		args = 5;
	else if (p_arg4.get_type() != Variant::NIL)
		args = 4;
	else if (p_arg3.get_type() != Variant::NIL)
		args = 3;
	else if (p_arg2.get_type() != Variant::NIL)
		args = 2;
	else if (p_arg1.get_type() != Variant::NIL)
		args = 1;
	else
		args = 0;

	data.args = args;
	data.arg[0] = p_arg1;
	data.arg[1] = p_arg2;
	data.arg[2] = p_arg3;
	data.arg[3] = p_arg4;
	data.arg[4] = p_arg5;

	pending_update++;
	interpolates.push_back(data);
	pending_update--;
	return true;
}

bool Tween::interpolate_deferred_callback(Object *p_object, real_t p_duration, String p_callback, VARIANT_ARG_DECLARE) {

	if (pending_update != 0) {
		_add_pending_command("interpolate_deferred_callback", p_object, p_duration, p_callback, p_arg1, p_arg2, p_arg3, p_arg4, p_arg5);
		return true;
	}
	ERR_FAIL_COND_V(p_object == NULL, false);
	ERR_FAIL_COND_V(!ObjectDB::instance_validate(p_object), false);
	ERR_FAIL_COND_V(p_duration < 0, false);

	ERR_EXPLAIN("Object has no callback named: %s" + p_callback);
	ERR_FAIL_COND_V(!p_object->has_method(p_callback), false);

	InterpolateData data;
	data.active = true;
	data.type = INTER_CALLBACK;
	data.finish = false;
	data.call_deferred = true;
	data.elapsed = 0;

	data.id = p_object->get_instance_id();
	data.key.push_back(p_callback);
	data.concatenated_key = p_callback;
	data.duration = p_duration;
	data.delay = 0;

	int args = 0;
	if (p_arg5.get_type() != Variant::NIL)
		args = 5;
	else if (p_arg4.get_type() != Variant::NIL)
		args = 4;
	else if (p_arg3.get_type() != Variant::NIL)
		args = 3;
	else if (p_arg2.get_type() != Variant::NIL)
		args = 2;
	else if (p_arg1.get_type() != Variant::NIL)
		args = 1;
	else
		args = 0;

	data.args = args;
	data.arg[0] = p_arg1;
	data.arg[1] = p_arg2;
	data.arg[2] = p_arg3;
	data.arg[3] = p_arg4;
	data.arg[4] = p_arg5;

	pending_update++;
	interpolates.push_back(data);
	pending_update--;
	return true;
}

bool Tween::follow_property(Object *p_object, NodePath p_property, Variant p_initial_val, Object *p_target, NodePath p_target_property, real_t p_duration, TransitionType p_trans_type, EaseType p_ease_type, real_t p_delay) {
	if (pending_update != 0) {
		_add_pending_command("follow_property", p_object, p_property, p_initial_val, p_target, p_target_property, p_duration, p_trans_type, p_ease_type, p_delay);
		return true;
	}
	p_property = p_property.get_as_property_path();
	p_target_property = p_target_property.get_as_property_path();

	if (p_initial_val.get_type() == Variant::NIL) p_initial_val = p_object->get_indexed(p_property.get_subnames());

	// convert INT to REAL is better for interpolaters
	if (p_initial_val.get_type() == Variant::INT) p_initial_val = p_initial_val.operator real_t();

	ERR_FAIL_COND_V(p_object == NULL, false);
	ERR_FAIL_COND_V(!ObjectDB::instance_validate(p_object), false);
	ERR_FAIL_COND_V(p_target == NULL, false);
	ERR_FAIL_COND_V(!ObjectDB::instance_validate(p_target), false);
	ERR_FAIL_COND_V(p_duration <= 0, false);
	ERR_FAIL_COND_V(p_trans_type < 0 || p_trans_type >= TRANS_COUNT, false);
	ERR_FAIL_COND_V(p_ease_type < 0 || p_ease_type >= EASE_COUNT, false);
	ERR_FAIL_COND_V(p_delay < 0, false);

	bool prop_valid = false;
	p_object->get_indexed(p_property.get_subnames(), &prop_valid);
	ERR_FAIL_COND_V(!prop_valid, false);

	bool target_prop_valid = false;
	Variant target_val = p_target->get_indexed(p_target_property.get_subnames(), &target_prop_valid);
	ERR_FAIL_COND_V(!target_prop_valid, false);

	// convert INT to REAL is better for interpolaters
	if (target_val.get_type() == Variant::INT) target_val = target_val.operator real_t();
	ERR_FAIL_COND_V(target_val.get_type() != p_initial_val.get_type(), false);

	InterpolateData data;
	data.active = true;
	data.type = FOLLOW_PROPERTY;
	data.finish = false;
	data.elapsed = 0;

	data.id = p_object->get_instance_id();
	data.key = p_property.get_subnames();
	data.concatenated_key = p_property.get_concatenated_subnames();
	data.initial_val = p_initial_val;
	data.target_id = p_target->get_instance_id();
	data.target_key = p_target_property.get_subnames();
	data.duration = p_duration;
	data.trans_type = p_trans_type;
	data.ease_type = p_ease_type;
	data.delay = p_delay;

	interpolates.push_back(data);
	return true;
}

bool Tween::follow_method(Object *p_object, StringName p_method, Variant p_initial_val, Object *p_target, StringName p_target_method, real_t p_duration, TransitionType p_trans_type, EaseType p_ease_type, real_t p_delay) {
	if (pending_update != 0) {
		_add_pending_command("follow_method", p_object, p_method, p_initial_val, p_target, p_target_method, p_duration, p_trans_type, p_ease_type, p_delay);
		return true;
	}
	// convert INT to REAL is better for interpolaters
	if (p_initial_val.get_type() == Variant::INT) p_initial_val = p_initial_val.operator real_t();

	ERR_FAIL_COND_V(p_object == NULL, false);
	ERR_FAIL_COND_V(!ObjectDB::instance_validate(p_object), false);
	ERR_FAIL_COND_V(p_target == NULL, false);
	ERR_FAIL_COND_V(!ObjectDB::instance_validate(p_target), false);
	ERR_FAIL_COND_V(p_duration <= 0, false);
	ERR_FAIL_COND_V(p_trans_type < 0 || p_trans_type >= TRANS_COUNT, false);
	ERR_FAIL_COND_V(p_ease_type < 0 || p_ease_type >= EASE_COUNT, false);
	ERR_FAIL_COND_V(p_delay < 0, false);

	ERR_EXPLAIN("Object has no method named: %s" + p_method);
	ERR_FAIL_COND_V(!p_object->has_method(p_method), false);
	ERR_EXPLAIN("Target has no method named: %s" + p_target_method);
	ERR_FAIL_COND_V(!p_target->has_method(p_target_method), false);

	Variant::CallError error;
	Variant target_val = p_target->call(p_target_method, NULL, 0, error);
	ERR_FAIL_COND_V(error.error != Variant::CallError::CALL_OK, false);

	// convert INT to REAL is better for interpolaters
	if (target_val.get_type() == Variant::INT) target_val = target_val.operator real_t();
	ERR_FAIL_COND_V(target_val.get_type() != p_initial_val.get_type(), false);

	InterpolateData data;
	data.active = true;
	data.type = FOLLOW_METHOD;
	data.finish = false;
	data.elapsed = 0;

	data.id = p_object->get_instance_id();
	data.key.push_back(p_method);
	data.concatenated_key = p_method;
	data.initial_val = p_initial_val;
	data.target_id = p_target->get_instance_id();
	data.target_key.push_back(p_target_method);
	data.duration = p_duration;
	data.trans_type = p_trans_type;
	data.ease_type = p_ease_type;
	data.delay = p_delay;

	interpolates.push_back(data);
	return true;
}

bool Tween::targeting_property(Object *p_object, NodePath p_property, Object *p_initial, NodePath p_initial_property, Variant p_final_val, real_t p_duration, TransitionType p_trans_type, EaseType p_ease_type, real_t p_delay) {

	if (pending_update != 0) {
		_add_pending_command("targeting_property", p_object, p_property, p_initial, p_initial_property, p_final_val, p_duration, p_trans_type, p_ease_type, p_delay);
		return true;
	}
	p_property = p_property.get_as_property_path();
	p_initial_property = p_initial_property.get_as_property_path();

	// convert INT to REAL is better for interpolaters
	if (p_final_val.get_type() == Variant::INT) p_final_val = p_final_val.operator real_t();

	ERR_FAIL_COND_V(p_object == NULL, false);
	ERR_FAIL_COND_V(!ObjectDB::instance_validate(p_object), false);
	ERR_FAIL_COND_V(p_initial == NULL, false);
	ERR_FAIL_COND_V(!ObjectDB::instance_validate(p_initial), false);
	ERR_FAIL_COND_V(p_duration <= 0, false);
	ERR_FAIL_COND_V(p_trans_type < 0 || p_trans_type >= TRANS_COUNT, false);
	ERR_FAIL_COND_V(p_ease_type < 0 || p_ease_type >= EASE_COUNT, false);
	ERR_FAIL_COND_V(p_delay < 0, false);

	bool prop_valid = false;
	p_object->get_indexed(p_property.get_subnames(), &prop_valid);
	ERR_FAIL_COND_V(!prop_valid, false);

	bool initial_prop_valid = false;
	Variant initial_val = p_initial->get_indexed(p_initial_property.get_subnames(), &initial_prop_valid);
	ERR_FAIL_COND_V(!initial_prop_valid, false);

	// convert INT to REAL is better for interpolaters
	if (initial_val.get_type() == Variant::INT) initial_val = initial_val.operator real_t();
	ERR_FAIL_COND_V(initial_val.get_type() != p_final_val.get_type(), false);

	InterpolateData data;
	data.active = true;
	data.type = TARGETING_PROPERTY;
	data.finish = false;
	data.elapsed = 0;

	data.id = p_object->get_instance_id();
	data.key = p_property.get_subnames();
	data.concatenated_key = p_property.get_concatenated_subnames();
	data.target_id = p_initial->get_instance_id();
	data.target_key = p_initial_property.get_subnames();
	data.initial_val = initial_val;
	data.final_val = p_final_val;
	data.duration = p_duration;
	data.trans_type = p_trans_type;
	data.ease_type = p_ease_type;
	data.delay = p_delay;

	if (!_calc_delta_val(data.initial_val, data.final_val, data.delta_val))
		return false;

	interpolates.push_back(data);
	return true;
}

bool Tween::targeting_method(Object *p_object, StringName p_method, Object *p_initial, StringName p_initial_method, Variant p_final_val, real_t p_duration, TransitionType p_trans_type, EaseType p_ease_type, real_t p_delay) {
	if (pending_update != 0) {
		_add_pending_command("targeting_method", p_object, p_method, p_initial, p_initial_method, p_final_val, p_duration, p_trans_type, p_ease_type, p_delay);
		return true;
	}
	// convert INT to REAL is better for interpolaters
	if (p_final_val.get_type() == Variant::INT) p_final_val = p_final_val.operator real_t();

	ERR_FAIL_COND_V(p_object == NULL, false);
	ERR_FAIL_COND_V(!ObjectDB::instance_validate(p_object), false);
	ERR_FAIL_COND_V(p_initial == NULL, false);
	ERR_FAIL_COND_V(!ObjectDB::instance_validate(p_initial), false);
	ERR_FAIL_COND_V(p_duration <= 0, false);
	ERR_FAIL_COND_V(p_trans_type < 0 || p_trans_type >= TRANS_COUNT, false);
	ERR_FAIL_COND_V(p_ease_type < 0 || p_ease_type >= EASE_COUNT, false);
	ERR_FAIL_COND_V(p_delay < 0, false);

	ERR_EXPLAIN("Object has no method named: %s" + p_method);
	ERR_FAIL_COND_V(!p_object->has_method(p_method), false);
	ERR_EXPLAIN("Initial Object has no method named: %s" + p_initial_method);
	ERR_FAIL_COND_V(!p_initial->has_method(p_initial_method), false);

	Variant::CallError error;
	Variant initial_val = p_initial->call(p_initial_method, NULL, 0, error);
	ERR_FAIL_COND_V(error.error != Variant::CallError::CALL_OK, false);

	// convert INT to REAL is better for interpolaters
	if (initial_val.get_type() == Variant::INT) initial_val = initial_val.operator real_t();
	ERR_FAIL_COND_V(initial_val.get_type() != p_final_val.get_type(), false);

	InterpolateData data;
	data.active = true;
	data.type = TARGETING_METHOD;
	data.finish = false;
	data.elapsed = 0;

	data.id = p_object->get_instance_id();
	data.key.push_back(p_method);
	data.concatenated_key = p_method;
	data.target_id = p_initial->get_instance_id();
	data.target_key.push_back(p_initial_method);
	data.initial_val = initial_val;
	data.final_val = p_final_val;
	data.duration = p_duration;
	data.trans_type = p_trans_type;
	data.ease_type = p_ease_type;
	data.delay = p_delay;

	if (!_calc_delta_val(data.initial_val, data.final_val, data.delta_val))
		return false;

	interpolates.push_back(data);
	return true;
}

Tween::Tween() {

	//String autoplay;
	tween_process_mode = TWEEN_PROCESS_IDLE;
	processing = false;
	active = false;
	repeat = false;
	speed_scale = 1;
	pending_update = 0;
}

Tween::~Tween() {
}
