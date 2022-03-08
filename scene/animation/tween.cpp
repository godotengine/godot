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

#include "core/method_bind_ext.gen.inc"

void Tween::_add_pending_command(StringName p_key, const Variant &p_arg1, const Variant &p_arg2, const Variant &p_arg3, const Variant &p_arg4, const Variant &p_arg5, const Variant &p_arg6, const Variant &p_arg7, const Variant &p_arg8, const Variant &p_arg9, const Variant &p_arg10) {
	// Add a new pending command and reference it
	pending_commands.push_back(PendingCommand());
	PendingCommand &cmd = pending_commands.back()->get();

	// Update the command with the target key
	cmd.key = p_key;

	// Determine command argument count
	int &count = cmd.args;
	if (p_arg10.get_type() != Variant::NIL) {
		count = 10;
	} else if (p_arg9.get_type() != Variant::NIL) {
		count = 9;
	} else if (p_arg8.get_type() != Variant::NIL) {
		count = 8;
	} else if (p_arg7.get_type() != Variant::NIL) {
		count = 7;
	} else if (p_arg6.get_type() != Variant::NIL) {
		count = 6;
	} else if (p_arg5.get_type() != Variant::NIL) {
		count = 5;
	} else if (p_arg4.get_type() != Variant::NIL) {
		count = 4;
	} else if (p_arg3.get_type() != Variant::NIL) {
		count = 3;
	} else if (p_arg2.get_type() != Variant::NIL) {
		count = 2;
	} else if (p_arg1.get_type() != Variant::NIL) {
		count = 1;
	} else {
		count = 0;
	}

	// Add the specified arguments to the command
	if (count > 0) {
		cmd.arg[0] = p_arg1;
	}
	if (count > 1) {
		cmd.arg[1] = p_arg2;
	}
	if (count > 2) {
		cmd.arg[2] = p_arg3;
	}
	if (count > 3) {
		cmd.arg[3] = p_arg4;
	}
	if (count > 4) {
		cmd.arg[4] = p_arg5;
	}
	if (count > 5) {
		cmd.arg[5] = p_arg6;
	}
	if (count > 6) {
		cmd.arg[6] = p_arg7;
	}
	if (count > 7) {
		cmd.arg[7] = p_arg8;
	}
	if (count > 8) {
		cmd.arg[8] = p_arg9;
	}
	if (count > 9) {
		cmd.arg[9] = p_arg10;
	}
}

void Tween::_process_pending_commands() {
	// For each pending command...
	for (List<PendingCommand>::Element *E = pending_commands.front(); E; E = E->next()) {
		// Get the command
		PendingCommand &cmd = E->get();
		Variant::CallError err;

		// Grab all of the arguments for the command
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

		// Execute the command (and retrieve any errors)
		this->call(cmd.key, (const Variant **)arg, cmd.args, err);
	}

	// Clear the pending commands
	pending_commands.clear();
}

bool Tween::_set(const StringName &p_name, const Variant &p_value) {
	// Set the correct attribute based on the given name
	String name = p_name;
	if (name == "playback/speed" || name == "speed") { // Backwards compatibility
		set_speed_scale(p_value);
		return true;

	} else if (name == "playback/active") {
		set_active(p_value);
		return true;

	} else if (name == "playback/repeat") {
		set_repeat(p_value);
		return true;
	}
	return false;
}

bool Tween::_get(const StringName &p_name, Variant &r_ret) const {
	// Get the correct attribute based on the given name
	String name = p_name;
	if (name == "playback/speed") { // Backwards compatibility
		r_ret = speed_scale;
		return true;

	} else if (name == "playback/active") {
		r_ret = is_active();
		return true;

	} else if (name == "playback/repeat") {
		r_ret = is_repeat();
		return true;
	}
	return false;
}

void Tween::_get_property_list(List<PropertyInfo> *p_list) const {
	// Add the property info for the Tween object
	p_list->push_back(PropertyInfo(Variant::BOOL, "playback/active", PROPERTY_HINT_NONE, ""));
	p_list->push_back(PropertyInfo(Variant::BOOL, "playback/repeat", PROPERTY_HINT_NONE, ""));
	p_list->push_back(PropertyInfo(Variant::REAL, "playback/speed", PROPERTY_HINT_RANGE, "-64,64,0.01"));
}

void Tween::_notification(int p_what) {
	// What notification did we receive?
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			// Are we not already active?
			if (!is_active()) {
				// Make sure that a previous process state was not saved
				// Only process if "processing" is set
				set_physics_process_internal(false);
				set_process_internal(false);
			}
		} break;

		case NOTIFICATION_READY: {
			// Do nothing
		} break;

		case NOTIFICATION_INTERNAL_PROCESS: {
			// Are we processing during physics time?
			if (tween_process_mode == TWEEN_PROCESS_PHYSICS) {
				// Do nothing since we aren't aligned with physics when we should be
				break;
			}

			// Should we update?
			if (is_active()) {
				// Update the tweens
				_tween_process(get_process_delta_time());
			}
		} break;

		case NOTIFICATION_INTERNAL_PHYSICS_PROCESS: {
			// Are we processing during 'regular' time?
			if (tween_process_mode == TWEEN_PROCESS_IDLE) {
				// Do nothing since we would only process during idle time
				break;
			}

			// Should we update?
			if (is_active()) {
				// Update the tweens
				_tween_process(get_physics_process_delta_time());
			}
		} break;

		case NOTIFICATION_EXIT_TREE: {
			// We've left the tree. Stop all tweens
			stop_all();
		} break;
	}
}

void Tween::_bind_methods() {
	// Bind getters and setters
	ClassDB::bind_method(D_METHOD("is_active"), &Tween::is_active);
	ClassDB::bind_method(D_METHOD("set_active", "active"), &Tween::set_active);

	ClassDB::bind_method(D_METHOD("is_repeat"), &Tween::is_repeat);
	ClassDB::bind_method(D_METHOD("set_repeat", "repeat"), &Tween::set_repeat);

	ClassDB::bind_method(D_METHOD("set_speed_scale", "speed"), &Tween::set_speed_scale);
	ClassDB::bind_method(D_METHOD("get_speed_scale"), &Tween::get_speed_scale);

	ClassDB::bind_method(D_METHOD("set_tween_process_mode", "mode"), &Tween::set_tween_process_mode);
	ClassDB::bind_method(D_METHOD("get_tween_process_mode"), &Tween::get_tween_process_mode);

	// Bind the various Tween control methods
	ClassDB::bind_method(D_METHOD("start"), &Tween::start);
	ClassDB::bind_method(D_METHOD("reset", "object", "key"), &Tween::reset, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("reset_all"), &Tween::reset_all);
	ClassDB::bind_method(D_METHOD("stop", "object", "key"), &Tween::stop, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("stop_all"), &Tween::stop_all);
	ClassDB::bind_method(D_METHOD("resume", "object", "key"), &Tween::resume, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("resume_all"), &Tween::resume_all);
	ClassDB::bind_method(D_METHOD("remove", "object", "key"), &Tween::remove, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("_remove_by_uid", "uid"), &Tween::_remove_by_uid);
	ClassDB::bind_method(D_METHOD("remove_all"), &Tween::remove_all);
	ClassDB::bind_method(D_METHOD("seek", "time"), &Tween::seek);
	ClassDB::bind_method(D_METHOD("tell"), &Tween::tell);
	ClassDB::bind_method(D_METHOD("get_runtime"), &Tween::get_runtime);

	// Bind interpolation and follow methods
	ClassDB::bind_method(D_METHOD("interpolate_property", "object", "property", "initial_val", "final_val", "duration", "trans_type", "ease_type", "delay"), &Tween::interpolate_property, DEFVAL(TRANS_LINEAR), DEFVAL(EASE_IN_OUT), DEFVAL(0));
	ClassDB::bind_method(D_METHOD("interpolate_method", "object", "method", "initial_val", "final_val", "duration", "trans_type", "ease_type", "delay"), &Tween::interpolate_method, DEFVAL(TRANS_LINEAR), DEFVAL(EASE_IN_OUT), DEFVAL(0));
	ClassDB::bind_method(D_METHOD("interpolate_callback", "object", "duration", "callback", "arg1", "arg2", "arg3", "arg4", "arg5", "arg6", "arg7", "arg8"), &Tween::interpolate_callback, DEFVAL(Variant()), DEFVAL(Variant()), DEFVAL(Variant()), DEFVAL(Variant()), DEFVAL(Variant()), DEFVAL(Variant()), DEFVAL(Variant()), DEFVAL(Variant()));
	ClassDB::bind_method(D_METHOD("interpolate_deferred_callback", "object", "duration", "callback", "arg1", "arg2", "arg3", "arg4", "arg5", "arg6", "arg7", "arg8"), &Tween::interpolate_deferred_callback, DEFVAL(Variant()), DEFVAL(Variant()), DEFVAL(Variant()), DEFVAL(Variant()), DEFVAL(Variant()), DEFVAL(Variant()), DEFVAL(Variant()), DEFVAL(Variant()));
	ClassDB::bind_method(D_METHOD("follow_property", "object", "property", "initial_val", "target", "target_property", "duration", "trans_type", "ease_type", "delay"), &Tween::follow_property, DEFVAL(TRANS_LINEAR), DEFVAL(EASE_IN_OUT), DEFVAL(0));
	ClassDB::bind_method(D_METHOD("follow_method", "object", "method", "initial_val", "target", "target_method", "duration", "trans_type", "ease_type", "delay"), &Tween::follow_method, DEFVAL(TRANS_LINEAR), DEFVAL(EASE_IN_OUT), DEFVAL(0));
	ClassDB::bind_method(D_METHOD("targeting_property", "object", "property", "initial", "initial_val", "final_val", "duration", "trans_type", "ease_type", "delay"), &Tween::targeting_property, DEFVAL(TRANS_LINEAR), DEFVAL(EASE_IN_OUT), DEFVAL(0));
	ClassDB::bind_method(D_METHOD("targeting_method", "object", "method", "initial", "initial_method", "final_val", "duration", "trans_type", "ease_type", "delay"), &Tween::targeting_method, DEFVAL(TRANS_LINEAR), DEFVAL(EASE_IN_OUT), DEFVAL(0));

	// Add the Tween signals
	ADD_SIGNAL(MethodInfo("tween_started", PropertyInfo(Variant::OBJECT, "object"), PropertyInfo(Variant::NODE_PATH, "key")));
	ADD_SIGNAL(MethodInfo("tween_step", PropertyInfo(Variant::OBJECT, "object"), PropertyInfo(Variant::NODE_PATH, "key"), PropertyInfo(Variant::REAL, "elapsed"), PropertyInfo(Variant::OBJECT, "value")));
	ADD_SIGNAL(MethodInfo("tween_completed", PropertyInfo(Variant::OBJECT, "object"), PropertyInfo(Variant::NODE_PATH, "key")));
	ADD_SIGNAL(MethodInfo("tween_all_completed"));

	// Add the properties and tie them to the getters and setters
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "repeat"), "set_repeat", "is_repeat");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "playback_process_mode", PROPERTY_HINT_ENUM, "Physics,Idle"), "set_tween_process_mode", "get_tween_process_mode");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "playback_speed", PROPERTY_HINT_RANGE, "-64,64,0.01"), "set_speed_scale", "get_speed_scale");

	// Bind Idle vs Physics process
	BIND_ENUM_CONSTANT(TWEEN_PROCESS_PHYSICS);
	BIND_ENUM_CONSTANT(TWEEN_PROCESS_IDLE);

	// Bind the Transition type constants
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

	// Bind the easing constants
	BIND_ENUM_CONSTANT(EASE_IN);
	BIND_ENUM_CONSTANT(EASE_OUT);
	BIND_ENUM_CONSTANT(EASE_IN_OUT);
	BIND_ENUM_CONSTANT(EASE_OUT_IN);
}

Variant Tween::_get_initial_val(const InterpolateData &p_data) const {
	// What type of data are we interpolating?
	switch (p_data.type) {
		case INTER_PROPERTY:
		case INTER_METHOD:
		case FOLLOW_PROPERTY:
		case FOLLOW_METHOD:
			// Simply use the given initial value
			return p_data.initial_val;

		case TARGETING_PROPERTY:
		case TARGETING_METHOD: {
			// Get the object that is being targeted
			Object *object = ObjectDB::get_instance(p_data.target_id);
			ERR_FAIL_COND_V(object == nullptr, p_data.initial_val);

			// Are we targeting a property or a method?
			Variant initial_val;
			if (p_data.type == TARGETING_PROPERTY) {
				// Get the property from the target object
				bool valid = false;
				initial_val = object->get_indexed(p_data.target_key, &valid);
				ERR_FAIL_COND_V(!valid, p_data.initial_val);
			} else {
				// Call the method and get the initial value from it
				Variant::CallError error;
				initial_val = object->call(p_data.target_key[0], nullptr, 0, error);
				ERR_FAIL_COND_V(error.error != Variant::CallError::CALL_OK, p_data.initial_val);
			}
			return initial_val;
		}

		case INTER_CALLBACK:
			// Callback does not have a special initial value
			break;
	}
	// If we've made it here, just return the delta value as the initial value
	return p_data.delta_val;
}

Variant Tween::_get_final_val(const InterpolateData &p_data) const {
	switch (p_data.type) {
		case FOLLOW_PROPERTY:
		case FOLLOW_METHOD: {
			// Get the object that is being followed
			Object *target = ObjectDB::get_instance(p_data.target_id);
			ERR_FAIL_COND_V(target == nullptr, p_data.initial_val);

			// We want to figure out the final value
			Variant final_val;
			if (p_data.type == FOLLOW_PROPERTY) {
				// Read the property as-is
				bool valid = false;
				final_val = target->get_indexed(p_data.target_key, &valid);
				ERR_FAIL_COND_V(!valid, p_data.initial_val);
			} else {
				// We're looking at a method. Call the method on the target object
				Variant::CallError error;
				final_val = target->call(p_data.target_key[0], nullptr, 0, error);
				ERR_FAIL_COND_V(error.error != Variant::CallError::CALL_OK, p_data.initial_val);
			}

			// If we're looking at an INT value, instead convert it to a REAL
			// This is better for interpolation
			if (final_val.get_type() == Variant::INT) {
				final_val = final_val.operator real_t();
			}

			return final_val;
		}
		default: {
			// If we're not following a final value/method, use the final value from the data
			return p_data.final_val;
		}
	}
}

Variant &Tween::_get_delta_val(InterpolateData &p_data) {
	// What kind of data are we interpolating?
	switch (p_data.type) {
		case INTER_PROPERTY:
		case INTER_METHOD:
			// Simply return the given delta value
			return p_data.delta_val;

		case FOLLOW_PROPERTY:
		case FOLLOW_METHOD: {
			// We're following an object, so grab that instance
			Object *target = ObjectDB::get_instance(p_data.target_id);
			ERR_FAIL_COND_V(target == nullptr, p_data.initial_val);

			// We want to figure out the final value
			Variant final_val;
			if (p_data.type == FOLLOW_PROPERTY) {
				// Read the property as-is
				bool valid = false;
				final_val = target->get_indexed(p_data.target_key, &valid);
				ERR_FAIL_COND_V(!valid, p_data.initial_val);
			} else {
				// We're looking at a method. Call the method on the target object
				Variant::CallError error;
				final_val = target->call(p_data.target_key[0], nullptr, 0, error);
				ERR_FAIL_COND_V(error.error != Variant::CallError::CALL_OK, p_data.initial_val);
			}

			// If we're looking at an INT value, instead convert it to a REAL
			// This is better for interpolation
			if (final_val.get_type() == Variant::INT) {
				final_val = final_val.operator real_t();
			}

			// Calculate the delta based on the initial value and the final value
			_calc_delta_val(p_data.initial_val, final_val, p_data.delta_val);
			return p_data.delta_val;
		}

		case TARGETING_PROPERTY:
		case TARGETING_METHOD: {
			// Grab the initial value from the data to calculate delta
			Variant initial_val = _get_initial_val(p_data);

			// If we're looking at an INT value, instead convert it to a REAL
			// This is better for interpolation
			if (initial_val.get_type() == Variant::INT) {
				initial_val = initial_val.operator real_t();
			}

			// Calculate the delta based on the initial value and the final value
			_calc_delta_val(initial_val, p_data.final_val, p_data.delta_val);
			return p_data.delta_val;
		}

		case INTER_CALLBACK:
			// Callbacks have no special delta
			break;
	}
	// If we've made it here, use the initial value as the delta
	return p_data.initial_val;
}

Variant Tween::_run_equation(InterpolateData &p_data) {
	// Get the initial and delta values from the data
	Variant initial_val = _get_initial_val(p_data);
	Variant &delta_val = _get_delta_val(p_data);
	Variant result;

#define APPLY_EQUATION(element) \
	r.element = _run_equation(p_data.trans_type, p_data.ease_type, p_data.elapsed - p_data.delay, i.element, d.element, p_data.duration);

	// What type of data are we interpolating?
	switch (initial_val.get_type()) {
		case Variant::BOOL:
			// Run the boolean specific equation (checking if it is at least 0.5)
			result = (_run_equation(p_data.trans_type, p_data.ease_type, p_data.elapsed - p_data.delay, initial_val, delta_val, p_data.duration)) >= 0.5;
			break;

		case Variant::INT:
			// Run the integer specific equation
			result = (int)_run_equation(p_data.trans_type, p_data.ease_type, p_data.elapsed - p_data.delay, (int)initial_val, (int)delta_val, p_data.duration);
			break;

		case Variant::REAL:
			// Run the REAL specific equation
			result = _run_equation(p_data.trans_type, p_data.ease_type, p_data.elapsed - p_data.delay, (real_t)initial_val, (real_t)delta_val, p_data.duration);
			break;

		case Variant::VECTOR2: {
			// Get vectors for initial and delta values
			Vector2 i = initial_val;
			Vector2 d = delta_val;
			Vector2 r;

			// Execute the equation and mutate the r vector
			// This uses the custom APPLY_EQUATION macro defined above
			APPLY_EQUATION(x);
			APPLY_EQUATION(y);
			result = r;
		} break;

		case Variant::RECT2: {
			// Get the Rect2 for initial and delta value
			Rect2 i = initial_val;
			Rect2 d = delta_val;
			Rect2 r;

			// Execute the equation for the position and size of Rect2
			APPLY_EQUATION(position.x);
			APPLY_EQUATION(position.y);
			APPLY_EQUATION(size.x);
			APPLY_EQUATION(size.y);
			result = r;
		} break;

		case Variant::VECTOR3: {
			// Get vectors for initial and delta values
			Vector3 i = initial_val;
			Vector3 d = delta_val;
			Vector3 r;

			// Execute the equation and mutate the r vector
			// This uses the custom APPLY_EQUATION macro defined above
			APPLY_EQUATION(x);
			APPLY_EQUATION(y);
			APPLY_EQUATION(z);
			result = r;
		} break;

		case Variant::TRANSFORM2D: {
			// Get the transforms for initial and delta values
			Transform2D i = initial_val;
			Transform2D d = delta_val;
			Transform2D r;

			// Execute the equation on the transforms and mutate the r transform
			// This uses the custom APPLY_EQUATION macro defined above
			APPLY_EQUATION(elements[0][0]);
			APPLY_EQUATION(elements[0][1]);
			APPLY_EQUATION(elements[1][0]);
			APPLY_EQUATION(elements[1][1]);
			APPLY_EQUATION(elements[2][0]);
			APPLY_EQUATION(elements[2][1]);
			result = r;
		} break;

		case Variant::QUAT: {
			// Get the quaternian for the initial and delta values
			Quat i = initial_val;
			Quat d = delta_val;
			Quat r;

			// Execute the equation on the quaternian values and mutate the r quaternian
			// This uses the custom APPLY_EQUATION macro defined above
			APPLY_EQUATION(x);
			APPLY_EQUATION(y);
			APPLY_EQUATION(z);
			APPLY_EQUATION(w);
			result = r;
		} break;

		case Variant::AABB: {
			// Get the AABB's for the initial and delta values
			AABB i = initial_val;
			AABB d = delta_val;
			AABB r;

			// Execute the equation for the position and size of the AABB's and mutate the r AABB
			// This uses the custom APPLY_EQUATION macro defined above
			APPLY_EQUATION(position.x);
			APPLY_EQUATION(position.y);
			APPLY_EQUATION(position.z);
			APPLY_EQUATION(size.x);
			APPLY_EQUATION(size.y);
			APPLY_EQUATION(size.z);
			result = r;
		} break;

		case Variant::BASIS: {
			// Get the basis for initial and delta values
			Basis i = initial_val;
			Basis d = delta_val;
			Basis r;

			// Execute the equation on all the basis and mutate the r basis
			// This uses the custom APPLY_EQUATION macro defined above
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

		case Variant::TRANSFORM: {
			// Get the transforms for the initial and delta values
			Transform i = initial_val;
			Transform d = delta_val;
			Transform r;

			// Execute the equation for each of the transforms and their origin and mutate the r transform
			// This uses the custom APPLY_EQUATION macro defined above
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
			// Get the Color for initial and delta value
			Color i = initial_val;
			Color d = delta_val;
			Color r;

			// Apply the equation on the Color RGBA, and mutate the r color
			// This uses the custom APPLY_EQUATION macro defined above
			APPLY_EQUATION(r);
			APPLY_EQUATION(g);
			APPLY_EQUATION(b);
			APPLY_EQUATION(a);
			result = r;
		} break;

		default: {
			// If unknown, just return the initial value
			result = initial_val;
		} break;
	};
#undef APPLY_EQUATION
	// Return the result that was computed
	return result;
}

bool Tween::_apply_tween_value(InterpolateData &p_data, Variant &value) {
	// Get the object we want to apply the new value to
	Object *object = ObjectDB::get_instance(p_data.id);
	ERR_FAIL_COND_V(object == nullptr, false);

	// What kind of data are we mutating?
	switch (p_data.type) {
		case INTER_PROPERTY:
		case FOLLOW_PROPERTY:
		case TARGETING_PROPERTY: {
			// Simply set the property on the object
			bool valid = false;
			object->set_indexed(p_data.key, value, &valid);
			return valid;
		}

		case INTER_METHOD:
		case FOLLOW_METHOD:
		case TARGETING_METHOD: {
			// We want to call the method on the target object
			Variant::CallError error;

			// Do we have a non-nil value passed in?
			if (value.get_type() != Variant::NIL) {
				// Pass it as an argument to the function call
				Variant *arg[1] = { &value };
				object->call(p_data.key[0], (const Variant **)arg, 1, error);
			} else {
				// Don't pass any argument
				object->call(p_data.key[0], nullptr, 0, error);
			}

			// Did we get an error from the function call?
			return error.error == Variant::CallError::CALL_OK;
		}

		case INTER_CALLBACK:
			// Nothing to apply for a callback
			break;
	};
	// No issues found!
	return true;
}

void Tween::_tween_process(float p_delta) {
	// Process all of the pending commands
	_process_pending_commands();

	// If the scale is 0, make no progress on the tweens
	if (speed_scale == 0) {
		return;
	}

	// Update the delta and whether we are pending an update
	p_delta *= speed_scale;
	pending_update++;

	// Are we repeating the interpolations?
	if (repeat) {
		// For each interpolation...
		bool repeats_finished = true;
		for (List<InterpolateData>::Element *E = interpolates.front(); E; E = E->next()) {
			// Get the data from it
			InterpolateData &data = E->get();

			// Is not finished?
			if (!data.finish) {
				// We aren't finished yet, no need to check the rest
				repeats_finished = false;
				break;
			}
		}

		// If we are all finished, we can reset all of the tweens
		if (repeats_finished) {
			reset_all();
		}
	}

	// Are all of the tweens complete?
	bool all_finished = true;

	// For each tween we wish to interpolate...
	for (List<InterpolateData>::Element *E = interpolates.front(); E; E = E->next()) {
		// Get the data from it
		InterpolateData &data = E->get();

		// Track if we hit one that isn't finished yet
		all_finished = all_finished && data.finish;

		// Is the data not active or already finished? No need to go any further
		if (!data.active || data.finish) {
			continue;
		}

		// Get the target object for this interpolation
		Object *object = ObjectDB::get_instance(data.id);
		if (object == nullptr) {
			continue;
		}

		// Are we still delaying this tween?
		bool prev_delaying = data.elapsed <= data.delay;
		data.elapsed += p_delta;
		if (data.elapsed < data.delay) {
			continue;
		} else if (prev_delaying) {
			// We can apply the tween's value to the data and emit that the tween has started
			_apply_tween_value(data, data.initial_val);
			emit_signal("tween_started", object, NodePath(Vector<StringName>(), data.key, false));
		}

		// Are we at the end of the tween?
		if (data.elapsed > (data.delay + data.duration)) {
			// Set the elapsed time to the end and mark this one as finished
			data.elapsed = data.delay + data.duration;
			data.finish = true;
		}

		// Are we interpolating a callback?
		if (data.type == INTER_CALLBACK) {
			// Is the tween completed?
			if (data.finish) {
				static_assert(VARIANT_ARG_MAX == 8, "This code needs to be updated if VARIANT_ARG_MAX != 8");

				// Are we calling this callback deferred or immediately?
				if (data.call_deferred) {
					// Run the deferred function callback, applying the correct number of arguments
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
						case 6:
							object->call_deferred(data.key[0], data.arg[0], data.arg[1], data.arg[2], data.arg[3], data.arg[4], data.arg[5]);
							break;
						case 7:
							object->call_deferred(data.key[0], data.arg[0], data.arg[1], data.arg[2], data.arg[3], data.arg[4], data.arg[5], data.arg[6]);
							break;
						case 8:
							object->call_deferred(data.key[0], data.arg[0], data.arg[1], data.arg[2], data.arg[3], data.arg[4], data.arg[5], data.arg[6], data.arg[7]);
							break;
					}
				} else {
					// Call the function directly with the arguments
					Variant::CallError error;
					Variant *arg[VARIANT_ARG_MAX] = {
						&data.arg[0],
						&data.arg[1],
						&data.arg[2],
						&data.arg[3],
						&data.arg[4],
						&data.arg[5],
						&data.arg[6],
						&data.arg[7],
					};
					object->call(data.key[0], (const Variant **)arg, data.args, error);
				}
			}
		} else {
			// We can apply the value directly
			Variant result = _run_equation(data);
			_apply_tween_value(data, result);

			// Emit that the tween has taken a step
			emit_signal("tween_step", object, NodePath(Vector<StringName>(), data.key, false), data.elapsed, result);
		}

		// Is the tween now finished?
		if (data.finish) {
			// Set it to the final value directly
			Variant final_val = _get_final_val(data);
			_apply_tween_value(data, final_val);

			// Emit the signal
			emit_signal("tween_completed", object, NodePath(Vector<StringName>(), data.key, false));

			// If we are not repeating the tween, remove it
			if (!repeat) {
				call_deferred("_remove_by_uid", data.uid);
			}
		} else if (!repeat) {
			// Check whether all tweens are finished
			all_finished = all_finished && data.finish;
		}
	}
	// One less update left to go
	pending_update--;

	// If all tweens are completed, we no longer need to be active
	if (all_finished) {
		set_active(false);
		emit_signal("tween_all_completed");
	}
}

void Tween::set_tween_process_mode(TweenProcessMode p_mode) {
	tween_process_mode = p_mode;
}

Tween::TweenProcessMode Tween::get_tween_process_mode() const {
	return tween_process_mode;
}

bool Tween::is_active() const {
	return is_processing_internal() || is_physics_processing_internal();
}

void Tween::set_active(bool p_active) {
	// Do nothing if it's the same active mode that we currently are
	if (is_active() == p_active) {
		return;
	}

	// Depending on physics or idle, set processing
	switch (tween_process_mode) {
		case TWEEN_PROCESS_IDLE:
			set_process_internal(p_active);
			break;
		case TWEEN_PROCESS_PHYSICS:
			set_physics_process_internal(p_active);
			break;
	}
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
	ERR_FAIL_COND_V_MSG(!is_inside_tree(), false, "Tween was not added to the SceneTree!");

	// Are there any pending updates?
	if (pending_update != 0) {
		// Start the tweens after deferring
		call_deferred("start");
		return true;
	}

	pending_update++;
	for (List<InterpolateData>::Element *E = interpolates.front(); E; E = E->next()) {
		InterpolateData &data = E->get();
		data.active = true;
	}
	pending_update--;

	// We want to be activated
	set_active(true);

	// Don't resume from current position if stop_all() function has been used
	if (was_stopped) {
		seek(0);
	}
	was_stopped = false;

	return true;
}

bool Tween::reset(Object *p_object, StringName p_key) {
	// Find all interpolations that use the same object and target string
	pending_update++;
	for (List<InterpolateData>::Element *E = interpolates.front(); E; E = E->next()) {
		// Get the target object
		InterpolateData &data = E->get();
		Object *object = ObjectDB::get_instance(data.id);
		if (object == nullptr) {
			continue;
		}

		// Do we have the correct object and key?
		if (object == p_object && (data.concatenated_key == p_key || p_key == "")) {
			// Reset the tween to the initial state
			data.elapsed = 0;
			data.finish = false;

			// Also apply the initial state if there isn't a delay
			if (data.delay == 0) {
				_apply_tween_value(data, data.initial_val);
			}
		}
	}
	pending_update--;
	return true;
}

bool Tween::reset_all() {
	// Go through all interpolations
	pending_update++;
	for (List<InterpolateData>::Element *E = interpolates.front(); E; E = E->next()) {
		// Get the target data and set it back to the initial state
		InterpolateData &data = E->get();
		data.elapsed = 0;
		data.finish = false;

		// If there isn't a delay, apply the value to the object
		if (data.delay == 0) {
			_apply_tween_value(data, data.initial_val);
		}
	}
	pending_update--;
	return true;
}

bool Tween::stop(Object *p_object, StringName p_key) {
	// Find the tween that has the given target object and string key
	pending_update++;
	for (List<InterpolateData>::Element *E = interpolates.front(); E; E = E->next()) {
		// Get the object the tween is targeting
		InterpolateData &data = E->get();
		Object *object = ObjectDB::get_instance(data.id);
		if (object == nullptr) {
			continue;
		}

		// Is this the correct object and does it have the given key?
		if (object == p_object && (data.concatenated_key == p_key || p_key == "")) {
			// Disable the tween
			data.active = false;
		}
	}
	pending_update--;
	return true;
}

bool Tween::stop_all() {
	// We no longer need to be active since all tweens have been stopped
	set_active(false);
	was_stopped = true;

	// For each interpolation...
	pending_update++;
	for (List<InterpolateData>::Element *E = interpolates.front(); E; E = E->next()) {
		// Simply set it inactive
		InterpolateData &data = E->get();
		data.active = false;
	}
	pending_update--;
	return true;
}

bool Tween::resume(Object *p_object, StringName p_key) {
	// We need to be activated
	// TODO: What if no tween is found??
	set_active(true);

	// Find the tween that uses the given target object and string key
	pending_update++;
	for (List<InterpolateData>::Element *E = interpolates.front(); E; E = E->next()) {
		// Grab the object
		InterpolateData &data = E->get();
		Object *object = ObjectDB::get_instance(data.id);
		if (object == nullptr) {
			continue;
		}

		// If the object and string key match, activate it
		if (object == p_object && (data.concatenated_key == p_key || p_key == "")) {
			data.active = true;
		}
	}
	pending_update--;
	return true;
}

bool Tween::resume_all() {
	// Set ourselves active so we can process tweens
	// TODO: What if there are no tweens? We get set to active for no reason!
	set_active(true);

	// For each interpolation...
	pending_update++;
	for (List<InterpolateData>::Element *E = interpolates.front(); E; E = E->next()) {
		// Simply grab it and set it to active
		InterpolateData &data = E->get();
		data.active = true;
	}
	pending_update--;
	return true;
}

bool Tween::remove(Object *p_object, StringName p_key) {
	// If we are still updating, call this function again later
	if (pending_update != 0) {
		call_deferred("remove", p_object, p_key);
		return true;
	}

	// For each interpolation...
	List<List<InterpolateData>::Element *> for_removal;
	for (List<InterpolateData>::Element *E = interpolates.front(); E; E = E->next()) {
		// Get the target object
		InterpolateData &data = E->get();
		Object *object = ObjectDB::get_instance(data.id);
		if (object == nullptr) {
			continue;
		}

		// If the target object and string key match, queue it for removal
		if (object == p_object && (data.concatenated_key == p_key || p_key == "")) {
			for_removal.push_back(E);
		}
	}

	// For each interpolation we wish to remove...
	for (List<List<InterpolateData>::Element *>::Element *E = for_removal.front(); E; E = E->next()) {
		// Erase it
		interpolates.erase(E->get());
	}
	return true;
}

void Tween::_remove_by_uid(int uid) {
	// If we are still updating, call this function again later
	if (pending_update != 0) {
		call_deferred("_remove_by_uid", uid);
		return;
	}

	// Find the interpolation that matches the given UID
	for (List<InterpolateData>::Element *E = interpolates.front(); E; E = E->next()) {
		if (uid == E->get().uid) {
			// It matches, erase it and stop looking
			E->erase();
			break;
		}
	}
}

void Tween::_push_interpolate_data(InterpolateData &p_data) {
	pending_update++;

	// Add the new interpolation
	p_data.uid = ++uid;
	interpolates.push_back(p_data);

	pending_update--;
}

bool Tween::remove_all() {
	// If we are still updating, call this function again later
	if (pending_update != 0) {
		call_deferred("remove_all");
		return true;
	}
	// We no longer need to be active
	set_active(false);

	// Clear out all interpolations and reset the uid
	interpolates.clear();
	uid = 0;

	return true;
}

bool Tween::seek(real_t p_time) {
	// Go through each interpolation...
	pending_update++;
	for (List<InterpolateData>::Element *E = interpolates.front(); E; E = E->next()) {
		// Get the target data
		InterpolateData &data = E->get();

		// Update the elapsed data to be set to the target time
		data.elapsed = p_time;

		// Are we at the end?
		if (data.elapsed < data.delay) {
			// There is still time left to go
			data.finish = false;
			continue;
		} else if (data.elapsed >= (data.delay + data.duration)) {
			// We are past the end of it, set the elapsed time to the end and mark as finished
			data.elapsed = (data.delay + data.duration);
			data.finish = true;
		} else {
			// We are not finished with this interpolation yet
			data.finish = false;
		}

		// If we are a callback, do nothing special
		if (data.type == INTER_CALLBACK) {
			continue;
		}

		// Run the equation on the data and apply the value
		Variant result = _run_equation(data);
		_apply_tween_value(data, result);
	}
	pending_update--;
	return true;
}

real_t Tween::tell() const {
	// We want to grab the position of the furthest along tween
	pending_update++;
	real_t pos = 0;

	// For each interpolation...
	for (const List<InterpolateData>::Element *E = interpolates.front(); E; E = E->next()) {
		// Get the data and figure out if it's position is further along than the previous ones
		const InterpolateData &data = E->get();
		if (data.elapsed > pos) {
			// Save it if so
			pos = data.elapsed;
		}
	}
	pending_update--;
	return pos;
}

real_t Tween::get_runtime() const {
	// If the tween isn't moving, it'll last forever
	if (speed_scale == 0) {
		return INFINITY;
	}

	pending_update++;

	// For each interpolation...
	real_t runtime = 0;
	for (const List<InterpolateData>::Element *E = interpolates.front(); E; E = E->next()) {
		// Get the tween data and see if it's runtime is greater than the previous tweens
		const InterpolateData &data = E->get();
		real_t t = data.delay + data.duration;
		if (t > runtime) {
			// This is the longest running tween
			runtime = t;
		}
	}
	pending_update--;

	// Adjust the runtime for the current speed scale
	return runtime / speed_scale;
}

bool Tween::_calc_delta_val(const Variant &p_initial_val, const Variant &p_final_val, Variant &p_delta_val) {
	// Get the initial, final, and delta values
	const Variant &initial_val = p_initial_val;
	const Variant &final_val = p_final_val;
	Variant &delta_val = p_delta_val;

	// What kind of data are we interpolating?
	switch (initial_val.get_type()) {
		case Variant::BOOL:
			// We'll treat booleans just like integers
		case Variant::INT:
			// Compute the integer delta
			delta_val = (int)final_val - (int)initial_val;
			break;

		case Variant::REAL:
			// Convert to REAL and find the delta
			delta_val = (real_t)final_val - (real_t)initial_val;
			break;

		case Variant::VECTOR2:
			// Convert to Vectors and find the delta
			delta_val = final_val.operator Vector2() - initial_val.operator Vector2();
			break;

		case Variant::RECT2: {
			// Build a new Rect2 and use the new position and sizes to make a delta
			Rect2 i = initial_val;
			Rect2 f = final_val;
			delta_val = Rect2(f.position - i.position, f.size - i.size);
		} break;

		case Variant::VECTOR3:
			// Convert to Vectors and find the delta
			delta_val = final_val.operator Vector3() - initial_val.operator Vector3();
			break;

		case Variant::TRANSFORM2D: {
			// Build a new transform which is the difference between the initial and final values
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
			// Convert to quaternianls and find the delta
			delta_val = final_val.operator Quat() - initial_val.operator Quat();
			break;

		case Variant::AABB: {
			// Build a new AABB and use the new position and sizes to make a delta
			AABB i = initial_val;
			AABB f = final_val;
			delta_val = AABB(f.position - i.position, f.size - i.size);
		} break;

		case Variant::BASIS: {
			// Build a new basis which is the delta between the initial and final values
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

		case Variant::TRANSFORM: {
			// Build a new transform which is the difference between the initial and final values
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
			// Make a new color which is the difference between each the color's RGBA attributes
			Color i = initial_val;
			Color f = final_val;
			delta_val = Color(f.r - i.r, f.g - i.g, f.b - i.b, f.a - i.a);
		} break;

		default: {
			static Variant::Type supported_types[] = {
				Variant::BOOL,
				Variant::INT,
				Variant::REAL,
				Variant::VECTOR2,
				Variant::RECT2,
				Variant::VECTOR3,
				Variant::TRANSFORM2D,
				Variant::QUAT,
				Variant::AABB,
				Variant::BASIS,
				Variant::TRANSFORM,
				Variant::COLOR,
			};

			int length = *(&supported_types + 1) - supported_types;
			String error_msg = "Invalid parameter type. Supported types are: ";
			for (int i = 0; i < length; i++) {
				if (i != 0) {
					error_msg += ", ";
				}
				error_msg += Variant::get_type_name(supported_types[i]);
			}
			error_msg += ".";
			ERR_PRINT(error_msg);
			return false;
		}
	};
	return true;
}

bool Tween::_build_interpolation(InterpolateType p_interpolation_type, Object *p_object, NodePath *p_property, StringName *p_method, Variant p_initial_val, Variant p_final_val, real_t p_duration, TransitionType p_trans_type, EaseType p_ease_type, real_t p_delay) {
	// TODO: Add initialization+implementation for remaining interpolation types
	// TODO: Fix this method's organization to take advantage of the type

	// Make a new interpolation data
	InterpolateData data;
	data.active = true;
	data.type = p_interpolation_type;
	data.finish = false;
	data.elapsed = 0;

	// Validate and apply interpolation data

	// Give it the object
	ERR_FAIL_COND_V_MSG(p_object == nullptr, false, "Invalid object provided to Tween.");
	data.id = p_object->get_instance_id();

	// Validate the initial and final values
	ERR_FAIL_COND_V_MSG(p_initial_val.get_type() != p_final_val.get_type(), false, "Initial value type '" + Variant::get_type_name(p_initial_val.get_type()) + "' does not match final value type '" + Variant::get_type_name(p_final_val.get_type()) + "'.");
	data.initial_val = p_initial_val;
	data.final_val = p_final_val;

	// Check the Duration
	ERR_FAIL_COND_V_MSG(p_duration < 0, false, "Only non-negative duration values allowed in Tweens.");
	data.duration = p_duration;

	// Tween Delay
	ERR_FAIL_COND_V_MSG(p_delay < 0, false, "Only non-negative delay values allowed in Tweens.");
	data.delay = p_delay;

	// Transition type
	ERR_FAIL_COND_V_MSG(p_trans_type < 0 || p_trans_type >= TRANS_COUNT, false, "Invalid transition type provided to Tween.");
	data.trans_type = p_trans_type;

	// Easing type
	ERR_FAIL_COND_V_MSG(p_ease_type < 0 || p_ease_type >= EASE_COUNT, false, "Invalid easing type provided to Tween.");
	data.ease_type = p_ease_type;

	// Is the property defined?
	if (p_property) {
		// Check that the object actually contains the given property
		bool prop_valid = false;
		p_object->get_indexed(p_property->get_subnames(), &prop_valid);
		ERR_FAIL_COND_V_MSG(!prop_valid, false, "Tween target object has no property named: " + p_property->get_concatenated_subnames() + ".");

		data.key = p_property->get_subnames();
		data.concatenated_key = p_property->get_concatenated_subnames();
	}

	// Is the method defined?
	if (p_method) {
		// Does the object even have the requested method?
		ERR_FAIL_COND_V_MSG(!p_object->has_method(*p_method), false, "Tween target object has no method named: " + *p_method + ".");

		data.key.push_back(*p_method);
		data.concatenated_key = *p_method;
	}

	// Is there not a valid delta?
	if (!_calc_delta_val(data.initial_val, data.final_val, data.delta_val)) {
		return false;
	}

	// Add this interpolation to the total
	_push_interpolate_data(data);
	return true;
}

bool Tween::interpolate_property(Object *p_object, NodePath p_property, Variant p_initial_val, Variant p_final_val, real_t p_duration, TransitionType p_trans_type, EaseType p_ease_type, real_t p_delay) {
	// If we are busy updating, call this function again later
	if (pending_update != 0) {
		_add_pending_command("interpolate_property", p_object, p_property, p_initial_val, p_final_val, p_duration, p_trans_type, p_ease_type, p_delay);
		return true;
	}

	// Check that the target object is valid
	ERR_FAIL_COND_V_MSG(p_object == nullptr, false, vformat("The Tween \"%s\"'s target node is `null`. Is the node reference correct?", get_name()));

	// Get the property from the node path
	p_property = p_property.get_as_property_path();

	// If no initial value given, grab the initial value from the object
	// TODO: Is this documented? This is very useful and removes a lot of clutter from tweens!
	if (p_initial_val.get_type() == Variant::NIL) {
		p_initial_val = p_object->get_indexed(p_property.get_subnames());
	}

	// Convert any integers into REALs as they are better for interpolation
	if (p_initial_val.get_type() == Variant::INT) {
		p_initial_val = p_initial_val.operator real_t();
	}
	if (p_final_val.get_type() == Variant::INT) {
		p_final_val = p_final_val.operator real_t();
	}

	// Build the interpolation data
	bool result = _build_interpolation(INTER_PROPERTY, p_object, &p_property, nullptr, p_initial_val, p_final_val, p_duration, p_trans_type, p_ease_type, p_delay);
	return result;
}

bool Tween::interpolate_method(Object *p_object, StringName p_method, Variant p_initial_val, Variant p_final_val, real_t p_duration, TransitionType p_trans_type, EaseType p_ease_type, real_t p_delay) {
	// If we are busy updating, call this function again later
	if (pending_update != 0) {
		_add_pending_command("interpolate_method", p_object, p_method, p_initial_val, p_final_val, p_duration, p_trans_type, p_ease_type, p_delay);
		return true;
	}

	// Check that the target object is valid
	ERR_FAIL_COND_V_MSG(p_object == nullptr, false, vformat("The Tween \"%s\"'s target node is `null`. Is the node reference correct?", get_name()));

	// Convert any integers into REALs as they are better for interpolation
	if (p_initial_val.get_type() == Variant::INT) {
		p_initial_val = p_initial_val.operator real_t();
	}
	if (p_final_val.get_type() == Variant::INT) {
		p_final_val = p_final_val.operator real_t();
	}

	// Build the interpolation data
	bool result = _build_interpolation(INTER_METHOD, p_object, nullptr, &p_method, p_initial_val, p_final_val, p_duration, p_trans_type, p_ease_type, p_delay);
	return result;
}

bool Tween::interpolate_callback(Object *p_object, real_t p_duration, String p_callback, VARIANT_ARG_DECLARE) {
	// If we are already updating, call this function again later
	if (pending_update != 0) {
		_add_pending_command("interpolate_callback", p_object, p_duration, p_callback, p_arg1, p_arg2, p_arg3, p_arg4, p_arg5);
		return true;
	}

	// Check that the target object is valid
	ERR_FAIL_COND_V(p_object == nullptr, false);

	// Duration cannot be negative
	ERR_FAIL_COND_V(p_duration < 0, false);

	// Check whether the object even has the callback
	ERR_FAIL_COND_V_MSG(!p_object->has_method(p_callback), false, "Object has no callback named: " + p_callback + ".");

	// Build a new InterpolationData
	InterpolateData data;
	data.active = true;
	data.type = INTER_CALLBACK;
	data.finish = false;
	data.call_deferred = false;
	data.elapsed = 0;

	// Give the data it's configuration
	data.id = p_object->get_instance_id();
	data.key.push_back(p_callback);
	data.concatenated_key = p_callback;
	data.duration = p_duration;
	data.delay = 0;

	// Add arguments to the interpolation
	int args = 0;
	if (p_arg5.get_type() != Variant::NIL) {
		args = 5;
	} else if (p_arg4.get_type() != Variant::NIL) {
		args = 4;
	} else if (p_arg3.get_type() != Variant::NIL) {
		args = 3;
	} else if (p_arg2.get_type() != Variant::NIL) {
		args = 2;
	} else if (p_arg1.get_type() != Variant::NIL) {
		args = 1;
	} else {
		args = 0;
	}

	data.args = args;
	data.arg[0] = p_arg1;
	data.arg[1] = p_arg2;
	data.arg[2] = p_arg3;
	data.arg[3] = p_arg4;
	data.arg[4] = p_arg5;

	// Add the new interpolation
	_push_interpolate_data(data);
	return true;
}

bool Tween::interpolate_deferred_callback(Object *p_object, real_t p_duration, String p_callback, VARIANT_ARG_DECLARE) {
	// If we are already updating, call this function again later
	if (pending_update != 0) {
		_add_pending_command("interpolate_deferred_callback", p_object, p_duration, p_callback, p_arg1, p_arg2, p_arg3, p_arg4, p_arg5);
		return true;
	}

	// Check that the target object is valid
	ERR_FAIL_COND_V(p_object == nullptr, false);

	// No negative durations allowed
	ERR_FAIL_COND_V(p_duration < 0, false);

	// Confirm the callback exists on the object
	ERR_FAIL_COND_V_MSG(!p_object->has_method(p_callback), false, "Object has no callback named: " + p_callback + ".");

	// Create a new InterpolateData for the callback
	InterpolateData data;
	data.active = true;
	data.type = INTER_CALLBACK;
	data.finish = false;
	data.call_deferred = true;
	data.elapsed = 0;

	// Give the data it's configuration
	data.id = p_object->get_instance_id();
	data.key.push_back(p_callback);
	data.concatenated_key = p_callback;
	data.duration = p_duration;
	data.delay = 0;

	// Collect arguments for the callback
	static_assert(VARIANT_ARG_MAX == 8, "This code needs to be updated if VARIANT_ARG_MAX != 8");
	int args = 0;
	if (p_arg8.get_type() != Variant::NIL) {
		args = 8;
	} else if (p_arg7.get_type() != Variant::NIL) {
		args = 7;
	} else if (p_arg6.get_type() != Variant::NIL) {
		args = 6;
	} else if (p_arg5.get_type() != Variant::NIL) {
		args = 5;
	} else if (p_arg4.get_type() != Variant::NIL) {
		args = 4;
	} else if (p_arg3.get_type() != Variant::NIL) {
		args = 3;
	} else if (p_arg2.get_type() != Variant::NIL) {
		args = 2;
	} else if (p_arg1.get_type() != Variant::NIL) {
		args = 1;
	} else {
		args = 0;
	}

	data.args = args;
	data.arg[0] = p_arg1;
	data.arg[1] = p_arg2;
	data.arg[2] = p_arg3;
	data.arg[3] = p_arg4;
	data.arg[4] = p_arg5;
	data.arg[5] = p_arg6;
	data.arg[6] = p_arg7;
	data.arg[7] = p_arg8;

	// Add the new interpolation
	_push_interpolate_data(data);
	return true;
}

bool Tween::follow_property(Object *p_object, NodePath p_property, Variant p_initial_val, Object *p_target, NodePath p_target_property, real_t p_duration, TransitionType p_trans_type, EaseType p_ease_type, real_t p_delay) {
	// If we are already updating, call this function again later
	if (pending_update != 0) {
		_add_pending_command("follow_property", p_object, p_property, p_initial_val, p_target, p_target_property, p_duration, p_trans_type, p_ease_type, p_delay);
		return true;
	}

	// Confirm the source and target objects are valid
	ERR_FAIL_NULL_V(p_object, false);
	ERR_FAIL_NULL_V(p_target, false);

	// Get the two properties from their paths
	p_property = p_property.get_as_property_path();
	p_target_property = p_target_property.get_as_property_path();

	// If no initial value is given, grab it from the source object
	// TODO: Is this documented? It's really helpful for decluttering tweens
	if (p_initial_val.get_type() == Variant::NIL) {
		p_initial_val = p_object->get_indexed(p_property.get_subnames());
	}

	// Convert initial INT values to REAL as they are better for interpolation
	if (p_initial_val.get_type() == Variant::INT) {
		p_initial_val = p_initial_val.operator real_t();
	}

	// No negative durations
	ERR_FAIL_COND_V(p_duration < 0, false);

	// Ensure transition and easing types are valid
	ERR_FAIL_COND_V(p_trans_type < 0 || p_trans_type >= TRANS_COUNT, false);
	ERR_FAIL_COND_V(p_ease_type < 0 || p_ease_type >= EASE_COUNT, false);

	// No negative delays
	ERR_FAIL_COND_V(p_delay < 0, false);

	// Confirm the source and target objects have the desired properties
	bool prop_valid = false;
	p_object->get_indexed(p_property.get_subnames(), &prop_valid);
	ERR_FAIL_COND_V(!prop_valid, false);

	bool target_prop_valid = false;
	Variant target_val = p_target->get_indexed(p_target_property.get_subnames(), &target_prop_valid);
	ERR_FAIL_COND_V(!target_prop_valid, false);

	// Convert target INT to REAL since it is better for interpolation
	if (target_val.get_type() == Variant::INT) {
		target_val = target_val.operator real_t();
	}

	// Verify that the target value and initial value are the same type
	ERR_FAIL_COND_V(target_val.get_type() != p_initial_val.get_type(), false);

	// Create a new InterpolateData
	InterpolateData data;
	data.active = true;
	data.type = FOLLOW_PROPERTY;
	data.finish = false;
	data.elapsed = 0;

	// Give the InterpolateData it's configuration
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

	// Add the interpolation
	_push_interpolate_data(data);
	return true;
}

bool Tween::follow_method(Object *p_object, StringName p_method, Variant p_initial_val, Object *p_target, StringName p_target_method, real_t p_duration, TransitionType p_trans_type, EaseType p_ease_type, real_t p_delay) {
	// If we are currently updating, call this function again later
	if (pending_update != 0) {
		_add_pending_command("follow_method", p_object, p_method, p_initial_val, p_target, p_target_method, p_duration, p_trans_type, p_ease_type, p_delay);
		return true;
	}
	// Convert initial INT values to REAL as they are better for interpolation
	if (p_initial_val.get_type() == Variant::INT) {
		p_initial_val = p_initial_val.operator real_t();
	}

	// Verify the source and target objects are valid
	ERR_FAIL_COND_V(p_object == nullptr, false);
	ERR_FAIL_COND_V(p_target == nullptr, false);

	// No negative durations
	ERR_FAIL_COND_V(p_duration < 0, false);

	// Ensure that the transition and ease types are valid
	ERR_FAIL_COND_V(p_trans_type < 0 || p_trans_type >= TRANS_COUNT, false);
	ERR_FAIL_COND_V(p_ease_type < 0 || p_ease_type >= EASE_COUNT, false);

	// No negative delays
	ERR_FAIL_COND_V(p_delay < 0, false);

	// Confirm both objects have the target methods
	ERR_FAIL_COND_V_MSG(!p_object->has_method(p_method), false, "Object has no method named: " + p_method + ".");
	ERR_FAIL_COND_V_MSG(!p_target->has_method(p_target_method), false, "Target has no method named: " + p_target_method + ".");

	// Call the method to get the target value
	Variant::CallError error;
	Variant target_val = p_target->call(p_target_method, nullptr, 0, error);
	ERR_FAIL_COND_V(error.error != Variant::CallError::CALL_OK, false);

	// Convert target INT values to REAL as they are better for interpolation
	if (target_val.get_type() == Variant::INT) {
		target_val = target_val.operator real_t();
	}
	ERR_FAIL_COND_V(target_val.get_type() != p_initial_val.get_type(), false);

	// Make the new InterpolateData for the method follow
	InterpolateData data;
	data.active = true;
	data.type = FOLLOW_METHOD;
	data.finish = false;
	data.elapsed = 0;

	// Give the data it's configuration
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

	// Add the new interpolation
	_push_interpolate_data(data);
	return true;
}

bool Tween::targeting_property(Object *p_object, NodePath p_property, Object *p_initial, NodePath p_initial_property, Variant p_final_val, real_t p_duration, TransitionType p_trans_type, EaseType p_ease_type, real_t p_delay) {
	// If we are currently updating, call this function again later
	if (pending_update != 0) {
		_add_pending_command("targeting_property", p_object, p_property, p_initial, p_initial_property, p_final_val, p_duration, p_trans_type, p_ease_type, p_delay);
		return true;
	}
	// Grab the target property and the target property
	p_property = p_property.get_as_property_path();
	p_initial_property = p_initial_property.get_as_property_path();

	// Convert the initial INT values to REAL as they are better for Interpolation
	if (p_final_val.get_type() == Variant::INT) {
		p_final_val = p_final_val.operator real_t();
	}

	// Verify both objects are valid
	ERR_FAIL_COND_V(p_object == nullptr, false);
	ERR_FAIL_COND_V(p_initial == nullptr, false);

	// No negative durations
	ERR_FAIL_COND_V(p_duration < 0, false);

	// Ensure transition and easing types are valid
	ERR_FAIL_COND_V(p_trans_type < 0 || p_trans_type >= TRANS_COUNT, false);
	ERR_FAIL_COND_V(p_ease_type < 0 || p_ease_type >= EASE_COUNT, false);

	// No negative delays
	ERR_FAIL_COND_V(p_delay < 0, false);

	// Ensure the initial and target properties exist on their objects
	bool prop_valid = false;
	p_object->get_indexed(p_property.get_subnames(), &prop_valid);
	ERR_FAIL_COND_V(!prop_valid, false);

	bool initial_prop_valid = false;
	Variant initial_val = p_initial->get_indexed(p_initial_property.get_subnames(), &initial_prop_valid);
	ERR_FAIL_COND_V(!initial_prop_valid, false);

	// Convert the initial INT value to REAL as it is better for interpolation
	if (initial_val.get_type() == Variant::INT) {
		initial_val = initial_val.operator real_t();
	}
	ERR_FAIL_COND_V(initial_val.get_type() != p_final_val.get_type(), false);

	// Build the InterpolateData object
	InterpolateData data;
	data.active = true;
	data.type = TARGETING_PROPERTY;
	data.finish = false;
	data.elapsed = 0;

	// Give the data it's configuration
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

	// Ensure there is a valid delta
	if (!_calc_delta_val(data.initial_val, data.final_val, data.delta_val)) {
		return false;
	}

	// Add the interpolation
	_push_interpolate_data(data);
	return true;
}

bool Tween::targeting_method(Object *p_object, StringName p_method, Object *p_initial, StringName p_initial_method, Variant p_final_val, real_t p_duration, TransitionType p_trans_type, EaseType p_ease_type, real_t p_delay) {
	// If we are currently updating, call this function again later
	if (pending_update != 0) {
		_add_pending_command("targeting_method", p_object, p_method, p_initial, p_initial_method, p_final_val, p_duration, p_trans_type, p_ease_type, p_delay);
		return true;
	}

	// Convert final INT values to REAL as they are better for interpolation
	if (p_final_val.get_type() == Variant::INT) {
		p_final_val = p_final_val.operator real_t();
	}

	// Make sure the given objects are valid
	ERR_FAIL_COND_V(p_object == nullptr, false);
	ERR_FAIL_COND_V(p_initial == nullptr, false);

	// No negative durations
	ERR_FAIL_COND_V(p_duration < 0, false);

	// Ensure transition and easing types are valid
	ERR_FAIL_COND_V(p_trans_type < 0 || p_trans_type >= TRANS_COUNT, false);
	ERR_FAIL_COND_V(p_ease_type < 0 || p_ease_type >= EASE_COUNT, false);

	// No negative delays
	ERR_FAIL_COND_V(p_delay < 0, false);

	// Make sure both objects have the given method
	ERR_FAIL_COND_V_MSG(!p_object->has_method(p_method), false, "Object has no method named: " + p_method + ".");
	ERR_FAIL_COND_V_MSG(!p_initial->has_method(p_initial_method), false, "Initial Object has no method named: " + p_initial_method + ".");

	// Call the method to get the initial value
	Variant::CallError error;
	Variant initial_val = p_initial->call(p_initial_method, nullptr, 0, error);
	ERR_FAIL_COND_V(error.error != Variant::CallError::CALL_OK, false);

	// Convert initial INT values to REAL as they aer better for interpolation
	if (initial_val.get_type() == Variant::INT) {
		initial_val = initial_val.operator real_t();
	}
	ERR_FAIL_COND_V(initial_val.get_type() != p_final_val.get_type(), false);

	// Build the new InterpolateData object
	InterpolateData data;
	data.active = true;
	data.type = TARGETING_METHOD;
	data.finish = false;
	data.elapsed = 0;

	// Configure the data
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

	// Ensure there is a valid delta
	if (!_calc_delta_val(data.initial_val, data.final_val, data.delta_val)) {
		return false;
	}

	// Add the interpolation
	_push_interpolate_data(data);
	return true;
}

Tween::Tween() {
	// Initialize tween attributes
	tween_process_mode = TWEEN_PROCESS_IDLE;
	repeat = false;
	speed_scale = 1;
	pending_update = 0;
	uid = 0;
}

Tween::~Tween() {
}
