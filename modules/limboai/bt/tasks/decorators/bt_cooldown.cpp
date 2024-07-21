/**
 * bt_cooldown.cpp
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#include "bt_cooldown.h"

#ifdef LIMBOAI_MODULE
#include "scene/main/scene_tree.h"
#endif

#ifdef LIMBOAI_GDEXTENSION
#include <godot_cpp/classes/scene_tree.hpp>
#endif

//**** Setters / Getters

void BTCooldown::set_duration(double p_value) {
	duration = p_value;
	emit_changed();
}

void BTCooldown::set_process_pause(bool p_value) {
	process_pause = p_value;
	emit_changed();
}

void BTCooldown::set_start_cooled(bool p_value) {
	start_cooled = p_value;
	emit_changed();
}

void BTCooldown::set_trigger_on_failure(bool p_value) {
	trigger_on_failure = p_value;
	emit_changed();
}

void BTCooldown::set_cooldown_state_var(const StringName &p_value) {
	cooldown_state_var = p_value;
	emit_changed();
}

//**** Task Implementation

String BTCooldown::_generate_name() {
	return vformat("Cooldown %s sec", Math::snapped(duration, 0.001));
}

void BTCooldown::_setup() {
	if (cooldown_state_var == StringName()) {
		cooldown_state_var = vformat("cooldown_%d", get_instance_id());
	}
	get_blackboard()->set_var(cooldown_state_var, false);
	if (start_cooled) {
		_chill();
	}
}

BT::Status BTCooldown::_tick(double p_delta) {
	ERR_FAIL_COND_V_MSG(get_child_count() == 0, FAILURE, "BT decorator has no child.");
	if (get_blackboard()->get_var(cooldown_state_var, true)) {
		return FAILURE;
	}
	Status status = get_child(0)->execute(p_delta);
	if (status == SUCCESS || (trigger_on_failure && status == FAILURE)) {
		_chill();
	}
	return status;
}

void BTCooldown::_chill() {
	get_blackboard()->set_var(cooldown_state_var, true);
	if (timer.is_valid()) {
		timer->set_time_left(duration);
	} else {
		timer = SCENE_TREE()->create_timer(duration, process_pause);
		ERR_FAIL_NULL(timer);
		timer->connect(LW_NAME(timeout), callable_mp(this, &BTCooldown::_on_timeout), CONNECT_ONE_SHOT);
	}
}

void BTCooldown::_on_timeout() {
	get_blackboard()->set_var(cooldown_state_var, false);
	timer.unref();
}

//**** Godot

void BTCooldown::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_duration", "duration"), &BTCooldown::set_duration);
	ClassDB::bind_method(D_METHOD("get_duration"), &BTCooldown::get_duration);
	ClassDB::bind_method(D_METHOD("set_process_pause", "enable"), &BTCooldown::set_process_pause);
	ClassDB::bind_method(D_METHOD("get_process_pause"), &BTCooldown::get_process_pause);
	ClassDB::bind_method(D_METHOD("set_start_cooled", "enable"), &BTCooldown::set_start_cooled);
	ClassDB::bind_method(D_METHOD("get_start_cooled"), &BTCooldown::get_start_cooled);
	ClassDB::bind_method(D_METHOD("set_trigger_on_failure", "enable"), &BTCooldown::set_trigger_on_failure);
	ClassDB::bind_method(D_METHOD("get_trigger_on_failure"), &BTCooldown::get_trigger_on_failure);
	ClassDB::bind_method(D_METHOD("set_cooldown_state_var", "variable"), &BTCooldown::set_cooldown_state_var);
	ClassDB::bind_method(D_METHOD("get_cooldown_state_var"), &BTCooldown::get_cooldown_state_var);
	ClassDB::bind_method(D_METHOD("_on_timeout"), &BTCooldown::_on_timeout);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "duration"), "set_duration", "get_duration");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "process_pause"), "set_process_pause", "get_process_pause");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "start_cooled"), "set_start_cooled", "get_start_cooled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "trigger_on_failure"), "set_trigger_on_failure", "get_trigger_on_failure");
	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "cooldown_state_var"), "set_cooldown_state_var", "get_cooldown_state_var");
}
