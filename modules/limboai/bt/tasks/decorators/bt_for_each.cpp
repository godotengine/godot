/**
 * bt_for_each.cpp
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#include "bt_for_each.h"

#include "../../../blackboard/blackboard.h"
#include "../../../util/limbo_utility.h"

#ifdef LIMBOAI_MODULE
#include "core/error/error_list.h"
#endif

//**** Setters / Getters

void BTForEach::set_array_var(const StringName &p_value) {
	array_var = p_value;
	emit_changed();
}

void BTForEach::set_save_var(const StringName &p_value) {
	save_var = p_value;
	emit_changed();
}

//**** Task Implementation

String BTForEach::_generate_name() {
	return vformat("ForEach %s in %s",
			LimboUtility::get_singleton()->decorate_var(save_var),
			LimboUtility::get_singleton()->decorate_var(array_var));
}

void BTForEach::_enter() {
	current_idx = 0;
}

BT::Status BTForEach::_tick(double p_delta) {
	ERR_FAIL_COND_V_MSG(get_child_count() == 0, FAILURE, "ForEach decorator has no child.");
	ERR_FAIL_COND_V_MSG(save_var == StringName(), FAILURE, "ForEach save variable is not set.");
	ERR_FAIL_COND_V_MSG(array_var == StringName(), FAILURE, "ForEach array variable is not set.");

	Array arr = get_blackboard()->get_var(array_var, Variant());
	if (arr.size() == 0) {
		return SUCCESS;
	}
	Variant elem = arr[current_idx];
	get_blackboard()->set_var(save_var, elem);

	Status status = get_child(0)->execute(p_delta);
	if (status == RUNNING) {
		return RUNNING;
	} else if (status == FAILURE) {
		return FAILURE;
	} else if (current_idx == (arr.size() - 1)) {
		return SUCCESS;
	} else {
		current_idx += 1;
		return RUNNING;
	}
}

//**** Godot

void BTForEach::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_array_var", "variable"), &BTForEach::set_array_var);
	ClassDB::bind_method(D_METHOD("get_array_var"), &BTForEach::get_array_var);
	ClassDB::bind_method(D_METHOD("set_save_var", "variable"), &BTForEach::set_save_var);
	ClassDB::bind_method(D_METHOD("get_save_var"), &BTForEach::get_save_var);

	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "array_var"), "set_array_var", "get_array_var");
	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "save_var"), "set_save_var", "get_save_var");
}
