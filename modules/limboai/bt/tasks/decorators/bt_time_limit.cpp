/**
 * bt_time_limit.cpp
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#include "bt_time_limit.h"

void BTTimeLimit::set_time_limit(double p_value) {
	time_limit = p_value;
	emit_changed();
}

String BTTimeLimit::_generate_name() {
	return vformat("TimeLimit %s sec", Math::snapped(time_limit, 0.001));
}

BT::Status BTTimeLimit::_tick(double p_delta) {
	ERR_FAIL_COND_V_MSG(get_child_count() == 0, FAILURE, "BT decorator has no child.");
	Status status = get_child(0)->execute(p_delta);
	if (status == RUNNING && get_elapsed_time() >= time_limit) {
		get_child(0)->abort();
		return FAILURE;
	}
	return status;
}

void BTTimeLimit::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_time_limit", "duration_sec"), &BTTimeLimit::set_time_limit);
	ClassDB::bind_method(D_METHOD("get_time_limit"), &BTTimeLimit::get_time_limit);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "time_limit"), "set_time_limit", "get_time_limit");
}
