/**
 * bt_random_wait.cpp
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#include "bt_random_wait.h"

#include "../../../util/limbo_compat.h"

String BTRandomWait::_generate_name() {
	return vformat("Wait %s to %s sec",
			Math::snapped(min_duration, 0.001),
			Math::snapped(max_duration, 0.001));
}

void BTRandomWait::_enter() {
	duration = RAND_RANGE(min_duration, max_duration);
}

BT::Status BTRandomWait::_tick(double p_delta) {
	if (get_elapsed_time() < duration) {
		return RUNNING;
	} else {
		return SUCCESS;
	}
}

void BTRandomWait::set_min_duration(double p_max_duration) {
	min_duration = p_max_duration;
	if (max_duration < min_duration) {
		set_max_duration(min_duration);
	}
	emit_changed();
}

void BTRandomWait::set_max_duration(double p_max_duration) {
	max_duration = p_max_duration;
	if (min_duration > max_duration) {
		set_min_duration(max_duration);
	}
	emit_changed();
}

void BTRandomWait::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_min_duration", "duration_sec"), &BTRandomWait::set_min_duration);
	ClassDB::bind_method(D_METHOD("get_min_duration"), &BTRandomWait::get_min_duration);
	ClassDB::bind_method(D_METHOD("set_max_duration", "duration_sec"), &BTRandomWait::set_max_duration);
	ClassDB::bind_method(D_METHOD("get_max_duration"), &BTRandomWait::get_max_duration);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "min_duration"), "set_min_duration", "get_min_duration");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "max_duration"), "set_max_duration", "get_max_duration");
}
