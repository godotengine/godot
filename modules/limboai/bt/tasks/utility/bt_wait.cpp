/**
 * bt_wait.cpp
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#include "bt_wait.h"

String BTWait::_generate_name() {
	return vformat("Wait %s sec", Math::snapped(duration, 0.001));
}

BT::Status BTWait::_tick(double p_delta) {
	if (get_elapsed_time() < duration) {
		return RUNNING;
	} else {
		return SUCCESS;
	}
}

void BTWait::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_duration", "duration_sec"), &BTWait::set_duration);
	ClassDB::bind_method(D_METHOD("get_duration"), &BTWait::get_duration);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "duration"), "set_duration", "get_duration");
}
