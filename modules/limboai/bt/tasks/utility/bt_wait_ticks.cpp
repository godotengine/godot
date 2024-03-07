/**
 * bt_wait_ticks.cpp
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#include "bt_wait_ticks.h"

String BTWaitTicks::_generate_name() {
	return vformat("WaitTicks x%d", num_ticks);
}

void BTWaitTicks::_enter() {
	num_passed = 0;
}

BT::Status BTWaitTicks::_tick(double p_delta) {
	if (num_passed < num_ticks) {
		num_passed += 1;
		return RUNNING;
	} else {
		return SUCCESS;
	}
}

void BTWaitTicks::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_num_ticks", "num_ticks"), &BTWaitTicks::set_num_ticks);
	ClassDB::bind_method(D_METHOD("get_num_ticks"), &BTWaitTicks::get_num_ticks);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "num_ticks"), "set_num_ticks", "get_num_ticks");
}
