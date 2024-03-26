/**
 * bt_always_succeed.cpp
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#include "bt_always_succeed.h"

BT::Status BTAlwaysSucceed::_tick(double p_delta) {
	if (get_child_count() > 0 && get_child(0)->execute(p_delta) == RUNNING) {
		return RUNNING;
	}
	return SUCCESS;
}
