/**
 * bt_repeat_until_failure.cpp
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#include "bt_repeat_until_failure.h"

BT::Status BTRepeatUntilFailure::_tick(double p_delta) {
	ERR_FAIL_COND_V_MSG(get_child_count() == 0, FAILURE, "BT decorator has no child.");
	if (get_child(0)->execute(p_delta) == FAILURE) {
		return SUCCESS;
	}
	return RUNNING;
}
