/**
 * bt_invert.cpp
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#include "bt_invert.h"

BT::Status BTInvert::_tick(double p_delta) {
	ERR_FAIL_COND_V_MSG(get_child_count() == 0, FAILURE, "BT decorator has no child.");
	Status status = get_child(0)->execute(p_delta);
	if (status == SUCCESS) {
		status = FAILURE;
	} else if (status == FAILURE) {
		status = SUCCESS;
	}
	return status;
}
