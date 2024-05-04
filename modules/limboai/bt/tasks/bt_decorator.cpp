/**
 * bt_decorator.cpp
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#include "bt_decorator.h"

PackedStringArray BTDecorator::get_configuration_warnings() {
	PackedStringArray warnings = BTTask::get_configuration_warnings();
	if (get_child_count_excluding_comments() != 1) {
		warnings.append("Decorator should have a single child task.");
	}
	return warnings;
}

BT::Status BTDecorator::_tick(double p_delta) {
	ERR_FAIL_COND_V_MSG(get_child_count() == 0, FAILURE, "BT decorator doesn't have a child.");
	return get_child(0)->execute(p_delta);
}
