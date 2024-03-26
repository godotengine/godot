/**
 * bt_action.cpp
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#include "bt_action.h"

PackedStringArray BTAction::get_configuration_warnings() {
	PackedStringArray warnings = BTTask::get_configuration_warnings();
	if (get_child_count_excluding_comments() != 0) {
		warnings.append("Action can't have child tasks.");
	}
	return warnings;
}
