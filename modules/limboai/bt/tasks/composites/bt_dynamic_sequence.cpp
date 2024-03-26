/**
 * bt_dynamic_sequence.cpp
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#include "bt_dynamic_sequence.h"

void BTDynamicSequence::_enter() {
	last_running_idx = 0;
}

BT::Status BTDynamicSequence::_tick(double p_delta) {
	Status status = SUCCESS;
	int i;
	for (i = 0; i < get_child_count(); i++) {
		status = get_child(i)->execute(p_delta);
		if (status != SUCCESS) {
			break;
		}
	}
	// If the last node ticked is earlier in the tree than the previous runner,
	// cancel previous runner.
	if (last_running_idx > i && get_child(last_running_idx)->get_status() == RUNNING) {
		get_child(last_running_idx)->abort();
	}
	last_running_idx = i;
	return status;
}
