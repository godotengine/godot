/**
 * bt_sequence.cpp
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#include "bt_sequence.h"

void BTSequence::_enter() {
	last_running_idx = 0;
}

BT::Status BTSequence::_tick(double p_delta) {
	Status status = SUCCESS;
	for (int i = last_running_idx; i < get_child_count(); i++) {
		status = get_child(i)->execute(p_delta);
		if (status != SUCCESS) {
			last_running_idx = i;
			break;
		}
	}
	return status;
}
