/**
 * bt_random_selector.cpp
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#include "bt_random_selector.h"

void BTRandomSelector::_enter() {
	last_running_idx = 0;
	if (indicies.size() != get_child_count()) {
		indicies.resize(get_child_count());
		for (int i = 0; i < get_child_count(); i++) {
			indicies[i] = i;
		}
	}
	indicies.shuffle();
}

BT::Status BTRandomSelector::_tick(double p_delta) {
	Status status = FAILURE;
	for (int i = last_running_idx; i < get_child_count(); i++) {
		status = get_child(indicies[i])->execute(p_delta);
		if (status != FAILURE) {
			last_running_idx = i;
			break;
		}
	}
	return status;
}
