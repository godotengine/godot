/**
 * bt_parallel.cpp
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#include "bt_parallel.h"

void BTParallel::_enter() {
	for (int i = 0; i < get_child_count(); i++) {
		get_child(i)->abort();
	}
}

BT::Status BTParallel::_tick(double p_delta) {
	int num_succeeded = 0;
	int num_failed = 0;
	BT::Status return_status = RUNNING;
	for (int i = 0; i < get_child_count(); i++) {
		Status status = BT::FRESH;
		Ref<BTTask> child = get_child(i);
		if (!repeat && (child->get_status() == FAILURE || child->get_status() == SUCCESS)) {
			status = child->get_status();
		} else {
			status = child->execute(p_delta);
		}
		if (status == FAILURE) {
			num_failed += 1;
			if (num_failed >= num_failures_required && return_status == RUNNING) {
				return_status = FAILURE;
			}
		} else if (status == SUCCESS) {
			num_succeeded += 1;
			if (num_succeeded >= num_successes_required && return_status == RUNNING) {
				return_status = SUCCESS;
			}
		}
	}
	if (!repeat && (num_failed + num_succeeded) == get_child_count() && return_status == RUNNING) {
		return_status = FAILURE;
	}
	return return_status;
}

void BTParallel::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_num_successes_required"), &BTParallel::get_num_successes_required);
	ClassDB::bind_method(D_METHOD("set_num_successes_required", "value"), &BTParallel::set_num_successes_required);
	ClassDB::bind_method(D_METHOD("get_num_failures_required"), &BTParallel::get_num_failures_required);
	ClassDB::bind_method(D_METHOD("set_num_failures_required", "value"), &BTParallel::set_num_failures_required);
	ClassDB::bind_method(D_METHOD("get_repeat"), &BTParallel::get_repeat);
	ClassDB::bind_method(D_METHOD("set_repeat", "enable"), &BTParallel::set_repeat);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "num_successes_required"), "set_num_successes_required", "get_num_successes_required");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "num_failures_required"), "set_num_failures_required", "get_num_failures_required");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "repeat"), "set_repeat", "get_repeat");
}
