/**
 * limbo_test.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef LIMBO_TEST_H
#define LIMBO_TEST_H

#define LIMBOAI_MODULE

#include "core/object/ref_counted.h"
#include "tests/test_macros.h"

#include "modules/limboai/bt/tasks/bt_action.h"

class CallbackCounter : public RefCounted {
	GDCLASS(CallbackCounter, RefCounted);

public:
	int num_callbacks = 0;

	void callback() { num_callbacks += 1; }
	void callback_delta(double delta) { num_callbacks += 1; }

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("callback"), &CallbackCounter::callback);
		ClassDB::bind_method(D_METHOD("callback_delta", "delta"), &CallbackCounter::callback_delta);
	}
};

class BTTestAction : public BTAction {
	GDCLASS(BTTestAction, BTAction);

public:
	Status ret_status = BTTask::SUCCESS;
	int num_entries = 0;
	int num_ticks = 0;
	int num_exits = 0;

protected:
	virtual void _enter() override { num_entries += 1; }
	virtual void _exit() override { num_exits += 1; }

	virtual Status _tick(double p_delta) override {
		num_ticks += 1;
		return ret_status;
	}

public:
	bool is_status_either(Status p_status1, Status p_status2) { return (get_status() == p_status1 || get_status() == p_status2); }

	BTTestAction(Status p_return_status) { ret_status = p_return_status; }
	BTTestAction() {}
};

#define CHECK_ENTRIES_TICKS_EXITS(m_task, m_entries, m_ticks, m_exits) \
	CHECK(m_task->num_entries == m_entries);                           \
	CHECK(m_task->num_ticks == m_ticks);                               \
	CHECK(m_task->num_exits == m_exits);

#define CHECK_ENTRIES_TICKS_EXITS_UP_TO(m_task, m_entries, m_ticks, m_exits) \
	CHECK(m_task->num_entries <= m_entries);                                 \
	CHECK(m_task->num_ticks <= m_ticks);                                     \
	CHECK(m_task->num_exits <= m_exits);

#define CHECK_STATUS_ENTRIES_TICKS_EXITS(m_task, m_status, m_entries, m_ticks, m_exits) \
	CHECK(m_task->get_status() == m_status);                                            \
	CHECK(m_task->num_entries == m_entries);                                            \
	CHECK(m_task->num_ticks == m_ticks);                                                \
	CHECK(m_task->num_exits == m_exits);

#endif // LIMBO_TEST_H
