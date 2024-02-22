/**
 * bt_state.h
 * =============================================================================
 * Copyright 2021-2023 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef BT_STATE_H
#define BT_STATE_H

#include "../hsm/limbo_state.h"

#include "../bt/behavior_tree.h"
#include "../bt/tasks/bt_task.h"

class BTState : public LimboState {
	GDCLASS(BTState, LimboState);

private:
	Ref<BehaviorTree> behavior_tree;
	Ref<BTTask> tree_instance;
	String success_event;
	String failure_event;

	void _update_blackboard_plan();

protected:
	static void _bind_methods();

	virtual void _setup() override;
	virtual void _exit() override;
	virtual void _update(double p_delta) override;

public:
	void set_behavior_tree(const Ref<BehaviorTree> &p_value);
	Ref<BehaviorTree> get_behavior_tree() const { return behavior_tree; }

	Ref<BTTask> get_tree_instance() const { return tree_instance; }

	void set_success_event(String p_success_event) { success_event = p_success_event; }
	String get_success_event() const { return success_event; }

	void set_failure_event(String p_failure_event) { failure_event = p_failure_event; }
	String get_failure_event() const { return failure_event; }

	BTState();

protected:
	void _notification(int p_notification);
};

#endif // BT_STATE_H
