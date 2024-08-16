/**
 * bt_state.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
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
	Ref<BTInstance> bt_instance;
	StringName success_event;
	StringName failure_event;
	Node *scene_root_hint = nullptr;
	bool monitor_performance = false;

protected:
	static void _bind_methods();

	void _notification(int p_notification);

	virtual bool _should_use_new_scope() const override { return true; }
	virtual void _update_blackboard_plan() override;

	virtual void _setup() override;
	virtual void _exit() override;
	virtual void _update(double p_delta) override;

public:
	void set_behavior_tree(const Ref<BehaviorTree> &p_value);
	Ref<BehaviorTree> get_behavior_tree() const { return behavior_tree; }

	Ref<BTInstance> get_bt_instance() const { return bt_instance; }

	void set_success_event(const StringName &p_success_event) { success_event = p_success_event; }
	StringName get_success_event() const { return success_event; }

	void set_failure_event(const StringName &p_failure_event) { failure_event = p_failure_event; }
	StringName get_failure_event() const { return failure_event; }

	void set_monitor_performance(bool p_monitor);
	bool get_monitor_performance() const { return monitor_performance; }

	void set_scene_root_hint(Node *p_node);

	BTState();
};

#endif // BT_STATE_H
