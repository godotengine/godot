/**
 * bt_subtree.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef BT_SUBTREE_H
#define BT_SUBTREE_H

#include "bt_new_scope.h"

#include "../../../bt/behavior_tree.h"

class BTSubtree : public BTNewScope {
	GDCLASS(BTSubtree, BTNewScope);
	TASK_CATEGORY(Decorators);

private:
	Ref<BehaviorTree> subtree;

protected:
	static void _bind_methods();

	virtual void _update_blackboard_plan() override;

	virtual String _generate_name() override;
	virtual Status _tick(double p_delta) override;

public:
	void set_subtree(const Ref<BehaviorTree> &p_value);
	Ref<BehaviorTree> get_subtree() const { return subtree; }

	virtual void initialize(Node *p_agent, const Ref<Blackboard> &p_blackboard, Node *p_scene_root) override;
	virtual PackedStringArray get_configuration_warnings() override;

	BTSubtree() = default;
	~BTSubtree();
};

#endif // BT_SUBTREE_H
