/**
 * behavior_tree.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef BEHAVIOR_TREE_H
#define BEHAVIOR_TREE_H

#include "../blackboard/blackboard_plan.h"
#include "tasks/bt_task.h"

#ifdef LIMBOAI_MODULE
#include "core/io/resource.h"
#endif // LIMBOAI_MODULE

#ifdef LIMBOAI_GDEXTENSION
#include <godot_cpp/classes/resource.hpp>
using namespace godot;
#endif // LIMBOAI_GDEXTENSION

class BehaviorTree : public Resource {
	GDCLASS(BehaviorTree, Resource);

private:
	String description;
	Ref<BlackboardPlan> blackboard_plan;
	Ref<BTTask> root_task;

	void _plan_changed();

#ifdef TOOLS_ENABLED
	void _set_editor_behavior_tree_hint();
	void _unset_editor_behavior_tree_hint();
#endif // TOOLS_ENABLED

protected:
	static void _bind_methods();

public:
#ifdef LIMBOAI_MODULE
	virtual bool editor_can_reload_from_file() override { return false; }
#endif

	void set_description(const String &p_value);
	String get_description() const { return description; }

	void set_blackboard_plan(const Ref<BlackboardPlan> &p_plan);
	Ref<BlackboardPlan> get_blackboard_plan() const { return blackboard_plan; }

	void set_root_task(const Ref<BTTask> &p_value);
	Ref<BTTask> get_root_task() const { return root_task; }

	Ref<BehaviorTree> clone() const;
	void copy_other(const Ref<BehaviorTree> &p_other);
	Ref<BTTask> instantiate(Node *p_agent, const Ref<Blackboard> &p_blackboard, Node *p_scene_root) const;

	BehaviorTree();
	~BehaviorTree();
};

#endif // BEHAVIOR_TREE_H
