/**
 * bt_player.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef BT_PLAYER_H
#define BT_PLAYER_H

#include "../blackboard/blackboard.h"
#include "../blackboard/blackboard_plan.h"
#include "behavior_tree.h"
#include "bt_instance.h"
#include "tasks/bt_task.h"

#ifdef LIMBOAI_MODULE
#include "scene/main/node.h"
#endif

#ifdef LIMBOAI_GDEXTENSION
#include <godot_cpp/classes/node.hpp>
#endif

class BTPlayer : public Node {
	GDCLASS(BTPlayer, Node);

public:
	enum UpdateMode : unsigned int {
		IDLE, // automatically call update() during NOTIFICATION_PROCESS
		PHYSICS, // automatically call update() during NOTIFICATION_PHYSICS
		MANUAL, // manually update state machine, user must call update(delta)
	};

private:
	Ref<BehaviorTree> behavior_tree;
	NodePath agent_node;
	Ref<BlackboardPlan> blackboard_plan;
	UpdateMode update_mode = UpdateMode::PHYSICS;
	bool active = true;
	Ref<Blackboard> blackboard;
	Node *scene_root_hint = nullptr;
	bool monitor_performance = false;

	Ref<BTInstance> bt_instance;

	void _load_tree();
	void _update_blackboard_plan();

protected:
	static void _bind_methods();

	void _notification(int p_notification);

public:
	void set_behavior_tree(const Ref<BehaviorTree> &p_tree);
	Ref<BehaviorTree> get_behavior_tree() const { return behavior_tree; };

	void set_agent_node(const NodePath &p_agent_node);
	NodePath get_agent_node() const { return agent_node; }

	void set_blackboard_plan(const Ref<BlackboardPlan> &p_plan);
	Ref<BlackboardPlan> get_blackboard_plan() const { return blackboard_plan; }

	void set_update_mode(UpdateMode p_mode);
	UpdateMode get_update_mode() const { return update_mode; }

	void set_active(bool p_active);
	bool get_active() const { return active; }

	Ref<Blackboard> get_blackboard() const { return blackboard; }
	void set_blackboard(const Ref<Blackboard> &p_blackboard) { blackboard = p_blackboard; }

	void set_monitor_performance(bool p_monitor_performance);
	bool get_monitor_performance() const { return monitor_performance; }

	void update(double p_delta);
	void restart();

	Ref<BTInstance> get_bt_instance() { return bt_instance; }
	void set_bt_instance(const Ref<BTInstance> &p_bt_instance);

	void set_scene_root_hint(Node *p_scene_root);

	BTPlayer();
	~BTPlayer();
};

#endif // BT_PLAYER_H
