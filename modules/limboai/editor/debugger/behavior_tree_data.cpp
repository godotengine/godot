/**
 * behavior_tree_data.cpp
 * =============================================================================
 * Copyright 2021-2023 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#include "behavior_tree_data.h"

#ifdef LIMBOAI_MODULE
#include "core/templates/list.h"
#endif

//**** BehaviorTreeData

void BehaviorTreeData::serialize(Array &p_arr) {
	p_arr.push_back(bt_player_path);
	p_arr.push_back(bt_resource_path);
	for (const TaskData &td : tasks) {
		p_arr.push_back(td.id);
		p_arr.push_back(td.name);
		p_arr.push_back(td.is_custom_name);
		p_arr.push_back(td.num_children);
		p_arr.push_back(td.status);
		p_arr.push_back(td.elapsed_time);
		p_arr.push_back(td.type_name);
		p_arr.push_back(td.script_path);
	}
}

void BehaviorTreeData::deserialize(const Array &p_arr) {
	ERR_FAIL_COND(tasks.size() != 0);
	ERR_FAIL_COND(p_arr.size() < 2);

	ERR_FAIL_COND(p_arr[0].get_type() != Variant::NODE_PATH);
	bt_player_path = p_arr[0];

	ERR_FAIL_COND(p_arr[1].get_type() != Variant::STRING);
	bt_resource_path = p_arr[1];

	int idx = 2;
	while (p_arr.size() > idx + 1) {
		ERR_FAIL_COND(p_arr.size() < idx + 7);
		ERR_FAIL_COND(p_arr[idx].get_type() != Variant::INT);
		ERR_FAIL_COND(p_arr[idx + 1].get_type() != Variant::STRING);
		ERR_FAIL_COND(p_arr[idx + 2].get_type() != Variant::BOOL);
		ERR_FAIL_COND(p_arr[idx + 3].get_type() != Variant::INT);
		ERR_FAIL_COND(p_arr[idx + 4].get_type() != Variant::INT);
		ERR_FAIL_COND(p_arr[idx + 5].get_type() != Variant::FLOAT);
		ERR_FAIL_COND(p_arr[idx + 6].get_type() != Variant::STRING);
		ERR_FAIL_COND(p_arr[idx + 7].get_type() != Variant::STRING);
		tasks.push_back(TaskData(p_arr[idx], p_arr[idx + 1], p_arr[idx + 2], p_arr[idx + 3], p_arr[idx + 4], p_arr[idx + 5], p_arr[idx + 6], p_arr[idx + 7]));
		idx += 8;
	}
}

BehaviorTreeData::BehaviorTreeData(const Ref<BTTask> &p_instance, const NodePath &p_player_path, const String &p_bt_resource) {
	bt_player_path = p_player_path;
	bt_resource_path = p_bt_resource;

	// Flatten tree into list depth first
	List<Ref<BTTask>> stack;
	stack.push_back(p_instance);
	int id = 0;
	while (stack.size()) {
		Ref<BTTask> task = stack[0];
		stack.pop_front();

		int num_children = task->get_child_count();
		for (int i = 0; i < num_children; i++) {
			stack.push_front(task->get_child(num_children - 1 - i));
		}

		String script_path;
		if (task->get_script()) {
			Ref<Resource> script = task->get_script();
			script_path = script->get_path();
		}

		tasks.push_back(TaskData(
				id,
				task->get_task_name(),
				!task->get_custom_name().is_empty(),
				num_children,
				task->get_status(),
				task->get_elapsed_time(),
				task->get_class(),
				script_path));
		id += 1;
	}
}
