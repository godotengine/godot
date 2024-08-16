/**
 * behavior_tree_data.cpp
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
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

Array BehaviorTreeData::serialize(const Ref<BTInstance> &p_instance) {
	Array arr;
	arr.push_back(uint64_t(p_instance->get_instance_id()));
	arr.push_back(p_instance->get_owner_node() ? p_instance->get_owner_node()->get_path() : NodePath());
	arr.push_back(p_instance->get_source_bt_path());

	// Flatten tree into list depth first
	List<Ref<BTTask>> stack;
	stack.push_back(p_instance->get_root_task());
	while (stack.size()) {
		Ref<BTTask> task = stack.front()->get();
		stack.pop_front();

		int num_children = task->get_child_count();
		for (int i = 0; i < num_children; i++) {
			stack.push_front(task->get_child(num_children - 1 - i));
		}

		String script_path;
		if (task->get_script()) {
			Ref<Resource> s = task->get_script();
			script_path = s->get_path();
		}

		arr.push_back(task->get_instance_id());
		arr.push_back(task->get_task_name());
		arr.push_back(!task->get_custom_name().is_empty());
		arr.push_back(num_children);
		arr.push_back(task->get_status());
		arr.push_back(task->get_elapsed_time());
		arr.push_back(task->get_class());
		arr.push_back(script_path);
	}

	return arr;
}

Ref<BehaviorTreeData> BehaviorTreeData::deserialize(const Array &p_array) {
	ERR_FAIL_COND_V(p_array.size() < 3, nullptr);
	ERR_FAIL_COND_V(p_array[0].get_type() != Variant::INT, nullptr);
	ERR_FAIL_COND_V(p_array[1].get_type() != Variant::NODE_PATH, nullptr);
	ERR_FAIL_COND_V(p_array[2].get_type() != Variant::STRING, nullptr);

	Ref<BehaviorTreeData> data = memnew(BehaviorTreeData);
	data->bt_instance_id = uint64_t(p_array[0]);
	data->node_owner_path = p_array[1];
	data->source_bt_path = p_array[2];

	int idx = 3;
	while (p_array.size() > idx + 1) {
		ERR_FAIL_COND_V(p_array.size() < idx + 7, nullptr);
		ERR_FAIL_COND_V(p_array[idx].get_type() != Variant::INT, nullptr);
		ERR_FAIL_COND_V(p_array[idx + 1].get_type() != Variant::STRING, nullptr);
		ERR_FAIL_COND_V(p_array[idx + 2].get_type() != Variant::BOOL, nullptr);
		ERR_FAIL_COND_V(p_array[idx + 3].get_type() != Variant::INT, nullptr);
		ERR_FAIL_COND_V(p_array[idx + 4].get_type() != Variant::INT, nullptr);
		ERR_FAIL_COND_V(p_array[idx + 5].get_type() != Variant::FLOAT, nullptr);
		ERR_FAIL_COND_V(p_array[idx + 6].get_type() != Variant::STRING, nullptr);
		ERR_FAIL_COND_V(p_array[idx + 7].get_type() != Variant::STRING, nullptr);
		data->tasks.push_back(TaskData(p_array[idx], p_array[idx + 1], p_array[idx + 2], p_array[idx + 3], p_array[idx + 4], p_array[idx + 5], p_array[idx + 6], p_array[idx + 7]));
		idx += 8;
	}

	return data;
}

Ref<BehaviorTreeData> BehaviorTreeData::create_from_bt_instance(const Ref<BTInstance> &p_bt_instance) {
	Ref<BehaviorTreeData> data = memnew(BehaviorTreeData);

	data->bt_instance_id = p_bt_instance->get_instance_id();
	data->node_owner_path = p_bt_instance->get_owner_node() ? p_bt_instance->get_owner_node()->get_path() : NodePath();
	data->source_bt_path = p_bt_instance->get_source_bt_path();

	// Flatten tree into list depth first
	List<Ref<BTTask>> stack;
	stack.push_back(p_bt_instance->get_root_task());
	while (stack.size()) {
		Ref<BTTask> task = stack.front()->get();
		stack.pop_front();

		int num_children = task->get_child_count();
		for (int i = 0; i < num_children; i++) {
			stack.push_front(task->get_child(num_children - 1 - i));
		}

		String script_path;
		if (task->get_script()) {
			Ref<Resource> s = task->get_script();
			script_path = s->get_path();
		}

		data->tasks.push_back(TaskData(
				task->get_instance_id(),
				task->get_task_name(),
				!task->get_custom_name().is_empty(),
				num_children,
				task->get_status(),
				task->get_elapsed_time(),
				task->get_class(),
				script_path));
	}
	return data;
}

void BehaviorTreeData::_bind_methods() {
	ClassDB::bind_static_method("BehaviorTreeData", D_METHOD("create_from_bt_instance", "bt_instance"), &BehaviorTreeData::create_from_bt_instance);
}

BehaviorTreeData::BehaviorTreeData() {
}
