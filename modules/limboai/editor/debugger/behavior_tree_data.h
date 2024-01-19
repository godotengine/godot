/**
 * behavior_tree_data.h
 * =============================================================================
 * Copyright 2021-2023 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef BEHAVIOR_TREE_DATA_H
#define BEHAVIOR_TREE_DATA_H

#include "../../bt/tasks/bt_task.h"

class BehaviorTreeData {
public:
	struct TaskData {
		int id = 0;
		String name;
		bool is_custom_name = false;
		int num_children = 0;
		int status = 0;
		double elapsed_time = 0.0;
		String type_name;
		String script_path;

		TaskData(int p_id, const String &p_name, bool p_is_custom_name, int p_num_children, int p_status, double p_elapsed_time, const String &p_type_name, const String &p_script_path) {
			id = p_id;
			name = p_name;
			is_custom_name = p_is_custom_name;
			num_children = p_num_children;
			status = p_status;
			elapsed_time = p_elapsed_time;
			type_name = p_type_name;
			script_path = p_script_path;
		}

		TaskData() {}
	};

	List<TaskData> tasks;
	NodePath bt_player_path;
	String bt_resource_path;

	void serialize(Array &p_arr);
	void deserialize(const Array &p_arr);

	BehaviorTreeData(const Ref<BTTask> &p_instance, const NodePath &p_player_path, const String &p_bt_resource);
	BehaviorTreeData() {}
};

#endif // BEHAVIOR_TREE_DATA_H
