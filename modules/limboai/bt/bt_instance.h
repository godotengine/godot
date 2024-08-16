/**
 * bt_instance.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */
#ifndef BT_INSTANCE_H
#define BT_INSTANCE_H

#include "tasks/bt_task.h"

class BTInstance : public RefCounted {
	GDCLASS(BTInstance, RefCounted);

private:
	Ref<BTTask> root_task;
	uint64_t owner_node_id = 0;
	String source_bt_path;
	BT::Status last_status = BT::FRESH;

#ifdef DEBUG_ENABLED
	bool monitor_performance = false;
	StringName monitor_id;
	double update_time_acc = 0.0;
	double update_time_n = 0.0;

	double _get_mean_update_time_msec_and_reset();
	void _add_custom_monitor();
	void _remove_custom_monitor();

#endif // * DEBUG_ENABLED

protected:
	static void _bind_methods();

public:
	_FORCE_INLINE_ Ref<BTTask> get_root_task() const { return root_task; }
	_FORCE_INLINE_ Node *get_owner_node() const { return owner_node_id ? Object::cast_to<Node>(ObjectDB::get_instance(ObjectID(owner_node_id))) : nullptr; }
	_FORCE_INLINE_ BT::Status get_last_status() const { return last_status; }
	_FORCE_INLINE_ String get_source_bt_path() const { return source_bt_path; }
	_FORCE_INLINE_ Node *get_agent() const { return root_task.is_valid() ? root_task->get_agent() : nullptr; }
	_FORCE_INLINE_ Ref<Blackboard> get_blackboard() const { return root_task.is_valid() ? root_task->get_blackboard() : Ref<Blackboard>(); }

	_FORCE_INLINE_ bool is_instance_valid() const { return root_task.is_valid(); }

	BT::Status update(double p_delta);

	void set_monitor_performance(bool p_monitor);
	bool get_monitor_performance() const;

	void register_with_debugger();
	void unregister_with_debugger();

	static Ref<BTInstance> create(Ref<BTTask> p_root_task, String p_source_bt_path, Node *p_owner_node);

	BTInstance() = default;
	~BTInstance();
};

#endif // BT_INSTANCE_H
