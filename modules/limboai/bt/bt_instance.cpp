/**
 * bt_instance.cpp
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#include "bt_instance.h"

#include "../editor/debugger/limbo_debugger.h"
#include "behavior_tree.h"

#ifdef LIMBOAI_MODULE
#include "core/os/time.h"
#include "main/performance.h"
#endif

#ifdef LIMBOAI_GDEXTENSION
#include <godot_cpp/classes/performance.hpp>
#include <godot_cpp/classes/time.hpp>
#endif

Ref<BTInstance> BTInstance::create(Ref<BTTask> p_root_task, String p_source_bt_path, Node *p_owner_node) {
	ERR_FAIL_NULL_V(p_root_task, nullptr);
	ERR_FAIL_NULL_V(p_owner_node, nullptr);
	Ref<BTInstance> inst;
	inst.instantiate();
	inst->root_task = p_root_task;
	inst->owner_node_id = p_owner_node->get_instance_id();
	inst->source_bt_path = p_source_bt_path;
	return inst;
}

BT::Status BTInstance::update(double p_delta) {
	ERR_FAIL_COND_V(!root_task.is_valid(), BT::FRESH);

#ifdef DEBUG_ENABLED
	double start = Time::get_singleton()->get_ticks_usec();
#endif

	last_status = root_task->execute(p_delta);
	emit_signal(LimboStringNames::get_singleton()->updated, last_status);

#ifdef DEBUG_ENABLED
	double end = Time::get_singleton()->get_ticks_usec();
	update_time_acc += (end - start);
	update_time_n += 1.0;
#endif
	return last_status;
}

void BTInstance::set_monitor_performance(bool p_monitor) {
#ifdef DEBUG_ENABLED
	monitor_performance = p_monitor;
	if (monitor_performance) {
		_add_custom_monitor();
	} else {
		_remove_custom_monitor();
	}
#endif
}

bool BTInstance::get_monitor_performance() const {
#ifdef DEBUG_ENABLED
	return monitor_performance;
#else
	return false;
#endif
}

void BTInstance::register_with_debugger() {
#ifdef DEBUG_ENABLED
	if (LimboDebugger::get_singleton()->is_active()) {
		LimboDebugger::get_singleton()->register_bt_instance(get_instance_id());
	}
#endif
}

void BTInstance::unregister_with_debugger() {
#ifdef DEBUG_ENABLED
	if (LimboDebugger::get_singleton()->is_active()) {
		LimboDebugger::get_singleton()->unregister_bt_instance(get_instance_id());
	}
#endif
}

#ifdef DEBUG_ENABLED

double BTInstance::_get_mean_update_time_msec_and_reset() {
	if (update_time_n) {
		double mean_time_msec = (update_time_acc * 0.001) / update_time_n;
		update_time_acc = 0.0;
		update_time_n = 0.0;
		return mean_time_msec;
	}
	return 0.0;
}

void BTInstance::_add_custom_monitor() {
	ERR_FAIL_NULL(get_owner_node());
	ERR_FAIL_NULL(root_task);
	ERR_FAIL_NULL(root_task->get_agent());

	if (monitor_id == StringName()) {
		monitor_id = vformat("LimboAI/update_ms|%s_%s_%s", root_task->get_agent()->get_name(), get_owner_node()->get_name(),
				String(itos(get_instance_id())).md5_text().substr(0, 4));
	}
	if (!Performance::get_singleton()->has_custom_monitor(monitor_id)) {
		PERFORMANCE_ADD_CUSTOM_MONITOR(monitor_id, callable_mp(this, &BTInstance::_get_mean_update_time_msec_and_reset));
	}
}

void BTInstance::_remove_custom_monitor() {
	if (monitor_id != StringName() && Performance::get_singleton()->has_custom_monitor(monitor_id)) {
		Performance::get_singleton()->remove_custom_monitor(monitor_id);
	}
}

#endif // * DEBUG_ENABLED

void BTInstance::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_root_task"), &BTInstance::get_root_task);
	ClassDB::bind_method(D_METHOD("get_owner_node"), &BTInstance::get_owner_node);
	ClassDB::bind_method(D_METHOD("get_last_status"), &BTInstance::get_last_status);
	ClassDB::bind_method(D_METHOD("get_source_bt_path"), &BTInstance::get_source_bt_path);
	ClassDB::bind_method(D_METHOD("get_agent"), &BTInstance::get_agent);
	ClassDB::bind_method(D_METHOD("get_blackboard"), &BTInstance::get_blackboard);

	ClassDB::bind_method(D_METHOD("is_instance_valid"), &BTInstance::is_instance_valid);

	ClassDB::bind_method(D_METHOD("set_monitor_performance", "monitor"), &BTInstance::set_monitor_performance);
	ClassDB::bind_method(D_METHOD("get_monitor_performance"), &BTInstance::get_monitor_performance);

	ClassDB::bind_method(D_METHOD("update", "delta"), &BTInstance::update);

	ClassDB::bind_method(D_METHOD("register_with_debugger"), &BTInstance::register_with_debugger);
	ClassDB::bind_method(D_METHOD("unregister_with_debugger"), &BTInstance::unregister_with_debugger);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "monitor_performance"), "set_monitor_performance", "get_monitor_performance");

	ADD_SIGNAL(MethodInfo("updated", PropertyInfo(Variant::INT, "status")));
	ADD_SIGNAL(MethodInfo("freed"));
}

BTInstance::~BTInstance() {
	emit_signal(LW_NAME(freed));
#ifdef DEBUG_ENABLED
	_remove_custom_monitor();
#endif
}
