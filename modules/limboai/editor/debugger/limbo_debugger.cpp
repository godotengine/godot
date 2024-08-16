/**
 * limbo_debugger.cpp
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#include "limbo_debugger.h"

#include "../../bt/bt_instance.h"
#include "../../bt/tasks/bt_task.h"
#include "../../util/limbo_compat.h"
#include "behavior_tree_data.h"

#ifdef LIMBOAI_MODULE
#include "core/debugger/engine_debugger.h"
#include "core/io/resource.h"
#include "core/string/node_path.h"
#include "scene/main/scene_tree.h"
#include "scene/main/window.h"
#endif // LIMBOAI_MODULE

#ifdef LIMBOAI_GDEXTENSION
#include <godot_cpp/classes/engine_debugger.hpp>
#include <godot_cpp/classes/scene_tree.hpp>
#include <godot_cpp/classes/window.hpp>
#endif // LIMBOAI_GDEXTENSION

//**** LimboDebugger

LimboDebugger *LimboDebugger::singleton = nullptr;

LimboDebugger::LimboDebugger() {
	singleton = this;
#if defined(DEBUG_ENABLED) && defined(LIMBOAI_MODULE)
	EngineDebugger::register_message_capture("limboai", EngineDebugger::Capture(nullptr, LimboDebugger::parse_message));
#elif defined(DEBUG_ENABLED) && defined(LIMBOAI_GDEXTENSION)
	EngineDebugger::get_singleton()->register_message_capture("limboai", callable_mp(this, &LimboDebugger::parse_message_gdext));
#endif
}

LimboDebugger::~LimboDebugger() {
	singleton = nullptr;
}

void LimboDebugger::initialize() {
	if (IS_DEBUGGER_ACTIVE()) {
		memnew(LimboDebugger);
	}
}

void LimboDebugger::deinitialize() {
	if (singleton) {
		memdelete(singleton);
	}
}

void LimboDebugger::_bind_methods() {
}

#ifdef DEBUG_ENABLED
Error LimboDebugger::parse_message(void *p_user, const String &p_msg, const Array &p_args, bool &r_captured) {
	r_captured = true;
	if (p_msg == "track_bt_player") {
		singleton->_track_tree(p_args[0]);
	} else if (p_msg == "untrack_bt_player") {
		singleton->_untrack_tree();
	} else if (p_msg == "start_session") {
		singleton->session_active = true;
		singleton->_send_active_bt_players();
	} else if (p_msg == "stop_session") {
		singleton->session_active = false;
	} else {
		r_captured = false;
	}
	return OK;
}

#ifdef LIMBOAI_GDEXTENSION
bool LimboDebugger::parse_message_gdext(const String &p_msg, const Array &p_args) {
	bool r_captured;
	LimboDebugger::parse_message(nullptr, p_msg, p_args, r_captured);
	return r_captured;
}
#endif // LIMBOAI_GDEXTENSION

void LimboDebugger::register_bt_instance(uint64_t p_instance_id) {
	ERR_FAIL_COND(p_instance_id == 0);
	if (!IS_DEBUGGER_ACTIVE()) {
		return;
	}

	BTInstance *inst = Object::cast_to<BTInstance>(OBJECT_DB_GET_INSTANCE(p_instance_id));
	ERR_FAIL_NULL(inst);
	ERR_FAIL_COND(!inst->is_instance_valid());

	if (active_bt_instances.has(p_instance_id)) {
		return;
	}

	if (!inst->is_connected(LW_NAME(freed), callable_mp(this, &LimboDebugger::unregister_bt_instance).bind(p_instance_id))) {
		inst->connect(LW_NAME(freed), callable_mp(this, &LimboDebugger::unregister_bt_instance).bind(p_instance_id));
	}

	active_bt_instances.insert(p_instance_id);
	if (session_active) {
		_send_active_bt_players();
	}
}

void LimboDebugger::unregister_bt_instance(uint64_t p_instance_id) {
	if (!active_bt_instances.has(p_instance_id)) {
		return;
	}

	if (tracked_instance_id == p_instance_id) {
		_untrack_tree();
	}
	active_bt_instances.erase(p_instance_id);

	if (session_active) {
		_send_active_bt_players();
	}
}

bool LimboDebugger::is_active() const {
	return IS_DEBUGGER_ACTIVE();
}

void LimboDebugger::_track_tree(uint64_t p_instance_id) {
	ERR_FAIL_COND(p_instance_id == 0);
	ERR_FAIL_COND(!active_bt_instances.has(p_instance_id));

	_untrack_tree();

	tracked_instance_id = p_instance_id;

	BTInstance *inst = Object::cast_to<BTInstance>(OBJECT_DB_GET_INSTANCE(p_instance_id));
	ERR_FAIL_NULL(inst);
	inst->connect(LW_NAME(updated), callable_mp(this, &LimboDebugger::_on_bt_instance_updated).bind(p_instance_id));
}

void LimboDebugger::_untrack_tree() {
	if (tracked_instance_id == 0) {
		return;
	}

	BTInstance *inst = Object::cast_to<BTInstance>(OBJECT_DB_GET_INSTANCE(tracked_instance_id));
	if (inst) {
		inst->disconnect(LW_NAME(updated), callable_mp(this, &LimboDebugger::_on_bt_instance_updated));
	}
	tracked_instance_id = 0;
}

void LimboDebugger::_send_active_bt_players() {
	Array arr;
	for (uint64_t instance_id : active_bt_instances) {
		arr.append(instance_id);
		BTInstance *inst = Object::cast_to<BTInstance>(OBJECT_DB_GET_INSTANCE(instance_id));
		if (inst == nullptr) {
			ERR_PRINT("LimboDebugger::_send_active_bt_players: Registered BTInstance not found (no longer exists?).");
			continue;
		}
		Node *owner_node = inst->get_owner_node();
		arr.append(owner_node ? owner_node->get_path() : NodePath());
	}
	EngineDebugger::get_singleton()->send_message("limboai:active_bt_players", arr);
}

void LimboDebugger::_on_bt_instance_updated(int _status, uint64_t p_instance_id) {
	if (p_instance_id != tracked_instance_id) {
		return;
	}
	BTInstance *inst = Object::cast_to<BTInstance>(OBJECT_DB_GET_INSTANCE(p_instance_id));
	ERR_FAIL_NULL(inst);
	Array arr = BehaviorTreeData::serialize(inst);
	EngineDebugger::get_singleton()->send_message("limboai:bt_update", arr);
}

#endif // ! DEBUG_ENABLED
