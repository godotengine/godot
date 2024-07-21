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

#include "../../bt/tasks/bt_task.h"
#include "../../util/limbo_compat.h"
#include "behavior_tree_data.h"

#ifdef LIMBOAI_MODULE
#include "core/debugger/engine_debugger.h"
#include "core/error/error_macros.h"
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
LimboDebugger *LimboDebugger::get_singleton() {
	return singleton;
}

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
#ifdef DEBUG_ENABLED

#ifdef LIMBOAI_GDEXTENSION
	ClassDB::bind_method(D_METHOD("parse_message_gdext"), &LimboDebugger::parse_message_gdext);
#endif
	ClassDB::bind_method(D_METHOD("_on_bt_updated", "status", "path"), &LimboDebugger::_on_bt_updated);
	ClassDB::bind_method(D_METHOD("_on_state_updated", "delta", "path"), &LimboDebugger::_on_state_updated);
#endif // ! DEBUG_ENABLED
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

void LimboDebugger::register_bt_instance(Ref<BTTask> p_instance, NodePath p_player_path) {
	ERR_FAIL_COND(p_instance.is_null());
	ERR_FAIL_COND(p_player_path.is_empty());
	if (active_trees.has(p_player_path)) {
		return;
	}

	active_trees.insert(p_player_path, p_instance);
	if (session_active) {
		_send_active_bt_players();
	}
}

void LimboDebugger::unregister_bt_instance(Ref<BTTask> p_instance, NodePath p_player_path) {
	ERR_FAIL_COND(p_instance.is_null());
	ERR_FAIL_COND(p_player_path.is_empty());
	ERR_FAIL_COND(!active_trees.has(p_player_path));

	if (tracked_player == p_player_path) {
		_untrack_tree();
	}
	active_trees.erase(p_player_path);

	if (session_active) {
		_send_active_bt_players();
	}
}

void LimboDebugger::_track_tree(NodePath p_path) {
	ERR_FAIL_COND(!active_trees.has(p_path));

	if (!tracked_player.is_empty()) {
		_untrack_tree();
	}

	Node *node = SCENE_TREE()->get_root()->get_node_or_null(p_path);
	ERR_FAIL_COND(node == nullptr);

	tracked_player = p_path;

	Ref<Resource> bt = node->get(LW_NAME(behavior_tree));

	if (bt.is_valid()) {
		bt_resource_path = bt->get_path();
	} else {
		bt_resource_path = "";
	}

	if (node->is_class("BTPlayer")) {
		node->connect(LW_NAME(updated), callable_mp(this, &LimboDebugger::_on_bt_updated).bind(p_path));
	} else if (node->is_class("BTState")) {
		node->connect(LW_NAME(updated), callable_mp(this, &LimboDebugger::_on_state_updated).bind(p_path));
	}
}

void LimboDebugger::_untrack_tree() {
	if (tracked_player.is_empty()) {
		return;
	}

	NodePath was_tracking = tracked_player;
	tracked_player = NodePath();

	Node *node = SCENE_TREE()->get_root()->get_node_or_null(was_tracking);
	ERR_FAIL_COND(node == nullptr);

	if (node->is_class("BTPlayer")) {
		node->disconnect(LW_NAME(updated), callable_mp(this, &LimboDebugger::_on_bt_updated));
	} else if (node->is_class("BTState")) {
		node->disconnect(LW_NAME(updated), callable_mp(this, &LimboDebugger::_on_state_updated));
	}
}

void LimboDebugger::_send_active_bt_players() {
	Array arr;
	for (KeyValue<NodePath, Ref<BTTask>> kv : active_trees) {
		arr.append(kv.key);
	}
	EngineDebugger::get_singleton()->send_message("limboai:active_bt_players", arr);
}

void LimboDebugger::_on_bt_updated(int _status, NodePath p_path) {
	if (p_path != tracked_player) {
		return;
	}
	Array arr = BehaviorTreeData::serialize(active_trees.get(tracked_player), tracked_player, bt_resource_path);
	EngineDebugger::get_singleton()->send_message("limboai:bt_update", arr);
}

void LimboDebugger::_on_state_updated(float _delta, NodePath p_path) {
	if (p_path != tracked_player) {
		return;
	}
	Array arr = BehaviorTreeData::serialize(active_trees.get(tracked_player), tracked_player, bt_resource_path);
	EngineDebugger::get_singleton()->send_message("limboai:bt_update", arr);
}

#endif // ! DEBUG_ENABLED
