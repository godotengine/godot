/**
 * limbo_debugger.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef LIMBO_DEBUGGER_H
#define LIMBO_DEBUGGER_H

#include "../../bt/tasks/bt_task.h"

#ifdef LIMBOAI_MODULE
#include "core/object/class_db.h"
#include "core/object/object.h"
#include "core/string/node_path.h"
#endif // LIMBOAI_MODULE

#ifdef LIMBOAI_GDEXTENSION
#include <godot_cpp/classes/object.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/node_path.hpp>
#endif // LIMBOAI_GDEXTENSION

class LimboDebugger : public Object {
	GDCLASS(LimboDebugger, Object);

private:
	static LimboDebugger *singleton;

	LimboDebugger();

public:
	static void initialize();
	static void deinitialize();
	static LimboDebugger *get_singleton();

	~LimboDebugger();

protected:
	static void _bind_methods();

#ifdef DEBUG_ENABLED
private:
	HashMap<NodePath, Ref<BTTask>> active_trees;
	NodePath tracked_player;
	String bt_resource_path;
	bool session_active = false;

	void _track_tree(NodePath p_path);
	void _untrack_tree();
	void _send_active_bt_players();

	void _on_bt_updated(int status, NodePath p_path);
	void _on_state_updated(float _delta, NodePath p_path);

public:
	static Error parse_message(void *p_user, const String &p_msg, const Array &p_args, bool &r_captured);
#ifdef LIMBOAI_GDEXTENSION
	bool parse_message_gdext(const String &p_msg, const Array &p_args);
#endif

	void register_bt_instance(Ref<BTTask> p_instance, NodePath p_player_path);
	void unregister_bt_instance(Ref<BTTask> p_instance, NodePath p_player_path);

#endif // ! DEBUG_ENABLED
};
#endif // LIMBO_DEBUGGER_H
