/*************************************************************************/
/*  character_net_controller.cpp                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "scene_rewinder.h"

void SceneRewinder::_bind_methods() {

	ClassDB::bind_method(D_METHOD("register_sync_variable", "node", "variable", "on_change_notify"), &SceneRewinder::register_sync_variable, DEFVAL(StringName()));
	ClassDB::bind_method(D_METHOD("unregister_sync_variable", "node", "variable"), &SceneRewinder::unregister_sync_variable);

	ClassDB::bind_method(D_METHOD("get_changed_event_name", "variable"), &SceneRewinder::get_changed_event_name);

	ClassDB::bind_method(D_METHOD("track_variable_changes", "node", "variable", "method"), &SceneRewinder::track_variable_changes);
	ClassDB::bind_method(D_METHOD("untrack_variable_changes", "node", "variable", "method"), &SceneRewinder::untrack_variable_changes);

	ClassDB::bind_method(D_METHOD("register_sync_action", "node", "action"), &SceneRewinder::register_sync_action);
	ClassDB::bind_method(D_METHOD("unregister_sync_action", "node", "action"), &SceneRewinder::unregister_sync_action);
	ClassDB::bind_method(D_METHOD("trigger_action", "node", "action"), &SceneRewinder::trigger_action);

	ClassDB::bind_method(D_METHOD("register_sync_process", "node", "func_name"), &SceneRewinder::register_sync_process);
	ClassDB::bind_method(D_METHOD("unregister_sync_process", "node", "func_name"), &SceneRewinder::unregister_sync_process);
}

void SceneRewinder::_notification(int p_what) {
}

SceneRewinder::SceneRewinder() {
}

void SceneRewinder::register_sync_variable(Node *p_node, StringName p_variable, StringName p_on_change_notify) {
}

void SceneRewinder::unregister_sync_variable(Node *p_node, StringName p_variable) {
}

StringName SceneRewinder::get_changed_event_name(StringName p_variable) {
	return "variable_" + p_variable + "_changed";
}

void SceneRewinder::track_variable_changes(Node *p_node, StringName p_variable, StringName p_method) {
}

void SceneRewinder::untrack_variable_changes(Node *p_node, StringName p_variable, StringName p_method) {
}

void SceneRewinder::register_sync_action(Node *p_node, StringName p_action) {
}

void SceneRewinder::unregister_sync_action(Node *p_node, StringName p_action) {
}

void SceneRewinder::trigger_action(Node *p_node, StringName p_action) {
}

void SceneRewinder::register_sync_process(Node *p_node, StringName p_func_name) {
}

void SceneRewinder::unregister_sync_process(Node *p_node, StringName p_func_name) {
}
