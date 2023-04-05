/**************************************************************************/
/*  scene_tree.cpp                                                        */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "scene_tree.h"

#include "core/config/project_settings.h"
#include "core/debugger/engine_debugger.h"
#include "core/input/input.h"
#include "core/io/dir_access.h"
#include "core/io/image_loader.h"
#include "core/io/marshalls.h"
#include "core/io/resource_loader.h"
#include "core/object/message_queue.h"
#include "core/os/keyboard.h"
#include "core/os/os.h"
#include "core/string/print_string.h"
#include "node.h"
#include "scene/animation/tween.h"
#include "scene/debugger/scene_debugger.h"
#include "scene/gui/control.h"
#include "scene/main/multiplayer_api.h"
#include "scene/main/viewport.h"
#include "scene/resources/environment.h"
#include "scene/resources/font.h"
#include "scene/resources/material.h"
#include "scene/resources/mesh.h"
#include "scene/resources/packed_scene.h"
#include "scene/resources/world_2d.h"
#include "scene/resources/world_3d.h"
#include "scene/scene_string_names.h"
#include "servers/display_server.h"
#include "servers/navigation_server_3d.h"
#include "servers/physics_server_2d.h"
#include "servers/physics_server_3d.h"
#include "window.h"

#include <stdio.h>
#include <stdlib.h>

void SceneTreeTimer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_time_left", "time"), &SceneTreeTimer::set_time_left);
	ClassDB::bind_method(D_METHOD("get_time_left"), &SceneTreeTimer::get_time_left);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "time_left", PROPERTY_HINT_NONE, "suffix:s"), "set_time_left", "get_time_left");

	ADD_SIGNAL(MethodInfo("timeout"));
}

void SceneTreeTimer::set_time_left(double p_time) {
	time_left = p_time;
}

double SceneTreeTimer::get_time_left() const {
	return time_left;
}

void SceneTreeTimer::set_process_always(bool p_process_always) {
	process_always = p_process_always;
}

bool SceneTreeTimer::is_process_always() {
	return process_always;
}

void SceneTreeTimer::set_process_in_physics(bool p_process_in_physics) {
	process_in_physics = p_process_in_physics;
}

bool SceneTreeTimer::is_process_in_physics() {
	return process_in_physics;
}

void SceneTreeTimer::set_ignore_time_scale(bool p_ignore) {
	ignore_time_scale = p_ignore;
}

bool SceneTreeTimer::is_ignore_time_scale() {
	return ignore_time_scale;
}

void SceneTreeTimer::release_connections() {
	List<Connection> signal_connections;
	get_all_signal_connections(&signal_connections);

	for (const Connection &connection : signal_connections) {
		disconnect(connection.signal.get_name(), connection.callable);
	}
}

SceneTreeTimer::SceneTreeTimer() {}

void SceneTree::tree_changed() {
	tree_version++;
	emit_signal(tree_changed_name);
}

void SceneTree::node_added(Node *p_node) {
	emit_signal(node_added_name, p_node);
}

void SceneTree::node_removed(Node *p_node) {
	if (current_scene == p_node) {
		current_scene = nullptr;
	}
	emit_signal(node_removed_name, p_node);
	if (call_lock > 0) {
		call_skip.insert(p_node);
	}
}

void SceneTree::node_renamed(Node *p_node) {
	emit_signal(node_renamed_name, p_node);
}

SceneTree::Group *SceneTree::add_to_group(const StringName &p_group, Node *p_node) {
	HashMap<StringName, Group>::Iterator E = group_map.find(p_group);
	if (!E) {
		E = group_map.insert(p_group, Group());
	}

	ERR_FAIL_COND_V_MSG(E->value.nodes.has(p_node), &E->value, "Already in group: " + p_group + ".");
	E->value.nodes.push_back(p_node);
	//E->value.last_tree_version=0;
	E->value.changed = true;
	return &E->value;
}

void SceneTree::remove_from_group(const StringName &p_group, Node *p_node) {
	HashMap<StringName, Group>::Iterator E = group_map.find(p_group);
	ERR_FAIL_COND(!E);

	E->value.nodes.erase(p_node);
	if (E->value.nodes.is_empty()) {
		group_map.remove(E);
	}
}

void SceneTree::make_group_changed(const StringName &p_group) {
	HashMap<StringName, Group>::Iterator E = group_map.find(p_group);
	if (E) {
		E->value.changed = true;
	}
}

void SceneTree::flush_transform_notifications() {
	SelfList<Node> *n = xform_change_list.first();
	while (n) {
		Node *node = n->self();
		SelfList<Node> *nx = n->next();
		xform_change_list.remove(n);
		n = nx;
		node->notification(NOTIFICATION_TRANSFORM_CHANGED);
	}
}

void SceneTree::_flush_ugc() {
	ugc_locked = true;

	while (unique_group_calls.size()) {
		HashMap<UGCall, Vector<Variant>, UGCall>::Iterator E = unique_group_calls.begin();

		const Variant **argptrs = (const Variant **)alloca(E->value.size() * sizeof(Variant *));

		for (int i = 0; i < E->value.size(); i++) {
			argptrs[i] = &E->value[i];
		}

		call_group_flagsp(GROUP_CALL_DEFAULT, E->key.group, E->key.call, argptrs, E->value.size());

		unique_group_calls.remove(E);
	}

	ugc_locked = false;
}

void SceneTree::_update_group_order(Group &g, bool p_use_priority) {
	if (!g.changed) {
		return;
	}
	if (g.nodes.is_empty()) {
		return;
	}

	Node **gr_nodes = g.nodes.ptrw();
	int gr_node_count = g.nodes.size();

	if (p_use_priority) {
		SortArray<Node *, Node::ComparatorWithPriority> node_sort;
		node_sort.sort(gr_nodes, gr_node_count);
	} else {
		SortArray<Node *, Node::Comparator> node_sort;
		node_sort.sort(gr_nodes, gr_node_count);
	}
	g.changed = false;
}

void SceneTree::call_group_flagsp(uint32_t p_call_flags, const StringName &p_group, const StringName &p_function, const Variant **p_args, int p_argcount) {
	HashMap<StringName, Group>::Iterator E = group_map.find(p_group);
	if (!E) {
		return;
	}
	Group &g = E->value;
	if (g.nodes.is_empty()) {
		return;
	}

	if (p_call_flags & GROUP_CALL_UNIQUE && p_call_flags & GROUP_CALL_DEFERRED) {
		ERR_FAIL_COND(ugc_locked);

		UGCall ug;
		ug.call = p_function;
		ug.group = p_group;

		if (unique_group_calls.has(ug)) {
			return;
		}

		Vector<Variant> args;
		for (int i = 0; i < p_argcount; i++) {
			args.push_back(*p_args[i]);
		}

		unique_group_calls[ug] = args;
		return;
	}

	_update_group_order(g);

	Vector<Node *> nodes_copy = g.nodes;
	Node **gr_nodes = nodes_copy.ptrw();
	int gr_node_count = nodes_copy.size();

	call_lock++;

	if (p_call_flags & GROUP_CALL_REVERSE) {
		for (int i = gr_node_count - 1; i >= 0; i--) {
			if (call_lock && call_skip.has(gr_nodes[i])) {
				continue;
			}

			if (!(p_call_flags & GROUP_CALL_DEFERRED)) {
				Callable::CallError ce;
				gr_nodes[i]->callp(p_function, p_args, p_argcount, ce);
			} else {
				MessageQueue::get_singleton()->push_callp(gr_nodes[i], p_function, p_args, p_argcount);
			}
		}

	} else {
		for (int i = 0; i < gr_node_count; i++) {
			if (call_lock && call_skip.has(gr_nodes[i])) {
				continue;
			}

			if (!(p_call_flags & GROUP_CALL_DEFERRED)) {
				Callable::CallError ce;
				gr_nodes[i]->callp(p_function, p_args, p_argcount, ce);
			} else {
				MessageQueue::get_singleton()->push_callp(gr_nodes[i], p_function, p_args, p_argcount);
			}
		}
	}

	call_lock--;
	if (call_lock == 0) {
		call_skip.clear();
	}
}

void SceneTree::notify_group_flags(uint32_t p_call_flags, const StringName &p_group, int p_notification) {
	HashMap<StringName, Group>::Iterator E = group_map.find(p_group);
	if (!E) {
		return;
	}
	Group &g = E->value;
	if (g.nodes.is_empty()) {
		return;
	}

	_update_group_order(g);

	Vector<Node *> nodes_copy = g.nodes;
	Node **gr_nodes = nodes_copy.ptrw();
	int gr_node_count = nodes_copy.size();

	call_lock++;

	if (p_call_flags & GROUP_CALL_REVERSE) {
		for (int i = gr_node_count - 1; i >= 0; i--) {
			if (call_lock && call_skip.has(gr_nodes[i])) {
				continue;
			}

			if (!(p_call_flags & GROUP_CALL_DEFERRED)) {
				gr_nodes[i]->notification(p_notification);
			} else {
				MessageQueue::get_singleton()->push_notification(gr_nodes[i], p_notification);
			}
		}

	} else {
		for (int i = 0; i < gr_node_count; i++) {
			if (call_lock && call_skip.has(gr_nodes[i])) {
				continue;
			}

			if (!(p_call_flags & GROUP_CALL_DEFERRED)) {
				gr_nodes[i]->notification(p_notification);
			} else {
				MessageQueue::get_singleton()->push_notification(gr_nodes[i], p_notification);
			}
		}
	}

	call_lock--;
	if (call_lock == 0) {
		call_skip.clear();
	}
}

void SceneTree::set_group_flags(uint32_t p_call_flags, const StringName &p_group, const String &p_name, const Variant &p_value) {
	HashMap<StringName, Group>::Iterator E = group_map.find(p_group);
	if (!E) {
		return;
	}
	Group &g = E->value;
	if (g.nodes.is_empty()) {
		return;
	}

	_update_group_order(g);

	Vector<Node *> nodes_copy = g.nodes;
	Node **gr_nodes = nodes_copy.ptrw();
	int gr_node_count = nodes_copy.size();

	call_lock++;

	if (p_call_flags & GROUP_CALL_REVERSE) {
		for (int i = gr_node_count - 1; i >= 0; i--) {
			if (call_lock && call_skip.has(gr_nodes[i])) {
				continue;
			}

			if (!(p_call_flags & GROUP_CALL_DEFERRED)) {
				gr_nodes[i]->set(p_name, p_value);
			} else {
				MessageQueue::get_singleton()->push_set(gr_nodes[i], p_name, p_value);
			}
		}

	} else {
		for (int i = 0; i < gr_node_count; i++) {
			if (call_lock && call_skip.has(gr_nodes[i])) {
				continue;
			}

			if (!(p_call_flags & GROUP_CALL_DEFERRED)) {
				gr_nodes[i]->set(p_name, p_value);
			} else {
				MessageQueue::get_singleton()->push_set(gr_nodes[i], p_name, p_value);
			}
		}
	}

	call_lock--;
	if (call_lock == 0) {
		call_skip.clear();
	}
}

void SceneTree::notify_group(const StringName &p_group, int p_notification) {
	notify_group_flags(GROUP_CALL_DEFAULT, p_group, p_notification);
}

void SceneTree::set_group(const StringName &p_group, const String &p_name, const Variant &p_value) {
	set_group_flags(GROUP_CALL_DEFAULT, p_group, p_name, p_value);
}

void SceneTree::initialize() {
	ERR_FAIL_COND(!root);
	initialized = true;
	root->_set_tree(this);
	MainLoop::initialize();
}

bool SceneTree::physics_process(double p_time) {
	root_lock++;

	current_frame++;

	flush_transform_notifications();

	MainLoop::physics_process(p_time);
	physics_process_time = p_time;

	emit_signal(SNAME("physics_frame"));

	_notify_group_pause(SNAME("_physics_process_internal"), Node::NOTIFICATION_INTERNAL_PHYSICS_PROCESS);
	call_group(SNAME("_picking_viewports"), SNAME("_process_picking"));
	_notify_group_pause(SNAME("_physics_process"), Node::NOTIFICATION_PHYSICS_PROCESS);
	_flush_ugc();
	MessageQueue::get_singleton()->flush(); //small little hack

	process_timers(p_time, true); //go through timers

	process_tweens(p_time, true);

	flush_transform_notifications();
	root_lock--;

	_flush_delete_queue();
	_call_idle_callbacks();

	return _quit;
}

bool SceneTree::process(double p_time) {
	root_lock++;

	MainLoop::process(p_time);

	process_time = p_time;

	if (multiplayer_poll) {
		multiplayer->poll();
		for (KeyValue<NodePath, Ref<MultiplayerAPI>> &E : custom_multiplayers) {
			E.value->poll();
		}
	}

	emit_signal(SNAME("process_frame"));

	MessageQueue::get_singleton()->flush(); //small little hack

	flush_transform_notifications();

	_notify_group_pause(SNAME("_process_internal"), Node::NOTIFICATION_INTERNAL_PROCESS);
	_notify_group_pause(SNAME("_process"), Node::NOTIFICATION_PROCESS);

	_flush_ugc();
	MessageQueue::get_singleton()->flush(); //small little hack
	flush_transform_notifications(); //transforms after world update, to avoid unnecessary enter/exit notifications

	root_lock--;

	_flush_delete_queue();

	process_timers(p_time, false); //go through timers

	process_tweens(p_time, false);

	flush_transform_notifications(); //additional transforms after timers update

	_call_idle_callbacks();

#ifdef TOOLS_ENABLED
#ifndef _3D_DISABLED
	if (Engine::get_singleton()->is_editor_hint()) {
		//simple hack to reload fallback environment if it changed from editor
		String env_path = GLOBAL_GET(SNAME("rendering/environment/defaults/default_environment"));
		env_path = env_path.strip_edges(); //user may have added a space or two
		String cpath;
		Ref<Environment> fallback = get_root()->get_world_3d()->get_fallback_environment();
		if (fallback.is_valid()) {
			cpath = fallback->get_path();
		}
		if (cpath != env_path) {
			if (!env_path.is_empty()) {
				fallback = ResourceLoader::load(env_path);
				if (fallback.is_null()) {
					//could not load fallback, set as empty
					ProjectSettings::get_singleton()->set("rendering/environment/defaults/default_environment", "");
				}
			} else {
				fallback.unref();
			}
			get_root()->get_world_3d()->set_fallback_environment(fallback);
		}
	}
#endif // _3D_DISABLED
#endif // TOOLS_ENABLED

	return _quit;
}

void SceneTree::process_timers(double p_delta, bool p_physics_frame) {
	List<Ref<SceneTreeTimer>>::Element *L = timers.back(); //last element

	for (List<Ref<SceneTreeTimer>>::Element *E = timers.front(); E;) {
		List<Ref<SceneTreeTimer>>::Element *N = E->next();
		if ((paused && !E->get()->is_process_always()) || (E->get()->is_process_in_physics() != p_physics_frame)) {
			if (E == L) {
				break; //break on last, so if new timers were added during list traversal, ignore them.
			}
			E = N;
			continue;
		}

		double time_left = E->get()->get_time_left();
		if (E->get()->is_ignore_time_scale()) {
			time_left -= Engine::get_singleton()->get_process_step();
		} else {
			time_left -= p_delta;
		}
		E->get()->set_time_left(time_left);

		if (time_left <= 0) {
			E->get()->emit_signal(SNAME("timeout"));
			timers.erase(E);
		}
		if (E == L) {
			break; //break on last, so if new timers were added during list traversal, ignore them.
		}
		E = N;
	}
}

void SceneTree::process_tweens(double p_delta, bool p_physics) {
	// This methods works similarly to how SceneTreeTimers are handled.
	List<Ref<Tween>>::Element *L = tweens.back();

	for (List<Ref<Tween>>::Element *E = tweens.front(); E;) {
		List<Ref<Tween>>::Element *N = E->next();
		// Don't process if paused or process mode doesn't match.
		if (!E->get()->can_process(paused) || (p_physics == (E->get()->get_process_mode() == Tween::TWEEN_PROCESS_IDLE))) {
			if (E == L) {
				break;
			}
			E = N;
			continue;
		}

		if (!E->get()->step(p_delta)) {
			E->get()->clear();
			tweens.erase(E);
		}
		if (E == L) {
			break;
		}
		E = N;
	}
}

void SceneTree::finalize() {
	_flush_delete_queue();

	_flush_ugc();

	initialized = false;

	MainLoop::finalize();

	if (root) {
		root->_set_tree(nullptr);
		root->_propagate_after_exit_tree();
		memdelete(root); //delete root
		root = nullptr;
	}

	// In case deletion of some objects was queued when destructing the `root`.
	// E.g. if `queue_free()` was called for some node outside the tree when handling NOTIFICATION_PREDELETE for some node in the tree.
	_flush_delete_queue();

	// Cleanup timers.
	for (Ref<SceneTreeTimer> &timer : timers) {
		timer->release_connections();
	}
	timers.clear();

	// Cleanup tweens.
	for (Ref<Tween> &tween : tweens) {
		tween->clear();
	}
	tweens.clear();
}

void SceneTree::quit(int p_exit_code) {
	OS::get_singleton()->set_exit_code(p_exit_code);
	_quit = true;
}

void SceneTree::_main_window_close() {
	if (accept_quit) {
		_quit = true;
	}
}

void SceneTree::_main_window_go_back() {
	if (quit_on_go_back) {
		_quit = true;
	}
}

void SceneTree::_main_window_focus_in() {
	Input *id = Input::get_singleton();
	if (id) {
		id->ensure_touch_mouse_raised();
	}
}

void SceneTree::_notification(int p_notification) {
	switch (p_notification) {
		case NOTIFICATION_TRANSLATION_CHANGED: {
			if (!Engine::get_singleton()->is_editor_hint()) {
				get_root()->propagate_notification(p_notification);
			}
		} break;

		case NOTIFICATION_OS_MEMORY_WARNING:
		case NOTIFICATION_OS_IME_UPDATE:
		case NOTIFICATION_WM_ABOUT:
		case NOTIFICATION_CRASH:
		case NOTIFICATION_APPLICATION_RESUMED:
		case NOTIFICATION_APPLICATION_PAUSED:
		case NOTIFICATION_APPLICATION_FOCUS_IN:
		case NOTIFICATION_APPLICATION_FOCUS_OUT: {
			// Pass these to nodes, since they are mirrored.
			get_root()->propagate_notification(p_notification);
		} break;
	}
}

bool SceneTree::is_auto_accept_quit() const {
	return accept_quit;
}

void SceneTree::set_auto_accept_quit(bool p_enable) {
	accept_quit = p_enable;
}

bool SceneTree::is_quit_on_go_back() const {
	return quit_on_go_back;
}

void SceneTree::set_quit_on_go_back(bool p_enable) {
	quit_on_go_back = p_enable;
}

#ifdef TOOLS_ENABLED

bool SceneTree::is_node_being_edited(const Node *p_node) const {
	return Engine::get_singleton()->is_editor_hint() && edited_scene_root && (edited_scene_root->is_ancestor_of(p_node) || edited_scene_root == p_node);
}
#endif

#ifdef DEBUG_ENABLED
void SceneTree::set_debug_collisions_hint(bool p_enabled) {
	debug_collisions_hint = p_enabled;
}

bool SceneTree::is_debugging_collisions_hint() const {
	return debug_collisions_hint;
}

void SceneTree::set_debug_paths_hint(bool p_enabled) {
	debug_paths_hint = p_enabled;
}

bool SceneTree::is_debugging_paths_hint() const {
	return debug_paths_hint;
}

void SceneTree::set_debug_navigation_hint(bool p_enabled) {
	debug_navigation_hint = p_enabled;
}

bool SceneTree::is_debugging_navigation_hint() const {
	return debug_navigation_hint;
}
#endif

void SceneTree::set_debug_collisions_color(const Color &p_color) {
	debug_collisions_color = p_color;
}

Color SceneTree::get_debug_collisions_color() const {
	return debug_collisions_color;
}

void SceneTree::set_debug_collision_contact_color(const Color &p_color) {
	debug_collision_contact_color = p_color;
}

Color SceneTree::get_debug_collision_contact_color() const {
	return debug_collision_contact_color;
}

void SceneTree::set_debug_paths_color(const Color &p_color) {
	debug_paths_color = p_color;
}

Color SceneTree::get_debug_paths_color() const {
	return debug_paths_color;
}

void SceneTree::set_debug_paths_width(float p_width) {
	debug_paths_width = p_width;
}

float SceneTree::get_debug_paths_width() const {
	return debug_paths_width;
}

Ref<Material> SceneTree::get_debug_paths_material() {
	if (debug_paths_material.is_valid()) {
		return debug_paths_material;
	}

	Ref<StandardMaterial3D> _debug_material = Ref<StandardMaterial3D>(memnew(StandardMaterial3D));
	_debug_material->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
	_debug_material->set_transparency(StandardMaterial3D::TRANSPARENCY_ALPHA);
	_debug_material->set_flag(StandardMaterial3D::FLAG_SRGB_VERTEX_COLOR, true);
	_debug_material->set_flag(StandardMaterial3D::FLAG_ALBEDO_FROM_VERTEX_COLOR, true);
	_debug_material->set_albedo(get_debug_paths_color());

	debug_paths_material = _debug_material;

	return debug_paths_material;
}

Ref<Material> SceneTree::get_debug_collision_material() {
	if (collision_material.is_valid()) {
		return collision_material;
	}

	Ref<StandardMaterial3D> line_material = Ref<StandardMaterial3D>(memnew(StandardMaterial3D));
	line_material->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
	line_material->set_transparency(StandardMaterial3D::TRANSPARENCY_ALPHA);
	line_material->set_flag(StandardMaterial3D::FLAG_SRGB_VERTEX_COLOR, true);
	line_material->set_flag(StandardMaterial3D::FLAG_ALBEDO_FROM_VERTEX_COLOR, true);
	line_material->set_albedo(get_debug_collisions_color());

	collision_material = line_material;

	return collision_material;
}

Ref<ArrayMesh> SceneTree::get_debug_contact_mesh() {
	if (debug_contact_mesh.is_valid()) {
		return debug_contact_mesh;
	}

	debug_contact_mesh = Ref<ArrayMesh>(memnew(ArrayMesh));

	Ref<StandardMaterial3D> mat = Ref<StandardMaterial3D>(memnew(StandardMaterial3D));
	mat->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
	mat->set_transparency(StandardMaterial3D::TRANSPARENCY_ALPHA);
	mat->set_flag(StandardMaterial3D::FLAG_SRGB_VERTEX_COLOR, true);
	mat->set_flag(StandardMaterial3D::FLAG_ALBEDO_FROM_VERTEX_COLOR, true);
	mat->set_albedo(get_debug_collision_contact_color());

	Vector3 diamond[6] = {
		Vector3(-1, 0, 0),
		Vector3(1, 0, 0),
		Vector3(0, -1, 0),
		Vector3(0, 1, 0),
		Vector3(0, 0, -1),
		Vector3(0, 0, 1)
	};

	/* clang-format off */
	int diamond_faces[8 * 3] = {
		0, 2, 4,
		0, 3, 4,
		1, 2, 4,
		1, 3, 4,
		0, 2, 5,
		0, 3, 5,
		1, 2, 5,
		1, 3, 5,
	};
	/* clang-format on */

	Vector<int> indices;
	for (int i = 0; i < 8 * 3; i++) {
		indices.push_back(diamond_faces[i]);
	}

	Vector<Vector3> vertices;
	for (int i = 0; i < 6; i++) {
		vertices.push_back(diamond[i] * 0.1);
	}

	Array arr;
	arr.resize(Mesh::ARRAY_MAX);
	arr[Mesh::ARRAY_VERTEX] = vertices;
	arr[Mesh::ARRAY_INDEX] = indices;

	debug_contact_mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, arr);
	debug_contact_mesh->surface_set_material(0, mat);

	return debug_contact_mesh;
}

void SceneTree::set_pause(bool p_enabled) {
	if (p_enabled == paused) {
		return;
	}
	paused = p_enabled;
	NavigationServer3D::get_singleton()->set_active(!p_enabled);
	PhysicsServer3D::get_singleton()->set_active(!p_enabled);
	PhysicsServer2D::get_singleton()->set_active(!p_enabled);
	if (get_root()) {
		get_root()->_propagate_pause_notification(p_enabled);
	}
}

bool SceneTree::is_paused() const {
	return paused;
}

void SceneTree::_notify_group_pause(const StringName &p_group, int p_notification) {
	HashMap<StringName, Group>::Iterator E = group_map.find(p_group);
	if (!E) {
		return;
	}
	Group &g = E->value;
	if (g.nodes.is_empty()) {
		return;
	}

	_update_group_order(g, p_notification == Node::NOTIFICATION_PROCESS || p_notification == Node::NOTIFICATION_INTERNAL_PROCESS || p_notification == Node::NOTIFICATION_PHYSICS_PROCESS || p_notification == Node::NOTIFICATION_INTERNAL_PHYSICS_PROCESS);

	//copy, so copy on write happens in case something is removed from process while being called
	//performance is not lost because only if something is added/removed the vector is copied.
	Vector<Node *> nodes_copy = g.nodes;

	int gr_node_count = nodes_copy.size();
	Node **gr_nodes = nodes_copy.ptrw();

	call_lock++;

	for (int i = 0; i < gr_node_count; i++) {
		Node *n = gr_nodes[i];
		if (call_lock && call_skip.has(n)) {
			continue;
		}

		if (!n->can_process()) {
			continue;
		}
		if (!n->can_process_notification(p_notification)) {
			continue;
		}

		n->notification(p_notification);
		//ERR_FAIL_COND(gr_node_count != g.nodes.size());
	}

	call_lock--;
	if (call_lock == 0) {
		call_skip.clear();
	}
}

void SceneTree::_call_input_pause(const StringName &p_group, CallInputType p_call_type, const Ref<InputEvent> &p_input, Viewport *p_viewport) {
	HashMap<StringName, Group>::Iterator E = group_map.find(p_group);
	if (!E) {
		return;
	}
	Group &g = E->value;
	if (g.nodes.is_empty()) {
		return;
	}

	_update_group_order(g);

	//copy, so copy on write happens in case something is removed from process while being called
	//performance is not lost because only if something is added/removed the vector is copied.
	Vector<Node *> nodes_copy = g.nodes;

	int gr_node_count = nodes_copy.size();
	Node **gr_nodes = nodes_copy.ptrw();

	call_lock++;

	Vector<ObjectID> no_context_node_ids; // Nodes may be deleted due to this shortcut input.

	for (int i = gr_node_count - 1; i >= 0; i--) {
		if (p_viewport->is_input_handled()) {
			break;
		}

		Node *n = gr_nodes[i];
		if (call_lock && call_skip.has(n)) {
			continue;
		}

		if (!n->can_process()) {
			continue;
		}

		switch (p_call_type) {
			case CALL_INPUT_TYPE_INPUT:
				n->_call_input(p_input);
				break;
			case CALL_INPUT_TYPE_SHORTCUT_INPUT: {
				const Control *c = Object::cast_to<Control>(n);
				if (c) {
					// If calling shortcut input on a control, ensure it respects the shortcut context.
					// Shortcut context (based on focus) only makes sense for controls (UI), so don't need to worry about it for nodes
					if (c->get_shortcut_context() == nullptr) {
						no_context_node_ids.append(n->get_instance_id());
						continue;
					}
					if (!c->is_focus_owner_in_shortcut_context()) {
						continue;
					}
				}
				n->_call_shortcut_input(p_input);
				break;
			}
			case CALL_INPUT_TYPE_UNHANDLED_INPUT:
				n->_call_unhandled_input(p_input);
				break;
			case CALL_INPUT_TYPE_UNHANDLED_KEY_INPUT:
				n->_call_unhandled_key_input(p_input);
				break;
		}
	}

	for (const ObjectID &id : no_context_node_ids) {
		if (p_viewport->is_input_handled()) {
			break;
		}
		Node *n = Object::cast_to<Node>(ObjectDB::get_instance(id));
		if (n) {
			n->_call_shortcut_input(p_input);
		}
	}

	call_lock--;
	if (call_lock == 0) {
		call_skip.clear();
	}
}

void SceneTree::_call_group_flags(const Variant **p_args, int p_argcount, Callable::CallError &r_error) {
	r_error.error = Callable::CallError::CALL_OK;

	ERR_FAIL_COND(p_argcount < 3);
	ERR_FAIL_COND(!p_args[0]->is_num());
	ERR_FAIL_COND(p_args[1]->get_type() != Variant::STRING_NAME && p_args[1]->get_type() != Variant::STRING);
	ERR_FAIL_COND(p_args[2]->get_type() != Variant::STRING_NAME && p_args[2]->get_type() != Variant::STRING);

	int flags = *p_args[0];
	StringName group = *p_args[1];
	StringName method = *p_args[2];

	call_group_flagsp(flags, group, method, p_args + 3, p_argcount - 3);
}

void SceneTree::_call_group(const Variant **p_args, int p_argcount, Callable::CallError &r_error) {
	r_error.error = Callable::CallError::CALL_OK;

	ERR_FAIL_COND(p_argcount < 2);
	ERR_FAIL_COND(p_args[0]->get_type() != Variant::STRING_NAME && p_args[0]->get_type() != Variant::STRING);
	ERR_FAIL_COND(p_args[1]->get_type() != Variant::STRING_NAME && p_args[1]->get_type() != Variant::STRING);

	StringName group = *p_args[0];
	StringName method = *p_args[1];

	call_group_flagsp(GROUP_CALL_DEFAULT, group, method, p_args + 2, p_argcount - 2);
}

int64_t SceneTree::get_frame() const {
	return current_frame;
}

TypedArray<Node> SceneTree::_get_nodes_in_group(const StringName &p_group) {
	TypedArray<Node> ret;
	HashMap<StringName, Group>::Iterator E = group_map.find(p_group);
	if (!E) {
		return ret;
	}

	_update_group_order(E->value); //update order just in case
	int nc = E->value.nodes.size();
	if (nc == 0) {
		return ret;
	}

	ret.resize(nc);

	Node **ptr = E->value.nodes.ptrw();
	for (int i = 0; i < nc; i++) {
		ret[i] = ptr[i];
	}

	return ret;
}

bool SceneTree::has_group(const StringName &p_identifier) const {
	return group_map.has(p_identifier);
}

Node *SceneTree::get_first_node_in_group(const StringName &p_group) {
	HashMap<StringName, Group>::Iterator E = group_map.find(p_group);
	if (!E) {
		return nullptr; // No group.
	}

	_update_group_order(E->value); // Update order just in case.

	if (E->value.nodes.is_empty()) {
		return nullptr;
	}

	return E->value.nodes[0];
}

void SceneTree::get_nodes_in_group(const StringName &p_group, List<Node *> *p_list) {
	HashMap<StringName, Group>::Iterator E = group_map.find(p_group);
	if (!E) {
		return;
	}

	_update_group_order(E->value); //update order just in case
	int nc = E->value.nodes.size();
	if (nc == 0) {
		return;
	}
	Node **ptr = E->value.nodes.ptrw();
	for (int i = 0; i < nc; i++) {
		p_list->push_back(ptr[i]);
	}
}

void SceneTree::_flush_delete_queue() {
	_THREAD_SAFE_METHOD_

	while (delete_queue.size()) {
		Object *obj = ObjectDB::get_instance(delete_queue.front()->get());
		if (obj) {
			memdelete(obj);
		}
		delete_queue.pop_front();
	}
}

void SceneTree::queue_delete(Object *p_object) {
	_THREAD_SAFE_METHOD_
	ERR_FAIL_NULL(p_object);
	p_object->_is_queued_for_deletion = true;
	delete_queue.push_back(p_object->get_instance_id());
}

int SceneTree::get_node_count() const {
	return node_count;
}

void SceneTree::set_edited_scene_root(Node *p_node) {
#ifdef TOOLS_ENABLED
	edited_scene_root = p_node;
#endif
}

Node *SceneTree::get_edited_scene_root() const {
#ifdef TOOLS_ENABLED
	return edited_scene_root;
#else
	return nullptr;
#endif
}

void SceneTree::set_current_scene(Node *p_scene) {
	ERR_FAIL_COND(p_scene && p_scene->get_parent() != root);
	current_scene = p_scene;
}

Node *SceneTree::get_current_scene() const {
	return current_scene;
}

void SceneTree::_change_scene(Node *p_to) {
	if (current_scene) {
		memdelete(current_scene);
		current_scene = nullptr;
	}

	// If we're quitting, abort.
	if (unlikely(_quit)) {
		if (p_to) { // Prevent memory leak.
			memdelete(p_to);
		}
		return;
	}

	if (p_to) {
		current_scene = p_to;
		root->add_child(p_to);
		root->update_mouse_cursor_shape();
	}
}

Error SceneTree::change_scene_to_file(const String &p_path) {
	Ref<PackedScene> new_scene = ResourceLoader::load(p_path);
	if (new_scene.is_null()) {
		return ERR_CANT_OPEN;
	}

	return change_scene_to_packed(new_scene);
}

Error SceneTree::change_scene_to_packed(const Ref<PackedScene> &p_scene) {
	ERR_FAIL_COND_V_MSG(p_scene.is_null(), ERR_INVALID_PARAMETER, "Can't change to a null scene. Use unload_current_scene() if you wish to unload it.");

	Node *new_scene = p_scene->instantiate();
	ERR_FAIL_COND_V(!new_scene, ERR_CANT_CREATE);

	call_deferred(SNAME("_change_scene"), new_scene);
	return OK;
}

Error SceneTree::reload_current_scene() {
	ERR_FAIL_COND_V(!current_scene, ERR_UNCONFIGURED);
	String fname = current_scene->get_scene_file_path();
	return change_scene_to_file(fname);
}

void SceneTree::unload_current_scene() {
	if (current_scene) {
		memdelete(current_scene);
		current_scene = nullptr;
	}
}

void SceneTree::add_current_scene(Node *p_current) {
	current_scene = p_current;
	root->add_child(p_current);
}

Ref<SceneTreeTimer> SceneTree::create_timer(double p_delay_sec, bool p_process_always, bool p_process_in_physics, bool p_ignore_time_scale) {
	Ref<SceneTreeTimer> stt;
	stt.instantiate();
	stt->set_process_always(p_process_always);
	stt->set_time_left(p_delay_sec);
	stt->set_process_in_physics(p_process_in_physics);
	stt->set_ignore_time_scale(p_ignore_time_scale);
	timers.push_back(stt);
	return stt;
}

Ref<Tween> SceneTree::create_tween() {
	Ref<Tween> tween = memnew(Tween(true));
	tweens.push_back(tween);
	return tween;
}

TypedArray<Tween> SceneTree::get_processed_tweens() {
	TypedArray<Tween> ret;
	ret.resize(tweens.size());

	int i = 0;
	for (const Ref<Tween> &tween : tweens) {
		ret[i] = tween;
		i++;
	}

	return ret;
}

Ref<MultiplayerAPI> SceneTree::get_multiplayer(const NodePath &p_for_path) const {
	Ref<MultiplayerAPI> out = multiplayer;
	for (const KeyValue<NodePath, Ref<MultiplayerAPI>> &E : custom_multiplayers) {
		const Vector<StringName> snames = E.key.get_names();
		const Vector<StringName> tnames = p_for_path.get_names();
		if (tnames.size() < snames.size()) {
			continue;
		}
		const StringName *sptr = snames.ptr();
		const StringName *nptr = tnames.ptr();
		bool valid = true;
		for (int i = 0; i < snames.size(); i++) {
			if (sptr[i] != nptr[i]) {
				valid = false;
				break;
			}
		}
		if (valid) {
			out = E.value;
			break;
		}
	}
	return out;
}

void SceneTree::set_multiplayer(Ref<MultiplayerAPI> p_multiplayer, const NodePath &p_root_path) {
	if (p_root_path.is_empty()) {
		ERR_FAIL_COND(!p_multiplayer.is_valid());
		if (multiplayer.is_valid()) {
			multiplayer->object_configuration_remove(nullptr, NodePath("/" + root->get_name()));
		}
		multiplayer = p_multiplayer;
		multiplayer->object_configuration_add(nullptr, NodePath("/" + root->get_name()));
	} else {
		if (custom_multiplayers.has(p_root_path)) {
			custom_multiplayers[p_root_path]->object_configuration_remove(nullptr, p_root_path);
		}
		if (p_multiplayer.is_valid()) {
			custom_multiplayers[p_root_path] = p_multiplayer;
			p_multiplayer->object_configuration_add(nullptr, p_root_path);
		}
	}
}

void SceneTree::set_multiplayer_poll_enabled(bool p_enabled) {
	multiplayer_poll = p_enabled;
}

bool SceneTree::is_multiplayer_poll_enabled() const {
	return multiplayer_poll;
}

void SceneTree::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_root"), &SceneTree::get_root);
	ClassDB::bind_method(D_METHOD("has_group", "name"), &SceneTree::has_group);

	ClassDB::bind_method(D_METHOD("is_auto_accept_quit"), &SceneTree::is_auto_accept_quit);
	ClassDB::bind_method(D_METHOD("set_auto_accept_quit", "enabled"), &SceneTree::set_auto_accept_quit);
	ClassDB::bind_method(D_METHOD("is_quit_on_go_back"), &SceneTree::is_quit_on_go_back);
	ClassDB::bind_method(D_METHOD("set_quit_on_go_back", "enabled"), &SceneTree::set_quit_on_go_back);

	ClassDB::bind_method(D_METHOD("set_debug_collisions_hint", "enable"), &SceneTree::set_debug_collisions_hint);
	ClassDB::bind_method(D_METHOD("is_debugging_collisions_hint"), &SceneTree::is_debugging_collisions_hint);
	ClassDB::bind_method(D_METHOD("set_debug_paths_hint", "enable"), &SceneTree::set_debug_paths_hint);
	ClassDB::bind_method(D_METHOD("is_debugging_paths_hint"), &SceneTree::is_debugging_paths_hint);
	ClassDB::bind_method(D_METHOD("set_debug_navigation_hint", "enable"), &SceneTree::set_debug_navigation_hint);
	ClassDB::bind_method(D_METHOD("is_debugging_navigation_hint"), &SceneTree::is_debugging_navigation_hint);

	ClassDB::bind_method(D_METHOD("set_edited_scene_root", "scene"), &SceneTree::set_edited_scene_root);
	ClassDB::bind_method(D_METHOD("get_edited_scene_root"), &SceneTree::get_edited_scene_root);

	ClassDB::bind_method(D_METHOD("set_pause", "enable"), &SceneTree::set_pause);
	ClassDB::bind_method(D_METHOD("is_paused"), &SceneTree::is_paused);

	ClassDB::bind_method(D_METHOD("create_timer", "time_sec", "process_always", "process_in_physics", "ignore_time_scale"), &SceneTree::create_timer, DEFVAL(true), DEFVAL(false), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("create_tween"), &SceneTree::create_tween);
	ClassDB::bind_method(D_METHOD("get_processed_tweens"), &SceneTree::get_processed_tweens);

	ClassDB::bind_method(D_METHOD("get_node_count"), &SceneTree::get_node_count);
	ClassDB::bind_method(D_METHOD("get_frame"), &SceneTree::get_frame);
	ClassDB::bind_method(D_METHOD("quit", "exit_code"), &SceneTree::quit, DEFVAL(EXIT_SUCCESS));

	ClassDB::bind_method(D_METHOD("queue_delete", "obj"), &SceneTree::queue_delete);

	MethodInfo mi;
	mi.name = "call_group_flags";
	mi.arguments.push_back(PropertyInfo(Variant::INT, "flags"));
	mi.arguments.push_back(PropertyInfo(Variant::STRING_NAME, "group"));
	mi.arguments.push_back(PropertyInfo(Variant::STRING_NAME, "method"));

	ClassDB::bind_vararg_method(METHOD_FLAGS_DEFAULT, "call_group_flags", &SceneTree::_call_group_flags, mi);

	ClassDB::bind_method(D_METHOD("notify_group_flags", "call_flags", "group", "notification"), &SceneTree::notify_group_flags);
	ClassDB::bind_method(D_METHOD("set_group_flags", "call_flags", "group", "property", "value"), &SceneTree::set_group_flags);

	MethodInfo mi2;
	mi2.name = "call_group";
	mi2.arguments.push_back(PropertyInfo(Variant::STRING_NAME, "group"));
	mi2.arguments.push_back(PropertyInfo(Variant::STRING_NAME, "method"));

	ClassDB::bind_vararg_method(METHOD_FLAGS_DEFAULT, "call_group", &SceneTree::_call_group, mi2);

	ClassDB::bind_method(D_METHOD("notify_group", "group", "notification"), &SceneTree::notify_group);
	ClassDB::bind_method(D_METHOD("set_group", "group", "property", "value"), &SceneTree::set_group);

	ClassDB::bind_method(D_METHOD("get_nodes_in_group", "group"), &SceneTree::_get_nodes_in_group);
	ClassDB::bind_method(D_METHOD("get_first_node_in_group", "group"), &SceneTree::get_first_node_in_group);

	ClassDB::bind_method(D_METHOD("set_current_scene", "child_node"), &SceneTree::set_current_scene);
	ClassDB::bind_method(D_METHOD("get_current_scene"), &SceneTree::get_current_scene);

	ClassDB::bind_method(D_METHOD("change_scene_to_file", "path"), &SceneTree::change_scene_to_file);
	ClassDB::bind_method(D_METHOD("change_scene_to_packed", "packed_scene"), &SceneTree::change_scene_to_packed);

	ClassDB::bind_method(D_METHOD("reload_current_scene"), &SceneTree::reload_current_scene);
	ClassDB::bind_method(D_METHOD("unload_current_scene"), &SceneTree::unload_current_scene);

	ClassDB::bind_method(D_METHOD("_change_scene"), &SceneTree::_change_scene);

	ClassDB::bind_method(D_METHOD("set_multiplayer", "multiplayer", "root_path"), &SceneTree::set_multiplayer, DEFVAL(NodePath()));
	ClassDB::bind_method(D_METHOD("get_multiplayer", "for_path"), &SceneTree::get_multiplayer, DEFVAL(NodePath()));
	ClassDB::bind_method(D_METHOD("set_multiplayer_poll_enabled", "enabled"), &SceneTree::set_multiplayer_poll_enabled);
	ClassDB::bind_method(D_METHOD("is_multiplayer_poll_enabled"), &SceneTree::is_multiplayer_poll_enabled);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "auto_accept_quit"), "set_auto_accept_quit", "is_auto_accept_quit");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "quit_on_go_back"), "set_quit_on_go_back", "is_quit_on_go_back");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "debug_collisions_hint"), "set_debug_collisions_hint", "is_debugging_collisions_hint");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "debug_paths_hint"), "set_debug_paths_hint", "is_debugging_paths_hint");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "debug_navigation_hint"), "set_debug_navigation_hint", "is_debugging_navigation_hint");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "paused"), "set_pause", "is_paused");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "edited_scene_root", PROPERTY_HINT_RESOURCE_TYPE, "Node", PROPERTY_USAGE_NONE), "set_edited_scene_root", "get_edited_scene_root");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "current_scene", PROPERTY_HINT_RESOURCE_TYPE, "Node", PROPERTY_USAGE_NONE), "set_current_scene", "get_current_scene");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "root", PROPERTY_HINT_RESOURCE_TYPE, "Node", PROPERTY_USAGE_NONE), "", "get_root");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "multiplayer_poll"), "set_multiplayer_poll_enabled", "is_multiplayer_poll_enabled");

	ADD_SIGNAL(MethodInfo("tree_changed"));
	ADD_SIGNAL(MethodInfo("tree_process_mode_changed")); //editor only signal, but due to API hash it can't be removed in run-time
	ADD_SIGNAL(MethodInfo("node_added", PropertyInfo(Variant::OBJECT, "node", PROPERTY_HINT_RESOURCE_TYPE, "Node")));
	ADD_SIGNAL(MethodInfo("node_removed", PropertyInfo(Variant::OBJECT, "node", PROPERTY_HINT_RESOURCE_TYPE, "Node")));
	ADD_SIGNAL(MethodInfo("node_renamed", PropertyInfo(Variant::OBJECT, "node", PROPERTY_HINT_RESOURCE_TYPE, "Node")));
	ADD_SIGNAL(MethodInfo("node_configuration_warning_changed", PropertyInfo(Variant::OBJECT, "node", PROPERTY_HINT_RESOURCE_TYPE, "Node")));

	ADD_SIGNAL(MethodInfo("process_frame"));
	ADD_SIGNAL(MethodInfo("physics_frame"));

	BIND_ENUM_CONSTANT(GROUP_CALL_DEFAULT);
	BIND_ENUM_CONSTANT(GROUP_CALL_REVERSE);
	BIND_ENUM_CONSTANT(GROUP_CALL_DEFERRED);
	BIND_ENUM_CONSTANT(GROUP_CALL_UNIQUE);
}

SceneTree *SceneTree::singleton = nullptr;

SceneTree::IdleCallback SceneTree::idle_callbacks[SceneTree::MAX_IDLE_CALLBACKS];
int SceneTree::idle_callback_count = 0;

void SceneTree::_call_idle_callbacks() {
	for (int i = 0; i < idle_callback_count; i++) {
		idle_callbacks[i]();
	}
}

void SceneTree::add_idle_callback(IdleCallback p_callback) {
	ERR_FAIL_COND(idle_callback_count >= MAX_IDLE_CALLBACKS);
	idle_callbacks[idle_callback_count++] = p_callback;
}

void SceneTree::get_argument_options(const StringName &p_function, int p_idx, List<String> *r_options) const {
	if (p_function == "change_scene_to_file") {
		Ref<DirAccess> dir_access = DirAccess::create(DirAccess::ACCESS_RESOURCES);
		List<String> directories;
		directories.push_back(dir_access->get_current_dir());

		while (!directories.is_empty()) {
			dir_access->change_dir(directories.back()->get());
			directories.pop_back();

			dir_access->list_dir_begin();
			String filename = dir_access->get_next();

			while (!filename.is_empty()) {
				if (filename == "." || filename == "..") {
					filename = dir_access->get_next();
					continue;
				}

				if (dir_access->dir_exists(filename)) {
					directories.push_back(dir_access->get_current_dir().path_join(filename));
				} else if (filename.ends_with(".tscn") || filename.ends_with(".scn")) {
					r_options->push_back("\"" + dir_access->get_current_dir().path_join(filename) + "\"");
				}

				filename = dir_access->get_next();
			}
		}
	}
}

SceneTree::SceneTree() {
	if (singleton == nullptr) {
		singleton = this;
	}
	debug_collisions_color = GLOBAL_DEF("debug/shapes/collision/shape_color", Color(0.0, 0.6, 0.7, 0.42));
	debug_collision_contact_color = GLOBAL_DEF("debug/shapes/collision/contact_color", Color(1.0, 0.2, 0.1, 0.8));
	debug_paths_color = GLOBAL_DEF("debug/shapes/paths/geometry_color", Color(0.1, 1.0, 0.7, 0.4));
	debug_paths_width = GLOBAL_DEF("debug/shapes/paths/geometry_width", 2.0);
	collision_debug_contacts = GLOBAL_DEF(PropertyInfo(Variant::INT, "debug/shapes/collision/max_contacts_displayed", PROPERTY_HINT_RANGE, "0,20000,1"), 10000);

	GLOBAL_DEF("debug/shapes/collision/draw_2d_outlines", true);

	Math::randomize();

	// Create with mainloop.

	root = memnew(Window);
	root->set_min_size(Size2i(64, 64)); // Define a very small minimum window size to prevent bugs such as GH-37242.
	root->set_process_mode(Node::PROCESS_MODE_PAUSABLE);
	root->set_name("root");
	root->set_title(GLOBAL_GET("application/config/name"));

#ifndef _3D_DISABLED
	if (!root->get_world_3d().is_valid()) {
		root->set_world_3d(Ref<World3D>(memnew(World3D)));
	}
	root->set_as_audio_listener_3d(true);
#endif // _3D_DISABLED

	// Initialize network state.
	set_multiplayer(MultiplayerAPI::create_default_interface());

	root->set_as_audio_listener_2d(true);
	current_scene = nullptr;

	const int msaa_mode_2d = GLOBAL_DEF_BASIC(PropertyInfo(Variant::INT, "rendering/anti_aliasing/quality/msaa_2d", PROPERTY_HINT_ENUM, String::utf8("Disabled (Fastest),2× (Average),4× (Slow),8× (Slowest)")), 0);
	root->set_msaa_2d(Viewport::MSAA(msaa_mode_2d));

	const int msaa_mode_3d = GLOBAL_DEF_BASIC(PropertyInfo(Variant::INT, "rendering/anti_aliasing/quality/msaa_3d", PROPERTY_HINT_ENUM, String::utf8("Disabled (Fastest),2× (Average),4× (Slow),8× (Slowest)")), 0);
	root->set_msaa_3d(Viewport::MSAA(msaa_mode_3d));

	const bool transparent_background = GLOBAL_DEF("rendering/viewport/transparent_background", false);
	root->set_transparent_background(transparent_background);

	const int ssaa_mode = GLOBAL_DEF_BASIC(PropertyInfo(Variant::INT, "rendering/anti_aliasing/quality/screen_space_aa", PROPERTY_HINT_ENUM, "Disabled (Fastest),FXAA (Fast)"), 0);
	root->set_screen_space_aa(Viewport::ScreenSpaceAA(ssaa_mode));

	const bool use_taa = GLOBAL_DEF_BASIC("rendering/anti_aliasing/quality/use_taa", false);
	root->set_use_taa(use_taa);

	const bool use_debanding = GLOBAL_DEF("rendering/anti_aliasing/quality/use_debanding", false);
	root->set_use_debanding(use_debanding);

	const bool use_occlusion_culling = GLOBAL_DEF("rendering/occlusion_culling/use_occlusion_culling", false);
	root->set_use_occlusion_culling(use_occlusion_culling);

	float mesh_lod_threshold = GLOBAL_DEF(PropertyInfo(Variant::FLOAT, "rendering/mesh_lod/lod_change/threshold_pixels", PROPERTY_HINT_RANGE, "0,1024,0.1"), 1.0);
	root->set_mesh_lod_threshold(mesh_lod_threshold);

	bool snap_2d_transforms = GLOBAL_DEF("rendering/2d/snap/snap_2d_transforms_to_pixel", false);
	root->set_snap_2d_transforms_to_pixel(snap_2d_transforms);

	bool snap_2d_vertices = GLOBAL_DEF("rendering/2d/snap/snap_2d_vertices_to_pixel", false);
	root->set_snap_2d_vertices_to_pixel(snap_2d_vertices);

	// We setup VRS for the main viewport here, in the editor this will have little effect.
	const int vrs_mode = GLOBAL_DEF(PropertyInfo(Variant::INT, "rendering/vrs/mode", PROPERTY_HINT_ENUM, String::utf8("Disabled,Texture,XR")), 0);
	root->set_vrs_mode(Viewport::VRSMode(vrs_mode));
	const String vrs_texture_path = String(GLOBAL_DEF(PropertyInfo(Variant::STRING, "rendering/vrs/texture", PROPERTY_HINT_FILE, "*.bmp,*.png,*.tga,*.webp"), String())).strip_edges();
	if (vrs_mode == 1 && !vrs_texture_path.is_empty()) {
		Ref<Image> vrs_image;
		vrs_image.instantiate();
		Error load_err = ImageLoader::load_image(vrs_texture_path, vrs_image);
		if (load_err) {
			ERR_PRINT("Non-existing or invalid VRS texture at '" + vrs_texture_path + "'.");
		} else {
			Ref<ImageTexture> vrs_texture;
			vrs_texture.instantiate();
			vrs_texture->create_from_image(vrs_image);
			root->set_vrs_texture(vrs_texture);
		}
	}

	int shadowmap_size = GLOBAL_DEF(PropertyInfo(Variant::INT, "rendering/lights_and_shadows/positional_shadow/atlas_size", PROPERTY_HINT_RANGE, "256,16384"), 4096);
	GLOBAL_DEF("rendering/lights_and_shadows/positional_shadow/atlas_size.mobile", 2048);
	bool shadowmap_16_bits = GLOBAL_DEF("rendering/lights_and_shadows/positional_shadow/atlas_16_bits", true);
	int atlas_q0 = GLOBAL_DEF(PropertyInfo(Variant::INT, "rendering/lights_and_shadows/positional_shadow/atlas_quadrant_0_subdiv", PROPERTY_HINT_ENUM, "Disabled,1 Shadow,4 Shadows,16 Shadows,64 Shadows,256 Shadows,1024 Shadows"), 2);
	int atlas_q1 = GLOBAL_DEF(PropertyInfo(Variant::INT, "rendering/lights_and_shadows/positional_shadow/atlas_quadrant_1_subdiv", PROPERTY_HINT_ENUM, "Disabled,1 Shadow,4 Shadows,16 Shadows,64 Shadows,256 Shadows,1024 Shadows"), 2);
	int atlas_q2 = GLOBAL_DEF(PropertyInfo(Variant::INT, "rendering/lights_and_shadows/positional_shadow/atlas_quadrant_2_subdiv", PROPERTY_HINT_ENUM, "Disabled,1 Shadow,4 Shadows,16 Shadows,64 Shadows,256 Shadows,1024 Shadows"), 3);
	int atlas_q3 = GLOBAL_DEF(PropertyInfo(Variant::INT, "rendering/lights_and_shadows/positional_shadow/atlas_quadrant_3_subdiv", PROPERTY_HINT_ENUM, "Disabled,1 Shadow,4 Shadows,16 Shadows,64 Shadows,256 Shadows,1024 Shadows"), 4);

	root->set_positional_shadow_atlas_size(shadowmap_size);
	root->set_positional_shadow_atlas_16_bits(shadowmap_16_bits);
	root->set_positional_shadow_atlas_quadrant_subdiv(0, Viewport::PositionalShadowAtlasQuadrantSubdiv(atlas_q0));
	root->set_positional_shadow_atlas_quadrant_subdiv(1, Viewport::PositionalShadowAtlasQuadrantSubdiv(atlas_q1));
	root->set_positional_shadow_atlas_quadrant_subdiv(2, Viewport::PositionalShadowAtlasQuadrantSubdiv(atlas_q2));
	root->set_positional_shadow_atlas_quadrant_subdiv(3, Viewport::PositionalShadowAtlasQuadrantSubdiv(atlas_q3));

	Viewport::SDFOversize sdf_oversize = Viewport::SDFOversize(int(GLOBAL_DEF(PropertyInfo(Variant::INT, "rendering/2d/sdf/oversize", PROPERTY_HINT_ENUM, "100%,120%,150%,200%"), 1)));
	root->set_sdf_oversize(sdf_oversize);
	Viewport::SDFScale sdf_scale = Viewport::SDFScale(int(GLOBAL_DEF(PropertyInfo(Variant::INT, "rendering/2d/sdf/scale", PROPERTY_HINT_ENUM, "100%,50%,25%"), 1)));
	root->set_sdf_scale(sdf_scale);

#ifndef _3D_DISABLED
	{ // Load default fallback environment.
		// Get possible extensions.
		List<String> exts;
		ResourceLoader::get_recognized_extensions_for_type("Environment", &exts);
		String ext_hint;
		for (const String &E : exts) {
			if (!ext_hint.is_empty()) {
				ext_hint += ",";
			}
			ext_hint += "*." + E;
		}
		// Get path.
		String env_path = GLOBAL_DEF(PropertyInfo(Variant::STRING, "rendering/environment/defaults/default_environment", PROPERTY_HINT_FILE, ext_hint), "");
		// Setup property.
		env_path = env_path.strip_edges();
		if (!env_path.is_empty()) {
			Ref<Environment> env = ResourceLoader::load(env_path);
			if (env.is_valid()) {
				root->get_world_3d()->set_fallback_environment(env);
			} else {
				if (Engine::get_singleton()->is_editor_hint()) {
					// File was erased, clear the field.
					ProjectSettings::get_singleton()->set("rendering/environment/defaults/default_environment", "");
				} else {
					// File was erased, notify user.
					ERR_PRINT(RTR("Default Environment as specified in Project Settings (Rendering -> Environment -> Default Environment) could not be loaded."));
				}
			}
		}
	}
#endif // _3D_DISABLED

	root->set_physics_object_picking(GLOBAL_DEF("physics/common/enable_object_picking", true));

	root->connect("close_requested", callable_mp(this, &SceneTree::_main_window_close));
	root->connect("go_back_requested", callable_mp(this, &SceneTree::_main_window_go_back));
	root->connect("focus_entered", callable_mp(this, &SceneTree::_main_window_focus_in));

#ifdef TOOLS_ENABLED
	edited_scene_root = nullptr;
#endif
}

SceneTree::~SceneTree() {
	if (root) {
		root->_set_tree(nullptr);
		root->_propagate_after_exit_tree();
		memdelete(root);
	}

	if (singleton == this) {
		singleton = nullptr;
	}
}
