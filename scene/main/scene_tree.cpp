/*************************************************************************/
/*  scene_tree.cpp                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "scene_tree.h"

#include "core/config/project_settings.h"
#include "core/debugger/engine_debugger.h"
#include "core/input/input.h"
#include "core/io/dir_access.h"
#include "core/io/marshalls.h"
#include "core/io/resource_loader.h"
#include "core/object/message_queue.h"
#include "core/os/keyboard.h"
#include "core/os/os.h"
#include "core/string/print_string.h"
#include "node.h"
#include "scene/animation/tween.h"
#include "scene/debugger/scene_debugger.h"
#include "scene/resources/font.h"
#include "scene/resources/material.h"
#include "scene/resources/mesh.h"
#include "scene/resources/packed_scene.h"
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

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "time_left"), "set_time_left", "get_time_left");

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

void SceneTreeTimer::set_ignore_time_scale(bool p_ignore) {
	ignore_time_scale = p_ignore;
}

bool SceneTreeTimer::is_ignore_time_scale() {
	return ignore_time_scale;
}

void SceneTreeTimer::release_connections() {
	List<Connection> connections;
	get_all_signal_connections(&connections);

	for (const Connection &connection : connections) {
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
	Map<StringName, Group>::Element *E = group_map.find(p_group);
	if (!E) {
		E = group_map.insert(p_group, Group());
	}

	ERR_FAIL_COND_V_MSG(E->get().nodes.find(p_node) != -1, &E->get(), "Already in group: " + p_group + ".");
	E->get().nodes.push_back(p_node);
	//E->get().last_tree_version=0;
	E->get().changed = true;
	return &E->get();
}

void SceneTree::remove_from_group(const StringName &p_group, Node *p_node) {
	Map<StringName, Group>::Element *E = group_map.find(p_group);
	ERR_FAIL_COND(!E);

	E->get().nodes.erase(p_node);
	if (E->get().nodes.is_empty()) {
		group_map.erase(E);
	}
}

void SceneTree::make_group_changed(const StringName &p_group) {
	Map<StringName, Group>::Element *E = group_map.find(p_group);
	if (E) {
		E->get().changed = true;
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
		Map<UGCall, Vector<Variant>>::Element *E = unique_group_calls.front();

		Variant v[VARIANT_ARG_MAX];
		for (int i = 0; i < E->get().size(); i++) {
			v[i] = E->get()[i];
		}

		static_assert(VARIANT_ARG_MAX == 8, "This code needs to be updated if VARIANT_ARG_MAX != 8");
		call_group_flags(GROUP_CALL_REALTIME, E->key().group, E->key().call, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);

		unique_group_calls.erase(E);
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

	Node **nodes = g.nodes.ptrw();
	int node_count = g.nodes.size();

	if (p_use_priority) {
		SortArray<Node *, Node::ComparatorWithPriority> node_sort;
		node_sort.sort(nodes, node_count);
	} else {
		SortArray<Node *, Node::Comparator> node_sort;
		node_sort.sort(nodes, node_count);
	}
	g.changed = false;
}

void SceneTree::call_group_flags(uint32_t p_call_flags, const StringName &p_group, const StringName &p_function, VARIANT_ARG_DECLARE) {
	Map<StringName, Group>::Element *E = group_map.find(p_group);
	if (!E) {
		return;
	}
	Group &g = E->get();
	if (g.nodes.is_empty()) {
		return;
	}

	if (p_call_flags & GROUP_CALL_UNIQUE && !(p_call_flags & GROUP_CALL_REALTIME)) {
		ERR_FAIL_COND(ugc_locked);

		UGCall ug;
		ug.call = p_function;
		ug.group = p_group;

		if (unique_group_calls.has(ug)) {
			return;
		}

		VARIANT_ARGPTRS;

		Vector<Variant> args;
		for (int i = 0; i < VARIANT_ARG_MAX; i++) {
			if (argptr[i]->get_type() == Variant::NIL) {
				break;
			}
			args.push_back(*argptr[i]);
		}

		unique_group_calls[ug] = args;
		return;
	}

	_update_group_order(g);

	Vector<Node *> nodes_copy = g.nodes;
	Node **nodes = nodes_copy.ptrw();
	int node_count = nodes_copy.size();

	call_lock++;

	if (p_call_flags & GROUP_CALL_REVERSE) {
		for (int i = node_count - 1; i >= 0; i--) {
			if (call_lock && call_skip.has(nodes[i])) {
				continue;
			}

			if (p_call_flags & GROUP_CALL_REALTIME) {
				nodes[i]->call(p_function, VARIANT_ARG_PASS);
			} else {
				MessageQueue::get_singleton()->push_call(nodes[i], p_function, VARIANT_ARG_PASS);
			}
		}

	} else {
		for (int i = 0; i < node_count; i++) {
			if (call_lock && call_skip.has(nodes[i])) {
				continue;
			}

			if (p_call_flags & GROUP_CALL_REALTIME) {
				nodes[i]->call(p_function, VARIANT_ARG_PASS);
			} else {
				MessageQueue::get_singleton()->push_call(nodes[i], p_function, VARIANT_ARG_PASS);
			}
		}
	}

	call_lock--;
	if (call_lock == 0) {
		call_skip.clear();
	}
}

void SceneTree::notify_group_flags(uint32_t p_call_flags, const StringName &p_group, int p_notification) {
	Map<StringName, Group>::Element *E = group_map.find(p_group);
	if (!E) {
		return;
	}
	Group &g = E->get();
	if (g.nodes.is_empty()) {
		return;
	}

	_update_group_order(g);

	Vector<Node *> nodes_copy = g.nodes;
	Node **nodes = nodes_copy.ptrw();
	int node_count = nodes_copy.size();

	call_lock++;

	if (p_call_flags & GROUP_CALL_REVERSE) {
		for (int i = node_count - 1; i >= 0; i--) {
			if (call_lock && call_skip.has(nodes[i])) {
				continue;
			}

			if (p_call_flags & GROUP_CALL_REALTIME) {
				nodes[i]->notification(p_notification);
			} else {
				MessageQueue::get_singleton()->push_notification(nodes[i], p_notification);
			}
		}

	} else {
		for (int i = 0; i < node_count; i++) {
			if (call_lock && call_skip.has(nodes[i])) {
				continue;
			}

			if (p_call_flags & GROUP_CALL_REALTIME) {
				nodes[i]->notification(p_notification);
			} else {
				MessageQueue::get_singleton()->push_notification(nodes[i], p_notification);
			}
		}
	}

	call_lock--;
	if (call_lock == 0) {
		call_skip.clear();
	}
}

void SceneTree::set_group_flags(uint32_t p_call_flags, const StringName &p_group, const String &p_name, const Variant &p_value) {
	Map<StringName, Group>::Element *E = group_map.find(p_group);
	if (!E) {
		return;
	}
	Group &g = E->get();
	if (g.nodes.is_empty()) {
		return;
	}

	_update_group_order(g);

	Vector<Node *> nodes_copy = g.nodes;
	Node **nodes = nodes_copy.ptrw();
	int node_count = nodes_copy.size();

	call_lock++;

	if (p_call_flags & GROUP_CALL_REVERSE) {
		for (int i = node_count - 1; i >= 0; i--) {
			if (call_lock && call_skip.has(nodes[i])) {
				continue;
			}

			if (p_call_flags & GROUP_CALL_REALTIME) {
				nodes[i]->set(p_name, p_value);
			} else {
				MessageQueue::get_singleton()->push_set(nodes[i], p_name, p_value);
			}
		}

	} else {
		for (int i = 0; i < node_count; i++) {
			if (call_lock && call_skip.has(nodes[i])) {
				continue;
			}

			if (p_call_flags & GROUP_CALL_REALTIME) {
				nodes[i]->set(p_name, p_value);
			} else {
				MessageQueue::get_singleton()->push_set(nodes[i], p_name, p_value);
			}
		}
	}

	call_lock--;
	if (call_lock == 0) {
		call_skip.clear();
	}
}

void SceneTree::call_group(const StringName &p_group, const StringName &p_function, VARIANT_ARG_DECLARE) {
	call_group_flags(0, p_group, p_function, VARIANT_ARG_PASS);
}

void SceneTree::notify_group(const StringName &p_group, int p_notification) {
	notify_group_flags(0, p_group, p_notification);
}

void SceneTree::set_group(const StringName &p_group, const String &p_name, const Variant &p_value) {
	set_group_flags(0, p_group, p_name, p_value);
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

	_notify_group_pause(SNAME("physics_process_internal"), Node::NOTIFICATION_INTERNAL_PHYSICS_PROCESS);
	call_group_flags(GROUP_CALL_REALTIME, SNAME("_picking_viewports"), SNAME("_process_picking"));
	_notify_group_pause(SNAME("physics_process"), Node::NOTIFICATION_PHYSICS_PROCESS);
	_flush_ugc();
	MessageQueue::get_singleton()->flush(); //small little hack

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
	}

	emit_signal(SNAME("process_frame"));

	MessageQueue::get_singleton()->flush(); //small little hack

	flush_transform_notifications();

	_notify_group_pause(SNAME("process_internal"), Node::NOTIFICATION_INTERNAL_PROCESS);
	_notify_group_pause(SNAME("process"), Node::NOTIFICATION_PROCESS);

	_flush_ugc();
	MessageQueue::get_singleton()->flush(); //small little hack
	flush_transform_notifications(); //transforms after world update, to avoid unnecessary enter/exit notifications

	root_lock--;

	_flush_delete_queue();

	//go through timers

	List<Ref<SceneTreeTimer>>::Element *L = timers.back(); //last element

	for (List<Ref<SceneTreeTimer>>::Element *E = timers.front(); E;) {
		List<Ref<SceneTreeTimer>>::Element *N = E->next();
		if (paused && !E->get()->is_process_always()) {
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
			time_left -= p_time;
		}
		E->get()->set_time_left(time_left);

		if (time_left < 0) {
			E->get()->emit_signal(SNAME("timeout"));
			timers.erase(E);
		}
		if (E == L) {
			break; //break on last, so if new timers were added during list traversal, ignore them.
		}
		E = N;
	}

	process_tweens(p_time, false);

	flush_transform_notifications(); //additional transforms after timers update

	_call_idle_callbacks();

#ifdef TOOLS_ENABLED
#ifndef _3D_DISABLED
	if (Engine::get_singleton()->is_editor_hint()) {
		//simple hack to reload fallback environment if it changed from editor
		String env_path = ProjectSettings::get_singleton()->get(SNAME("rendering/environment/defaults/default_environment"));
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

void SceneTree::process_tweens(float p_delta, bool p_physics) {
	// This methods works similarly to how SceneTreeTimers are handled.
	List<Ref<Tween>>::Element *L = tweens.back();

	for (List<Ref<Tween>>::Element *E = tweens.front(); E;) {
		List<Ref<Tween>>::Element *N = E->next();
		// Don't process if paused or process mode doesn't match.
		if ((paused && E->get()->should_pause()) || (p_physics == (E->get()->get_process_mode() == Tween::TWEEN_PROCESS_IDLE))) {
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
			get_root()->propagate_notification(p_notification); //pass these to nodes, since they are mirrored
		} break;

		default:
			break;
	};
};

void SceneTree::set_auto_accept_quit(bool p_enable) {
	accept_quit = p_enable;
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

void SceneTree::set_debug_navigation_color(const Color &p_color) {
	debug_navigation_color = p_color;
}

Color SceneTree::get_debug_navigation_color() const {
	return debug_navigation_color;
}

void SceneTree::set_debug_navigation_disabled_color(const Color &p_color) {
	debug_navigation_disabled_color = p_color;
}

Color SceneTree::get_debug_navigation_disabled_color() const {
	return debug_navigation_disabled_color;
}

Ref<Material> SceneTree::get_debug_navigation_material() {
	if (navigation_material.is_valid()) {
		return navigation_material;
	}

	Ref<StandardMaterial3D> line_material = Ref<StandardMaterial3D>(memnew(StandardMaterial3D));
	line_material->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
	line_material->set_transparency(StandardMaterial3D::TRANSPARENCY_ALPHA);
	line_material->set_flag(StandardMaterial3D::FLAG_SRGB_VERTEX_COLOR, true);
	line_material->set_flag(StandardMaterial3D::FLAG_ALBEDO_FROM_VERTEX_COLOR, true);
	line_material->set_albedo(get_debug_navigation_color());

	navigation_material = line_material;

	return navigation_material;
}

Ref<Material> SceneTree::get_debug_navigation_disabled_material() {
	if (navigation_disabled_material.is_valid()) {
		return navigation_disabled_material;
	}

	Ref<StandardMaterial3D> line_material = Ref<StandardMaterial3D>(memnew(StandardMaterial3D));
	line_material->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
	line_material->set_transparency(StandardMaterial3D::TRANSPARENCY_ALPHA);
	line_material->set_flag(StandardMaterial3D::FLAG_SRGB_VERTEX_COLOR, true);
	line_material->set_flag(StandardMaterial3D::FLAG_ALBEDO_FROM_VERTEX_COLOR, true);
	line_material->set_albedo(get_debug_navigation_disabled_color());

	navigation_disabled_material = line_material;

	return navigation_disabled_material;
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
	Map<StringName, Group>::Element *E = group_map.find(p_group);
	if (!E) {
		return;
	}
	Group &g = E->get();
	if (g.nodes.is_empty()) {
		return;
	}

	_update_group_order(g, p_notification == Node::NOTIFICATION_PROCESS || p_notification == Node::NOTIFICATION_INTERNAL_PROCESS || p_notification == Node::NOTIFICATION_PHYSICS_PROCESS || p_notification == Node::NOTIFICATION_INTERNAL_PHYSICS_PROCESS);

	//copy, so copy on write happens in case something is removed from process while being called
	//performance is not lost because only if something is added/removed the vector is copied.
	Vector<Node *> nodes_copy = g.nodes;

	int node_count = nodes_copy.size();
	Node **nodes = nodes_copy.ptrw();

	call_lock++;

	for (int i = 0; i < node_count; i++) {
		Node *n = nodes[i];
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
		//ERR_FAIL_COND(node_count != g.nodes.size());
	}

	call_lock--;
	if (call_lock == 0) {
		call_skip.clear();
	}
}

void SceneTree::_call_input_pause(const StringName &p_group, CallInputType p_call_type, const Ref<InputEvent> &p_input, Viewport *p_viewport) {
	Map<StringName, Group>::Element *E = group_map.find(p_group);
	if (!E) {
		return;
	}
	Group &g = E->get();
	if (g.nodes.is_empty()) {
		return;
	}

	_update_group_order(g);

	//copy, so copy on write happens in case something is removed from process while being called
	//performance is not lost because only if something is added/removed the vector is copied.
	Vector<Node *> nodes_copy = g.nodes;

	int node_count = nodes_copy.size();
	Node **nodes = nodes_copy.ptrw();

	call_lock++;

	for (int i = node_count - 1; i >= 0; i--) {
		if (p_viewport->is_input_handled()) {
			break;
		}

		Node *n = nodes[i];
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
			case CALL_INPUT_TYPE_UNHANDLED_INPUT:
				n->_call_unhandled_input(p_input);
				break;
			case CALL_INPUT_TYPE_UNHANDLED_KEY_INPUT:
				n->_call_unhandled_key_input(p_input);
				break;
		}
	}

	call_lock--;
	if (call_lock == 0) {
		call_skip.clear();
	}
}

Variant SceneTree::_call_group_flags(const Variant **p_args, int p_argcount, Callable::CallError &r_error) {
	r_error.error = Callable::CallError::CALL_OK;

	ERR_FAIL_COND_V(p_argcount < 3, Variant());
	ERR_FAIL_COND_V(!p_args[0]->is_num(), Variant());
	ERR_FAIL_COND_V(p_args[1]->get_type() != Variant::STRING_NAME && p_args[1]->get_type() != Variant::STRING, Variant());
	ERR_FAIL_COND_V(p_args[2]->get_type() != Variant::STRING_NAME && p_args[2]->get_type() != Variant::STRING, Variant());

	int flags = *p_args[0];
	StringName group = *p_args[1];
	StringName method = *p_args[2];
	Variant v[VARIANT_ARG_MAX];

	for (int i = 0; i < MIN(p_argcount - 3, 5); i++) {
		v[i] = *p_args[i + 3];
	}

	static_assert(VARIANT_ARG_MAX == 8, "This code needs to be updated if VARIANT_ARG_MAX != 8");
	call_group_flags(flags, group, method, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
	return Variant();
}

Variant SceneTree::_call_group(const Variant **p_args, int p_argcount, Callable::CallError &r_error) {
	r_error.error = Callable::CallError::CALL_OK;

	ERR_FAIL_COND_V(p_argcount < 2, Variant());
	ERR_FAIL_COND_V(p_args[0]->get_type() != Variant::STRING_NAME && p_args[0]->get_type() != Variant::STRING, Variant());
	ERR_FAIL_COND_V(p_args[1]->get_type() != Variant::STRING_NAME && p_args[1]->get_type() != Variant::STRING, Variant());

	StringName group = *p_args[0];
	StringName method = *p_args[1];
	Variant v[VARIANT_ARG_MAX];

	for (int i = 0; i < MIN(p_argcount - 2, 5); i++) {
		v[i] = *p_args[i + 2];
	}

	static_assert(VARIANT_ARG_MAX == 8, "This code needs to be updated if VARIANT_ARG_MAX != 8");
	call_group_flags(0, group, method, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
	return Variant();
}

int64_t SceneTree::get_frame() const {
	return current_frame;
}

Array SceneTree::_get_nodes_in_group(const StringName &p_group) {
	Array ret;
	Map<StringName, Group>::Element *E = group_map.find(p_group);
	if (!E) {
		return ret;
	}

	_update_group_order(E->get()); //update order just in case
	int nc = E->get().nodes.size();
	if (nc == 0) {
		return ret;
	}

	ret.resize(nc);

	Node **ptr = E->get().nodes.ptrw();
	for (int i = 0; i < nc; i++) {
		ret[i] = ptr[i];
	}

	return ret;
}

bool SceneTree::has_group(const StringName &p_identifier) const {
	return group_map.has(p_identifier);
}

Node *SceneTree::get_first_node_in_group(const StringName &p_group) {
	Map<StringName, Group>::Element *E = group_map.find(p_group);
	if (!E) {
		return nullptr; //no group
	}

	_update_group_order(E->get()); //update order just in case

	if (E->get().nodes.size() == 0) {
		return nullptr;
	}

	return E->get().nodes[0];
}

void SceneTree::get_nodes_in_group(const StringName &p_group, List<Node *> *p_list) {
	Map<StringName, Group>::Element *E = group_map.find(p_group);
	if (!E) {
		return;
	}

	_update_group_order(E->get()); //update order just in case
	int nc = E->get().nodes.size();
	if (nc == 0) {
		return;
	}
	Node **ptr = E->get().nodes.ptrw();
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
	}
}

Error SceneTree::change_scene(const String &p_path) {
	Ref<PackedScene> new_scene = ResourceLoader::load(p_path);
	if (new_scene.is_null()) {
		return ERR_CANT_OPEN;
	}

	return change_scene_to(new_scene);
}

Error SceneTree::change_scene_to(const Ref<PackedScene> &p_scene) {
	Node *new_scene = nullptr;
	if (p_scene.is_valid()) {
		new_scene = p_scene->instantiate();
		ERR_FAIL_COND_V(!new_scene, ERR_CANT_CREATE);
	}

	call_deferred(SNAME("_change_scene"), new_scene);
	return OK;
}

Error SceneTree::reload_current_scene() {
	ERR_FAIL_COND_V(!current_scene, ERR_UNCONFIGURED);
	String fname = current_scene->get_scene_file_path();
	return change_scene(fname);
}

void SceneTree::add_current_scene(Node *p_current) {
	current_scene = p_current;
	root->add_child(p_current);
}

Ref<SceneTreeTimer> SceneTree::create_timer(double p_delay_sec, bool p_process_always) {
	Ref<SceneTreeTimer> stt;
	stt.instantiate();
	stt->set_process_always(p_process_always);
	stt->set_time_left(p_delay_sec);
	timers.push_back(stt);
	return stt;
}

Ref<Tween> SceneTree::create_tween() {
	Ref<Tween> tween;
	tween.instantiate();
	tween->set_valid(true);
	tweens.push_back(tween);
	return tween;
}

Array SceneTree::get_processed_tweens() {
	Array ret;
	ret.resize(tweens.size());

	int i = 0;
	for (const Ref<Tween> &tween : tweens) {
		ret[i] = tween;
		i++;
	}

	return ret;
}

Ref<MultiplayerAPI> SceneTree::get_multiplayer() const {
	return multiplayer;
}

void SceneTree::set_multiplayer_poll_enabled(bool p_enabled) {
	multiplayer_poll = p_enabled;
}

bool SceneTree::is_multiplayer_poll_enabled() const {
	return multiplayer_poll;
}

void SceneTree::set_multiplayer(Ref<MultiplayerAPI> p_multiplayer) {
	ERR_FAIL_COND(!p_multiplayer.is_valid());

	multiplayer = p_multiplayer;
	multiplayer->set_root_node(root);
}

void SceneTree::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_root"), &SceneTree::get_root);
	ClassDB::bind_method(D_METHOD("has_group", "name"), &SceneTree::has_group);

	ClassDB::bind_method(D_METHOD("set_auto_accept_quit", "enabled"), &SceneTree::set_auto_accept_quit);
	ClassDB::bind_method(D_METHOD("set_quit_on_go_back", "enabled"), &SceneTree::set_quit_on_go_back);

	ClassDB::bind_method(D_METHOD("set_debug_collisions_hint", "enable"), &SceneTree::set_debug_collisions_hint);
	ClassDB::bind_method(D_METHOD("is_debugging_collisions_hint"), &SceneTree::is_debugging_collisions_hint);
	ClassDB::bind_method(D_METHOD("set_debug_navigation_hint", "enable"), &SceneTree::set_debug_navigation_hint);
	ClassDB::bind_method(D_METHOD("is_debugging_navigation_hint"), &SceneTree::is_debugging_navigation_hint);

	ClassDB::bind_method(D_METHOD("set_edited_scene_root", "scene"), &SceneTree::set_edited_scene_root);
	ClassDB::bind_method(D_METHOD("get_edited_scene_root"), &SceneTree::get_edited_scene_root);

	ClassDB::bind_method(D_METHOD("set_pause", "enable"), &SceneTree::set_pause);
	ClassDB::bind_method(D_METHOD("is_paused"), &SceneTree::is_paused);

	ClassDB::bind_method(D_METHOD("create_timer", "time_sec", "process_always"), &SceneTree::create_timer, DEFVAL(true));
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

	ClassDB::bind_method(D_METHOD("change_scene", "path"), &SceneTree::change_scene);
	ClassDB::bind_method(D_METHOD("change_scene_to", "packed_scene"), &SceneTree::change_scene_to);

	ClassDB::bind_method(D_METHOD("reload_current_scene"), &SceneTree::reload_current_scene);

	ClassDB::bind_method(D_METHOD("_change_scene"), &SceneTree::_change_scene);

	ClassDB::bind_method(D_METHOD("set_multiplayer", "multiplayer"), &SceneTree::set_multiplayer);
	ClassDB::bind_method(D_METHOD("get_multiplayer"), &SceneTree::get_multiplayer);
	ClassDB::bind_method(D_METHOD("set_multiplayer_poll_enabled", "enabled"), &SceneTree::set_multiplayer_poll_enabled);
	ClassDB::bind_method(D_METHOD("is_multiplayer_poll_enabled"), &SceneTree::is_multiplayer_poll_enabled);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "debug_collisions_hint"), "set_debug_collisions_hint", "is_debugging_collisions_hint");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "debug_navigation_hint"), "set_debug_navigation_hint", "is_debugging_navigation_hint");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "paused"), "set_pause", "is_paused");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "edited_scene_root", PROPERTY_HINT_RESOURCE_TYPE, "Node", PROPERTY_USAGE_NONE), "set_edited_scene_root", "get_edited_scene_root");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "current_scene", PROPERTY_HINT_RESOURCE_TYPE, "Node", PROPERTY_USAGE_NONE), "set_current_scene", "get_current_scene");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "root", PROPERTY_HINT_RESOURCE_TYPE, "Node", PROPERTY_USAGE_NONE), "", "get_root");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "multiplayer", PROPERTY_HINT_RESOURCE_TYPE, "MultiplayerAPI", PROPERTY_USAGE_NONE), "set_multiplayer", "get_multiplayer");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "multiplayer_poll"), "set_multiplayer_poll_enabled", "is_multiplayer_poll_enabled");

	ADD_SIGNAL(MethodInfo("tree_changed"));
	ADD_SIGNAL(MethodInfo("tree_process_mode_changed")); //editor only signal, but due to API hash it can't be removed in run-time
	ADD_SIGNAL(MethodInfo("node_added", PropertyInfo(Variant::OBJECT, "node", PROPERTY_HINT_RESOURCE_TYPE, "Node")));
	ADD_SIGNAL(MethodInfo("node_removed", PropertyInfo(Variant::OBJECT, "node", PROPERTY_HINT_RESOURCE_TYPE, "Node")));
	ADD_SIGNAL(MethodInfo("node_renamed", PropertyInfo(Variant::OBJECT, "node", PROPERTY_HINT_RESOURCE_TYPE, "Node")));
	ADD_SIGNAL(MethodInfo("node_configuration_warning_changed", PropertyInfo(Variant::OBJECT, "node", PROPERTY_HINT_RESOURCE_TYPE, "Node")));

	ADD_SIGNAL(MethodInfo("process_frame"));
	ADD_SIGNAL(MethodInfo("physics_frame"));

	ADD_SIGNAL(MethodInfo("files_dropped", PropertyInfo(Variant::PACKED_STRING_ARRAY, "files"), PropertyInfo(Variant::INT, "screen")));

	BIND_ENUM_CONSTANT(GROUP_CALL_DEFAULT);
	BIND_ENUM_CONSTANT(GROUP_CALL_REVERSE);
	BIND_ENUM_CONSTANT(GROUP_CALL_REALTIME);
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
	if (p_function == "change_scene") {
		DirAccessRef dir_access = DirAccess::create(DirAccess::ACCESS_RESOURCES);
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
					directories.push_back(dir_access->get_current_dir().plus_file(filename));
				} else if (filename.ends_with(".tscn") || filename.ends_with(".scn")) {
					r_options->push_back("\"" + dir_access->get_current_dir().plus_file(filename) + "\"");
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
	debug_navigation_color = GLOBAL_DEF("debug/shapes/navigation/geometry_color", Color(0.1, 1.0, 0.7, 0.4));
	debug_navigation_disabled_color = GLOBAL_DEF("debug/shapes/navigation/disabled_geometry_color", Color(1.0, 0.7, 0.1, 0.4));
	collision_debug_contacts = GLOBAL_DEF("debug/shapes/collision/max_contacts_displayed", 10000);
	ProjectSettings::get_singleton()->set_custom_property_info("debug/shapes/collision/max_contacts_displayed", PropertyInfo(Variant::INT, "debug/shapes/collision/max_contacts_displayed", PROPERTY_HINT_RANGE, "0,20000,1")); // No negative

	GLOBAL_DEF("debug/shapes/collision/draw_2d_outlines", true);

	Math::randomize();

	// Create with mainloop.

	root = memnew(Window);
	root->set_process_mode(Node::PROCESS_MODE_PAUSABLE);
	root->set_name("root");
#ifndef _3D_DISABLED
	if (!root->get_world_3d().is_valid()) {
		root->set_world_3d(Ref<World3D>(memnew(World3D)));
	}
	root->set_as_audio_listener_3d(true);
#endif // _3D_DISABLED

	// Initialize network state.
	set_multiplayer(Ref<MultiplayerAPI>(memnew(MultiplayerAPI)));

	root->set_as_audio_listener_2d(true);
	current_scene = nullptr;

	const int msaa_mode = GLOBAL_DEF("rendering/anti_aliasing/quality/msaa", 0);
	ProjectSettings::get_singleton()->set_custom_property_info("rendering/anti_aliasing/quality/msaa", PropertyInfo(Variant::INT, "rendering/anti_aliasing/quality/msaa", PROPERTY_HINT_ENUM, String::utf8("Disabled (Fastest),2× (Average),4× (Slow),8× (Slowest)")));
	root->set_msaa(Viewport::MSAA(msaa_mode));

	const int ssaa_mode = GLOBAL_DEF("rendering/anti_aliasing/quality/screen_space_aa", 0);
	ProjectSettings::get_singleton()->set_custom_property_info("rendering/anti_aliasing/quality/screen_space_aa", PropertyInfo(Variant::INT, "rendering/anti_aliasing/quality/screen_space_aa", PROPERTY_HINT_ENUM, "Disabled (Fastest),FXAA (Fast)"));
	root->set_screen_space_aa(Viewport::ScreenSpaceAA(ssaa_mode));

	const bool use_debanding = GLOBAL_DEF("rendering/anti_aliasing/quality/use_debanding", false);
	root->set_use_debanding(use_debanding);

	const bool use_occlusion_culling = GLOBAL_DEF("rendering/occlusion_culling/use_occlusion_culling", false);
	root->set_use_occlusion_culling(use_occlusion_culling);

	float mesh_lod_threshold = GLOBAL_DEF("rendering/mesh_lod/lod_change/threshold_pixels", 1.0);
	ProjectSettings::get_singleton()->set_custom_property_info("rendering/mesh_lod/lod_change/threshold_pixels", PropertyInfo(Variant::FLOAT, "rendering/mesh_lod/lod_change/threshold_pixels", PROPERTY_HINT_RANGE, "0,1024,0.1"));
	root->set_mesh_lod_threshold(mesh_lod_threshold);

	bool snap_2d_transforms = GLOBAL_DEF("rendering/2d/snap/snap_2d_transforms_to_pixel", false);
	root->set_snap_2d_transforms_to_pixel(snap_2d_transforms);

	bool snap_2d_vertices = GLOBAL_DEF("rendering/2d/snap/snap_2d_vertices_to_pixel", false);
	root->set_snap_2d_vertices_to_pixel(snap_2d_vertices);

	int shadowmap_size = GLOBAL_DEF("rendering/shadows/shadow_atlas/size", 4096);
	ProjectSettings::get_singleton()->set_custom_property_info("rendering/shadows/shadow_atlas/size", PropertyInfo(Variant::INT, "rendering/shadows/shadow_atlas/size", PROPERTY_HINT_RANGE, "256,16384"));
	GLOBAL_DEF("rendering/shadows/shadow_atlas/size.mobile", 2048);
	bool shadowmap_16_bits = GLOBAL_DEF("rendering/shadows/shadow_atlas/16_bits", true);
	int atlas_q0 = GLOBAL_DEF("rendering/shadows/shadow_atlas/quadrant_0_subdiv", 2);
	int atlas_q1 = GLOBAL_DEF("rendering/shadows/shadow_atlas/quadrant_1_subdiv", 2);
	int atlas_q2 = GLOBAL_DEF("rendering/shadows/shadow_atlas/quadrant_2_subdiv", 3);
	int atlas_q3 = GLOBAL_DEF("rendering/shadows/shadow_atlas/quadrant_3_subdiv", 4);
	ProjectSettings::get_singleton()->set_custom_property_info("rendering/shadows/shadow_atlas/quadrant_0_subdiv", PropertyInfo(Variant::INT, "rendering/shadows/shadow_atlas/quadrant_0_subdiv", PROPERTY_HINT_ENUM, "Disabled,1 Shadow,4 Shadows,16 Shadows,64 Shadows,256 Shadows,1024 Shadows"));
	ProjectSettings::get_singleton()->set_custom_property_info("rendering/shadows/shadow_atlas/quadrant_1_subdiv", PropertyInfo(Variant::INT, "rendering/shadows/shadow_atlas/quadrant_1_subdiv", PROPERTY_HINT_ENUM, "Disabled,1 Shadow,4 Shadows,16 Shadows,64 Shadows,256 Shadows,1024 Shadows"));
	ProjectSettings::get_singleton()->set_custom_property_info("rendering/shadows/shadow_atlas/quadrant_2_subdiv", PropertyInfo(Variant::INT, "rendering/shadows/shadow_atlas/quadrant_2_subdiv", PROPERTY_HINT_ENUM, "Disabled,1 Shadow,4 Shadows,16 Shadows,64 Shadows,256 Shadows,1024 Shadows"));
	ProjectSettings::get_singleton()->set_custom_property_info("rendering/shadows/shadow_atlas/quadrant_3_subdiv", PropertyInfo(Variant::INT, "rendering/shadows/shadow_atlas/quadrant_3_subdiv", PROPERTY_HINT_ENUM, "Disabled,1 Shadow,4 Shadows,16 Shadows,64 Shadows,256 Shadows,1024 Shadows"));

	root->set_shadow_atlas_size(shadowmap_size);
	root->set_shadow_atlas_16_bits(shadowmap_16_bits);
	root->set_shadow_atlas_quadrant_subdiv(0, Viewport::ShadowAtlasQuadrantSubdiv(atlas_q0));
	root->set_shadow_atlas_quadrant_subdiv(1, Viewport::ShadowAtlasQuadrantSubdiv(atlas_q1));
	root->set_shadow_atlas_quadrant_subdiv(2, Viewport::ShadowAtlasQuadrantSubdiv(atlas_q2));
	root->set_shadow_atlas_quadrant_subdiv(3, Viewport::ShadowAtlasQuadrantSubdiv(atlas_q3));

	Viewport::SDFOversize sdf_oversize = Viewport::SDFOversize(int(GLOBAL_DEF("rendering/2d/sdf/oversize", 1)));
	root->set_sdf_oversize(sdf_oversize);
	Viewport::SDFScale sdf_scale = Viewport::SDFScale(int(GLOBAL_DEF("rendering/2d/sdf/scale", 1)));
	root->set_sdf_scale(sdf_scale);

	ProjectSettings::get_singleton()->set_custom_property_info("rendering/2d/sdf/oversize", PropertyInfo(Variant::INT, "rendering/2d/sdf/oversize", PROPERTY_HINT_ENUM, "100%,120%,150%,200%"));
	ProjectSettings::get_singleton()->set_custom_property_info("rendering/2d/sdf/scale", PropertyInfo(Variant::INT, "rendering/2d/sdf/scale", PROPERTY_HINT_ENUM, "100%,50%,25%"));

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
		String env_path = GLOBAL_DEF("rendering/environment/defaults/default_environment", "");
		// Setup property.
		ProjectSettings::get_singleton()->set_custom_property_info("rendering/environment/defaults/default_environment", PropertyInfo(Variant::STRING, "rendering/viewport/default_environment", PROPERTY_HINT_FILE, ext_hint));
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
