/*************************************************************************/
/*  scene_tree.cpp                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "editor/editor_node.h"
#include "io/marshalls.h"
#include "io/resource_loader.h"
#include "message_queue.h"
#include "node.h"
#include "os/keyboard.h"
#include "os/os.h"
#include "print_string.h"
#include "project_settings.h"
#include "scene/resources/dynamic_font.h"
#include "scene/resources/material.h"
#include "scene/resources/mesh.h"
#include "scene/resources/packed_scene.h"
#include "scene/scene_string_names.h"
#include "servers/physics_2d_server.h"
#include "servers/physics_server.h"
#include "viewport.h"

#include <stdio.h>

void SceneTreeTimer::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_time_left", "time"), &SceneTreeTimer::set_time_left);
	ClassDB::bind_method(D_METHOD("get_time_left"), &SceneTreeTimer::get_time_left);

	ADD_SIGNAL(MethodInfo("timeout"));
}

void SceneTreeTimer::set_time_left(float p_time) {
	time_left = p_time;
}

float SceneTreeTimer::get_time_left() const {
	return time_left;
}

void SceneTreeTimer::set_pause_mode_process(bool p_pause_mode_process) {
	if (process_pause != p_pause_mode_process) {
		process_pause = p_pause_mode_process;
	}
}

bool SceneTreeTimer::is_pause_mode_process() {
	return process_pause;
}

SceneTreeTimer::SceneTreeTimer() {
	time_left = 0;
	process_pause = true;
}

void SceneTree::tree_changed() {

	tree_version++;
	emit_signal(tree_changed_name);
}

void SceneTree::node_added(Node *p_node) {

	emit_signal(node_added_name, p_node);
}

void SceneTree::node_removed(Node *p_node) {

	if (current_scene == p_node) {
		current_scene = NULL;
	}
	emit_signal(node_removed_name, p_node);
	if (call_lock > 0)
		call_skip.insert(p_node);
}

SceneTree::Group *SceneTree::add_to_group(const StringName &p_group, Node *p_node) {

	Map<StringName, Group>::Element *E = group_map.find(p_group);
	if (!E) {
		E = group_map.insert(p_group, Group());
	}

	if (E->get().nodes.find(p_node) != -1) {
		ERR_EXPLAIN("Already in group: " + p_group);
		ERR_FAIL_V(&E->get());
	}
	E->get().nodes.push_back(p_node);
	//E->get().last_tree_version=0;
	E->get().changed = true;
	return &E->get();
}

void SceneTree::remove_from_group(const StringName &p_group, Node *p_node) {

	Map<StringName, Group>::Element *E = group_map.find(p_group);
	ERR_FAIL_COND(!E);

	E->get().nodes.erase(p_node);
	if (E->get().nodes.empty())
		group_map.erase(E);
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

		Map<UGCall, Vector<Variant> >::Element *E = unique_group_calls.front();

		Variant v[VARIANT_ARG_MAX];
		for (int i = 0; i < E->get().size(); i++)
			v[i] = E->get()[i];

		call_group_flags(GROUP_CALL_REALTIME, E->key().group, E->key().call, v[0], v[1], v[2], v[3], v[4]);

		unique_group_calls.erase(E);
	}

	ugc_locked = false;
}

void SceneTree::_update_group_order(Group &g) {

	if (!g.changed)
		return;
	if (g.nodes.empty())
		return;

	Node **nodes = &g.nodes[0];
	int node_count = g.nodes.size();

	SortArray<Node *, Node::Comparator> node_sort;
	node_sort.sort(nodes, node_count);
	g.changed = false;
}

void SceneTree::call_group_flags(uint32_t p_call_flags, const StringName &p_group, const StringName &p_function, VARIANT_ARG_DECLARE) {

	Map<StringName, Group>::Element *E = group_map.find(p_group);
	if (!E)
		return;
	Group &g = E->get();
	if (g.nodes.empty())
		return;

	if (p_call_flags & GROUP_CALL_UNIQUE && !(p_call_flags & GROUP_CALL_REALTIME)) {

		ERR_FAIL_COND(ugc_locked);

		UGCall ug;
		ug.call = p_function;
		ug.group = p_group;

		if (unique_group_calls.has(ug))
			return;

		VARIANT_ARGPTRS;

		Vector<Variant> args;
		for (int i = 0; i < VARIANT_ARG_MAX; i++) {
			if (argptr[i]->get_type() == Variant::NIL)
				break;
			args.push_back(*argptr[i]);
		}

		unique_group_calls[ug] = args;
		return;
	}

	_update_group_order(g);

	Vector<Node *> nodes_copy = g.nodes;
	Node **nodes = &nodes_copy[0];
	int node_count = nodes_copy.size();

	call_lock++;

	if (p_call_flags & GROUP_CALL_REVERSE) {

		for (int i = node_count - 1; i >= 0; i--) {

			if (call_lock && call_skip.has(nodes[i]))
				continue;

			if (p_call_flags & GROUP_CALL_REALTIME) {
				if (p_call_flags & GROUP_CALL_MULTILEVEL)
					nodes[i]->call_multilevel(p_function, VARIANT_ARG_PASS);
				else
					nodes[i]->call(p_function, VARIANT_ARG_PASS);
			} else
				MessageQueue::get_singleton()->push_call(nodes[i], p_function, VARIANT_ARG_PASS);
		}

	} else {

		for (int i = 0; i < node_count; i++) {

			if (call_lock && call_skip.has(nodes[i]))
				continue;

			if (p_call_flags & GROUP_CALL_REALTIME) {
				if (p_call_flags & GROUP_CALL_MULTILEVEL)
					nodes[i]->call_multilevel(p_function, VARIANT_ARG_PASS);
				else
					nodes[i]->call(p_function, VARIANT_ARG_PASS);
			} else
				MessageQueue::get_singleton()->push_call(nodes[i], p_function, VARIANT_ARG_PASS);
		}
	}

	call_lock--;
	if (call_lock == 0)
		call_skip.clear();
}

void SceneTree::notify_group_flags(uint32_t p_call_flags, const StringName &p_group, int p_notification) {

	Map<StringName, Group>::Element *E = group_map.find(p_group);
	if (!E)
		return;
	Group &g = E->get();
	if (g.nodes.empty())
		return;

	_update_group_order(g);

	Vector<Node *> nodes_copy = g.nodes;
	Node **nodes = &nodes_copy[0];
	int node_count = nodes_copy.size();

	call_lock++;

	if (p_call_flags & GROUP_CALL_REVERSE) {

		for (int i = node_count - 1; i >= 0; i--) {

			if (call_lock && call_skip.has(nodes[i]))
				continue;

			if (p_call_flags & GROUP_CALL_REALTIME)
				nodes[i]->notification(p_notification);
			else
				MessageQueue::get_singleton()->push_notification(nodes[i], p_notification);
		}

	} else {

		for (int i = 0; i < node_count; i++) {

			if (call_lock && call_skip.has(nodes[i]))
				continue;

			if (p_call_flags & GROUP_CALL_REALTIME)
				nodes[i]->notification(p_notification);
			else
				MessageQueue::get_singleton()->push_notification(nodes[i], p_notification);
		}
	}

	call_lock--;
	if (call_lock == 0)
		call_skip.clear();
}

void SceneTree::set_group_flags(uint32_t p_call_flags, const StringName &p_group, const String &p_name, const Variant &p_value) {

	Map<StringName, Group>::Element *E = group_map.find(p_group);
	if (!E)
		return;
	Group &g = E->get();
	if (g.nodes.empty())
		return;

	_update_group_order(g);

	Vector<Node *> nodes_copy = g.nodes;
	Node **nodes = &nodes_copy[0];
	int node_count = nodes_copy.size();

	call_lock++;

	if (p_call_flags & GROUP_CALL_REVERSE) {

		for (int i = node_count - 1; i >= 0; i--) {

			if (call_lock && call_skip.has(nodes[i]))
				continue;

			if (p_call_flags & GROUP_CALL_REALTIME)
				nodes[i]->set(p_name, p_value);
			else
				MessageQueue::get_singleton()->push_set(nodes[i], p_name, p_value);
		}

	} else {

		for (int i = 0; i < node_count; i++) {

			if (call_lock && call_skip.has(nodes[i]))
				continue;

			if (p_call_flags & GROUP_CALL_REALTIME)
				nodes[i]->set(p_name, p_value);
			else
				MessageQueue::get_singleton()->push_set(nodes[i], p_name, p_value);
		}
	}

	call_lock--;
	if (call_lock == 0)
		call_skip.clear();
}

void SceneTree::call_group(const StringName &p_group, const StringName &p_function, VARIANT_ARG_DECLARE) {

	call_group_flags(0, p_group, VARIANT_ARG_PASS);
}

void SceneTree::notify_group(const StringName &p_group, int p_notification) {

	notify_group_flags(0, p_group, p_notification);
}

void SceneTree::set_group(const StringName &p_group, const String &p_name, const Variant &p_value) {

	set_group_flags(0, p_group, p_name, p_value);
}

void SceneTree::set_input_as_handled() {

	input_handled = true;
}

void SceneTree::input_text(const String &p_text) {

	root_lock++;

	call_group_flags(GROUP_CALL_REALTIME, "_viewports", "_vp_input_text", p_text); //special one for GUI, as controls use their own process check

	root_lock--;
}

bool SceneTree::is_input_handled() {
	return input_handled;
}

void SceneTree::input_event(const Ref<InputEvent> &p_event) {

	if (Engine::get_singleton()->is_editor_hint() && (Object::cast_to<InputEventJoypadButton>(p_event.ptr()) || Object::cast_to<InputEventJoypadMotion>(*p_event)))
		return; //avoid joy input on editor

	root_lock++;
	//last_id=p_event.ID;

	input_handled = false;

	Ref<InputEvent> ev = p_event;
	ev->set_id(++last_id); //this should work better

	MainLoop::input_event(ev);

	call_group_flags(GROUP_CALL_REALTIME, "_viewports", "_vp_input", ev); //special one for GUI, as controls use their own process check

	if (ScriptDebugger::get_singleton() && ScriptDebugger::get_singleton()->is_remote()) {
		//quit from game window using F8
		Ref<InputEventKey> k = ev;
		if (k.is_valid() && k->is_pressed() && !k->is_echo() && k->get_scancode() == KEY_F8) {
			ScriptDebugger::get_singleton()->request_quit();
		}
	}

	_flush_ugc();
	root_lock--;
	//MessageQueue::get_singleton()->flush(); //flushing here causes UI and other places slowness

	root_lock++;

	if (!input_handled) {
		call_group_flags(GROUP_CALL_REALTIME, "_viewports", "_vp_unhandled_input", ev); //special one for GUI, as controls use their own process check
		_flush_ugc();
		//		input_handled = true; - no reason to set this as handled
		root_lock--;
		//MessageQueue::get_singleton()->flush(); //flushing here causes UI and other places slowness
	} else {
		//		input_handled = true; - no reason to set this as handled
		root_lock--;
	}

	_call_idle_callbacks();
}

void SceneTree::init() {

	//_quit=false;
	initialized = true;
	input_handled = false;

	pause = false;

	root->_set_tree(this);
	MainLoop::init();
}

bool SceneTree::iteration(float p_time) {

	root_lock++;

	current_frame++;

	flush_transform_notifications();

	MainLoop::iteration(p_time);
	physics_process_time = p_time;

	emit_signal("physics_frame");

	_notify_group_pause("physics_process_internal", Node::NOTIFICATION_INTERNAL_PHYSICS_PROCESS);
	_notify_group_pause("physics_process", Node::NOTIFICATION_PHYSICS_PROCESS);
	_flush_ugc();
	MessageQueue::get_singleton()->flush(); //small little hack
	flush_transform_notifications();
	call_group_flags(GROUP_CALL_REALTIME, "_viewports", "update_worlds");
	root_lock--;

	_flush_delete_queue();
	_call_idle_callbacks();

	return _quit;
}

bool SceneTree::idle(float p_time) {

	//print_line("ram: "+itos(OS::get_singleton()->get_static_memory_usage())+" sram: "+itos(OS::get_singleton()->get_dynamic_memory_usage()));
	//print_line("node count: "+itos(get_node_count()));
	//print_line("TEXTURE RAM: "+itos(VS::get_singleton()->get_render_info(VS::INFO_TEXTURE_MEM_USED)));

	root_lock++;

	MainLoop::idle(p_time);

	idle_process_time = p_time;

	_network_poll();

	emit_signal("idle_frame");

	MessageQueue::get_singleton()->flush(); //small little hack

	flush_transform_notifications();

	_notify_group_pause("idle_process_internal", Node::NOTIFICATION_INTERNAL_PROCESS);
	_notify_group_pause("idle_process", Node::NOTIFICATION_PROCESS);

	Size2 win_size = Size2(OS::get_singleton()->get_video_mode().width, OS::get_singleton()->get_video_mode().height);
	if (win_size != last_screen_size) {

		if (use_font_oversampling) {
			DynamicFontAtSize::font_oversampling = OS::get_singleton()->get_window_size().width / root->get_visible_rect().size.width;
			DynamicFont::update_oversampling();
		}

		last_screen_size = win_size;
		_update_root_rect();

		emit_signal("screen_resized");
	}

	_flush_ugc();
	MessageQueue::get_singleton()->flush(); //small little hack
	flush_transform_notifications(); //transforms after world update, to avoid unnecessary enter/exit notifications
	call_group_flags(GROUP_CALL_REALTIME, "_viewports", "update_worlds");

	root_lock--;

	_flush_delete_queue();

	//go through timers

	for (List<Ref<SceneTreeTimer> >::Element *E = timers.front(); E;) {

		List<Ref<SceneTreeTimer> >::Element *N = E->next();
		if (pause && !E->get()->is_pause_mode_process()) {
			E = N;
			continue;
		}
		float time_left = E->get()->get_time_left();
		time_left -= p_time;
		E->get()->set_time_left(time_left);

		if (time_left < 0) {
			E->get()->emit_signal("timeout");
			timers.erase(E);
		}
		E = N;
	}

	_call_idle_callbacks();

#ifdef TOOLS_ENABLED

	if (Engine::get_singleton()->is_editor_hint()) {
		//simple hack to reload fallback environment if it changed from editor
		String env_path = ProjectSettings::get_singleton()->get("rendering/environment/default_environment");
		env_path = env_path.strip_edges(); //user may have added a space or two
		String cpath;
		Ref<Environment> fallback = get_root()->get_world()->get_fallback_environment();
		if (fallback.is_valid()) {
			cpath = fallback->get_path();
		}
		if (cpath != env_path) {

			if (env_path != String()) {
				fallback = ResourceLoader::load(env_path);
				if (fallback.is_null()) {
					//could not load fallback, set as empty
					ProjectSettings::get_singleton()->set("rendering/environment/default_environment", "");
				}
			} else {
				fallback.unref();
			}
			get_root()->get_world()->set_fallback_environment(fallback);
		}
	}

#endif

	return _quit;
}

void SceneTree::finish() {

	_flush_delete_queue();

	_flush_ugc();

	initialized = false;

	MainLoop::finish();

	if (root) {
		root->_set_tree(NULL);
		memdelete(root); //delete root
	}
}

void SceneTree::quit() {

	_quit = true;
}

void SceneTree::_notification(int p_notification) {

	switch (p_notification) {

		case NOTIFICATION_WM_QUIT_REQUEST: {

			get_root()->propagate_notification(p_notification);

			if (accept_quit) {
				_quit = true;
				break;
			}
		} break;
		case NOTIFICATION_WM_GO_BACK_REQUEST: {

			get_root()->propagate_notification(p_notification);

			if (quit_on_go_back) {
				_quit = true;
				break;
			}
		} break;
		case NOTIFICATION_OS_MEMORY_WARNING:
		case NOTIFICATION_WM_FOCUS_IN:
		case NOTIFICATION_WM_FOCUS_OUT: {

			get_root()->propagate_notification(p_notification);
		} break;
		case NOTIFICATION_TRANSLATION_CHANGED: {
			if (!Engine::get_singleton()->is_editor_hint()) {
				get_root()->propagate_notification(Node::NOTIFICATION_TRANSLATION_CHANGED);
			}
		} break;
		case NOTIFICATION_WM_UNFOCUS_REQUEST: {

			notify_group_flags(GROUP_CALL_REALTIME | GROUP_CALL_MULTILEVEL, "input", NOTIFICATION_WM_UNFOCUS_REQUEST);

		} break;

		case NOTIFICATION_WM_ABOUT: {

#ifdef TOOLS_ENABLED
			if (EditorNode::get_singleton()) {
				EditorNode::get_singleton()->show_about();
			} else {
#endif
				get_root()->propagate_notification(p_notification);
#ifdef TOOLS_ENABLED
			}
#endif
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

	return Engine::get_singleton()->is_editor_hint() && edited_scene_root && (edited_scene_root->is_a_parent_of(p_node) || edited_scene_root == p_node);
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

	if (navigation_material.is_valid())
		return navigation_material;

	Ref<SpatialMaterial> line_material = Ref<SpatialMaterial>(memnew(SpatialMaterial));
	line_material->set_flag(SpatialMaterial::FLAG_UNSHADED, true);
	line_material->set_feature(SpatialMaterial::FEATURE_TRANSPARENT, true);
	line_material->set_flag(SpatialMaterial::FLAG_SRGB_VERTEX_COLOR, true);
	line_material->set_flag(SpatialMaterial::FLAG_ALBEDO_FROM_VERTEX_COLOR, true);
	line_material->set_albedo(get_debug_navigation_color());

	navigation_material = line_material;

	return navigation_material;
}

Ref<Material> SceneTree::get_debug_navigation_disabled_material() {

	if (navigation_disabled_material.is_valid())
		return navigation_disabled_material;

	Ref<SpatialMaterial> line_material = Ref<SpatialMaterial>(memnew(SpatialMaterial));
	line_material->set_flag(SpatialMaterial::FLAG_UNSHADED, true);
	line_material->set_feature(SpatialMaterial::FEATURE_TRANSPARENT, true);
	line_material->set_flag(SpatialMaterial::FLAG_SRGB_VERTEX_COLOR, true);
	line_material->set_flag(SpatialMaterial::FLAG_ALBEDO_FROM_VERTEX_COLOR, true);
	line_material->set_albedo(get_debug_navigation_disabled_color());

	navigation_disabled_material = line_material;

	return navigation_disabled_material;
}
Ref<Material> SceneTree::get_debug_collision_material() {

	if (collision_material.is_valid())
		return collision_material;

	Ref<SpatialMaterial> line_material = Ref<SpatialMaterial>(memnew(SpatialMaterial));
	line_material->set_flag(SpatialMaterial::FLAG_UNSHADED, true);
	line_material->set_feature(SpatialMaterial::FEATURE_TRANSPARENT, true);
	line_material->set_flag(SpatialMaterial::FLAG_SRGB_VERTEX_COLOR, true);
	line_material->set_flag(SpatialMaterial::FLAG_ALBEDO_FROM_VERTEX_COLOR, true);
	line_material->set_albedo(get_debug_collisions_color());

	collision_material = line_material;

	return collision_material;
}

Ref<ArrayMesh> SceneTree::get_debug_contact_mesh() {

	if (debug_contact_mesh.is_valid())
		return debug_contact_mesh;

	debug_contact_mesh = Ref<ArrayMesh>(memnew(ArrayMesh));

	Ref<SpatialMaterial> mat = Ref<SpatialMaterial>(memnew(SpatialMaterial));
	mat->set_flag(SpatialMaterial::FLAG_UNSHADED, true);
	mat->set_feature(SpatialMaterial::FEATURE_TRANSPARENT, true);
	mat->set_flag(SpatialMaterial::FLAG_SRGB_VERTEX_COLOR, true);
	mat->set_flag(SpatialMaterial::FLAG_ALBEDO_FROM_VERTEX_COLOR, true);
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

	PoolVector<int> indices;
	for (int i = 0; i < 8 * 3; i++)
		indices.push_back(diamond_faces[i]);

	PoolVector<Vector3> vertices;
	for (int i = 0; i < 6; i++)
		vertices.push_back(diamond[i] * 0.1);

	Array arr;
	arr.resize(Mesh::ARRAY_MAX);
	arr[Mesh::ARRAY_VERTEX] = vertices;
	arr[Mesh::ARRAY_INDEX] = indices;

	debug_contact_mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, arr);
	debug_contact_mesh->surface_set_material(0, mat);

	return debug_contact_mesh;
}

void SceneTree::set_pause(bool p_enabled) {

	if (p_enabled == pause)
		return;
	pause = p_enabled;
	PhysicsServer::get_singleton()->set_active(!p_enabled);
	Physics2DServer::get_singleton()->set_active(!p_enabled);
	if (get_root())
		get_root()->propagate_notification(p_enabled ? Node::NOTIFICATION_PAUSED : Node::NOTIFICATION_UNPAUSED);
}

bool SceneTree::is_paused() const {

	return pause;
}

void SceneTree::_call_input_pause(const StringName &p_group, const StringName &p_method, const Ref<InputEvent> &p_input) {

	Map<StringName, Group>::Element *E = group_map.find(p_group);
	if (!E)
		return;
	Group &g = E->get();
	if (g.nodes.empty())
		return;

	_update_group_order(g);

	//copy, so copy on write happens in case something is removed from process while being called
	//performance is not lost because only if something is added/removed the vector is copied.
	Vector<Node *> nodes_copy = g.nodes;

	int node_count = nodes_copy.size();
	Node **nodes = &nodes_copy[0];

	Variant arg = p_input;
	const Variant *v[1] = { &arg };

	call_lock++;

	for (int i = node_count - 1; i >= 0; i--) {

		if (input_handled)
			break;

		Node *n = nodes[i];
		if (call_lock && call_skip.has(n))
			continue;

		if (!n->can_process())
			continue;

		n->call_multilevel(p_method, (const Variant **)v, 1);
		//ERR_FAIL_COND(node_count != g.nodes.size());
	}

	call_lock--;
	if (call_lock == 0)
		call_skip.clear();
}

void SceneTree::_notify_group_pause(const StringName &p_group, int p_notification) {

	Map<StringName, Group>::Element *E = group_map.find(p_group);
	if (!E)
		return;
	Group &g = E->get();
	if (g.nodes.empty())
		return;

	_update_group_order(g);

	//copy, so copy on write happens in case something is removed from process while being called
	//performance is not lost because only if something is added/removed the vector is copied.
	Vector<Node *> nodes_copy = g.nodes;

	int node_count = nodes_copy.size();
	Node **nodes = &nodes_copy[0];

	call_lock++;

	for (int i = 0; i < node_count; i++) {

		Node *n = nodes[i];
		if (call_lock && call_skip.has(n))
			continue;

		if (!n->can_process())
			continue;

		n->notification(p_notification);
		//ERR_FAIL_COND(node_count != g.nodes.size());
	}

	call_lock--;
	if (call_lock == 0)
		call_skip.clear();
}

/*
void SceneMainLoop::_update_listener_2d() {

	if (listener_2d.is_valid()) {

		SpatialSound2DServer::get_singleton()->listener_set_space( listener_2d, world_2d->get_sound_space() );
	}

}
*/

uint32_t SceneTree::get_last_event_id() const {

	return last_id;
}

Variant SceneTree::_call_group_flags(const Variant **p_args, int p_argcount, Variant::CallError &r_error) {

	r_error.error = Variant::CallError::CALL_OK;

	ERR_FAIL_COND_V(p_argcount < 3, Variant());
	ERR_FAIL_COND_V(!p_args[0]->is_num(), Variant());
	ERR_FAIL_COND_V(p_args[1]->get_type() != Variant::STRING, Variant());
	ERR_FAIL_COND_V(p_args[2]->get_type() != Variant::STRING, Variant());

	int flags = *p_args[0];
	StringName group = *p_args[1];
	StringName method = *p_args[2];
	Variant v[VARIANT_ARG_MAX];

	for (int i = 0; i < MIN(p_argcount - 3, 5); i++) {

		v[i] = *p_args[i + 3];
	}

	call_group_flags(flags, group, method, v[0], v[1], v[2], v[3], v[4]);
	return Variant();
}

Variant SceneTree::_call_group(const Variant **p_args, int p_argcount, Variant::CallError &r_error) {

	r_error.error = Variant::CallError::CALL_OK;

	ERR_FAIL_COND_V(p_argcount < 2, Variant());
	ERR_FAIL_COND_V(p_args[0]->get_type() != Variant::STRING, Variant());
	ERR_FAIL_COND_V(p_args[1]->get_type() != Variant::STRING, Variant());

	StringName group = *p_args[0];
	StringName method = *p_args[1];
	Variant v[VARIANT_ARG_MAX];

	for (int i = 0; i < MIN(p_argcount - 2, 5); i++) {

		v[i] = *p_args[i + 2];
	}

	call_group_flags(0, group, method, v[0], v[1], v[2], v[3], v[4]);
	return Variant();
}

int64_t SceneTree::get_frame() const {

	return current_frame;
}

Array SceneTree::_get_nodes_in_group(const StringName &p_group) {

	Array ret;
	Map<StringName, Group>::Element *E = group_map.find(p_group);
	if (!E)
		return ret;

	_update_group_order(E->get()); //update order just in case
	int nc = E->get().nodes.size();
	if (nc == 0)
		return ret;

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
void SceneTree::get_nodes_in_group(const StringName &p_group, List<Node *> *p_list) {

	Map<StringName, Group>::Element *E = group_map.find(p_group);
	if (!E)
		return;

	_update_group_order(E->get()); //update order just in case
	int nc = E->get().nodes.size();
	if (nc == 0)
		return;
	Node **ptr = E->get().nodes.ptrw();
	for (int i = 0; i < nc; i++) {

		p_list->push_back(ptr[i]);
	}
}

static void _fill_array(Node *p_node, Array &array, int p_level) {

	array.push_back(p_level);
	array.push_back(p_node->get_name());
	array.push_back(p_node->get_class());
	array.push_back(p_node->get_instance_id());
	for (int i = 0; i < p_node->get_child_count(); i++) {

		_fill_array(p_node->get_child(i), array, p_level + 1);
	}
}

void SceneTree::_debugger_request_tree(void *self) {

	SceneTree *sml = (SceneTree *)self;

	Array arr;
	_fill_array(sml->root, arr, 0);
	ScriptDebugger::get_singleton()->send_message("scene_tree", arr);
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

void SceneTree::_update_root_rect() {

	if (stretch_mode == STRETCH_MODE_DISABLED) {

		root->set_size((last_screen_size / stretch_shrink).floor());
		root->set_attach_to_screen_rect(Rect2(Point2(), last_screen_size));
		root->set_size_override_stretch(false);
		root->set_size_override(false, Size2());
		return; //user will take care
	}

	//actual screen video mode
	Size2 video_mode = Size2(OS::get_singleton()->get_video_mode().width, OS::get_singleton()->get_video_mode().height);
	Size2 desired_res = stretch_min;

	Size2 viewport_size;
	Size2 screen_size;

	float viewport_aspect = desired_res.aspect();
	float video_mode_aspect = video_mode.aspect();

	if (stretch_aspect == STRETCH_ASPECT_IGNORE || ABS(viewport_aspect - video_mode_aspect) < CMP_EPSILON) {
		//same aspect or ignore aspect
		viewport_size = desired_res;
		screen_size = video_mode;
	} else if (viewport_aspect < video_mode_aspect) {
		// screen ratio is smaller vertically

		if (stretch_aspect == STRETCH_ASPECT_KEEP_HEIGHT || stretch_aspect == STRETCH_ASPECT_EXPAND) {

			//will stretch horizontally
			viewport_size.x = desired_res.y * video_mode_aspect;
			viewport_size.y = desired_res.y;
			screen_size = video_mode;

		} else {
			//will need black bars
			viewport_size = desired_res;
			screen_size.x = video_mode.y * viewport_aspect;
			screen_size.y = video_mode.y;
		}
	} else {
		//screen ratio is smaller horizontally
		if (stretch_aspect == STRETCH_ASPECT_KEEP_WIDTH || stretch_aspect == STRETCH_ASPECT_EXPAND) {

			//will stretch horizontally
			viewport_size.x = desired_res.x;
			viewport_size.y = desired_res.x / video_mode_aspect;
			screen_size = video_mode;

		} else {
			//will need black bars
			viewport_size = desired_res;
			screen_size.x = video_mode.x;
			screen_size.y = video_mode.x / viewport_aspect;
		}
	}

	screen_size = screen_size.floor();
	viewport_size = viewport_size.floor();

	Size2 margin;
	Size2 offset;
	//black bars and margin
	if (stretch_aspect != STRETCH_ASPECT_EXPAND && screen_size.x < video_mode.x) {
		margin.x = Math::round((video_mode.x - screen_size.x) / 2.0);
		VisualServer::get_singleton()->black_bars_set_margins(margin.x, 0, margin.x, 0);
		offset.x = Math::round(margin.x * viewport_size.y / screen_size.y);
	} else if (stretch_aspect != STRETCH_ASPECT_EXPAND && screen_size.y < video_mode.y) {
		margin.y = Math::round((video_mode.y - screen_size.y) / 2.0);
		VisualServer::get_singleton()->black_bars_set_margins(0, margin.y, 0, margin.y);
		offset.y = Math::round(margin.y * viewport_size.x / screen_size.x);
	} else {
		VisualServer::get_singleton()->black_bars_set_margins(0, 0, 0, 0);
	}

	//print_line("VP SIZE: "+viewport_size+" OFFSET: "+offset+" = "+(offset*2+viewport_size));
	//print_line("SS: "+video_mode);
	switch (stretch_mode) {
		case STRETCH_MODE_2D: {

			root->set_size((screen_size / stretch_shrink).floor());
			root->set_attach_to_screen_rect(Rect2(margin, screen_size));
			root->set_size_override_stretch(true);
			root->set_size_override(true, (viewport_size / stretch_shrink).floor());

		} break;
		case STRETCH_MODE_VIEWPORT: {

			root->set_size((viewport_size / stretch_shrink).floor());
			root->set_attach_to_screen_rect(Rect2(margin, screen_size));
			root->set_size_override_stretch(false);
			root->set_size_override(false, Size2());

		} break;
	}
}

void SceneTree::set_screen_stretch(StretchMode p_mode, StretchAspect p_aspect, const Size2 p_minsize, real_t p_shrink) {

	stretch_mode = p_mode;
	stretch_aspect = p_aspect;
	stretch_min = p_minsize;
	stretch_shrink = p_shrink;
	_update_root_rect();
}

#ifdef TOOLS_ENABLED
void SceneTree::set_edited_scene_root(Node *p_node) {
	edited_scene_root = p_node;
}

Node *SceneTree::get_edited_scene_root() const {

	return edited_scene_root;
}
#endif

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
		current_scene = NULL;
	}

	if (p_to) {
		current_scene = p_to;
		root->add_child(p_to);
	}
}

Error SceneTree::change_scene(const String &p_path) {

	Ref<PackedScene> new_scene = ResourceLoader::load(p_path);
	if (new_scene.is_null())
		return ERR_CANT_OPEN;

	return change_scene_to(new_scene);
}
Error SceneTree::change_scene_to(const Ref<PackedScene> &p_scene) {

	Node *new_scene = NULL;
	if (p_scene.is_valid()) {
		new_scene = p_scene->instance();
		ERR_FAIL_COND_V(!new_scene, ERR_CANT_CREATE);
	}

	call_deferred("_change_scene", new_scene);
	return OK;
}
Error SceneTree::reload_current_scene() {

	ERR_FAIL_COND_V(!current_scene, ERR_UNCONFIGURED);
	String fname = current_scene->get_filename();
	return change_scene(fname);
}

void SceneTree::add_current_scene(Node *p_current) {

	current_scene = p_current;
	root->add_child(p_current);
}
#ifdef DEBUG_ENABLED

void SceneTree::_live_edit_node_path_func(const NodePath &p_path, int p_id) {

	live_edit_node_path_cache[p_id] = p_path;
}

void SceneTree::_live_edit_res_path_func(const String &p_path, int p_id) {

	live_edit_resource_cache[p_id] = p_path;
}

void SceneTree::_live_edit_node_set_func(int p_id, const StringName &p_prop, const Variant &p_value) {

	if (!live_edit_node_path_cache.has(p_id))
		return;

	NodePath np = live_edit_node_path_cache[p_id];
	Node *base = NULL;
	if (root->has_node(live_edit_root))
		base = root->get_node(live_edit_root);

	Map<String, Set<Node *> >::Element *E = live_scene_edit_cache.find(live_edit_scene);
	if (!E)
		return; //scene not editable

	for (Set<Node *>::Element *F = E->get().front(); F; F = F->next()) {

		Node *n = F->get();

		if (base && !base->is_a_parent_of(n))
			continue;

		if (!n->has_node(np))
			continue;
		Node *n2 = n->get_node(np);

		n2->set(p_prop, p_value);
	}
}

void SceneTree::_live_edit_node_set_res_func(int p_id, const StringName &p_prop, const String &p_value) {

	RES r = ResourceLoader::load(p_value);
	if (!r.is_valid())
		return;
	_live_edit_node_set_func(p_id, p_prop, r);
}
void SceneTree::_live_edit_node_call_func(int p_id, const StringName &p_method, VARIANT_ARG_DECLARE) {

	if (!live_edit_node_path_cache.has(p_id))
		return;

	NodePath np = live_edit_node_path_cache[p_id];
	Node *base = NULL;
	if (root->has_node(live_edit_root))
		base = root->get_node(live_edit_root);

	Map<String, Set<Node *> >::Element *E = live_scene_edit_cache.find(live_edit_scene);
	if (!E)
		return; //scene not editable

	for (Set<Node *>::Element *F = E->get().front(); F; F = F->next()) {

		Node *n = F->get();

		if (base && !base->is_a_parent_of(n))
			continue;

		if (!n->has_node(np))
			continue;
		Node *n2 = n->get_node(np);

		n2->call(p_method, VARIANT_ARG_PASS);
	}
}
void SceneTree::_live_edit_res_set_func(int p_id, const StringName &p_prop, const Variant &p_value) {

	if (!live_edit_resource_cache.has(p_id))
		return;

	String resp = live_edit_resource_cache[p_id];

	if (!ResourceCache::has(resp))
		return;

	RES r = ResourceCache::get(resp);
	if (!r.is_valid())
		return;

	r->set(p_prop, p_value);
}
void SceneTree::_live_edit_res_set_res_func(int p_id, const StringName &p_prop, const String &p_value) {

	RES r = ResourceLoader::load(p_value);
	if (!r.is_valid())
		return;
	_live_edit_res_set_func(p_id, p_prop, r);
}
void SceneTree::_live_edit_res_call_func(int p_id, const StringName &p_method, VARIANT_ARG_DECLARE) {

	if (!live_edit_resource_cache.has(p_id))
		return;

	String resp = live_edit_resource_cache[p_id];

	if (!ResourceCache::has(resp))
		return;

	RES r = ResourceCache::get(resp);
	if (!r.is_valid())
		return;

	r->call(p_method, VARIANT_ARG_PASS);
}

void SceneTree::_live_edit_root_func(const NodePath &p_scene_path, const String &p_scene_from) {

	live_edit_root = p_scene_path;
	live_edit_scene = p_scene_from;
}

void SceneTree::_live_edit_create_node_func(const NodePath &p_parent, const String &p_type, const String &p_name) {

	Node *base = NULL;
	if (root->has_node(live_edit_root))
		base = root->get_node(live_edit_root);

	Map<String, Set<Node *> >::Element *E = live_scene_edit_cache.find(live_edit_scene);
	if (!E)
		return; //scene not editable

	for (Set<Node *>::Element *F = E->get().front(); F; F = F->next()) {

		Node *n = F->get();

		if (base && !base->is_a_parent_of(n))
			continue;

		if (!n->has_node(p_parent))
			continue;
		Node *n2 = n->get_node(p_parent);

		Node *no = Object::cast_to<Node>(ClassDB::instance(p_type));
		if (!no) {
			continue;
		}

		no->set_name(p_name);
		n2->add_child(no);
	}
}
void SceneTree::_live_edit_instance_node_func(const NodePath &p_parent, const String &p_path, const String &p_name) {

	Ref<PackedScene> ps = ResourceLoader::load(p_path);

	if (!ps.is_valid())
		return;

	Node *base = NULL;
	if (root->has_node(live_edit_root))
		base = root->get_node(live_edit_root);

	Map<String, Set<Node *> >::Element *E = live_scene_edit_cache.find(live_edit_scene);
	if (!E)
		return; //scene not editable

	for (Set<Node *>::Element *F = E->get().front(); F; F = F->next()) {

		Node *n = F->get();

		if (base && !base->is_a_parent_of(n))
			continue;

		if (!n->has_node(p_parent))
			continue;
		Node *n2 = n->get_node(p_parent);

		Node *no = ps->instance();
		no->set_name(p_name);

		n2->add_child(no);
	}
}
void SceneTree::_live_edit_remove_node_func(const NodePath &p_at) {

	Node *base = NULL;
	if (root->has_node(live_edit_root))
		base = root->get_node(live_edit_root);

	Map<String, Set<Node *> >::Element *E = live_scene_edit_cache.find(live_edit_scene);
	if (!E)
		return; //scene not editable

	for (Set<Node *>::Element *F = E->get().front(); F;) {

		Set<Node *>::Element *N = F->next();

		Node *n = F->get();

		if (base && !base->is_a_parent_of(n))
			continue;

		if (!n->has_node(p_at))
			continue;
		Node *n2 = n->get_node(p_at);

		memdelete(n2);

		F = N;
	}
}
void SceneTree::_live_edit_remove_and_keep_node_func(const NodePath &p_at, ObjectID p_keep_id) {

	Node *base = NULL;
	if (root->has_node(live_edit_root))
		base = root->get_node(live_edit_root);

	Map<String, Set<Node *> >::Element *E = live_scene_edit_cache.find(live_edit_scene);
	if (!E)
		return; //scene not editable

	for (Set<Node *>::Element *F = E->get().front(); F;) {

		Set<Node *>::Element *N = F->next();

		Node *n = F->get();

		if (base && !base->is_a_parent_of(n))
			continue;

		if (!n->has_node(p_at))
			continue;

		Node *n2 = n->get_node(p_at);

		n2->get_parent()->remove_child(n2);

		live_edit_remove_list[n][p_keep_id] = n2;

		F = N;
	}
}
void SceneTree::_live_edit_restore_node_func(ObjectID p_id, const NodePath &p_at, int p_at_pos) {

	Node *base = NULL;
	if (root->has_node(live_edit_root))
		base = root->get_node(live_edit_root);

	Map<String, Set<Node *> >::Element *E = live_scene_edit_cache.find(live_edit_scene);
	if (!E)
		return; //scene not editable

	for (Set<Node *>::Element *F = E->get().front(); F;) {

		Set<Node *>::Element *N = F->next();

		Node *n = F->get();

		if (base && !base->is_a_parent_of(n))
			continue;

		if (!n->has_node(p_at))
			continue;
		Node *n2 = n->get_node(p_at);

		Map<Node *, Map<ObjectID, Node *> >::Element *EN = live_edit_remove_list.find(n);

		if (!EN)
			continue;

		Map<ObjectID, Node *>::Element *FN = EN->get().find(p_id);

		if (!FN)
			continue;
		n2->add_child(FN->get());

		EN->get().erase(FN);

		if (EN->get().size() == 0) {
			live_edit_remove_list.erase(EN);
		}

		F = N;
	}
}
void SceneTree::_live_edit_duplicate_node_func(const NodePath &p_at, const String &p_new_name) {

	Node *base = NULL;
	if (root->has_node(live_edit_root))
		base = root->get_node(live_edit_root);

	Map<String, Set<Node *> >::Element *E = live_scene_edit_cache.find(live_edit_scene);
	if (!E)
		return; //scene not editable

	for (Set<Node *>::Element *F = E->get().front(); F; F = F->next()) {

		Node *n = F->get();

		if (base && !base->is_a_parent_of(n))
			continue;

		if (!n->has_node(p_at))
			continue;
		Node *n2 = n->get_node(p_at);

		Node *dup = n2->duplicate(true);

		if (!dup)
			continue;

		dup->set_name(p_new_name);
		n2->get_parent()->add_child(dup);
	}
}
void SceneTree::_live_edit_reparent_node_func(const NodePath &p_at, const NodePath &p_new_place, const String &p_new_name, int p_at_pos) {

	Node *base = NULL;
	if (root->has_node(live_edit_root))
		base = root->get_node(live_edit_root);

	Map<String, Set<Node *> >::Element *E = live_scene_edit_cache.find(live_edit_scene);
	if (!E)
		return; //scene not editable

	for (Set<Node *>::Element *F = E->get().front(); F; F = F->next()) {

		Node *n = F->get();

		if (base && !base->is_a_parent_of(n))
			continue;

		if (!n->has_node(p_at))
			continue;
		Node *nfrom = n->get_node(p_at);

		if (!n->has_node(p_new_place))
			continue;
		Node *nto = n->get_node(p_new_place);

		nfrom->get_parent()->remove_child(nfrom);
		nfrom->set_name(p_new_name);

		nto->add_child(nfrom);
		if (p_at_pos >= 0)
			nto->move_child(nfrom, p_at_pos);
	}
}

#endif

void SceneTree::drop_files(const Vector<String> &p_files, int p_from_screen) {

	emit_signal("files_dropped", p_files, p_from_screen);
	MainLoop::drop_files(p_files, p_from_screen);
}

Ref<SceneTreeTimer> SceneTree::create_timer(float p_delay_sec, bool p_process_pause) {

	Ref<SceneTreeTimer> stt;
	stt.instance();
	stt->set_pause_mode_process(p_process_pause);
	stt->set_time_left(p_delay_sec);
	timers.push_back(stt);
	return stt;
}

void SceneTree::_network_peer_connected(int p_id) {

	connected_peers.insert(p_id);
	path_get_cache.insert(p_id, PathGetCache());

	emit_signal("network_peer_connected", p_id);
}

void SceneTree::_network_peer_disconnected(int p_id) {

	connected_peers.erase(p_id);
	path_get_cache.erase(p_id); //I no longer need your cache, sorry
	emit_signal("network_peer_disconnected", p_id);
}

void SceneTree::_connected_to_server() {

	emit_signal("connected_to_server");
}

void SceneTree::_connection_failed() {

	emit_signal("connection_failed");
}

void SceneTree::_server_disconnected() {

	emit_signal("server_disconnected");
}

void SceneTree::set_network_peer(const Ref<NetworkedMultiplayerPeer> &p_network_peer) {
	if (network_peer.is_valid()) {
		network_peer->disconnect("peer_connected", this, "_network_peer_connected");
		network_peer->disconnect("peer_disconnected", this, "_network_peer_disconnected");
		network_peer->disconnect("connection_succeeded", this, "_connected_to_server");
		network_peer->disconnect("connection_failed", this, "_connection_failed");
		network_peer->disconnect("server_disconnected", this, "_server_disconnected");
		connected_peers.clear();
		path_get_cache.clear();
		path_send_cache.clear();
		last_send_cache_id = 1;
	}

	ERR_EXPLAIN("Supplied NetworkedNetworkPeer must be connecting or connected.");
	ERR_FAIL_COND(p_network_peer.is_valid() && p_network_peer->get_connection_status() == NetworkedMultiplayerPeer::CONNECTION_DISCONNECTED);

	network_peer = p_network_peer;

	if (network_peer.is_valid()) {
		network_peer->connect("peer_connected", this, "_network_peer_connected");
		network_peer->connect("peer_disconnected", this, "_network_peer_disconnected");
		network_peer->connect("connection_succeeded", this, "_connected_to_server");
		network_peer->connect("connection_failed", this, "_connection_failed");
		network_peer->connect("server_disconnected", this, "_server_disconnected");
	}
}

bool SceneTree::is_network_server() const {

	ERR_FAIL_COND_V(!network_peer.is_valid(), false);
	return network_peer->is_server();
}

bool SceneTree::has_network_peer() const {
	return network_peer.is_valid();
}

int SceneTree::get_network_unique_id() const {

	ERR_FAIL_COND_V(!network_peer.is_valid(), 0);
	return network_peer->get_unique_id();
}

Vector<int> SceneTree::get_network_connected_peers() const {
	ERR_FAIL_COND_V(!network_peer.is_valid(), Vector<int>());

	Vector<int> ret;
	for (Set<int>::Element *E = connected_peers.front(); E; E = E->next()) {
		ret.push_back(E->get());
	}

	return ret;
}

int SceneTree::get_rpc_sender_id() const {
	return rpc_sender_id;
}

void SceneTree::set_refuse_new_network_connections(bool p_refuse) {
	ERR_FAIL_COND(!network_peer.is_valid());
	network_peer->set_refuse_new_connections(p_refuse);
}

bool SceneTree::is_refusing_new_network_connections() const {

	ERR_FAIL_COND_V(!network_peer.is_valid(), false);

	return network_peer->is_refusing_new_connections();
}

void SceneTree::_rpc(Node *p_from, int p_to, bool p_unreliable, bool p_set, const StringName &p_name, const Variant **p_arg, int p_argcount) {

	if (network_peer.is_null()) {
		ERR_EXPLAIN("Attempt to remote call/set when networking is not active in SceneTree.");
		ERR_FAIL();
	}

	if (network_peer->get_connection_status() == NetworkedMultiplayerPeer::CONNECTION_CONNECTING) {
		ERR_EXPLAIN("Attempt to remote call/set when networking is not connected yet in SceneTree.");
		ERR_FAIL();
	}

	if (network_peer->get_connection_status() == NetworkedMultiplayerPeer::CONNECTION_DISCONNECTED) {
		ERR_EXPLAIN("Attempt to remote call/set when networking is disconnected.");
		ERR_FAIL();
	}

	if (p_argcount > 255) {
		ERR_EXPLAIN("Too many arguments >255.");
		ERR_FAIL();
	}

	if (p_to != 0 && !connected_peers.has(ABS(p_to))) {
		if (p_to == get_network_unique_id()) {
			ERR_EXPLAIN("Attempt to remote call/set yourself! unique ID: " + itos(get_network_unique_id()));
		} else {
			ERR_EXPLAIN("Attempt to remote call unexisting ID: " + itos(p_to));
		}

		ERR_FAIL();
	}

	NodePath from_path = p_from->get_path();
	ERR_FAIL_COND(from_path.is_empty());

	//see if the path is cached
	PathSentCache *psc = path_send_cache.getptr(from_path);
	if (!psc) {
		//path is not cached, create
		path_send_cache[from_path] = PathSentCache();
		psc = path_send_cache.getptr(from_path);
		psc->id = last_send_cache_id++;
	}

	//create base packet, lots of harcode because it must be tight

	int ofs = 0;

#define MAKE_ROOM(m_amount) \
	if (packet_cache.size() < m_amount) packet_cache.resize(m_amount);

	//encode type
	MAKE_ROOM(1);
	packet_cache[0] = p_set ? NETWORK_COMMAND_REMOTE_SET : NETWORK_COMMAND_REMOTE_CALL;
	ofs += 1;

	//encode ID
	MAKE_ROOM(ofs + 4);
	encode_uint32(psc->id, &packet_cache[ofs]);
	ofs += 4;

	//encode function name
	CharString name = String(p_name).utf8();
	int len = encode_cstring(name.get_data(), NULL);
	MAKE_ROOM(ofs + len);
	encode_cstring(name.get_data(), &packet_cache[ofs]);
	ofs += len;

	if (p_set) {
		//set argument
		Error err = encode_variant(*p_arg[0], NULL, len);
		ERR_FAIL_COND(err != OK);
		MAKE_ROOM(ofs + len);
		encode_variant(*p_arg[0], &packet_cache[ofs], len);
		ofs += len;

	} else {
		//call arguments
		MAKE_ROOM(ofs + 1);
		packet_cache[ofs] = p_argcount;
		ofs += 1;
		for (int i = 0; i < p_argcount; i++) {
			Error err = encode_variant(*p_arg[i], NULL, len);
			ERR_FAIL_COND(err != OK);
			MAKE_ROOM(ofs + len);
			encode_variant(*p_arg[i], &packet_cache[ofs], len);
			ofs += len;
		}
	}

	//see if all peers have cached path (is so, call can be fast)
	bool has_all_peers = true;

	List<int> peers_to_add; //if one is missing, take note to add it

	for (Set<int>::Element *E = connected_peers.front(); E; E = E->next()) {

		if (p_to < 0 && E->get() == -p_to)
			continue; //continue, excluded

		if (p_to > 0 && E->get() != p_to)
			continue; //continue, not for this peer

		Map<int, bool>::Element *F = psc->confirmed_peers.find(E->get());

		if (!F || F->get() == false) {
			//path was not cached, or was cached but is unconfirmed
			if (!F) {
				//not cached at all, take note
				peers_to_add.push_back(E->get());
			}

			has_all_peers = false;
		}
	}

	//those that need to be added, send a message for this

	for (List<int>::Element *E = peers_to_add.front(); E; E = E->next()) {

		//encode function name
		CharString pname = String(from_path).utf8();
		int len = encode_cstring(pname.get_data(), NULL);

		Vector<uint8_t> packet;

		packet.resize(1 + 4 + len);
		packet[0] = NETWORK_COMMAND_SIMPLIFY_PATH;
		encode_uint32(psc->id, &packet[1]);
		encode_cstring(pname.get_data(), &packet[5]);

		network_peer->set_target_peer(E->get()); //to all of you
		network_peer->set_transfer_mode(NetworkedMultiplayerPeer::TRANSFER_MODE_RELIABLE);
		network_peer->put_packet(packet.ptr(), packet.size());

		psc->confirmed_peers.insert(E->get(), false); //insert into confirmed, but as false since it was not confirmed
	}

	//take chance and set transfer mode, since all send methods will use it
	network_peer->set_transfer_mode(p_unreliable ? NetworkedMultiplayerPeer::TRANSFER_MODE_UNRELIABLE : NetworkedMultiplayerPeer::TRANSFER_MODE_RELIABLE);

	if (has_all_peers) {

		//they all have verified paths, so send fast
		network_peer->set_target_peer(p_to); //to all of you
		network_peer->put_packet(packet_cache.ptr(), ofs); //a message with love
	} else {
		//not all verified path, so send one by one

		//apend path at the end, since we will need it for some packets
		CharString pname = String(from_path).utf8();
		int path_len = encode_cstring(pname.get_data(), NULL);
		MAKE_ROOM(ofs + path_len);
		encode_cstring(pname.get_data(), &packet_cache[ofs]);

		for (Set<int>::Element *E = connected_peers.front(); E; E = E->next()) {

			if (p_to < 0 && E->get() == -p_to)
				continue; //continue, excluded

			if (p_to > 0 && E->get() != p_to)
				continue; //continue, not for this peer

			Map<int, bool>::Element *F = psc->confirmed_peers.find(E->get());
			ERR_CONTINUE(!F); //should never happen

			network_peer->set_target_peer(E->get()); //to this one specifically

			if (F->get() == true) {
				//this one confirmed path, so use id
				encode_uint32(psc->id, &packet_cache[1]);
				network_peer->put_packet(packet_cache.ptr(), ofs);
			} else {
				//this one did not confirm path yet, so use entire path (sorry!)
				encode_uint32(0x80000000 | ofs, &packet_cache[1]); //offset to path and flag
				network_peer->put_packet(packet_cache.ptr(), ofs + path_len);
			}
		}
	}
}

void SceneTree::_network_process_packet(int p_from, const uint8_t *p_packet, int p_packet_len) {

	ERR_FAIL_COND(p_packet_len < 5);

	uint8_t packet_type = p_packet[0];

	switch (packet_type) {

		case NETWORK_COMMAND_REMOTE_CALL:
		case NETWORK_COMMAND_REMOTE_SET: {

			ERR_FAIL_COND(p_packet_len < 5);
			uint32_t target = decode_uint32(&p_packet[1]);

			Node *node = NULL;

			if (target & 0x80000000) {
				//use full path (not cached yet)

				int ofs = target & 0x7FFFFFFF;
				ERR_FAIL_COND(ofs >= p_packet_len);

				String paths;
				paths.parse_utf8((const char *)&p_packet[ofs], p_packet_len - ofs);

				NodePath np = paths;

				node = get_root()->get_node(np);
				if (node == NULL) {
					ERR_EXPLAIN("Failed to get path from RPC: " + String(np));
					ERR_FAIL_COND(node == NULL);
				}
			} else {
				//use cached path
				int id = target;

				Map<int, PathGetCache>::Element *E = path_get_cache.find(p_from);
				ERR_FAIL_COND(!E);

				Map<int, PathGetCache::NodeInfo>::Element *F = E->get().nodes.find(id);
				ERR_FAIL_COND(!F);

				PathGetCache::NodeInfo *ni = &F->get();
				//do proper caching later

				node = get_root()->get_node(ni->path);
				if (node == NULL) {
					ERR_EXPLAIN("Failed to get cached path from RPC: " + String(ni->path));
					ERR_FAIL_COND(node == NULL);
				}
			}

			ERR_FAIL_COND(p_packet_len < 6);

			//detect cstring end
			int len_end = 5;
			for (; len_end < p_packet_len; len_end++) {
				if (p_packet[len_end] == 0) {
					break;
				}
			}

			ERR_FAIL_COND(len_end >= p_packet_len);

			StringName name = String::utf8((const char *)&p_packet[5]);

			if (packet_type == NETWORK_COMMAND_REMOTE_CALL) {

				if (!node->can_call_rpc(name, p_from))
					return;

				int ofs = len_end + 1;

				ERR_FAIL_COND(ofs >= p_packet_len);

				int argc = p_packet[ofs];
				Vector<Variant> args;
				Vector<const Variant *> argp;
				args.resize(argc);
				argp.resize(argc);

				ofs++;

				for (int i = 0; i < argc; i++) {

					ERR_FAIL_COND(ofs >= p_packet_len);
					int vlen;
					Error err = decode_variant(args[i], &p_packet[ofs], p_packet_len - ofs, &vlen);
					ERR_FAIL_COND(err != OK);
					//args[i]=p_packet[3+i];
					argp[i] = &args[i];
					ofs += vlen;
				}

				Variant::CallError ce;

				node->call(name, (const Variant **)argp.ptr(), argc, ce);
				if (ce.error != Variant::CallError::CALL_OK) {
					String error = Variant::get_call_error_text(node, name, (const Variant **)argp.ptr(), argc, ce);
					error = "RPC - " + error;
					ERR_PRINTS(error);
				}

			} else {

				if (!node->can_call_rset(name, p_from))
					return;

				int ofs = len_end + 1;

				ERR_FAIL_COND(ofs >= p_packet_len);

				Variant value;
				decode_variant(value, &p_packet[ofs], p_packet_len - ofs);

				bool valid;

				node->set(name, value, &valid);
				if (!valid) {
					String error = "Error setting remote property '" + String(name) + "', not found in object of type " + node->get_class();
					ERR_PRINTS(error);
				}
			}

		} break;
		case NETWORK_COMMAND_SIMPLIFY_PATH: {

			ERR_FAIL_COND(p_packet_len < 5);
			int id = decode_uint32(&p_packet[1]);

			String paths;
			paths.parse_utf8((const char *)&p_packet[5], p_packet_len - 5);

			NodePath path = paths;

			if (!path_get_cache.has(p_from)) {
				path_get_cache[p_from] = PathGetCache();
			}

			PathGetCache::NodeInfo ni;
			ni.path = path;
			ni.instance = 0;

			path_get_cache[p_from].nodes[id] = ni;

			{
				//send ack

				//encode path
				CharString pname = String(path).utf8();
				int len = encode_cstring(pname.get_data(), NULL);

				Vector<uint8_t> packet;

				packet.resize(1 + len);
				packet[0] = NETWORK_COMMAND_CONFIRM_PATH;
				encode_cstring(pname.get_data(), &packet[1]);

				network_peer->set_transfer_mode(NetworkedMultiplayerPeer::TRANSFER_MODE_RELIABLE);
				network_peer->set_target_peer(p_from);
				network_peer->put_packet(packet.ptr(), packet.size());
			}
		} break;
		case NETWORK_COMMAND_CONFIRM_PATH: {

			String paths;
			paths.parse_utf8((const char *)&p_packet[1], p_packet_len - 1);

			NodePath path = paths;

			PathSentCache *psc = path_send_cache.getptr(path);
			ERR_FAIL_COND(!psc);

			Map<int, bool>::Element *E = psc->confirmed_peers.find(p_from);
			ERR_FAIL_COND(!E);
			E->get() = true;
		} break;
	}
}

void SceneTree::_network_poll() {

	if (!network_peer.is_valid() || network_peer->get_connection_status() == NetworkedMultiplayerPeer::CONNECTION_DISCONNECTED)
		return;

	network_peer->poll();

	if (!network_peer.is_valid()) //it's possible that polling might have resulted in a disconnection, so check here
		return;

	while (network_peer->get_available_packet_count()) {

		int sender = network_peer->get_packet_peer();
		const uint8_t *packet;
		int len;

		Error err = network_peer->get_packet(&packet, len);
		if (err != OK) {
			ERR_PRINT("Error getting packet!");
		}

		rpc_sender_id = sender;
		_network_process_packet(sender, packet, len);
		rpc_sender_id = 0;

		if (!network_peer.is_valid()) {
			break; //it's also possible that a packet or RPC caused a disconnection, so also check here
		}
	}
}

void SceneTree::_bind_methods() {

	//ClassDB::bind_method(D_METHOD("call_group","call_flags","group","method","arg1","arg2"),&SceneMainLoop::_call_group,DEFVAL(Variant()),DEFVAL(Variant()));

	ClassDB::bind_method(D_METHOD("get_root"), &SceneTree::get_root);
	ClassDB::bind_method(D_METHOD("has_group", "name"), &SceneTree::has_group);

	ClassDB::bind_method(D_METHOD("set_auto_accept_quit", "enabled"), &SceneTree::set_auto_accept_quit);

	ClassDB::bind_method(D_METHOD("set_debug_collisions_hint", "enable"), &SceneTree::set_debug_collisions_hint);
	ClassDB::bind_method(D_METHOD("is_debugging_collisions_hint"), &SceneTree::is_debugging_collisions_hint);
	ClassDB::bind_method(D_METHOD("set_debug_navigation_hint", "enable"), &SceneTree::set_debug_navigation_hint);
	ClassDB::bind_method(D_METHOD("is_debugging_navigation_hint"), &SceneTree::is_debugging_navigation_hint);

#ifdef TOOLS_ENABLED
	ClassDB::bind_method(D_METHOD("set_edited_scene_root", "scene"), &SceneTree::set_edited_scene_root);
	ClassDB::bind_method(D_METHOD("get_edited_scene_root"), &SceneTree::get_edited_scene_root);
#endif

	ClassDB::bind_method(D_METHOD("set_pause", "enable"), &SceneTree::set_pause);
	ClassDB::bind_method(D_METHOD("is_paused"), &SceneTree::is_paused);
	ClassDB::bind_method(D_METHOD("set_input_as_handled"), &SceneTree::set_input_as_handled);
	ClassDB::bind_method(D_METHOD("is_input_handled"), &SceneTree::is_input_handled);

	ClassDB::bind_method(D_METHOD("create_timer", "time_sec", "pause_mode_process"), &SceneTree::create_timer, DEFVAL(true));

	ClassDB::bind_method(D_METHOD("get_node_count"), &SceneTree::get_node_count);
	ClassDB::bind_method(D_METHOD("get_frame"), &SceneTree::get_frame);
	ClassDB::bind_method(D_METHOD("quit"), &SceneTree::quit);

	ClassDB::bind_method(D_METHOD("set_screen_stretch", "mode", "aspect", "minsize", "shrink"), &SceneTree::set_screen_stretch, DEFVAL(1));

	ClassDB::bind_method(D_METHOD("queue_delete", "obj"), &SceneTree::queue_delete);

	MethodInfo mi;
	mi.name = "call_group_flags";
	mi.arguments.push_back(PropertyInfo(Variant::INT, "flags"));
	mi.arguments.push_back(PropertyInfo(Variant::STRING, "group"));
	mi.arguments.push_back(PropertyInfo(Variant::STRING, "method"));

	ClassDB::bind_vararg_method(METHOD_FLAGS_DEFAULT, "call_group_flags", &SceneTree::_call_group_flags, mi);

	ClassDB::bind_method(D_METHOD("notify_group_flags", "call_flags", "group", "notification"), &SceneTree::notify_group_flags);
	ClassDB::bind_method(D_METHOD("set_group_flags", "call_flags", "group", "property", "value"), &SceneTree::set_group_flags);

	MethodInfo mi2;
	mi2.name = "call_group";
	mi2.arguments.push_back(PropertyInfo(Variant::STRING, "group"));
	mi2.arguments.push_back(PropertyInfo(Variant::STRING, "method"));

	ClassDB::bind_vararg_method(METHOD_FLAGS_DEFAULT, "call_group", &SceneTree::_call_group, mi2);

	ClassDB::bind_method(D_METHOD("notify_group", "group", "notification"), &SceneTree::notify_group);
	ClassDB::bind_method(D_METHOD("set_group", "group", "property", "value"), &SceneTree::set_group);

	ClassDB::bind_method(D_METHOD("get_nodes_in_group", "group"), &SceneTree::_get_nodes_in_group);

	ClassDB::bind_method(D_METHOD("set_current_scene", "child_node"), &SceneTree::set_current_scene);
	ClassDB::bind_method(D_METHOD("get_current_scene"), &SceneTree::get_current_scene);

	ClassDB::bind_method(D_METHOD("change_scene", "path"), &SceneTree::change_scene);
	ClassDB::bind_method(D_METHOD("change_scene_to", "packed_scene"), &SceneTree::change_scene_to);

	ClassDB::bind_method(D_METHOD("reload_current_scene"), &SceneTree::reload_current_scene);

	ClassDB::bind_method(D_METHOD("_change_scene"), &SceneTree::_change_scene);

	ClassDB::bind_method(D_METHOD("set_network_peer", "peer"), &SceneTree::set_network_peer);
	ClassDB::bind_method(D_METHOD("is_network_server"), &SceneTree::is_network_server);
	ClassDB::bind_method(D_METHOD("has_network_peer"), &SceneTree::has_network_peer);
	ClassDB::bind_method(D_METHOD("get_network_connected_peers"), &SceneTree::get_network_connected_peers);
	ClassDB::bind_method(D_METHOD("get_network_unique_id"), &SceneTree::get_network_unique_id);
	ClassDB::bind_method(D_METHOD("get_rpc_sender_id"), &SceneTree::get_rpc_sender_id);
	ClassDB::bind_method(D_METHOD("set_refuse_new_network_connections", "refuse"), &SceneTree::set_refuse_new_network_connections);
	ClassDB::bind_method(D_METHOD("is_refusing_new_network_connections"), &SceneTree::is_refusing_new_network_connections);
	ClassDB::bind_method(D_METHOD("_network_peer_connected"), &SceneTree::_network_peer_connected);
	ClassDB::bind_method(D_METHOD("_network_peer_disconnected"), &SceneTree::_network_peer_disconnected);
	ClassDB::bind_method(D_METHOD("_connected_to_server"), &SceneTree::_connected_to_server);
	ClassDB::bind_method(D_METHOD("_connection_failed"), &SceneTree::_connection_failed);
	ClassDB::bind_method(D_METHOD("_server_disconnected"), &SceneTree::_server_disconnected);

	ClassDB::bind_method(D_METHOD("set_use_font_oversampling", "enable"), &SceneTree::set_use_font_oversampling);
	ClassDB::bind_method(D_METHOD("is_using_font_oversampling"), &SceneTree::is_using_font_oversampling);

	ADD_SIGNAL(MethodInfo("tree_changed"));
	ADD_SIGNAL(MethodInfo("node_added", PropertyInfo(Variant::OBJECT, "node")));
	ADD_SIGNAL(MethodInfo("node_removed", PropertyInfo(Variant::OBJECT, "node")));
	ADD_SIGNAL(MethodInfo("screen_resized"));
	ADD_SIGNAL(MethodInfo("node_configuration_warning_changed", PropertyInfo(Variant::OBJECT, "node")));

	ADD_SIGNAL(MethodInfo("idle_frame"));
	ADD_SIGNAL(MethodInfo("physics_frame"));

	ADD_SIGNAL(MethodInfo("files_dropped", PropertyInfo(Variant::POOL_STRING_ARRAY, "files"), PropertyInfo(Variant::INT, "screen")));
	ADD_SIGNAL(MethodInfo("network_peer_connected", PropertyInfo(Variant::INT, "id")));
	ADD_SIGNAL(MethodInfo("network_peer_disconnected", PropertyInfo(Variant::INT, "id")));
	ADD_SIGNAL(MethodInfo("connected_to_server"));
	ADD_SIGNAL(MethodInfo("connection_failed"));
	ADD_SIGNAL(MethodInfo("server_disconnected"));

	BIND_ENUM_CONSTANT(GROUP_CALL_DEFAULT);
	BIND_ENUM_CONSTANT(GROUP_CALL_REVERSE);
	BIND_ENUM_CONSTANT(GROUP_CALL_REALTIME);
	BIND_ENUM_CONSTANT(GROUP_CALL_UNIQUE);

	BIND_ENUM_CONSTANT(STRETCH_MODE_DISABLED);
	BIND_ENUM_CONSTANT(STRETCH_MODE_2D);
	BIND_ENUM_CONSTANT(STRETCH_MODE_VIEWPORT);

	BIND_ENUM_CONSTANT(STRETCH_ASPECT_IGNORE);
	BIND_ENUM_CONSTANT(STRETCH_ASPECT_KEEP);
	BIND_ENUM_CONSTANT(STRETCH_ASPECT_KEEP_WIDTH);
	BIND_ENUM_CONSTANT(STRETCH_ASPECT_KEEP_HEIGHT);
	BIND_ENUM_CONSTANT(STRETCH_ASPECT_EXPAND);
}

SceneTree *SceneTree::singleton = NULL;

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

void SceneTree::set_use_font_oversampling(bool p_oversampling) {

	use_font_oversampling = p_oversampling;
	if (use_font_oversampling) {
		DynamicFontAtSize::font_oversampling = OS::get_singleton()->get_window_size().width / root->get_visible_rect().size.width;
	} else {
		DynamicFontAtSize::font_oversampling = 1.0;
	}
}

bool SceneTree::is_using_font_oversampling() const {
	return use_font_oversampling;
}

SceneTree::SceneTree() {

	singleton = this;
	_quit = false;
	accept_quit = true;
	quit_on_go_back = true;
	initialized = false;
#ifdef DEBUG_ENABLED
	debug_collisions_hint = false;
	debug_navigation_hint = false;
#endif
	debug_collisions_color = GLOBAL_DEF("debug/shapes/collision/shape_color", Color(0.0, 0.6, 0.7, 0.5));
	debug_collision_contact_color = GLOBAL_DEF("debug/shapes/collision/contact_color", Color(1.0, 0.2, 0.1, 0.8));
	debug_navigation_color = GLOBAL_DEF("debug/shapes/navigation/geometry_color", Color(0.1, 1.0, 0.7, 0.4));
	debug_navigation_disabled_color = GLOBAL_DEF("debug/shapes/navigation/disabled_geometry_color", Color(1.0, 0.7, 0.1, 0.4));
	collision_debug_contacts = GLOBAL_DEF("debug/shapes/collision/max_contacts_displayed", 10000);

	tree_version = 1;
	physics_process_time = 1;
	idle_process_time = 1;
	last_id = 1;
	root = NULL;
	current_frame = 0;
	tree_changed_name = "tree_changed";
	node_added_name = "node_added";
	node_removed_name = "node_removed";
	ugc_locked = false;
	call_lock = 0;
	root_lock = 0;
	node_count = 0;
	rpc_sender_id = 0;

	//create with mainloop

	root = memnew(Viewport);
	root->set_name("root");
	if (!root->get_world().is_valid())
		root->set_world(Ref<World>(memnew(World)));

	//root->set_world_2d( Ref<World2D>( memnew( World2D )));
	root->set_as_audio_listener(true);
	root->set_as_audio_listener_2d(true);
	current_scene = NULL;

	int ref_atlas_size = GLOBAL_DEF("rendering/quality/reflections/atlas_size", 2048);
	int ref_atlas_subdiv = GLOBAL_DEF("rendering/quality/reflections/atlas_subdiv", 8);
	int msaa_mode = GLOBAL_DEF("rendering/quality/filters/msaa", 0);
	ProjectSettings::get_singleton()->set_custom_property_info("rendering/quality/filters/msaa", PropertyInfo(Variant::INT, "rendering/quality/filters/msaa", PROPERTY_HINT_ENUM, "Disabled,2x,4x,8x,16x"));
	root->set_msaa(Viewport::MSAA(msaa_mode));

	GLOBAL_DEF("rendering/quality/depth/hdr", true);
	GLOBAL_DEF("rendering/quality/depth/hdr.mobile", false);

	bool hdr = GLOBAL_GET("rendering/quality/depth/hdr");
	root->set_hdr(hdr);

	VS::get_singleton()->scenario_set_reflection_atlas_size(root->get_world()->get_scenario(), ref_atlas_size, ref_atlas_subdiv);

	{ //load default fallback environment
		//get possible extensions
		List<String> exts;
		ResourceLoader::get_recognized_extensions_for_type("Environment", &exts);
		String ext_hint;
		for (List<String>::Element *E = exts.front(); E; E = E->next()) {
			if (ext_hint != String())
				ext_hint += ",";
			ext_hint += "*." + E->get();
		}
		//get path
		String env_path = GLOBAL_DEF("rendering/environment/default_environment", "");
		//setup property
		ProjectSettings::get_singleton()->set_custom_property_info("rendering/environment/default_environment", PropertyInfo(Variant::STRING, "rendering/viewport/default_environment", PROPERTY_HINT_FILE, ext_hint));
		env_path = env_path.strip_edges();
		if (env_path != String()) {
			Ref<Environment> env = ResourceLoader::load(env_path);
			if (env.is_valid()) {
				root->get_world()->set_fallback_environment(env);
			} else {
				if (Engine::get_singleton()->is_editor_hint()) {
					//file was erased, clear the field.
					ProjectSettings::get_singleton()->set("rendering/environment/default_environment", "");
				} else {
					//file was erased, notify user.
					ERR_PRINTS(RTR("Default Environment as specified in Project Setings (Rendering -> Environment -> Default Environment) could not be loaded."));
				}
			}
		}
	}

	stretch_mode = STRETCH_MODE_DISABLED;
	stretch_aspect = STRETCH_ASPECT_IGNORE;
	stretch_shrink = 1;

	last_screen_size = Size2(OS::get_singleton()->get_video_mode().width, OS::get_singleton()->get_video_mode().height);
	_update_root_rect();

	if (ScriptDebugger::get_singleton()) {
		ScriptDebugger::get_singleton()->set_request_scene_tree_message_func(_debugger_request_tree, this);
	}

	root->set_physics_object_picking(GLOBAL_DEF("physics/common/enable_object_picking", true));

#ifdef TOOLS_ENABLED
	edited_scene_root = NULL;
#endif

#ifdef DEBUG_ENABLED

	live_edit_funcs.udata = this;
	live_edit_funcs.node_path_func = _live_edit_node_path_funcs;
	live_edit_funcs.res_path_func = _live_edit_res_path_funcs;
	live_edit_funcs.node_set_func = _live_edit_node_set_funcs;
	live_edit_funcs.node_set_res_func = _live_edit_node_set_res_funcs;
	live_edit_funcs.node_call_func = _live_edit_node_call_funcs;
	live_edit_funcs.res_set_func = _live_edit_res_set_funcs;
	live_edit_funcs.res_set_res_func = _live_edit_res_set_res_funcs;
	live_edit_funcs.res_call_func = _live_edit_res_call_funcs;
	live_edit_funcs.root_func = _live_edit_root_funcs;

	live_edit_funcs.tree_create_node_func = _live_edit_create_node_funcs;
	live_edit_funcs.tree_instance_node_func = _live_edit_instance_node_funcs;
	live_edit_funcs.tree_remove_node_func = _live_edit_remove_node_funcs;
	live_edit_funcs.tree_remove_and_keep_node_func = _live_edit_remove_and_keep_node_funcs;
	live_edit_funcs.tree_restore_node_func = _live_edit_restore_node_funcs;
	live_edit_funcs.tree_duplicate_node_func = _live_edit_duplicate_node_funcs;
	live_edit_funcs.tree_reparent_node_func = _live_edit_reparent_node_funcs;

	if (ScriptDebugger::get_singleton()) {
		ScriptDebugger::get_singleton()->set_live_edit_funcs(&live_edit_funcs);
	}

	live_edit_root = NodePath("/root");

	last_send_cache_id = 1;

#endif

	use_font_oversampling = false;
}

SceneTree::~SceneTree() {
}
