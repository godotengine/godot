/*************************************************************************/
/*  scene_main_loop.cpp                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "scene_main_loop.h"

#include "globals.h"
#include "io/resource_loader.h"
#include "message_queue.h"
#include "node.h"
#include "os/keyboard.h"
#include "os/os.h"
#include "print_string.h"
#include "scene/resources/material.h"
#include "scene/resources/mesh.h"
#include "scene/resources/packed_scene.h"
#include "scene/scene_string_names.h"
#include "servers/physics_2d_server.h"
#include "servers/physics_server.h"
#include "servers/spatial_sound_2d_server.h"
#include "viewport.h"
#include <stdio.h>

void SceneTree::tree_changed() {

	tree_version++;
	emit_signal(tree_changed_name);
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

void SceneTree::_flush_transform_notifications() {

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

		call_group(GROUP_CALL_REALTIME, E->key().group, E->key().call, v[0], v[1], v[2], v[3], v[4]);

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

void SceneTree::call_group(uint32_t p_call_flags, const StringName &p_group, const StringName &p_function, VARIANT_ARG_DECLARE) {

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
				if (p_call_flags & GROUP_CALL_MULIILEVEL)
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
				if (p_call_flags & GROUP_CALL_MULIILEVEL)
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

void SceneTree::notify_group(uint32_t p_call_flags, const StringName &p_group, int p_notification) {

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

void SceneTree::set_group(uint32_t p_call_flags, const StringName &p_group, const String &p_name, const Variant &p_value) {

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

void SceneTree::set_input_as_handled() {

	input_handled = true;
}

void SceneTree::input_text(const String &p_text) {

	root_lock++;

	call_group(GROUP_CALL_REALTIME, "_viewports", "_vp_input_text", p_text); //special one for GUI, as controls use their own process check

	root_lock--;
}

bool SceneTree::is_input_handled() {
	return input_handled;
}

void SceneTree::input_event(const InputEvent &p_event) {

	if (is_editor_hint() && (p_event.type == InputEvent::JOYSTICK_MOTION || p_event.type == InputEvent::JOYSTICK_BUTTON))
		return; //avoid joy input on editor

	root_lock++;
	//last_id=p_event.ID;

	input_handled = false;

	InputEvent ev = p_event;
	ev.ID = ++last_id; //this should work better
#if 0
	switch(ev.type) {

		case InputEvent::MOUSE_BUTTON: {

			Matrix32 ai = root->get_final_transform().affine_inverse();
			Vector2 g = ai.xform(Vector2(ev.mouse_button.global_x,ev.mouse_button.global_y));
			Vector2 l = ai.xform(Vector2(ev.mouse_button.x,ev.mouse_button.y));
			ev.mouse_button.x=l.x;
			ev.mouse_button.y=l.y;
			ev.mouse_button.global_x=g.x;
			ev.mouse_button.global_y=g.y;

		} break;
		case InputEvent::MOUSE_MOTION: {

			Matrix32 ai = root->get_final_transform().affine_inverse();
			Vector2 g = ai.xform(Vector2(ev.mouse_motion.global_x,ev.mouse_motion.global_y));
			Vector2 l = ai.xform(Vector2(ev.mouse_motion.x,ev.mouse_motion.y));
			Vector2 r = ai.xform(Vector2(ev.mouse_motion.relative_x,ev.mouse_motion.relative_y));
			ev.mouse_motion.x=l.x;
			ev.mouse_motion.y=l.y;
			ev.mouse_motion.global_x=g.x;
			ev.mouse_motion.global_y=g.y;
			ev.mouse_motion.relative_x=r.x;
			ev.mouse_motion.relative_y=r.y;

		} break;
		case InputEvent::SCREEN_TOUCH: {

			Matrix32 ai = root->get_final_transform().affine_inverse();
			Vector2 t = ai.xform(Vector2(ev.screen_touch.x,ev.screen_touch.y));
			ev.screen_touch.x=t.x;
			ev.screen_touch.y=t.y;

		} break;
		case InputEvent::SCREEN_DRAG: {

			Matrix32 ai = root->get_final_transform().affine_inverse();
			Vector2 t = ai.xform(Vector2(ev.screen_drag.x,ev.screen_drag.y));
			Vector2 r = ai.xform(Vector2(ev.screen_drag.relative_x,ev.screen_drag.relative_y));
			Vector2 s = ai.xform(Vector2(ev.screen_drag.speed_x,ev.screen_drag.speed_y));
			ev.screen_drag.x=t.x;
			ev.screen_drag.y=t.y;
			ev.screen_drag.relative_x=r.x;
			ev.screen_drag.relative_y=r.y;
			ev.screen_drag.speed_x=s.x;
			ev.screen_drag.speed_y=s.y;
		} break;
	}

#endif

	MainLoop::input_event(ev);
#if 0
	_call_input_pause("input","_input",ev);

	call_group(GROUP_CALL_REVERSE|GROUP_CALL_REALTIME|GROUP_CALL_MULIILEVEL,"_gui_input","_gui_input",p_event); //special one for GUI, as controls use their own process check

	//call_group(GROUP_CALL_REVERSE|GROUP_CALL_REALTIME|GROUP_CALL_MULIILEVEL,"input","_input",ev);

	/*if (ev.type==InputEvent::KEY && ev.key.pressed && !ev.key.echo && ev.key.scancode==KEY_F12) {

		print_line("RAM: "+itos(Memory::get_static_mem_usage()));
		print_line("DRAM: "+itos(Memory::get_dynamic_mem_usage()));
	}
*/
	//if (ev.type==InputEvent::KEY && ev.key.pressed && !ev.key.echo && ev.key.scancode==KEY_F11) {

	//	Memory::dump_static_mem_to_file("memdump.txt");
	//}

	//transform for the rest
#else

	call_group(GROUP_CALL_REALTIME, "_viewports", "_vp_input", ev); //special one for GUI, as controls use their own process check

#endif
	if (ScriptDebugger::get_singleton() && ScriptDebugger::get_singleton()->is_remote() && ev.type == InputEvent::KEY && ev.key.pressed && !ev.key.echo && ev.key.scancode == KEY_F8) {

		ScriptDebugger::get_singleton()->request_quit();
	}

	_flush_ugc();
	root_lock--;
	MessageQueue::get_singleton()->flush(); //small little hack

	root_lock++;

	if (!input_handled) {

#if 0
		_call_input_pause("unhandled_input","_unhandled_input",ev);
		//call_group(GROUP_CALL_REVERSE|GROUP_CALL_REALTIME|GROUP_CALL_MULIILEVEL,"unhandled_input","_unhandled_input",ev);
		if (!input_handled && ev.type==InputEvent::KEY) {
			_call_input_pause("unhandled_key_input","_unhandled_key_input",ev);
			//call_group(GROUP_CALL_REVERSE|GROUP_CALL_REALTIME|GROUP_CALL_MULIILEVEL,"unhandled_key_input","_unhandled_key_input",ev);
		}
#else

		call_group(GROUP_CALL_REALTIME, "_viewports", "_vp_unhandled_input", ev); //special one for GUI, as controls use their own process check

#endif
		input_handled = true;
		_flush_ugc();
		root_lock--;
		MessageQueue::get_singleton()->flush(); //small little hack
	} else {
		input_handled = true;
		root_lock--;
	}
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

	_flush_transform_notifications();

	MainLoop::iteration(p_time);
	fixed_process_time = p_time;

	emit_signal("fixed_frame");

	_notify_group_pause("fixed_process", Node::NOTIFICATION_FIXED_PROCESS);
	_flush_ugc();
	_flush_transform_notifications();
	call_group(GROUP_CALL_REALTIME, "_viewports", "update_worlds");
	root_lock--;

	_flush_delete_queue();

	return _quit;
}

bool SceneTree::idle(float p_time) {

	//	print_line("ram: "+itos(OS::get_singleton()->get_static_memory_usage())+" sram: "+itos(OS::get_singleton()->get_dynamic_memory_usage()));
	//	print_line("node count: "+itos(get_node_count()));
	//	print_line("TEXTURE RAM: "+itos(VS::get_singleton()->get_render_info(VS::INFO_TEXTURE_MEM_USED)));

	root_lock++;

	MainLoop::idle(p_time);

	idle_process_time = p_time;

	emit_signal("idle_frame");

	_flush_transform_notifications();

	_notify_group_pause("idle_process", Node::NOTIFICATION_PROCESS);

	Size2 win_size = Size2(OS::get_singleton()->get_video_mode().width, OS::get_singleton()->get_video_mode().height);
	if (win_size != last_screen_size) {

		last_screen_size = win_size;
		_update_root_rect();

		emit_signal("screen_resized");
	}

	_flush_ugc();
	_flush_transform_notifications(); //transforms after world update, to avoid unnecesary enter/exit notifications
	call_group(GROUP_CALL_REALTIME, "_viewports", "update_worlds");

	root_lock--;

	_flush_delete_queue();

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
		case NOTIFICATION_OS_MEMORY_WARNING:
		case NOTIFICATION_WM_MOUSE_ENTER:
		case NOTIFICATION_WM_MOUSE_EXIT:
		case NOTIFICATION_WM_FOCUS_IN:
		case NOTIFICATION_WM_FOCUS_OUT: {

			get_root()->propagate_notification(p_notification);
		} break;
		case NOTIFICATION_WM_UNFOCUS_REQUEST: {

			notify_group(GROUP_CALL_REALTIME | GROUP_CALL_MULIILEVEL, "input", NOTIFICATION_WM_UNFOCUS_REQUEST);

		} break;

		default:
			break;
	};
};

void SceneTree::set_auto_accept_quit(bool p_enable) {

	accept_quit = p_enable;
}

#ifdef TOOLS_ENABLED
void SceneTree::set_editor_hint(bool p_enabled) {

	editor_hint = p_enabled;
}

bool SceneTree::is_node_being_edited(const Node *p_node) const {

	return editor_hint && edited_scene_root && (edited_scene_root->is_a_parent_of(p_node) || edited_scene_root == p_node);
}

bool SceneTree::is_editor_hint() const {

	return editor_hint;
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

	Ref<FixedMaterial> line_material = Ref<FixedMaterial>(memnew(FixedMaterial));
	line_material->set_flag(Material::FLAG_UNSHADED, true);
	line_material->set_line_width(3.0);
	line_material->set_fixed_flag(FixedMaterial::FLAG_USE_ALPHA, true);
	line_material->set_fixed_flag(FixedMaterial::FLAG_USE_COLOR_ARRAY, true);
	line_material->set_parameter(FixedMaterial::PARAM_DIFFUSE, get_debug_navigation_color());

	navigation_material = line_material;

	return navigation_material;
}

Ref<Material> SceneTree::get_debug_navigation_disabled_material() {

	if (navigation_disabled_material.is_valid())
		return navigation_disabled_material;

	Ref<FixedMaterial> line_material = Ref<FixedMaterial>(memnew(FixedMaterial));
	line_material->set_flag(Material::FLAG_UNSHADED, true);
	line_material->set_line_width(3.0);
	line_material->set_fixed_flag(FixedMaterial::FLAG_USE_ALPHA, true);
	line_material->set_fixed_flag(FixedMaterial::FLAG_USE_COLOR_ARRAY, true);
	line_material->set_parameter(FixedMaterial::PARAM_DIFFUSE, get_debug_navigation_disabled_color());

	navigation_disabled_material = line_material;

	return navigation_disabled_material;
}
Ref<Material> SceneTree::get_debug_collision_material() {

	if (collision_material.is_valid())
		return collision_material;

	Ref<FixedMaterial> line_material = Ref<FixedMaterial>(memnew(FixedMaterial));
	line_material->set_flag(Material::FLAG_UNSHADED, true);
	line_material->set_line_width(3.0);
	line_material->set_fixed_flag(FixedMaterial::FLAG_USE_ALPHA, true);
	line_material->set_fixed_flag(FixedMaterial::FLAG_USE_COLOR_ARRAY, true);
	line_material->set_parameter(FixedMaterial::PARAM_DIFFUSE, get_debug_collisions_color());

	collision_material = line_material;

	return collision_material;
}

Ref<Mesh> SceneTree::get_debug_contact_mesh() {

	if (debug_contact_mesh.is_valid())
		return debug_contact_mesh;

	debug_contact_mesh = Ref<Mesh>(memnew(Mesh));

	Ref<FixedMaterial> mat = memnew(FixedMaterial);
	mat->set_flag(Material::FLAG_UNSHADED, true);
	mat->set_flag(Material::FLAG_DOUBLE_SIDED, true);
	mat->set_fixed_flag(FixedMaterial::FLAG_USE_ALPHA, true);
	mat->set_parameter(FixedMaterial::PARAM_DIFFUSE, get_debug_collision_contact_color());

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

	DVector<int> indices;
	for (int i = 0; i < 8 * 3; i++)
		indices.push_back(diamond_faces[i]);

	DVector<Vector3> vertices;
	for (int i = 0; i < 6; i++)
		vertices.push_back(diamond[i] * 0.1);

	Array arr;
	arr.resize(Mesh::ARRAY_MAX);
	arr[Mesh::ARRAY_VERTEX] = vertices;
	arr[Mesh::ARRAY_INDEX] = indices;

	debug_contact_mesh->add_surface(Mesh::PRIMITIVE_TRIANGLES, arr);
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

void SceneTree::_call_input_pause(const StringName &p_group, const StringName &p_method, const InputEvent &p_input) {

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

		Variant::CallError ce;
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

Variant SceneTree::_call_group(const Variant **p_args, int p_argcount, Variant::CallError &r_error) {

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

	call_group(flags, group, method, v[0], v[1], v[2], v[3], v[4]);
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

	Node **ptr = E->get().nodes.ptr();
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
	Node **ptr = E->get().nodes.ptr();
	for (int i = 0; i < nc; i++) {

		p_list->push_back(ptr[i]);
	}
}

static void _fill_array(Node *p_node, Array &array, int p_level) {

	array.push_back(p_level);
	array.push_back(p_node->get_name());
	array.push_back(p_node->get_type());
	array.push_back(p_node->get_instance_ID());
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
	delete_queue.push_back(p_object->get_instance_ID());
}

int SceneTree::get_node_count() const {

	return node_count;
}

void SceneTree::_update_root_rect() {

	if (stretch_mode == STRETCH_MODE_DISABLED) {
		root->set_rect(Rect2(Point2(), last_screen_size));
		return; //user will take care
	}

	//actual screen video mode
	Size2 video_mode = Size2(OS::get_singleton()->get_video_mode().width, OS::get_singleton()->get_video_mode().height);
	Size2 desired_res = stretch_min;

	Size2 viewport_size;
	Size2 screen_size;

	float viewport_aspect = desired_res.get_aspect();
	float video_mode_aspect = video_mode.get_aspect();

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

	//	print_line("VP SIZE: "+viewport_size+" OFFSET: "+offset+" = "+(offset*2+viewport_size));
	//	print_line("SS: "+video_mode);
	switch (stretch_mode) {
		case STRETCH_MODE_2D: {

			//			root->set_rect(Rect2(Point2(),video_mode));
			root->set_as_render_target(false);
			root->set_rect(Rect2(margin, screen_size));
			root->set_size_override_stretch(true);
			root->set_size_override(true, viewport_size);

		} break;
		case STRETCH_MODE_VIEWPORT: {

			root->set_rect(Rect2(Point2(), viewport_size));
			root->set_size_override_stretch(false);
			root->set_size_override(false, Size2());
			root->set_as_render_target(true);
			root->set_render_target_update_mode(Viewport::RENDER_TARGET_UPDATE_ALWAYS);
			root->set_render_target_to_screen_rect(Rect2(margin, screen_size));

		} break;
	}
}

void SceneTree::set_screen_stretch(StretchMode p_mode, StretchAspect p_aspect, const Size2 p_minsize) {

	stretch_mode = p_mode;
	stretch_aspect = p_aspect;
	stretch_min = p_minsize;
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

		Object *o = ObjectTypeDB::instance(p_type);
		if (!o)
			continue;
		Node *no = o->cast_to<Node>();
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

void SceneTree::_bind_methods() {

	//ObjectTypeDB::bind_method(_MD("call_group","call_flags","group","method","arg1","arg2"),&SceneMainLoop::_call_group,DEFVAL(Variant()),DEFVAL(Variant()));
	ObjectTypeDB::bind_method(_MD("notify_group", "call_flags", "group", "notification"), &SceneTree::notify_group);
	ObjectTypeDB::bind_method(_MD("set_group", "call_flags", "group", "property", "value"), &SceneTree::set_group);

	ObjectTypeDB::bind_method(_MD("get_nodes_in_group", "group"), &SceneTree::_get_nodes_in_group);

	ObjectTypeDB::bind_method(_MD("get_root:Viewport"), &SceneTree::get_root);
	ObjectTypeDB::bind_method(_MD("has_group", "name"), &SceneTree::has_group);

	ObjectTypeDB::bind_method(_MD("set_auto_accept_quit", "enabled"), &SceneTree::set_auto_accept_quit);

	ObjectTypeDB::bind_method(_MD("set_editor_hint", "enable"), &SceneTree::set_editor_hint);
	ObjectTypeDB::bind_method(_MD("is_editor_hint"), &SceneTree::is_editor_hint);
	ObjectTypeDB::bind_method(_MD("set_debug_collisions_hint", "enable"), &SceneTree::set_debug_collisions_hint);
	ObjectTypeDB::bind_method(_MD("is_debugging_collisions_hint"), &SceneTree::is_debugging_collisions_hint);
	ObjectTypeDB::bind_method(_MD("set_debug_navigation_hint", "enable"), &SceneTree::set_debug_navigation_hint);
	ObjectTypeDB::bind_method(_MD("is_debugging_navigation_hint"), &SceneTree::is_debugging_navigation_hint);

#ifdef TOOLS_ENABLED
	ObjectTypeDB::bind_method(_MD("set_edited_scene_root", "scene"), &SceneTree::set_edited_scene_root);
	ObjectTypeDB::bind_method(_MD("get_edited_scene_root"), &SceneTree::get_edited_scene_root);
#endif

	ObjectTypeDB::bind_method(_MD("set_pause", "enable"), &SceneTree::set_pause);
	ObjectTypeDB::bind_method(_MD("is_paused"), &SceneTree::is_paused);
	ObjectTypeDB::bind_method(_MD("set_input_as_handled"), &SceneTree::set_input_as_handled);
	ObjectTypeDB::bind_method(_MD("is_input_handled"), &SceneTree::is_input_handled);

	ObjectTypeDB::bind_method(_MD("get_node_count"), &SceneTree::get_node_count);
	ObjectTypeDB::bind_method(_MD("get_frame"), &SceneTree::get_frame);
	ObjectTypeDB::bind_method(_MD("quit"), &SceneTree::quit);

	ObjectTypeDB::bind_method(_MD("set_screen_stretch", "mode", "aspect", "minsize"), &SceneTree::set_screen_stretch);

	ObjectTypeDB::bind_method(_MD("queue_delete", "obj"), &SceneTree::queue_delete);

	MethodInfo mi;
	mi.name = "call_group";
	mi.arguments.push_back(PropertyInfo(Variant::INT, "flags"));
	mi.arguments.push_back(PropertyInfo(Variant::STRING, "group"));
	mi.arguments.push_back(PropertyInfo(Variant::STRING, "method"));
	Vector<Variant> defargs;
	for (int i = 0; i < VARIANT_ARG_MAX; i++) {
		mi.arguments.push_back(PropertyInfo(Variant::NIL, "arg" + itos(i)));
		defargs.push_back(Variant());
	}

	ObjectTypeDB::bind_native_method(METHOD_FLAGS_DEFAULT, "call_group", &SceneTree::_call_group, mi, defargs);

	ObjectTypeDB::bind_method(_MD("set_current_scene", "child_node:Node"), &SceneTree::set_current_scene);
	ObjectTypeDB::bind_method(_MD("get_current_scene:Node"), &SceneTree::get_current_scene);

	ObjectTypeDB::bind_method(_MD("change_scene", "path"), &SceneTree::change_scene);
	ObjectTypeDB::bind_method(_MD("change_scene_to", "packed_scene:PackedScene"), &SceneTree::change_scene_to);

	ObjectTypeDB::bind_method(_MD("reload_current_scene"), &SceneTree::reload_current_scene);

	ObjectTypeDB::bind_method(_MD("_change_scene"), &SceneTree::_change_scene);

	ADD_SIGNAL(MethodInfo("tree_changed"));
	ADD_SIGNAL(MethodInfo("node_removed", PropertyInfo(Variant::OBJECT, "node")));
	ADD_SIGNAL(MethodInfo("screen_resized"));
	ADD_SIGNAL(MethodInfo("node_configuration_warning_changed", PropertyInfo(Variant::OBJECT, "node")));

	ADD_SIGNAL(MethodInfo("idle_frame"));
	ADD_SIGNAL(MethodInfo("fixed_frame"));

	ADD_SIGNAL(MethodInfo("files_dropped", PropertyInfo(Variant::STRING_ARRAY, "files"), PropertyInfo(Variant::INT, "screen")));

	BIND_CONSTANT(GROUP_CALL_DEFAULT);
	BIND_CONSTANT(GROUP_CALL_REVERSE);
	BIND_CONSTANT(GROUP_CALL_REALTIME);
	BIND_CONSTANT(GROUP_CALL_UNIQUE);

	BIND_CONSTANT(STRETCH_MODE_DISABLED);
	BIND_CONSTANT(STRETCH_MODE_2D);
	BIND_CONSTANT(STRETCH_MODE_VIEWPORT);
	BIND_CONSTANT(STRETCH_ASPECT_IGNORE);
	BIND_CONSTANT(STRETCH_ASPECT_KEEP);
	BIND_CONSTANT(STRETCH_ASPECT_KEEP_WIDTH);
	BIND_CONSTANT(STRETCH_ASPECT_KEEP_HEIGHT);
}

SceneTree *SceneTree::singleton = NULL;

SceneTree::SceneTree() {

	singleton = this;
	_quit = false;
	accept_quit = true;
	initialized = false;
#ifdef TOOLS_ENABLED
	editor_hint = false;
#endif
#ifdef DEBUG_ENABLED
	debug_collisions_hint = false;
	debug_navigation_hint = false;
#endif
	debug_collisions_color = GLOBAL_DEF("debug/collision_shape_color", Color(0.0, 0.6, 0.7, 0.5));
	debug_collision_contact_color = GLOBAL_DEF("debug/collision_contact_color", Color(1.0, 0.2, 0.1, 0.8));
	debug_navigation_color = GLOBAL_DEF("debug/navigation_geometry_color", Color(0.1, 1.0, 0.7, 0.4));
	debug_navigation_disabled_color = GLOBAL_DEF("debug/navigation_disabled_geometry_color", Color(1.0, 0.7, 0.1, 0.4));
	collision_debug_contacts = GLOBAL_DEF("debug/collision_max_contacts_displayed", 10000);

	tree_version = 1;
	fixed_process_time = 1;
	idle_process_time = 1;
	last_id = 1;
	root = NULL;
	current_frame = 0;
	tree_changed_name = "tree_changed";
	node_removed_name = "node_removed";
	ugc_locked = false;
	call_lock = 0;
	root_lock = 0;
	node_count = 0;

	//create with mainloop

	root = memnew(Viewport);
	root->set_name("root");
	root->set_world(Ref<World>(memnew(World)));
	//root->set_world_2d( Ref<World2D>( memnew( World2D )));
	root->set_as_audio_listener(true);
	root->set_as_audio_listener_2d(true);
	current_scene = NULL;

	stretch_mode = STRETCH_MODE_DISABLED;
	stretch_aspect = STRETCH_ASPECT_IGNORE;

	last_screen_size = Size2(OS::get_singleton()->get_video_mode().width, OS::get_singleton()->get_video_mode().height);
	root->set_rect(Rect2(Point2(), last_screen_size));

	if (ScriptDebugger::get_singleton()) {
		ScriptDebugger::get_singleton()->set_request_scene_tree_message_func(_debugger_request_tree, this);
	}

	root->set_physics_object_picking(GLOBAL_DEF("physics/enable_object_picking", true));

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

#endif
}

SceneTree::~SceneTree() {
}
