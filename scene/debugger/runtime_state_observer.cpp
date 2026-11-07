/**************************************************************************/
/*  runtime_state_observer.cpp                                            */
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

#include "runtime_state_observer.h"

#ifdef DEBUG_ENABLED

#include "core/debugger/engine_debugger.h"
#include "core/object/callable_mp.h"
#include "core/os/os.h"
#include "scene/main/node.h"
#include "scene/main/scene_tree.h"
#include "scene/main/viewport.h"
#include "scene/main/window.h"

#ifndef _2D_DISABLED
#include "scene/2d/node_2d.h"
#include "scene/2d/sprite_2d.h"
#endif // _2D_DISABLED

#ifndef _3D_DISABLED
#include "scene/3d/mesh_instance_3d.h"
#include "scene/3d/visual_instance_3d.h"
#endif // _3D_DISABLED

// New nodes announced in one message. More are announced in the next messages, so a large
// scene doesn't send one huge packet when the view opens.
static constexpr int MAX_ADDED_PER_MESSAGE = 2048;

static String _get_file_resource_path(const Ref<Resource> &p_resource) {
	if (p_resource.is_null()) {
		return String();
	}
	const String path = p_resource->get_path();
	// Built-in resources (scene::id) can't be loaded by the editor on their own.
	if (path.is_empty() || path.contains("::")) {
		return String();
	}
	return path;
}

Error RuntimeStateObserver::_capture(void *p_user, const String &p_msg, const Array &p_args, bool &r_captured) {
	r_captured = true;
	if (p_msg == "setup") {
		ERR_FAIL_NULL_V(singleton, ERR_UNCONFIGURED);
		singleton->_setup(p_args);
		return OK;
	}
	r_captured = false;
	return OK;
}

void RuntimeStateObserver::_setup(const Array &p_args) {
	ERR_FAIL_COND(p_args.size() < 6);
	enabled = p_args[0];
	const double rate_hz = CLAMP(double(p_args[1]), 1.0, 240.0);
	interval_usec = uint64_t(1000000.0 / rate_hz);
	scene_path = p_args[2];
	observe_3d = p_args[3];
	observe_2d = p_args[4];
	max_nodes = MAX(1, int(p_args[5]));

	// The editor drops its state when it sends a setup, so start over.
	_reset();
	_set_processing(enabled);
}

void RuntimeStateObserver::_set_processing(bool p_processing) {
	SceneTree *tree = SceneTree::get_singleton();
	if (!tree || processing == p_processing) {
		return;
	}
	processing = p_processing;
	if (processing) {
		tree->connect(SNAME("process_frame"), callable_mp(this, &RuntimeStateObserver::_process_frame));
	} else {
		tree->disconnect(SNAME("process_frame"), callable_mp(this, &RuntimeStateObserver::_process_frame));
		tracked.clear();
	}
}

void RuntimeStateObserver::_reset() {
	tracked.clear();
	next_id = 1;
	root_id = ObjectID();
	reset_pending = true;
	last_send_usec = 0;
}

Node *RuntimeStateObserver::_find_root() const {
	SceneTree *tree = SceneTree::get_singleton();
	if (!tree) {
		return nullptr;
	}
	Node *current = tree->get_current_scene();
	if (!current) {
		return nullptr;
	}
	if (scene_path.is_empty() || current->get_scene_file_path() == scene_path) {
		return current;
	}

	// The edited scene may be instanced inside the running scene (a level inside a main scene).
	LocalVector<Node *> stack;
	stack.push_back(current);
	int visited = 0;
	while (!stack.is_empty() && visited < max_nodes) {
		Node *n = stack[stack.size() - 1];
		stack.resize(stack.size() - 1);
		visited++;
		if (n->get_scene_file_path() == scene_path) {
			return n;
		}
		for (int i = n->get_child_count(false) - 1; i >= 0; i--) {
			stack.push_back(n->get_child(i, false));
		}
	}
	return current;
}

void RuntimeStateObserver::_process_frame() {
	if (!enabled) {
		return;
	}
	const uint64_t now = OS::get_singleton()->get_ticks_usec();
	if (last_send_usec != 0 && now - last_send_usec < interval_usec) {
		return;
	}
	last_send_usec = now;

	Node *root = ObjectDB::get_instance<Node>(root_id);
	if (!root || !root->is_inside_tree()) {
		if (root_id.is_valid()) {
			// The observed scene was freed (scene change); start over with the new one.
			tracked.clear();
			next_id = 1;
			reset_pending = true;
		}
		root = _find_root();
		root_id = root ? root->get_instance_id() : ObjectID();
		if (!root) {
			return;
		}
	}

	tick++;

	Array added;
	PackedInt32Array removed;
	PackedInt32Array ids_3d;
	PackedFloat32Array xforms_3d;
	PackedInt32Array ids_2d;
	PackedFloat32Array xforms_2d;
	PackedInt32Array shown;
	PackedInt32Array hidden;

	int count = 0;
	walk_stack.clear();
	walk_stack.push_back(root);
	while (!walk_stack.is_empty()) {
		Node *n = walk_stack[walk_stack.size() - 1];
		walk_stack.resize(walk_stack.size() - 1);

		// Nodes inside a SubViewport render to their own world or canvas, not the editor's view.
		if (n != root && Object::cast_to<Viewport>(n)) {
			continue;
		}
		for (int i = n->get_child_count(false) - 1; i >= 0; i--) {
			walk_stack.push_back(n->get_child(i, false));
		}

		Kind kind;
		bool observed = false;
		Transform3D xform_3d;
		Transform2D xform_2d;
		String resource;
#ifndef _3D_DISABLED
		if (observe_3d) {
			if (VisualInstance3D *vi = Object::cast_to<VisualInstance3D>(n)) {
				kind = KIND_3D;
				observed = true;
				xform_3d = vi->get_global_transform();
				if (MeshInstance3D *mi = Object::cast_to<MeshInstance3D>(vi)) {
					resource = _get_file_resource_path(mi->get_mesh());
				}
			}
		}
#endif // _3D_DISABLED
#ifndef _2D_DISABLED
		if (!observed && observe_2d) {
			if (Node2D *n2d = Object::cast_to<Node2D>(n)) {
				kind = KIND_2D;
				observed = true;
				xform_2d = n2d->get_global_transform();
				if (Sprite2D *sprite = Object::cast_to<Sprite2D>(n2d)) {
					resource = _get_file_resource_path(sprite->get_texture());
				}
			}
		}
#endif // _2D_DISABLED
		if (!observed || n == root) {
			// The root itself is matched by the editor's edited scene root, nothing to send.
			continue;
		}
		if (count >= max_nodes) {
			continue;
		}

		const ObjectID oid = n->get_instance_id();
		Tracked *t = tracked.getptr(oid);
		const bool is_new = !t;
		if (is_new) {
			if (added.size() >= MAX_ADDED_PER_MESSAGE) {
				continue; // Announced in a later message.
			}
			Tracked nt;
			nt.id = next_id++;
			nt.kind = kind;
			t = &tracked.insert(oid, nt)->value;

			// What the editor needs to show a node that isn't in the edited scene:
			// the scene instance it comes from, or its own mesh or texture.
			String owner_path;
			String owner_scene;
			Node *instance_root = n->get_scene_file_path().is_empty() ? n->get_owner() : n;
			if (instance_root && instance_root != root && root->is_ancestor_of(instance_root) && !instance_root->get_scene_file_path().is_empty()) {
				owner_path = String(root->get_path_to(instance_root));
				owner_scene = instance_root->get_scene_file_path();
			}

			Array entry;
			entry.resize(ADDED_MAX);
			entry[ADDED_ID] = t->id;
			entry[ADDED_KIND] = kind;
			entry[ADDED_PATH] = String(root->get_path_to(n));
			entry[ADDED_CLASS] = n->get_class();
			entry[ADDED_OWNER_PATH] = owner_path;
			entry[ADDED_OWNER_SCENE] = owner_scene;
			entry[ADDED_RESOURCE] = resource;
			added.push_back(entry);
		}
		t->seen_tick = tick;
		count++;

		if (kind == KIND_3D) {
			if (is_new || !t->xform_3d.is_equal_approx(xform_3d)) {
				t->xform_3d = xform_3d;
				ids_3d.push_back(t->id);
				for (int r = 0; r < 3; r++) {
					xforms_3d.push_back(xform_3d.basis.rows[r].x);
					xforms_3d.push_back(xform_3d.basis.rows[r].y);
					xforms_3d.push_back(xform_3d.basis.rows[r].z);
				}
				xforms_3d.push_back(xform_3d.origin.x);
				xforms_3d.push_back(xform_3d.origin.y);
				xforms_3d.push_back(xform_3d.origin.z);
			}
		} else {
			if (is_new || !t->xform_2d.is_equal_approx(xform_2d)) {
				t->xform_2d = xform_2d;
				ids_2d.push_back(t->id);
				for (int c = 0; c < 3; c++) {
					xforms_2d.push_back(xform_2d.columns[c].x);
					xforms_2d.push_back(xform_2d.columns[c].y);
				}
			}
		}

		const bool visible = Object::cast_to<CanvasItem>(n) ? Object::cast_to<CanvasItem>(n)->is_visible_in_tree() : true;
#ifndef _3D_DISABLED
		const bool visible_3d = Object::cast_to<Node3D>(n) ? Object::cast_to<Node3D>(n)->is_visible_in_tree() : visible;
#else
		const bool visible_3d = visible;
#endif // _3D_DISABLED
		const bool is_visible = kind == KIND_3D ? visible_3d : visible;
		if (is_new || t->visible != is_visible) {
			t->visible = is_visible;
			if (is_visible) {
				shown.push_back(t->id);
			} else {
				hidden.push_back(t->id);
			}
		}
	}

	LocalVector<ObjectID> gone;
	for (const KeyValue<ObjectID, Tracked> &E : tracked) {
		if (E.value.seen_tick != tick) {
			gone.push_back(E.key);
			removed.push_back(E.value.id);
		}
	}
	for (const ObjectID &oid : gone) {
		tracked.erase(oid);
	}

	const bool changed = reset_pending || !added.is_empty() || !removed.is_empty() || !ids_3d.is_empty() || !ids_2d.is_empty() || !shown.is_empty() || !hidden.is_empty();
	if (!changed) {
		return;
	}

	Array msg;
	msg.push_back(reset_pending);
	msg.push_back(root->get_scene_file_path());
	msg.push_back(added);
	msg.push_back(removed);
	msg.push_back(ids_3d);
	msg.push_back(xforms_3d);
	msg.push_back(ids_2d);
	msg.push_back(xforms_2d);
	msg.push_back(shown);
	msg.push_back(hidden);
	reset_pending = false;
	EngineDebugger::get_singleton()->send_message("runtime_observer:state", msg);
}

RuntimeStateObserver::RuntimeStateObserver() {
	singleton = this;
	EngineDebugger::register_message_capture("runtime_observer", EngineDebugger::Capture(nullptr, RuntimeStateObserver::_capture));
}

RuntimeStateObserver::~RuntimeStateObserver() {
	_set_processing(false);
	EngineDebugger::unregister_message_capture("runtime_observer");
	singleton = nullptr;
}

#endif // DEBUG_ENABLED
