/**************************************************************************/
/*  live_runtime_plugin.cpp                                               */
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

#include "live_runtime_plugin.h"

#include "core/io/resource_loader.h"
#include "core/object/callable_mp.h"
#include "editor/editor_node.h"
#include "editor/editor_string_names.h"
#include "editor/scene/3d/node_3d_editor_plugin.h"
#include "editor/scene/canvas_item_editor_plugin.h"
#include "editor/settings/editor_settings.h"
#include "scene/2d/node_2d.h"
#include "scene/2d/sprite_2d.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/3d/visual_instance_3d.h"
#include "scene/debugger/runtime_state_observer.h"
#include "scene/gui/button.h"
#include "scene/main/viewport.h"
#include "scene/resources/packed_scene.h"
#include "servers/rendering/rendering_server.h"

// Fields of one entry of the "added" array, see RuntimeStateObserver.
enum {
	ADDED_ID,
	ADDED_KIND,
	ADDED_PATH,
	ADDED_CLASS,
	ADDED_OWNER_PATH,
	ADDED_OWNER_SCENE,
	ADDED_RESOURCE,
	ADDED_MAX,
};

// Fields of the "state" message.
enum {
	STATE_RESET,
	STATE_ROOT_PATH,
	STATE_ADDED,
	STATE_REMOVED,
	STATE_IDS_3D,
	STATE_XFORMS_3D,
	STATE_IDS_2D,
	STATE_XFORMS_2D,
	STATE_SHOWN,
	STATE_HIDDEN,
	STATE_MAX,
};

void LiveRuntimeDebugger::_bind_methods() {
	ADD_SIGNAL(MethodInfo("status_changed"));
}

bool LiveRuntimeDebugger::has_capture(const String &p_capture) const {
	return p_capture == "runtime_observer";
}

void LiveRuntimeDebugger::setup_session(int p_session_id) {
	Ref<EditorDebuggerSession> session = get_session(p_session_id);
	ERR_FAIL_COND(session.is_null());
	session->connect("started", callable_mp(this, &LiveRuntimeDebugger::_session_started).bind(p_session_id));
	session->connect("stopped", callable_mp(this, &LiveRuntimeDebugger::_session_stopped).bind(p_session_id));
}

void LiveRuntimeDebugger::_session_started(int p_session) {
	update_observing();
}

void LiveRuntimeDebugger::_session_stopped(int p_session) {
	if (p_session == observed_session) {
		_clear();
		observing = false;
		observed_session = -1;
		_set_status(String());
	}
	update_observing();
}

bool LiveRuntimeDebugger::capture(const String &p_message, const Array &p_data, int p_session) {
	if (p_message == "runtime_observer:state") {
		if (observing && p_session == observed_session) {
			_msg_state(p_data);
		}
		return true;
	}
	return false;
}

void LiveRuntimeDebugger::set_mode(Mode p_mode) {
	if (mode == p_mode) {
		return;
	}
	mode = p_mode;
	update_observing();
}

void LiveRuntimeDebugger::update_observing() {
	// Observe only while a view that can show it is visible: nothing is sent for hidden views.
	const bool want_3d = Node3DEditor::get_singleton() && Node3DEditor::get_singleton()->is_visible_in_tree();
	const bool want_2d = CanvasItemEditor::get_singleton() && CanvasItemEditor::get_singleton()->is_visible_in_tree();

	int session_id = -1;
	if (mode == MODE_LIVE && (want_3d || want_2d)) {
		Array all_sessions = get_sessions();
		for (int i = 0; i < all_sessions.size(); i++) {
			Ref<EditorDebuggerSession> session = all_sessions[i];
			if (session.is_valid() && session->is_active()) {
				session_id = i; // The first running instance is shown.
				break;
			}
		}
	}

	Node *edited = EditorNode::get_singleton()->get_edited_scene();
	const String scene_path = edited ? edited->get_scene_file_path() : String();

	// Every setup restarts the stream, so the editor's state starts over too.
	_clear();

	if (observing && observed_session != session_id) {
		Ref<EditorDebuggerSession> old_session = get_session(observed_session);
		if (old_session.is_valid() && old_session->is_active()) {
			old_session->send_message("runtime_observer:setup", { false, 1, String(), false, false, 1 });
		}
	}

	observing = session_id != -1;
	observed_session = session_id;
	observe_3d = want_3d;
	observe_2d = want_2d;

	if (!observing) {
		if (mode != MODE_LIVE) {
			_set_status(String());
		} else if (!want_3d && !want_2d) {
			_set_status(TTR("Live Runtime: paused while the 2D and 3D views are hidden."));
		} else {
			_set_status(TTR("Live Runtime: waiting for a running game."));
		}
		return;
	}

	const int rate = EDITOR_GET("run/live_runtime/update_rate");
	const int max_nodes = EDITOR_GET("run/live_runtime/max_nodes");
	Ref<EditorDebuggerSession> session = get_session(session_id);
	session->send_message("runtime_observer:setup", { true, rate, scene_path, want_3d, want_2d, max_nodes });
	_set_status(TTR("Live Runtime: connecting..."));
}

Node *LiveRuntimeDebugger::_get_ghost_parent(bool p_2d) {
	ObjectID &id = p_2d ? ghost_parent_2d : ghost_parent_3d;
	Node *parent = ObjectDB::get_instance<Node>(id);
	if (parent) {
		return parent;
	}
	if (p_2d) {
		parent = memnew(Node2D);
		parent->set_name("LiveRuntimeGhosts2D");
	} else {
		parent = memnew(Node3D);
		parent->set_name("LiveRuntimeGhosts3D");
	}
	// Ghosts don't run: they only show what the game reports.
	parent->set_process_mode(Node::PROCESS_MODE_DISABLED);
	// An internal child of the editor's scene root: rendered by the editor viewports,
	// but never part of the edited scene, its tree or its saved file.
	EditorNode::get_singleton()->get_scene_root()->add_child(parent, false, Node::INTERNAL_MODE_BACK);
	id = parent->get_instance_id();
	return parent;
}

void LiveRuntimeDebugger::_hide_ghost_visuals(Node *p_node) {
	// A ghost's visuals stay hidden until the game reports them, so parts the game removed don't show up.
	if (VisualInstance3D *vi = Object::cast_to<VisualInstance3D>(p_node)) {
		RS::get_singleton()->instance_set_visible(vi->get_instance(), false);
	}
	for (int i = 0; i < p_node->get_child_count(); i++) {
		_hide_ghost_visuals(p_node->get_child(i));
	}
}

Node *LiveRuntimeDebugger::_make_ghost(const String &p_path, Node *p_node) {
	const bool is_2d = Object::cast_to<CanvasItem>(p_node) != nullptr;
	_get_ghost_parent(is_2d)->add_child(p_node);
	_hide_ghost_visuals(p_node);
	ghost_roots[p_path] = p_node->get_instance_id();
	return p_node;
}

Node *LiveRuntimeDebugger::_resolve(const Array &p_added, bool &r_ghost, String &r_ghost_root) {
	r_ghost = false;
	const String path = p_added[ADDED_PATH];
	const int kind = p_added[ADDED_KIND];

	auto matches_kind = [kind](Node *p_node) {
		return p_node && (kind == KIND_3D ? Object::cast_to<VisualInstance3D>(p_node) != nullptr : Object::cast_to<Node2D>(p_node) != nullptr);
	};

	// 1. The node of the edited scene at the same path.
	Node *edited = EditorNode::get_singleton()->get_edited_scene();
	if (edited) {
		Node *n = edited->get_node_or_null(NodePath(path));
		if (matches_kind(n)) {
			return n;
		}
	}

	r_ghost = true;

	// 2. Inside a ghost created earlier (the longest matching runtime path wins).
	String best_root;
	for (const KeyValue<String, ObjectID> &E : ghost_roots) {
		if ((path == E.key || path.begins_with(E.key + "/")) && E.key.length() > best_root.length()) {
			best_root = E.key;
		}
	}
	if (!best_root.is_empty()) {
		Node *ghost = ObjectDB::get_instance<Node>(ghost_roots[best_root]);
		if (ghost) {
			Node *n = path == best_root ? ghost : ghost->get_node_or_null(NodePath(path.substr(best_root.length() + 1)));
			if (matches_kind(n)) {
				r_ghost_root = best_root;
				return n;
			}
		}
	}

	// 3. A ghost of the scene instance the node comes from.
	const String owner_path = p_added[ADDED_OWNER_PATH];
	const String owner_scene = p_added[ADDED_OWNER_SCENE];
	if (!owner_path.is_empty() && !owner_scene.is_empty() && !ghost_roots.has(owner_path) && (path == owner_path || path.begins_with(owner_path + "/"))) {
		Ref<PackedScene> scene;
		if (ghost_scenes.has(owner_scene)) {
			scene = ghost_scenes[owner_scene];
		} else {
			scene = ResourceLoader::load(owner_scene, "PackedScene");
			ghost_scenes[owner_scene] = scene; // Also caches failures.
		}
		if (scene.is_valid()) {
			Node *instance = scene->instantiate(PackedScene::GEN_EDIT_STATE_DISABLED);
			if (instance) {
				_make_ghost(owner_path, instance);
				Node *n = path == owner_path ? instance : instance->get_node_or_null(NodePath(path.substr(owner_path.length() + 1)));
				if (matches_kind(n)) {
					r_ghost_root = owner_path;
					return n;
				}
			}
		}
	}

	// 4. A ghost made from the node's own mesh or texture.
	const String resource = p_added[ADDED_RESOURCE];
	if (!resource.is_empty()) {
		if (kind == KIND_3D) {
			Ref<Mesh> mesh = ResourceLoader::load(resource, "Mesh");
			if (mesh.is_valid()) {
				MeshInstance3D *mi = memnew(MeshInstance3D);
				mi->set_mesh(mesh);
				r_ghost_root = path;
				return _make_ghost(path, mi);
			}
		} else {
			Ref<Texture2D> texture = ResourceLoader::load(resource, "Texture2D");
			if (texture.is_valid()) {
				Sprite2D *sprite = memnew(Sprite2D);
				sprite->set_texture(texture);
				r_ghost_root = path;
				return _make_ghost(path, sprite);
			}
		}
	}

	return nullptr;
}

void LiveRuntimeDebugger::_apply_entry_visibility(Entry &p_entry) {
	Node *n = ObjectDB::get_instance<Node>(p_entry.node);
	if (VisualInstance3D *vi = Object::cast_to<VisualInstance3D>(n)) {
		RS::get_singleton()->instance_set_visible(vi->get_instance(), p_entry.visible);
	} else if (CanvasItem *ci = Object::cast_to<CanvasItem>(n)) {
		RS::get_singleton()->canvas_item_set_visible(ci->get_canvas_item(), p_entry.visible);
	}
}

void LiveRuntimeDebugger::_apply_entry_2d(const Entry &p_entry) {
	Node2D *n2d = ObjectDB::get_instance<Node2D>(p_entry.node);
	if (!n2d) {
		return;
	}
	// The rendering server takes the transform relative to the parent canvas item, so
	// express the runtime global transform relative to the parent's runtime transform.
	Transform2D parent_global;
	if (!n2d->is_set_as_top_level()) {
		Node *p = n2d->get_parent();
		while (p && !Object::cast_to<CanvasItem>(p)) {
			p = p->get_parent();
		}
		if (CanvasItem *parent_ci = Object::cast_to<CanvasItem>(p)) {
			const uint32_t *parent_entry = node_entries.getptr(parent_ci->get_instance_id());
			if (parent_entry && entries.has(*parent_entry) && entries[*parent_entry].kind == KIND_2D) {
				parent_global = entries[*parent_entry].xform_2d;
			} else {
				parent_global = parent_ci->get_global_transform();
			}
		}
	}
	RS::get_singleton()->canvas_item_set_transform(n2d->get_canvas_item(), parent_global.affine_inverse() * p_entry.xform_2d);
}

void LiveRuntimeDebugger::_restore_entry(const Entry &p_entry) {
	if (p_entry.ghost) {
		return; // Ghosts are freed.
	}
	Node *n = ObjectDB::get_instance<Node>(p_entry.node);
	if (VisualInstance3D *vi = Object::cast_to<VisualInstance3D>(n)) {
		if (vi->is_inside_tree()) {
			RS::get_singleton()->instance_set_transform(vi->get_instance(), vi->get_global_transform());
			RS::get_singleton()->instance_set_visible(vi->get_instance(), vi->is_visible_in_tree());
		}
	} else if (Node2D *n2d = Object::cast_to<Node2D>(n)) {
		RS::get_singleton()->canvas_item_set_transform(n2d->get_canvas_item(), n2d->get_transform());
		RS::get_singleton()->canvas_item_set_visible(n2d->get_canvas_item(), n2d->is_visible_in_tree());
	}
}

void LiveRuntimeDebugger::_clear() {
	for (const KeyValue<uint32_t, Entry> &E : entries) {
		_restore_entry(E.value);
	}
	entries.clear();
	node_entries.clear();
	for (const KeyValue<String, ObjectID> &E : ghost_roots) {
		Node *ghost = ObjectDB::get_instance<Node>(E.value);
		if (ghost) {
			ghost->queue_free();
		}
	}
	ghost_roots.clear();
	ghost_refs.clear();
	unresolved_count = 0;
}

void LiveRuntimeDebugger::_release_ghost(const String &p_root) {
	int *refs = ghost_refs.getptr(p_root);
	if (!refs || --(*refs) > 0) {
		return;
	}
	// The game freed the whole instance (a despawned object): free its ghost.
	ghost_refs.erase(p_root);
	Node *ghost = ObjectDB::get_instance<Node>(ghost_roots[p_root]);
	if (ghost) {
		ghost->queue_free();
	}
	ghost_roots.erase(p_root);
}

void LiveRuntimeDebugger::_msg_state(const Array &p_data) {
	ERR_FAIL_COND(p_data.size() < STATE_MAX);

	if (bool(p_data[STATE_RESET])) {
		_clear();
	}

	const Array added = p_data[STATE_ADDED];
	for (int i = 0; i < added.size(); i++) {
		const Array a = added[i];
		ERR_CONTINUE(a.size() < ADDED_MAX);
		Entry entry;
		entry.kind = Kind(int(a[ADDED_KIND]));
		Node *n = _resolve(a, entry.ghost, entry.ghost_root);
		if (n) {
			entry.node = n->get_instance_id();
			node_entries[entry.node] = uint32_t(a[ADDED_ID]);
			if (entry.ghost) {
				ghost_refs[entry.ghost_root]++;
			}
		} else {
			unresolved_count++;
		}
		entries[uint32_t(a[ADDED_ID])] = entry;
	}

	const PackedInt32Array removed = p_data[STATE_REMOVED];
	for (int id : removed) {
		Entry *entry = entries.getptr(id);
		if (!entry) {
			continue;
		}
		if (!entry->node.is_valid()) {
			unresolved_count--;
			entries.erase(id);
			continue;
		}
		// The node is gone in the game: hide it.
		node_entries.erase(entry->node);
		entry->visible = false;
		_apply_entry_visibility(*entry);
		if (entry->ghost) {
			_release_ghost(entry->ghost_root);
			entries.erase(id);
		}
		// An edited node keeps its entry (ids are never reused), so it stays hidden until the view is restored.
	}

	const PackedInt32Array ids_3d = p_data[STATE_IDS_3D];
	const PackedFloat32Array xforms_3d = p_data[STATE_XFORMS_3D];
	ERR_FAIL_COND(xforms_3d.size() != ids_3d.size() * 12);
	const float *x3 = xforms_3d.ptr();
	for (int i = 0; i < ids_3d.size(); i++) {
		Entry *entry = entries.getptr(ids_3d[i]);
		const float *v = x3 + i * 12;
		if (!entry) {
			continue;
		}
		entry->xform_3d = Transform3D(Basis(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]), Vector3(v[9], v[10], v[11]));
		VisualInstance3D *vi = ObjectDB::get_instance<VisualInstance3D>(entry->node);
		if (vi) {
			RS::get_singleton()->instance_set_transform(vi->get_instance(), entry->xform_3d);
		}
	}

	const PackedInt32Array ids_2d = p_data[STATE_IDS_2D];
	const PackedFloat32Array xforms_2d = p_data[STATE_XFORMS_2D];
	ERR_FAIL_COND(xforms_2d.size() != ids_2d.size() * 6);
	const float *x2 = xforms_2d.ptr();
	for (int i = 0; i < ids_2d.size(); i++) {
		Entry *entry = entries.getptr(ids_2d[i]);
		const float *v = x2 + i * 6;
		if (entry) {
			entry->xform_2d = Transform2D(Vector2(v[0], v[1]), Vector2(v[2], v[3]), Vector2(v[4], v[5]));
		}
	}
	if (!ids_2d.is_empty()) {
		// A parent's move changes its children's local transforms, so update all 2D nodes.
		for (const KeyValue<uint32_t, Entry> &E : entries) {
			if (E.value.kind == KIND_2D && E.value.node.is_valid() && node_entries.has(E.value.node)) {
				_apply_entry_2d(E.value);
			}
		}
	}

	const PackedInt32Array shown = p_data[STATE_SHOWN];
	for (int id : shown) {
		if (Entry *entry = entries.getptr(id)) {
			entry->visible = true;
			_apply_entry_visibility(*entry);
		}
	}
	const PackedInt32Array hidden = p_data[STATE_HIDDEN];
	for (int id : hidden) {
		if (Entry *entry = entries.getptr(id)) {
			entry->visible = false;
			_apply_entry_visibility(*entry);
		}
	}

	const String root_path = p_data[STATE_ROOT_PATH];
	Node *edited = EditorNode::get_singleton()->get_edited_scene();
	String new_status;
	if (edited && !root_path.is_empty() && root_path != edited->get_scene_file_path()) {
		new_status = vformat(TTR("Live Runtime: the game runs \"%s\", not the edited scene; only nodes with matching paths and spawned instances are shown."), root_path.get_file());
	} else {
		new_status = vformat(TTR("Live Runtime: %d nodes, %d spawned instances, %d not shown."), node_entries.size(), ghost_roots.size(), unresolved_count);
	}
	_set_status(new_status);
}

void LiveRuntimeDebugger::_set_status(const String &p_status) {
	if (status == p_status) {
		return;
	}
	status = p_status;
	emit_signal(SNAME("status_changed"));
}

LiveRuntimeDebugger::~LiveRuntimeDebugger() {
	_clear();
}

///////////////////////////////////////////////////////////////////////////////

Button *LiveRuntimePlugin::_make_button() {
	Button *button = memnew(Button);
	button->set_theme_type_variation(SceneStringName(FlatButton));
	button->set_toggle_mode(true);
	button->set_text(TTRC("Live"));
	button->set_tooltip_text(TTRC("Live Runtime: while the game runs, this view shows the running scene (moving, spawned and removed objects). The editor camera stays free, and the edited scene is not changed.\nOff: the view shows the edited scene (Standard)."));
	button->set_accessibility_name(TTRC("Live Runtime View"));
	button->connect(SceneStringName(toggled), callable_mp(this, &LiveRuntimePlugin::_button_toggled));
	return button;
}

void LiveRuntimePlugin::_button_toggled(bool p_pressed) {
	button_3d->set_pressed_no_signal(p_pressed);
	button_2d->set_pressed_no_signal(p_pressed);
	debugger->set_mode(p_pressed ? LiveRuntimeDebugger::MODE_LIVE : LiveRuntimeDebugger::MODE_STANDARD);
}

void LiveRuntimePlugin::_status_changed() {
	const String status = debugger->get_status();
	for (Button *button : { button_3d, button_2d }) {
		button->set_tooltip_text(status.is_empty() ? TTR("Live Runtime view (off: Standard).") : status);
	}
}

void LiveRuntimePlugin::_queue_update() {
	if (update_queued) {
		return;
	}
	update_queued = true;
	callable_mp(this, &LiveRuntimePlugin::_update_observing).call_deferred();
}

void LiveRuntimePlugin::_update_observing() {
	update_queued = false;
	if (debugger->get_mode() == LiveRuntimeDebugger::MODE_LIVE) {
		debugger->update_observing();
	}
}

void LiveRuntimePlugin::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			add_debugger_plugin(debugger);

			button_3d = _make_button();
			Node3DEditor::get_singleton()->add_control_to_menu_panel(button_3d);
			button_2d = _make_button();
			CanvasItemEditor::get_singleton()->add_control_to_menu_panel(button_2d);

			// Observe only what can be seen: follow the visibility of both views and the edited scene.
			Node3DEditor::get_singleton()->connect(SceneStringName(visibility_changed), callable_mp(this, &LiveRuntimePlugin::_queue_update));
			CanvasItemEditor::get_singleton()->connect(SceneStringName(visibility_changed), callable_mp(this, &LiveRuntimePlugin::_queue_update));
			EditorNode::get_singleton()->connect("scene_changed", callable_mp(this, &LiveRuntimePlugin::_queue_update));
			debugger->connect("status_changed", callable_mp(this, &LiveRuntimePlugin::_status_changed));
		} break;

		case NOTIFICATION_EXIT_TREE: {
			remove_debugger_plugin(debugger);
		} break;
	}
}

LiveRuntimePlugin::LiveRuntimePlugin() {
	debugger.instantiate();
}
