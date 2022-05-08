/*************************************************************************/
/*  multiplayer_spawner.cpp                                              */
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

#include "multiplayer_spawner.h"

#include "core/io/marshalls.h"
#include "core/multiplayer/multiplayer_api.h"
#include "scene/main/window.h"
#include "scene/scene_string_names.h"

void MultiplayerSpawner::_bind_methods() {
	ClassDB::bind_method(D_METHOD("spawn", "data"), &MultiplayerSpawner::spawn, DEFVAL(Variant()));

	ClassDB::bind_method(D_METHOD("get_spawnable_scenes"), &MultiplayerSpawner::get_spawnable_scenes);
	ClassDB::bind_method(D_METHOD("set_spawnable_scenes", "scenes"), &MultiplayerSpawner::set_spawnable_scenes);
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "replication", PROPERTY_HINT_ARRAY_TYPE, vformat("%s/%s:%s", Variant::OBJECT, PROPERTY_HINT_RESOURCE_TYPE, "PackedScene"), (PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_SCRIPT_VARIABLE)), "set_spawnable_scenes", "get_spawnable_scenes");

	ClassDB::bind_method(D_METHOD("get_spawn_path"), &MultiplayerSpawner::get_spawn_path);
	ClassDB::bind_method(D_METHOD("set_spawn_path", "path"), &MultiplayerSpawner::set_spawn_path);
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "spawn_path", PROPERTY_HINT_NONE, ""), "set_spawn_path", "get_spawn_path");

	ClassDB::bind_method(D_METHOD("get_spawn_limit"), &MultiplayerSpawner::get_spawn_limit);
	ClassDB::bind_method(D_METHOD("set_spawn_limit", "limit"), &MultiplayerSpawner::set_spawn_limit);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "spawn_limit", PROPERTY_HINT_RANGE, "0,1024,1,or_greater"), "set_spawn_limit", "get_spawn_limit");

	ClassDB::bind_method(D_METHOD("set_auto_spawning", "enabled"), &MultiplayerSpawner::set_auto_spawning);
	ClassDB::bind_method(D_METHOD("is_auto_spawning"), &MultiplayerSpawner::is_auto_spawning);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "auto_spawn"), "set_auto_spawning", "is_auto_spawning");

	GDVIRTUAL_BIND(_spawn_custom, "data");

	ADD_SIGNAL(MethodInfo("despawned", PropertyInfo(Variant::INT, "scene_id"), PropertyInfo(Variant::OBJECT, "node", PROPERTY_HINT_RESOURCE_TYPE, "Node")));
	ADD_SIGNAL(MethodInfo("spawned", PropertyInfo(Variant::INT, "scene_id"), PropertyInfo(Variant::OBJECT, "node", PROPERTY_HINT_RESOURCE_TYPE, "Node")));
}

void MultiplayerSpawner::_update_spawn_node() {
#ifdef TOOLS_ENABLED
	if (Engine::get_singleton()->is_editor_hint()) {
		return;
	}
#endif
	if (spawn_node.is_valid()) {
		Node *node = Object::cast_to<Node>(ObjectDB::get_instance(spawn_node));
		if (node && node->is_connected("child_entered_tree", callable_mp(this, &MultiplayerSpawner::_node_added))) {
			node->disconnect("child_entered_tree", callable_mp(this, &MultiplayerSpawner::_node_added));
		}
	}
	Node *node = spawn_path.is_empty() && is_inside_tree() ? nullptr : get_node_or_null(spawn_path);
	if (node) {
		spawn_node = node->get_instance_id();
		if (auto_spawn) {
			node->connect("child_entered_tree", callable_mp(this, &MultiplayerSpawner::_node_added));
		}
	} else {
		spawn_node = ObjectID();
	}
}

void MultiplayerSpawner::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_POST_ENTER_TREE: {
			_update_spawn_node();
		} break;

		case NOTIFICATION_EXIT_TREE: {
			_update_spawn_node();

			for (const KeyValue<ObjectID, SpawnInfo> &E : tracked_nodes) {
				Node *node = Object::cast_to<Node>(ObjectDB::get_instance(E.key));
				ERR_CONTINUE(!node);
				node->disconnect(SceneStringNames::get_singleton()->tree_exiting, callable_mp(this, &MultiplayerSpawner::_node_exit));
				// This is unlikely, but might still crash the engine.
				if (node->is_connected(SceneStringNames::get_singleton()->ready, callable_mp(this, &MultiplayerSpawner::_node_ready))) {
					node->disconnect(SceneStringNames::get_singleton()->ready, callable_mp(this, &MultiplayerSpawner::_node_ready));
				}
				get_multiplayer()->despawn(node, this);
			}
			tracked_nodes.clear();
		} break;
	}
}

void MultiplayerSpawner::_node_added(Node *p_node) {
	if (!get_multiplayer()->has_multiplayer_peer() || !is_multiplayer_authority()) {
		return;
	}
	if (tracked_nodes.has(p_node->get_instance_id())) {
		return;
	}
	const Node *parent = get_spawn_node();
	if (!parent || p_node->get_parent() != parent) {
		return;
	}
	int id = get_scene_id(p_node->get_scene_file_path());
	if (id == INVALID_ID) {
		return;
	}
	const String name = p_node->get_name();
	ERR_FAIL_COND_MSG(name.validate_node_name() != name, vformat("Unable to auto-spawn node with reserved name: %s. Make sure to add your replicated scenes via 'add_child(node, true)' to produce valid names.", name));
	_track(p_node, Variant(), id);
}

void MultiplayerSpawner::set_auto_spawning(bool p_enabled) {
	auto_spawn = p_enabled;
	_update_spawn_node();
}

bool MultiplayerSpawner::is_auto_spawning() const {
	return auto_spawn;
}

TypedArray<PackedScene> MultiplayerSpawner::get_spawnable_scenes() {
	return spawnable_scenes;
}

void MultiplayerSpawner::set_spawnable_scenes(TypedArray<PackedScene> p_scenes) {
	spawnable_scenes = p_scenes;
}

NodePath MultiplayerSpawner::get_spawn_path() const {
	return spawn_path;
}

void MultiplayerSpawner::set_spawn_path(const NodePath &p_path) {
	spawn_path = p_path;
	_update_spawn_node();
}

void MultiplayerSpawner::_track(Node *p_node, const Variant &p_argument, int p_scene_id) {
	ObjectID oid = p_node->get_instance_id();
	if (!tracked_nodes.has(oid)) {
		tracked_nodes[oid] = SpawnInfo(p_argument.duplicate(true), p_scene_id);
		p_node->connect(SceneStringNames::get_singleton()->tree_exiting, callable_mp(this, &MultiplayerSpawner::_node_exit), varray(p_node->get_instance_id()), CONNECT_ONESHOT);
		p_node->connect(SceneStringNames::get_singleton()->ready, callable_mp(this, &MultiplayerSpawner::_node_ready), varray(p_node->get_instance_id()), CONNECT_ONESHOT);
	}
}

void MultiplayerSpawner::_node_ready(ObjectID p_id) {
	get_multiplayer()->spawn(ObjectDB::get_instance(p_id), this);
}

void MultiplayerSpawner::_node_exit(ObjectID p_id) {
	Node *node = Object::cast_to<Node>(ObjectDB::get_instance(p_id));
	ERR_FAIL_COND(!node);
	if (tracked_nodes.has(p_id)) {
		tracked_nodes.erase(p_id);
		get_multiplayer()->despawn(node, this);
	}
}

int MultiplayerSpawner::get_scene_id(const String &p_scene) const {
	for (int i = 0; i < spawnable_scenes.size(); i++) {
		Ref<PackedScene> ps = spawnable_scenes[i];
		ERR_CONTINUE(ps.is_null());
		if (ps->get_path() == p_scene) {
			return i;
		}
	}
	return INVALID_ID;
}

int MultiplayerSpawner::get_spawn_id(const ObjectID &p_id) const {
	const SpawnInfo *info = tracked_nodes.getptr(p_id);
	return info ? info->id : INVALID_ID;
}

const Variant MultiplayerSpawner::get_spawn_argument(const ObjectID &p_id) const {
	const SpawnInfo *info = tracked_nodes.getptr(p_id);
	return info ? info->args : Variant();
}

Node *MultiplayerSpawner::instantiate_scene(int p_id) {
	ERR_FAIL_COND_V_MSG(spawn_limit && spawn_limit <= tracked_nodes.size(), nullptr, "Spawn limit reached!");
	ERR_FAIL_INDEX_V(p_id, spawnable_scenes.size(), nullptr);
	Ref<PackedScene> scene = spawnable_scenes[p_id];
	ERR_FAIL_COND_V(scene.is_null(), nullptr);
	return scene->instantiate();
}

Node *MultiplayerSpawner::instantiate_custom(const Variant &p_data) {
	ERR_FAIL_COND_V_MSG(spawn_limit && spawn_limit <= tracked_nodes.size(), nullptr, "Spawn limit reached!");
	Object *obj = nullptr;
	Node *node = nullptr;
	if (GDVIRTUAL_CALL(_spawn_custom, p_data, obj)) {
		node = Object::cast_to<Node>(obj);
	}
	return node;
}

Node *MultiplayerSpawner::spawn(const Variant &p_data) {
	ERR_FAIL_COND_V(!is_inside_tree() || !get_multiplayer()->has_multiplayer_peer() || !is_multiplayer_authority(), nullptr);
	ERR_FAIL_COND_V_MSG(spawn_limit && spawn_limit <= tracked_nodes.size(), nullptr, "Spawn limit reached!");
	ERR_FAIL_COND_V_MSG(!GDVIRTUAL_IS_OVERRIDDEN(_spawn_custom), nullptr, "Custom spawn requires the '_spawn_custom' virtual method to be implemented via script.");

	Node *parent = get_spawn_node();
	ERR_FAIL_COND_V_MSG(!parent, nullptr, "Cannot find spawn node.");

	Node *node = instantiate_custom(p_data);
	ERR_FAIL_COND_V_MSG(!node, nullptr, "The '_spawn_custom' implementation must return a valid Node.");

	_track(node, p_data);
	parent->add_child(node, true);
	return node;
}
