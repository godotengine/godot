/*************************************************************************/
/*  multiplayer_synchronizer.cpp                                         */
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

#include "multiplayer_synchronizer.h"

#include "core/config/engine.h"
#include "core/multiplayer/multiplayer_api.h"

Object *MultiplayerSynchronizer::_get_prop_target(Object *p_obj, const NodePath &p_path) {
	if (p_path.get_name_count() == 0) {
		return p_obj;
	}
	Node *node = Object::cast_to<Node>(p_obj);
	ERR_FAIL_COND_V_MSG(!node || !node->has_node(p_path), nullptr, vformat("Node '%s' not found.", p_path));
	return node->get_node(p_path);
}

void MultiplayerSynchronizer::_stop() {
	Node *node = is_inside_tree() ? get_node_or_null(root_path) : nullptr;
	if (node) {
		get_multiplayer()->replication_stop(node, this);
	}
}

void MultiplayerSynchronizer::_start() {
	Node *node = is_inside_tree() ? get_node_or_null(root_path) : nullptr;
	if (node) {
		get_multiplayer()->replication_start(node, this);
	}
}

Error MultiplayerSynchronizer::get_state(const List<NodePath> &p_properties, Object *p_obj, Vector<Variant> &r_variant, Vector<const Variant *> &r_variant_ptrs) {
	ERR_FAIL_COND_V(!p_obj, ERR_INVALID_PARAMETER);
	r_variant.resize(p_properties.size());
	r_variant_ptrs.resize(r_variant.size());
	int i = 0;
	for (const NodePath &prop : p_properties) {
		bool valid = false;
		const Object *obj = _get_prop_target(p_obj, prop);
		ERR_FAIL_COND_V(!obj, FAILED);
		r_variant.write[i] = obj->get(prop.get_concatenated_subnames(), &valid);
		r_variant_ptrs.write[i] = &r_variant[i];
		ERR_FAIL_COND_V_MSG(!valid, ERR_INVALID_DATA, vformat("Property '%s' not found.", prop));
		i++;
	}
	return OK;
}

Error MultiplayerSynchronizer::set_state(const List<NodePath> &p_properties, Object *p_obj, const Vector<Variant> &p_state) {
	ERR_FAIL_COND_V(!p_obj, ERR_INVALID_PARAMETER);
	int i = 0;
	for (const NodePath &prop : p_properties) {
		Object *obj = _get_prop_target(p_obj, prop);
		ERR_FAIL_COND_V(!obj, FAILED);
		obj->set(prop.get_concatenated_subnames(), p_state[i]);
		i += 1;
	}
	return OK;
}

void MultiplayerSynchronizer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_root_path", "path"), &MultiplayerSynchronizer::set_root_path);
	ClassDB::bind_method(D_METHOD("get_root_path"), &MultiplayerSynchronizer::get_root_path);
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "root_path"), "set_root_path", "get_root_path");

	ClassDB::bind_method(D_METHOD("set_replication_interval", "milliseconds"), &MultiplayerSynchronizer::set_replication_interval);
	ClassDB::bind_method(D_METHOD("get_replication_interval"), &MultiplayerSynchronizer::get_replication_interval);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "replication_interval", PROPERTY_HINT_RANGE, "0,5,0.001"), "set_replication_interval", "get_replication_interval");

	ClassDB::bind_method(D_METHOD("set_replication_config", "config"), &MultiplayerSynchronizer::set_replication_config);
	ClassDB::bind_method(D_METHOD("get_replication_config"), &MultiplayerSynchronizer::get_replication_config);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "congiruation", PROPERTY_HINT_RESOURCE_TYPE, "SceneReplicationConfig"), "set_replication_config", "get_replication_config");
}

void MultiplayerSynchronizer::_notification(int p_what) {
#ifdef TOOLS_ENABLED
	if (Engine::get_singleton()->is_editor_hint()) {
		return;
	}
#endif
	if (root_path.is_empty()) {
		return;
	}

	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			_start();
		} break;

		case NOTIFICATION_EXIT_TREE: {
			_stop();
		} break;
	}
}

void MultiplayerSynchronizer::set_replication_interval(double p_interval) {
	ERR_FAIL_COND_MSG(p_interval < 0, "Interval must be greater or equal to 0 (where 0 means default)");
	interval_msec = uint64_t(p_interval * 1000);
}

double MultiplayerSynchronizer::get_replication_interval() const {
	return double(interval_msec) / 1000.0;
}

uint64_t MultiplayerSynchronizer::get_replication_interval_msec() const {
	return interval_msec;
}

void MultiplayerSynchronizer::set_replication_config(Ref<SceneReplicationConfig> p_config) {
	replication_config = p_config;
}

Ref<SceneReplicationConfig> MultiplayerSynchronizer::get_replication_config() {
	return replication_config;
}

void MultiplayerSynchronizer::set_root_path(const NodePath &p_path) {
	_stop();
	root_path = p_path;
	_start();
}

NodePath MultiplayerSynchronizer::get_root_path() const {
	return root_path;
}

void MultiplayerSynchronizer::set_multiplayer_authority(int p_peer_id, bool p_recursive) {
	Node *node = is_inside_tree() ? get_node_or_null(root_path) : nullptr;
	if (!node) {
		Node::set_multiplayer_authority(p_peer_id, p_recursive);
		return;
	}
	get_multiplayer()->replication_stop(node, this);
	Node::set_multiplayer_authority(p_peer_id, p_recursive);
	get_multiplayer()->replication_start(node, this);
}
