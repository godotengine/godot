/**************************************************************************/
/*  saveload_synchronizer.cpp                                             */
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

#include "saveload_synchronizer.h"

#include "core/config/engine.h"
#include "scene/main/saveload_api.h"

Object *SaveloadSynchronizer::_get_prop_target(Object *p_obj, const NodePath &p_path) {
	if (p_path.get_name_count() == 0) {
		return p_obj;
	}
	Node *node = Object::cast_to<Node>(p_obj);
	ERR_FAIL_COND_V_MSG(!node || !node->has_node(p_path), nullptr, vformat("Node '%s' not found.", p_path));
	return node->get_node(p_path);
}

void SaveloadSynchronizer::_stop() {
#ifdef TOOLS_ENABLED
	if (Engine::get_singleton()->is_editor_hint()) {
		return;
	}
#endif
	root_node_cache = ObjectID();
	reset();
	Node *node = is_inside_tree() ? get_node_or_null(root_path) : nullptr;
	if (node) {
		get_saveload()->object_configuration_remove(node, this);
	}
}

void SaveloadSynchronizer::_start() {
#ifdef TOOLS_ENABLED
	if (Engine::get_singleton()->is_editor_hint()) {
		return;
	}
#endif
	root_node_cache = ObjectID();
	reset();
	Node *node = is_inside_tree() ? get_node_or_null(root_path) : nullptr;
	if (node) {
		root_node_cache = node->get_instance_id();
		get_saveload()->object_configuration_add(node, this);
		_update_process();
	}
}

void SaveloadSynchronizer::_update_process() {
#ifdef TOOLS_ENABLED
	if (Engine::get_singleton()->is_editor_hint()) {
		return;
	}
#endif
	Node *node = is_inside_tree() ? get_node_or_null(root_path) : nullptr;
	if (!node) {
		return;
	}
	set_process_internal(false);
	set_physics_process_internal(false);
//	if (!visibility_filters.size()) {
//		return;
//	}
//	switch (visibility_update_mode) {
//		case VISIBILITY_PROCESS_IDLE:
//			set_process_internal(true);
//			break;
//		case VISIBILITY_PROCESS_PHYSICS:
//			set_physics_process_internal(true);
//			break;
//		case VISIBILITY_PROCESS_NONE:
//			break;
//	}
}

Node *SaveloadSynchronizer::get_root_node() {
	return root_node_cache.is_valid() ? Object::cast_to<Node>(ObjectDB::get_instance(root_node_cache)) : nullptr;
}

void SaveloadSynchronizer::reset() {
//	net_id = 0;
	last_sync_usec = 0;
	last_inbound_sync = 0;
}

uint32_t SaveloadSynchronizer::get_net_id() const {
	return net_id;
}

void SaveloadSynchronizer::set_net_id(uint32_t p_net_id) {
	net_id = p_net_id;
}

bool SaveloadSynchronizer::update_outbound_sync_time(uint64_t p_usec) {
	if (last_sync_usec == p_usec) {
		// last_sync_usec has been updated in this frame.
		return true;
	}
	if (p_usec < last_sync_usec + sync_interval_usec) {
		// Too soon, should skip this synchronization frame.
		return false;
	}
	last_sync_usec = p_usec;
	return true;
}

bool SaveloadSynchronizer::update_inbound_sync_time(uint16_t p_network_time) {
	if (p_network_time <= last_inbound_sync && last_inbound_sync - p_network_time < 32767) {
		return false;
	}
	last_inbound_sync = p_network_time;
	return true;
}

PackedStringArray SaveloadSynchronizer::get_configuration_warnings() const {
	PackedStringArray warnings = Node::get_configuration_warnings();

	if (root_path.is_empty() || !has_node(root_path)) {
		warnings.push_back(RTR("A valid NodePath must be set in the \"Root Path\" property in order for SaveloadSynchronizer to be able to synchronize properties."));
	}

	return warnings;
}

Error SaveloadSynchronizer::get_state(const List<NodePath> &p_properties, Object *p_obj, Vector<Variant> &r_variant, Vector<const Variant *> &r_variant_ptrs) {
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

Dictionary SaveloadSynchronizer::get_state_wrapper() {
	Vector<Variant> vars;
	Vector<const Variant *> varp;
	const List<NodePath> props = get_saveload_config()->get_sync_properties();
	get_state(props, get_root_node(), vars, varp);
	Dictionary dict;
	for (int i = 0; i < vars.size(); ++i) {
		dict[props[i]] = vars[i];
	}
	return dict;
}

Error SaveloadSynchronizer::set_state(const List<NodePath> &p_properties, Object *p_obj, const Vector<Variant> &p_state) {
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

//bool SaveloadSynchronizer::is_visibility_public() const {
//	return peer_visibility.has(0);
//}
//
//void SaveloadSynchronizer::set_visibility_public(bool p_visible) {
//	set_visibility_for(0, p_visible);
//}
//
//bool SaveloadSynchronizer::is_visible_to(int p_peer) {
//	if (visibility_filters.size()) {
//		Variant arg = p_peer;
//		const Variant *argv[1] = { &arg };
//		for (Callable filter : visibility_filters) {
//			Variant ret;
//			Callable::CallError err;
//			filter.callp(argv, 1, ret, err);
//			ERR_FAIL_COND_V(err.error != Callable::CallError::CALL_OK || ret.get_type() != Variant::BOOL, false);
//			if (!ret.operator bool()) {
//				return false;
//			}
//		}
//	}
//	return peer_visibility.has(0) || peer_visibility.has(p_peer);
//}
//
//void SaveloadSynchronizer::add_visibility_filter(Callable p_callback) {
//	visibility_filters.insert(p_callback);
//	_update_process();
//}
//
//void SaveloadSynchronizer::remove_visibility_filter(Callable p_callback) {
//	visibility_filters.erase(p_callback);
//	_update_process();
//}
//
//void SaveloadSynchronizer::set_visibility_for(int p_peer, bool p_visible) {
//	if (peer_visibility.has(p_peer) == p_visible) {
//		return;
//	}
//	if (p_visible) {
//		peer_visibility.insert(p_peer);
//	} else {
//		peer_visibility.erase(p_peer);
//	}
//	update_visibility(p_peer);
//}
//
//bool SaveloadSynchronizer::get_visibility_for(int p_peer) const {
//	return peer_visibility.has(p_peer);
//}
//
//void SaveloadSynchronizer::set_visibility_update_mode(VisibilityUpdateMode p_mode) {
//	visibility_update_mode = p_mode;
//	_update_process();
//}
//
//SaveloadSynchronizer::VisibilityUpdateMode SaveloadSynchronizer::get_visibility_update_mode() const {
//	return visibility_update_mode;
//}
//
void SaveloadSynchronizer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_root_path", "path"), &SaveloadSynchronizer::set_root_path);
	ClassDB::bind_method(D_METHOD("get_root_path"), &SaveloadSynchronizer::get_root_path);

	ClassDB::bind_method(D_METHOD("set_saveload_interval", "milliseconds"), &SaveloadSynchronizer::set_saveload_interval);
	ClassDB::bind_method(D_METHOD("get_saveload_interval"), &SaveloadSynchronizer::get_saveload_interval);

	ClassDB::bind_method(D_METHOD("set_delta_interval", "milliseconds"), &SaveloadSynchronizer::set_delta_interval);
	ClassDB::bind_method(D_METHOD("get_delta_interval"), &SaveloadSynchronizer::get_delta_interval);

	ClassDB::bind_method(D_METHOD("set_saveload_config", "config"), &SaveloadSynchronizer::set_saveload_config);
	ClassDB::bind_method(D_METHOD("get_saveload_config"), &SaveloadSynchronizer::get_saveload_config);

	ClassDB::bind_method(D_METHOD("get_state_wrapper"), &SaveloadSynchronizer::get_state_wrapper);

//	ClassDB::bind_method(D_METHOD("set_visibility_update_mode", "mode"), &SaveloadSynchronizer::set_visibility_update_mode);
//	ClassDB::bind_method(D_METHOD("get_visibility_update_mode"), &SaveloadSynchronizer::get_visibility_update_mode);
//	ClassDB::bind_method(D_METHOD("update_visibility", "for_peer"), &SaveloadSynchronizer::update_visibility, DEFVAL(0));
//
//	ClassDB::bind_method(D_METHOD("set_visibility_public", "visible"), &SaveloadSynchronizer::set_visibility_public);
//	ClassDB::bind_method(D_METHOD("is_visibility_public"), &SaveloadSynchronizer::is_visibility_public);
//
//	ClassDB::bind_method(D_METHOD("add_visibility_filter", "filter"), &SaveloadSynchronizer::add_visibility_filter);
//	ClassDB::bind_method(D_METHOD("remove_visibility_filter", "filter"), &SaveloadSynchronizer::remove_visibility_filter);
//	ClassDB::bind_method(D_METHOD("set_visibility_for", "peer", "visible"), &SaveloadSynchronizer::set_visibility_for);
//	ClassDB::bind_method(D_METHOD("get_visibility_for", "peer"), &SaveloadSynchronizer::get_visibility_for);

	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "root_path"), "set_root_path", "get_root_path");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "saveload_interval", PROPERTY_HINT_RANGE, "0,5,0.001,suffix:s"), "set_saveload_interval", "get_saveload_interval");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "delta_interval", PROPERTY_HINT_RANGE, "0,5,0.001,suffix:s"), "set_delta_interval", "get_delta_interval");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "saveload_config", PROPERTY_HINT_RESOURCE_TYPE, "SceneSaveloadConfig", PROPERTY_USAGE_NO_EDITOR), "set_saveload_config", "get_saveload_config");
//	ADD_PROPERTY(PropertyInfo(Variant::INT, "visibility_update_mode", PROPERTY_HINT_ENUM, "Idle,Physics,None"), "set_visibility_update_mode", "get_visibility_update_mode");
//	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "public_visibility"), "set_visibility_public", "is_visibility_public");

//	BIND_ENUM_CONSTANT(VISIBILITY_PROCESS_IDLE);
//	BIND_ENUM_CONSTANT(VISIBILITY_PROCESS_PHYSICS);
//	BIND_ENUM_CONSTANT(VISIBILITY_PROCESS_NONE);

	ADD_SIGNAL(MethodInfo("synchronized"));
	ADD_SIGNAL(MethodInfo("delta_synchronized"));
//	ADD_SIGNAL(MethodInfo("visibility_changed", PropertyInfo(Variant::INT, "for_peer")));
}

void SaveloadSynchronizer::_notification(int p_what) {
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

		case NOTIFICATION_INTERNAL_PROCESS:
		case NOTIFICATION_INTERNAL_PHYSICS_PROCESS: {
//			update_visibility(0);
		} break;
	}
}

void SaveloadSynchronizer::set_saveload_interval(double p_interval) {
	ERR_FAIL_COND_MSG(p_interval < 0, "Interval must be greater or equal to 0 (where 0 means default)");
	sync_interval_usec = uint64_t(p_interval * 1000 * 1000);
}

double SaveloadSynchronizer::get_saveload_interval() const {
	return double(sync_interval_usec) / 1000.0 / 1000.0;
}

void SaveloadSynchronizer::set_delta_interval(double p_interval) {
	ERR_FAIL_COND_MSG(p_interval < 0, "Interval must be greater or equal to 0 (where 0 means default)");
	delta_interval_usec = uint64_t(p_interval * 1000 * 1000);
}

double SaveloadSynchronizer::get_delta_interval() const {
	return double(delta_interval_usec) / 1000.0 / 1000.0;
}

void SaveloadSynchronizer::set_saveload_config(Ref<SceneSaveloadConfig> p_config) {
	saveload_config = p_config;
}

Ref<SceneSaveloadConfig> SaveloadSynchronizer::get_saveload_config() {
	return saveload_config;
}

//void SaveloadSynchronizer::update_visibility(int p_for_peer) {
//#ifdef TOOLS_ENABLED
//	if (Engine::get_singleton()->is_editor_hint()) {
//		return;
//	}
//#endif
//	Node *node = is_inside_tree() ? get_node_or_null(root_path) : nullptr;
//	if (node && get_saveload()->has_multiplayer_peer() && is_multiplayer_authority()) {
//		emit_signal(SNAME("visibility_changed"), p_for_peer);
//	}
//}
//
void SaveloadSynchronizer::set_root_path(const NodePath &p_path) {
	_stop();
	root_path = p_path;
	_start();
}

NodePath SaveloadSynchronizer::get_root_path() const {
	return root_path;
}

Error SaveloadSynchronizer::_watch_changes(uint64_t p_usec) {
	ERR_FAIL_COND_V(saveload_config.is_null(), FAILED);
	const List<NodePath> props = saveload_config->get_watch_properties();
	if (props.size() != watchers.size()) {
		watchers.resize(props.size());
	}
	if (props.size() == 0) {
		return OK;
	}
	Node *node = get_root_node();
	ERR_FAIL_COND_V(!node, FAILED);
	int idx = -1;
	Watcher *ptr = watchers.ptrw();
	for (const NodePath &prop : props) {
		idx++;
		bool valid = false;
		const Object *obj = _get_prop_target(node, prop);
		ERR_CONTINUE_MSG(!obj, vformat("Node not found for property '%s'.", prop));
		Variant v = obj->get(prop.get_concatenated_subnames(), &valid);
		ERR_CONTINUE_MSG(!valid, vformat("Property '%s' not found.", prop));
		Watcher &w = ptr[idx];
		if (w.prop != prop) {
			w.prop = prop;
			w.value = v.duplicate(true);
			w.last_change_usec = p_usec;
		} else if (!w.value.hash_compare(v)) {
			w.value = v.duplicate(true);
			w.last_change_usec = p_usec;
		}
	}
	return OK;
}

List<Variant> SaveloadSynchronizer::get_delta_state(uint64_t p_cur_usec, uint64_t p_last_usec, uint64_t &r_indexes) {
	r_indexes = 0;
	List<Variant> out;

	if (last_watch_usec == p_cur_usec) {
		// We already watched for changes in this frame.

	} else if (p_cur_usec < p_last_usec + delta_interval_usec) {
		// Too soon skip delta synchronization.
		return out;

	} else {
		// Watch for changes.
		Error err = _watch_changes(p_cur_usec);
		ERR_FAIL_COND_V(err != OK, out);
		last_watch_usec = p_cur_usec;
	}

	const Watcher *ptr = watchers.size() ? watchers.ptr() : nullptr;
	for (int i = 0; i < watchers.size(); i++) {
		const Watcher &w = ptr[i];
		if (w.last_change_usec <= p_last_usec) {
			continue;
		}
		out.push_back(w.value);
		r_indexes |= 1ULL << i;
	}
	return out;
}

List<NodePath> SaveloadSynchronizer::get_delta_properties(uint64_t p_indexes) {
	List<NodePath> out;
	ERR_FAIL_COND_V(saveload_config.is_null(), out);
	const List<NodePath> watch_props = saveload_config->get_watch_properties();
	int idx = 0;
	for (const NodePath &prop : watch_props) {
		if ((p_indexes & (1ULL << idx)) == 0) {
			continue;
		}
		out.push_back(prop);
		idx++;
	}
	return out;
}

SaveloadSynchronizer::SaveloadSynchronizer() {
	// Publicly visible by default.
//	peer_visibility.insert(0);
}
