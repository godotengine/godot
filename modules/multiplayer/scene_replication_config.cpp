/**************************************************************************/
/*  scene_replication_config.cpp                                          */
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

#include "scene_replication_config.h"

#include "scene/main/multiplayer_api.h"
#include "scene/main/node.h"

bool SceneReplicationConfig::_set(const StringName &p_name, const Variant &p_value) {
	String prop_name = p_name;

	if (prop_name.begins_with("properties/")) {
		int idx = prop_name.get_slicec('/', 1).to_int();
		String what = prop_name.get_slicec('/', 2);

		if (properties.size() == idx && what == "path") {
			ERR_FAIL_COND_V(p_value.get_type() != Variant::NODE_PATH, false);
			NodePath path = p_value;
			ERR_FAIL_COND_V(path.is_empty() || path.get_subname_count() == 0, false);
			add_property(path);
			return true;
		}
		ERR_FAIL_COND_V(p_value.get_type() != Variant::BOOL, false);
		ERR_FAIL_INDEX_V(idx, properties.size(), false);
		ReplicationProperty &prop = properties[idx];
		if (what == "sync") {
			if ((bool)p_value == prop.sync) {
				return true;
			}
			prop.sync = p_value;
			if (prop.sync) {
				sync_props.push_back(prop.name);
			} else {
				sync_props.erase(prop.name);
			}
			return true;
		} else if (what == "spawn") {
			if ((bool)p_value == prop.spawn) {
				return true;
			}
			prop.spawn = p_value;
			if (prop.spawn) {
				spawn_props.push_back(prop.name);
			} else {
				spawn_props.erase(prop.name);
			}
			return true;
		} else if (what == "watch") {
			prop.watch = p_value;
			if (prop.watch) {
				watch_props.push_back(prop.name);
			} else {
				watch_props.erase(prop.name);
			}
			return true;
		}
	}
	return false;
}

bool SceneReplicationConfig::_get(const StringName &p_name, Variant &r_ret) const {
	String prop_name = p_name;

	if (prop_name.begins_with("properties/")) {
		int idx = prop_name.get_slicec('/', 1).to_int();
		String what = prop_name.get_slicec('/', 2);
		ERR_FAIL_INDEX_V(idx, properties.size(), false);
		const ReplicationProperty &prop = properties[idx];
		if (what == "path") {
			r_ret = prop.name;
			return true;
		} else if (what == "sync") {
			r_ret = prop.sync;
			return true;
		} else if (what == "spawn") {
			r_ret = prop.spawn;
			return true;
		} else if (what == "watch") {
			r_ret = prop.watch;
			return true;
		}
	}
	return false;
}

void SceneReplicationConfig::_get_property_list(List<PropertyInfo> *p_list) const {
	for (int i = 0; i < properties.size(); i++) {
		p_list->push_back(PropertyInfo(Variant::STRING, "properties/" + itos(i) + "/path", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL));
		p_list->push_back(PropertyInfo(Variant::STRING, "properties/" + itos(i) + "/spawn", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL));
		p_list->push_back(PropertyInfo(Variant::STRING, "properties/" + itos(i) + "/sync", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL));
		p_list->push_back(PropertyInfo(Variant::STRING, "properties/" + itos(i) + "/watch", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL));
	}
}

TypedArray<NodePath> SceneReplicationConfig::get_properties() const {
	TypedArray<NodePath> paths;
	for (const ReplicationProperty &prop : properties) {
		paths.push_back(prop.name);
	}
	return paths;
}

void SceneReplicationConfig::add_property(const NodePath &p_path, int p_index) {
	ERR_FAIL_COND(properties.find(p_path));

	if (p_index < 0 || p_index == properties.size()) {
		properties.push_back(ReplicationProperty(p_path));
		sync_props.push_back(p_path);
		spawn_props.push_back(p_path);
		return;
	}

	ERR_FAIL_INDEX(p_index, properties.size());

	List<ReplicationProperty>::Element *I = properties.front();
	int c = 0;
	while (c < p_index) {
		I = I->next();
		c++;
	}
	properties.insert_before(I, ReplicationProperty(p_path));
	sync_props.clear();
	spawn_props.clear();
	for (const ReplicationProperty &prop : properties) {
		if (prop.sync) {
			sync_props.push_back(prop.name);
		}
		if (prop.spawn) {
			spawn_props.push_back(prop.name);
		}
	}
}

void SceneReplicationConfig::remove_property(const NodePath &p_path) {
	properties.erase(p_path);
	sync_props.erase(p_path);
	spawn_props.erase(p_path);
}

bool SceneReplicationConfig::has_property(const NodePath &p_path) const {
	for (int i = 0; i < properties.size(); i++) {
		if (properties[i].name == p_path) {
			return true;
		}
	}
	return false;
}

int SceneReplicationConfig::property_get_index(const NodePath &p_path) const {
	for (int i = 0; i < properties.size(); i++) {
		if (properties[i].name == p_path) {
			return i;
		}
	}
	ERR_FAIL_V(-1);
}

bool SceneReplicationConfig::property_get_spawn(const NodePath &p_path) {
	List<ReplicationProperty>::Element *E = properties.find(p_path);
	ERR_FAIL_COND_V(!E, false);
	return E->get().spawn;
}

void SceneReplicationConfig::property_set_spawn(const NodePath &p_path, bool p_enabled) {
	List<ReplicationProperty>::Element *E = properties.find(p_path);
	ERR_FAIL_COND(!E);
	if (E->get().spawn == p_enabled) {
		return;
	}
	E->get().spawn = p_enabled;
	spawn_props.clear();
	for (const ReplicationProperty &prop : properties) {
		if (prop.spawn) {
			spawn_props.push_back(prop.name);
		}
	}
}

bool SceneReplicationConfig::property_get_sync(const NodePath &p_path) {
	List<ReplicationProperty>::Element *E = properties.find(p_path);
	ERR_FAIL_COND_V(!E, false);
	return E->get().sync;
}

void SceneReplicationConfig::property_set_sync(const NodePath &p_path, bool p_enabled) {
	List<ReplicationProperty>::Element *E = properties.find(p_path);
	ERR_FAIL_COND(!E);
	if (E->get().sync == p_enabled) {
		return;
	}
	E->get().sync = p_enabled;
	sync_props.clear();
	for (const ReplicationProperty &prop : properties) {
		if (prop.sync) {
			sync_props.push_back(prop.name);
		}
	}
}

bool SceneReplicationConfig::property_get_watch(const NodePath &p_path) {
	List<ReplicationProperty>::Element *E = properties.find(p_path);
	ERR_FAIL_COND_V(!E, false);
	return E->get().watch;
}

void SceneReplicationConfig::property_set_watch(const NodePath &p_path, bool p_enabled) {
	List<ReplicationProperty>::Element *E = properties.find(p_path);
	ERR_FAIL_COND(!E);
	if (E->get().watch == p_enabled) {
		return;
	}
	E->get().watch = p_enabled;
	watch_props.clear();
	for (const ReplicationProperty &prop : properties) {
		if (prop.watch) {
			watch_props.push_back(p_path);
		}
	}
}

void SceneReplicationConfig::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_properties"), &SceneReplicationConfig::get_properties);
	ClassDB::bind_method(D_METHOD("add_property", "path", "index"), &SceneReplicationConfig::add_property, DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("has_property", "path"), &SceneReplicationConfig::has_property);
	ClassDB::bind_method(D_METHOD("remove_property", "path"), &SceneReplicationConfig::remove_property);
	ClassDB::bind_method(D_METHOD("property_get_index", "path"), &SceneReplicationConfig::property_get_index);
	ClassDB::bind_method(D_METHOD("property_get_spawn", "path"), &SceneReplicationConfig::property_get_spawn);
	ClassDB::bind_method(D_METHOD("property_set_spawn", "path", "enabled"), &SceneReplicationConfig::property_set_spawn);
	ClassDB::bind_method(D_METHOD("property_get_sync", "path"), &SceneReplicationConfig::property_get_sync);
	ClassDB::bind_method(D_METHOD("property_set_sync", "path", "enabled"), &SceneReplicationConfig::property_set_sync);
	ClassDB::bind_method(D_METHOD("property_get_watch", "path"), &SceneReplicationConfig::property_get_watch);
	ClassDB::bind_method(D_METHOD("property_set_watch", "path", "enabled"), &SceneReplicationConfig::property_set_watch);
}
