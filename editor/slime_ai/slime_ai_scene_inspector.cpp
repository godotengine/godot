/**************************************************************************/
/*  slime_ai_scene_inspector.cpp                                        */
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

#include "slime_ai_scene_inspector.h"

#include "core/config/project_settings.h"
#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/io/resource_saver.h"
#include "core/os/os.h"
#include "scene/2d/node_2d.h"
#include "scene/3d/node_3d.h"
#include "scene/resources/packed_scene.h"

namespace SlimeAI {

String SceneInspector::scene_ref(Node *p_root) {
	return p_root ? vformat("scene:%d", uint64_t(p_root->get_instance_id())) : String();
}

String SceneInspector::object_ref(Node *p_root, Node *p_node) {
	if (!p_root || !p_node || (p_node != p_root && !p_root->is_ancestor_of(p_node))) {
		return String();
	}
	return vformat("%s:%d:%s", scene_ref(p_root), uint64_t(p_node->get_instance_id()), String(p_root->get_path_to(p_node)));
}

Node *SceneInspector::resolve(Node *p_root, const String &p_ref) {
	if (!p_root || !p_ref.begins_with(scene_ref(p_root) + ":")) {
		return nullptr;
	}
	const String remainder = p_ref.substr(scene_ref(p_root).length() + 1);
	const int separator = remainder.find(":");
	if (separator < 1) {
		return nullptr;
	}
	const ObjectID id = ObjectID(remainder.substr(0, separator).to_int());
	Node *node = Object::cast_to<Node>(ObjectDB::get_instance(id));
	if (!node || (node != p_root && !p_root->is_ancestor_of(node)) || String(p_root->get_path_to(node)) != remainder.substr(separator + 1)) {
		return nullptr;
	}
	return node;
}

struct ScenePage {
	int offset = 0;
	int limit = 0;
	int max_depth = 0;
	int visited = 0;
	bool depth_truncated = false;
	Array nodes;
};

static void _append_nodes(Node *p_root, Node *p_node, ScenePage &r_page, int p_depth) {
	if (r_page.visited > r_page.offset + r_page.limit) {
		return;
	}
	const int index = r_page.visited++;
	if (index >= r_page.offset && r_page.nodes.size() < r_page.limit) {
	Dictionary item;
	item["ref"] = SceneInspector::object_ref(p_root, p_node);
	item["path"] = String(p_root->get_path_to(p_node));
	item["class_name"] = p_node->get_class();
	item["name"] = String(p_node->get_name());
	item["owner_ref"] = SceneInspector::object_ref(p_root, p_node->get_owner());
	item["instance_path"] = p_node == p_root ? String() : p_node->get_scene_file_path();
	if (Node2D *node_2d = Object::cast_to<Node2D>(p_node)) {
		Dictionary position;
		position["type"] = "Vector2";
		Array value;
		value.push_back(node_2d->get_position().x);
		value.push_back(node_2d->get_position().y);
		position["value"] = value;
		item["position"] = position;
		item["visible"] = node_2d->is_visible();
	} else if (Node3D *node_3d = Object::cast_to<Node3D>(p_node)) {
		Dictionary position;
		position["type"] = "Vector3";
		Array value;
		value.push_back(node_3d->get_position().x);
		value.push_back(node_3d->get_position().y);
		value.push_back(node_3d->get_position().z);
		position["value"] = value;
		item["position"] = position;
		item["visible"] = node_3d->is_visible();
	}
	r_page.nodes.push_back(item);
	}
	if (p_depth >= r_page.max_depth) {
		r_page.depth_truncated |= p_node->get_child_count(false) > 0;
		return;
	}
	for (int i = 0; i < p_node->get_child_count(false); i++) {
		_append_nodes(p_root, p_node->get_child(i, false), r_page, p_depth + 1);
	}
}

Dictionary SceneInspector::inspect(Node *p_root, bool p_editor_unsaved, int p_offset, int p_limit, int p_max_depth) {
	Dictionary result;
	if (!p_root) {
		result["error"] = "STALE_REFERENCE";
		return result;
	}
	const String scene_path = p_root->get_scene_file_path();
	result["scene_ref"] = scene_ref(p_root);
	result["scene_path"] = scene_path;
	result["source"] = scene_path.is_empty() ? "unsaved_scene" : "saved_scene_with_live_editor_state";
	result["editor_unsaved"] = p_editor_unsaved || scene_path.is_empty();
	result["runtime_state"] = "not_inspected";
	result["staged_state"] = "not_inspected";
	result["root_class"] = p_root->get_class();
	result["root_ref"] = object_ref(p_root, p_root);
	ScenePage page;
	page.offset = CLAMP(p_offset, 0, 4096);
	page.limit = CLAMP(p_limit, 1, 512);
	page.max_depth = CLAMP(p_max_depth, 0, 64);
	_append_nodes(p_root, p_root, page, 0);
	result["nodes"] = page.nodes;
	result["page_offset"] = page.offset;
	result["page_limit"] = page.limit;
	result["max_depth"] = page.max_depth;
	result["next_offset"] = page.visited > page.offset + page.limit ? page.offset + page.limit : -1;
	result["truncated"] = page.visited > page.offset + page.limit || page.depth_truncated;

	Ref<PackedScene> packed;
	packed.instantiate();
	if (packed->pack(p_root) != OK) {
		result["error"] = "UNSUPPORTED_OPERATION";
		return result;
	}
	// Serialize to a per-project user cache. This captures unsaved storage properties
	// without mutating the edited tree or the project scene file.
	const String cache_dir = ProjectSettings::get_singleton()->globalize_path("user://slime_ai");
	if (DirAccess::make_dir_recursive_absolute(cache_dir) != OK) {
		result["error"] = "CAPABILITY_UNAVAILABLE";
		return result;
	}
	const String scratch = cache_dir.path_join(vformat("inspection_%d_%d.tscn", uint64_t(p_root->get_instance_id()), OS::get_singleton()->get_ticks_usec()));
	if (ResourceSaver::save(packed, scratch) != OK) {
		DirAccess::remove_absolute(scratch);
		result["error"] = "CAPABILITY_UNAVAILABLE";
		return result;
	}
	const String live_hash = FileAccess::get_sha256(scratch);
	DirAccess::remove_absolute(scratch);
	if (live_hash.is_empty()) {
		result["error"] = "CAPABILITY_UNAVAILABLE";
		return result;
	}
	const String disk_hash = scene_path.is_empty() ? "" : FileAccess::get_sha256(scene_path);
	result["disk_revision"] = disk_hash;
	result["revision"] = (scene_ref(p_root) + ":" + live_hash + ":" + disk_hash).sha256_text();
	return result;
}

} // namespace SlimeAI
