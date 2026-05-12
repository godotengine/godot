/**************************************************************************/
/*  scene_inspector.cpp                                                   */
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

#include "scene_inspector.h"

#include "core/io/json.h"
#include "core/io/resource_loader.h"
#include "core/object/object.h"
#include "scene/main/node.h"
#include "scene/resources/packed_scene.h"

Dictionary SceneInspector::inspect(const String &p_scene_path) {
	Dictionary result;
	result["scene"] = p_scene_path;

	Error err = OK;
	Ref<PackedScene> ps = ResourceLoader::load(p_scene_path, "PackedScene", ResourceFormatLoader::CACHE_MODE_REUSE, &err);
	if (ps.is_null() || err != OK) {
		result["error"] = vformat("Failed to load PackedScene from '%s' (error %d).", p_scene_path, (int)err);
		return result;
	}

	Node *root = ps->instantiate(PackedScene::GEN_EDIT_STATE_DISABLED);
	if (!root) {
		result["error"] = vformat("Failed to instantiate scene '%s'.", p_scene_path);
		return result;
	}

	result["root"] = dump_node(root);
	memdelete(root);
	return result;
}

String SceneInspector::inspect_to_json(const String &p_scene_path) {
	Dictionary d = inspect(p_scene_path);
	return JSON::stringify(d, "", false, true);
}

Dictionary SceneInspector::dump_node(Node *p_node) {
	Dictionary d;
	d["name"] = String(p_node->get_name());
	d["class"] = p_node->get_class();

	Dictionary props;
	List<PropertyInfo> plist;
	p_node->get_property_list(&plist);
	for (const PropertyInfo &pi : plist) {
		// Skip non-data entries (groups, categories, subgroups, and internal-only fields).
		const uint32_t skip_mask = PROPERTY_USAGE_GROUP | PROPERTY_USAGE_CATEGORY | PROPERTY_USAGE_SUBGROUP | PROPERTY_USAGE_INTERNAL;
		if (pi.usage & skip_mask) {
			continue;
		}
		// Only editor-visible properties — keeps output focused on what a human/tool would care about.
		if (!(pi.usage & PROPERTY_USAGE_EDITOR)) {
			continue;
		}
		Variant value = p_node->get(pi.name);
		props[pi.name] = value;
	}
	if (!props.is_empty()) {
		d["properties"] = props;
	}

	int child_count = p_node->get_child_count();
	if (child_count > 0) {
		Array children;
		for (int i = 0; i < child_count; i++) {
			children.push_back(dump_node(p_node->get_child(i)));
		}
		d["children"] = children;
	}
	return d;
}
