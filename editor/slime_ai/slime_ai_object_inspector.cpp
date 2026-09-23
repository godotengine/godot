/**************************************************************************/
/*  slime_ai_object_inspector.cpp                                         */
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

#include "slime_ai_object_inspector.h"

#include "editor/slime_ai/slime_ai_scene_inspector.h"
#include "scene/2d/node_2d.h"
#include "scene/2d/sprite_2d.h"
#include "scene/3d/node_3d.h"

namespace SlimeAI {

Dictionary ObjectInspector::inspect(Node *p_root, const String &p_node_ref, bool p_editor_unsaved) {
	Dictionary result;
	Node *node = SceneInspector::resolve(p_root, p_node_ref);
	if (!node) {
		result["error"] = "STALE_REFERENCE";
		result["recovery"] = "Inspect the current scene and select the node again.";
		return result;
	}
	const Dictionary scene = SceneInspector::inspect(p_root, p_editor_unsaved);
	if (scene.has("error")) {
		return scene;
	}
	result["scene_ref"] = scene["scene_ref"];
	result["revision"] = scene["revision"];
	result["disk_revision"] = scene["disk_revision"];
	result["source"] = "live_editor";
	result["node_ref"] = p_node_ref;
	result["path"] = String(p_root->get_path_to(node));
	result["class_name"] = node->get_class();
	result["name"] = String(node->get_name());
	result["owner_ref"] = SceneInspector::object_ref(p_root, node->get_owner());
	result["script_attached"] = node->get_script_instance() != nullptr;
	Dictionary properties;
	if (Node2D *node_2d = Object::cast_to<Node2D>(node)) {
		Dictionary position;
		position["type"] = "Vector2";
		Array coordinates;
		coordinates.push_back(node_2d->get_position().x);
		coordinates.push_back(node_2d->get_position().y);
		position["value"] = coordinates;
		properties["position"] = position;
		Dictionary visible;
		visible["type"] = "bool";
		visible["value"] = node_2d->is_visible();
		properties["visible"] = visible;
	} else if (Node3D *node_3d = Object::cast_to<Node3D>(node)) {
		Dictionary position;
		position["type"] = "Vector3";
		Array coordinates;
		coordinates.push_back(node_3d->get_position().x);
		coordinates.push_back(node_3d->get_position().y);
		coordinates.push_back(node_3d->get_position().z);
		position["value"] = coordinates;
		properties["position"] = position;
		Dictionary visible;
		visible["type"] = "bool";
		visible["value"] = node_3d->is_visible();
		properties["visible"] = visible;
	}
	result["properties"] = properties;
	Dictionary resources;
	if (Sprite2D *sprite = Object::cast_to<Sprite2D>(node)) {
		Ref<Texture2D> texture = sprite->get_texture();
		if (texture.is_valid()) {
			Dictionary summary;
			summary["resource_ref"] = vformat("resource:%d", uint64_t(texture->get_instance_id()));
			summary["class_name"] = texture->get_class();
			summary["path"] = texture->get_path();
			summary["contents"] = "not_inspected";
			resources["texture"] = summary;
		}
	}
	result["resources"] = resources;
	result["unsupported_properties"] = "Only native position, visibility, and Sprite2D texture identity are inspected in P02.";
	return result;
}

} // namespace SlimeAI
