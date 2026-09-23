/**************************************************************************/
/*  test_slime_ai_extra_inspection.cpp                                    */
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

#include "tests/test_macros.h"

TEST_FORCE_LINK(test_slime_ai_extra_inspection)

#include "editor/slime_ai/slime_ai_api_describer.h"
#include "editor/slime_ai/slime_ai_object_inspector.h"
#include "editor/slime_ai/slime_ai_project_inspector.h"
#include "editor/slime_ai/slime_ai_scene_inspector.h"
#include "scene/2d/node_2d.h"
#include "scene/2d/sprite_2d.h"
#include "scene/resources/image_texture.h"

TEST_CASE("[SlimeAI][Inspect] object values are live and stale references fail") {
	Node2D *root = memnew(Node2D);
	root->set_name("Root");
	Node2D *child = memnew(Node2D);
	child->set_name("Marker");
	child->set_position(Vector2(12, 34));
	child->set_visible(false);
	root->add_child(child);
	child->set_owner(root);
	const String reference = SlimeAI::SceneInspector::object_ref(root, child);
	const Dictionary before = SlimeAI::ObjectInspector::inspect(root, reference, true);
	REQUIRE_FALSE(before.has("error"));
	CHECK(before["source"] == "live_editor");
	CHECK(before["class_name"] == "Node2D");
	const Dictionary values = before["properties"];
	const Dictionary position = values["position"];
	const Array coordinates = position["value"];
	CHECK(coordinates.size() == 2);
	CHECK(double(coordinates[0]) == 12.0);
	CHECK(double(coordinates[1]) == 34.0);
	const Dictionary visible = values["visible"];
	CHECK(bool(visible["value"]) == false);
	CHECK(root->get_child_count() == 1);
	CHECK(child->get_position() == Vector2(12, 34));
	root->remove_child(child);
	memdelete(child);
	const Dictionary stale = SlimeAI::ObjectInspector::inspect(root, reference, true);
	CHECK(stale["error"] == "STALE_REFERENCE");
	memdelete(root);
}

TEST_CASE("[SlimeAI][Inspect] native API metadata is separate from enabled action") {
	const Dictionary position = SlimeAI::ApiDescriber::describe(SNAME("Node2D"), SNAME("position"));
	REQUIRE_FALSE(position.has("error"));
	CHECK(position["type"] == "Vector2");
	CHECK(bool(position["enabled_action"]));
	const Dictionary method = SlimeAI::ApiDescriber::describe(SNAME("Node2D"), SNAME("queue_free"), true);
	REQUIRE_FALSE(method.has("error"));
	CHECK_FALSE(bool(method["enabled_action"]));
	const Dictionary unsupported = SlimeAI::ApiDescriber::describe(SNAME("Node2D"), SNAME("not_a_native_property"));
	CHECK(unsupported["error"] == "UNSUPPORTED_OPERATION");
}

TEST_CASE("[SlimeAI][Inspect] project overview bounds asset enumeration") {
	const Dictionary project = SlimeAI::ProjectInspector::inspect(1);
	REQUIRE_FALSE(project.has("error"));
	CHECK(String(project["project_ref"]).begins_with("project:"));
	CHECK(project["asset_scope"] == "res:// root files only");
	const Array assets = project["root_asset_summaries"];
	CHECK(assets.size() <= 1);
	CHECK(project.has("open_scenes"));
	CHECK(project.has("unsaved_buffers"));
	CHECK(project.has("engine"));
}

TEST_CASE("[SlimeAI][Inspect] scene pages have typed values and stable revision") {
	Node2D *root = memnew(Node2D);
	root->set_name("Root");
	for (int i = 0; i < 3; i++) {
		Node2D *child = memnew(Node2D);
		child->set_name(vformat("Node%d", i));
		child->set_position(Vector2(i, i + 1));
		root->add_child(child);
		child->set_owner(root);
	}
	const Dictionary first = SlimeAI::SceneInspector::inspect(root, true, 0, 2);
	const Dictionary second = SlimeAI::SceneInspector::inspect(root, true, 2, 2);
	REQUIRE_FALSE(first.has("error"));
	REQUIRE_FALSE(second.has("error"));
	CHECK(first["revision"] == second["revision"]);
	CHECK(int(first["next_offset"]) == 2);
	CHECK(int(second["next_offset"]) == -1);
	const Array nodes = second["nodes"];
	REQUIRE(nodes.size() == 2);
	const Dictionary item = nodes[0];
	const Dictionary position = item["position"];
	CHECK(position["type"] == "Vector2");
	const Array value = position["value"];
	CHECK(double(value[0]) == 1.0);
	CHECK(double(value[1]) == 2.0);
	memdelete(root);
}

TEST_CASE("[SlimeAI][Inspect] Unicode names and shared native texture identity") {
	Node2D *root = memnew(Node2D);
	root->set_name("Root");
	Ref<ImageTexture> texture;
	texture.instantiate();
	Sprite2D *first = memnew(Sprite2D);
	first->set_name(String::utf8("\xE6\x98\x9F"));
	first->set_texture(texture);
	root->add_child(first);
	first->set_owner(root);
	Sprite2D *second = memnew(Sprite2D);
	second->set_name("Second");
	second->set_texture(texture);
	root->add_child(second);
	second->set_owner(root);
	const Dictionary first_inspection = SlimeAI::ObjectInspector::inspect(root, SlimeAI::SceneInspector::object_ref(root, first), true);
	const Dictionary second_inspection = SlimeAI::ObjectInspector::inspect(root, SlimeAI::SceneInspector::object_ref(root, second), true);
	REQUIRE_FALSE(first_inspection.has("error"));
	REQUIRE_FALSE(second_inspection.has("error"));
	CHECK(first_inspection["name"] == String::utf8("\xE6\x98\x9F"));
	const Dictionary first_resources = first_inspection["resources"];
	const Dictionary second_resources = second_inspection["resources"];
	REQUIRE(first_resources.has("texture"));
	REQUIRE(second_resources.has("texture"));
	const Dictionary first_texture = first_resources["texture"];
	const Dictionary second_texture = second_resources["texture"];
	CHECK(first_texture["resource_ref"] == second_texture["resource_ref"]);
	CHECK(first_texture["contents"] == "not_inspected");
	memdelete(root);
}
