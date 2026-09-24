/**************************************************************************/
/*  test_slime_ai_p03_3d.cpp                                              */
/**************************************************************************/

#include "tests/test_macros.h"

TEST_FORCE_LINK(test_slime_ai_p03_3d)

#include "core/io/dir_access.h"
#include "core/io/resource_loader.h"
#include "core/io/resource_saver.h"
#include "core/os/os.h"
#include "editor/slime_ai/slime_ai_scene_inspector.h"
#include "editor/slime_ai/slime_ai_scene_transaction.h"
#include "scene/3d/node_3d.h"
#include "scene/resources/packed_scene.h"

TEST_CASE("[SlimeAI][TX3D] native create, undo, redo, save, and reopen") {
	const String suffix = vformat("%d", OS::get_singleton()->get_ticks_usec());
	const String base = OS::get_singleton()->get_user_data_dir().path_join("slime_ai_3d_" + suffix);
	Node3D *root = memnew(Node3D);
	root->set_name("Root3D");
	{
		Dictionary position;
		position["type"] = "Vector3";
		Array value;
		value.push_back(1.0);
		value.push_back(2.0);
		value.push_back(3.0);
		position["value"] = value;
		Dictionary properties;
		properties["position"] = position;
		Dictionary operation;
		operation["op"] = "create_child";
		operation["parent_ref"] = SlimeAI::SceneInspector::object_ref(root, root);
		operation["class_name"] = "Node3D";
		operation["name"] = "AI_3D_Marker";
		operation["properties"] = properties;
		Array operations;
		operations.push_back(operation);
		const Dictionary inspection = SlimeAI::SceneInspector::inspect(root, true);
		REQUIRE_FALSE(inspection.has("error"));
		Dictionary proposal;
		proposal["scene_ref"] = inspection["scene_ref"];
		proposal["base_revision"] = inspection["revision"];
		proposal["operations"] = operations;
		SlimeAI::SceneTransaction tx(base + ".journal.json");
		const Dictionary preview = tx.preview(root, "native-3d-create-" + suffix, proposal, true);
		REQUIRE(preview["status"] == "preview");
		CHECK(root->get_child_count() == 0);
		REQUIRE(tx.grant(preview["preview_id"])["status"] == "granted");
		REQUIRE(tx.apply(root, preview["preview_id"], true)["status"] == "applied");
		REQUIRE(root->get_child_count() == 1);
		Node3D *child = Object::cast_to<Node3D>(root->get_child(0));
		REQUIRE(child != nullptr);
		CHECK(child->get_position() == Vector3(1, 2, 3));
		CHECK(tx.undo(root)["status"] == "undone");
		CHECK(root->get_child_count() == 0);
		CHECK(tx.redo(root)["status"] == "redone");
		CHECK(root->get_child_count() == 1);
		Ref<PackedScene> packed;
		packed.instantiate();
		REQUIRE(packed->pack(root) == OK);
		REQUIRE(ResourceSaver::save(packed, base + ".tscn") == OK);
	}
	memdelete(root);
	Ref<PackedScene> reopened = ResourceLoader::load(base + ".tscn");
	REQUIRE(reopened.is_valid());
	Node *reloaded = reopened->instantiate();
	REQUIRE(reloaded != nullptr);
	REQUIRE(reloaded->get_child_count() == 1);
	Node3D *persisted = Object::cast_to<Node3D>(reloaded->get_child(0));
	REQUIRE(persisted != nullptr);
	CHECK(persisted->get_position() == Vector3(1, 2, 3));
	CHECK(persisted->get_owner() == reloaded);
	memdelete(reloaded);
	DirAccess::remove_absolute(base + ".journal.json");
	DirAccess::remove_absolute(base + ".tscn");
}
