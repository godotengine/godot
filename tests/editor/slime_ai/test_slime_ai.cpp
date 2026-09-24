/**************************************************************************/
/*  test_slime_ai.cpp                                                   */
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

TEST_FORCE_LINK(test_slime_ai)

#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/io/resource_loader.h"
#include "core/io/resource_saver.h"
#include "core/os/os.h"
#include "editor/slime_ai/slime_ai_protocol.h"
#include "editor/slime_ai/slime_ai_scene_inspector.h"
#include "editor/slime_ai/slime_ai_scene_transaction.h"
#include "editor/slime_ai/slime_ai_service_client.h"
#include "scene/2d/node_2d.h"
#include "scene/resources/packed_scene.h"

namespace TestSlimeAI {

static String _temp_path(const String &p_extension) {
	return OS::get_singleton()->get_user_data_dir().path_join(vformat("slime_ai_test_%d.%s", OS::get_singleton()->get_ticks_usec(), p_extension));
}

static Dictionary _proposal(Node2D *p_root, const Dictionary &p_inspection, const String &p_parent_ref = String()) {
	Dictionary position;
	position["type"] = "Vector2";
	Array values;
	values.push_back(48);
	values.push_back(24);
	position["value"] = values;
	Dictionary properties;
	properties["position"] = position;
	Dictionary operation;
	operation["op"] = "create_child";
	operation["parent_ref"] = p_parent_ref.is_empty() ? SlimeAI::SceneInspector::object_ref(p_root, p_root) : p_parent_ref;
	operation["class_name"] = "Node2D";
	operation["name"] = "AI_Marker";
	operation["properties"] = properties;
	Array operations;
	operations.push_back(operation);
	Dictionary proposal;
	proposal["scene_ref"] = p_inspection["scene_ref"];
	proposal["base_revision"] = p_inspection["revision"];
	proposal["operations"] = operations;
	return proposal;
}

static Dictionary _proposal_for(Node2D *p_root, const Dictionary &p_operation) {
	const Dictionary inspection = SlimeAI::SceneInspector::inspect(p_root, true);
	Array operations;
	operations.push_back(p_operation);
	Dictionary proposal;
	proposal["scene_ref"] = inspection["scene_ref"];
	proposal["base_revision"] = inspection["revision"];
	proposal["operations"] = operations;
	return proposal;
}

static String _error_code(const Dictionary &p_result) {
	return p_result.has("error") && p_result["error"].get_type() == Variant::DICTIONARY ? String(Dictionary(p_result["error"])["code"]) : String();
}

TEST_CASE("[SlimeAI][IPC] fragmented UTF-8 and duplicate key rejection") {
	SlimeAI::FrameDecoder decoder;
	Vector<Dictionary> frames;
	const String valid = "{\"protocol_version\":\"1.0\",\"request_id\":\"é\",\"status\":\"ok\",\"result\":{}}\n";
	const CharString bytes = valid.utf8();
	for (int i = 0; i < bytes.length(); i++) {
		CHECK(decoder.feed((const uint8_t *)bytes.get_data() + i, 1, frames));
	}
	REQUIRE(frames.size() == 1);
	String error;
	CHECK(SlimeAI::validate_envelope(frames[0], "é", error));
	CHECK_FALSE(SlimeAI::validate_envelope(frames[0], "other", error));
	decoder.reset();
	frames.clear();
	const char *duplicate = "{\"a\":1,\"a\":2}\n";
	CHECK_FALSE(decoder.feed((const uint8_t *)duplicate, strlen(duplicate), frames));
	decoder.reset();
	const uint8_t bad_utf8[] = { 0xc3, 0x28, '\n' };
	CHECK_FALSE(decoder.feed(bad_utf8, 3, frames));
}

TEST_CASE("[SlimeAI][IPC] overlong and partial frames fail closed") {
	SlimeAI::FrameDecoder decoder;
	Vector<Dictionary> frames;
	Vector<uint8_t> oversized;
	oversized.resize(SlimeAI::MAX_FRAME_BYTES + 1);
	oversized.fill('x');
	CHECK_FALSE(decoder.feed(oversized.ptr(), oversized.size(), frames));
	CHECK(frames.is_empty());
	decoder.reset();
	const char *partial = "{\"status\":\"ok\"";
	CHECK(decoder.feed((const uint8_t *)partial, strlen(partial), frames));
	CHECK(frames.is_empty());
}

TEST_CASE("[SlimeAI][IPC] missing service reports unavailable") {
	SlimeAI::ServiceClient client;
	CHECK_FALSE(client.start(_temp_path("missing.ts")));
	CHECK(client.get_state() == "unavailable");
}

TEST_CASE("[SlimeAI][IPC] private fake service handshake and disconnect") {
	const String script = OS::get_singleton()->get_executable_path().get_base_dir().get_base_dir().path_join("tools/slime_ai/agent_service/src/main.ts");
	SlimeAI::ServiceClient client;
	REQUIRE(client.start(script));
	for (int i = 0; i < 500 && !client.is_ready(); i++) {
		Vector<Dictionary> responses;
		client.poll(responses);
		OS::get_singleton()->delay_usec(2000);
	}
	REQUIRE(client.is_ready());
	Dictionary params;
	params["scene_ref"] = "scene-test";
	params["base_revision"] = "revision-test";
	params["parent_ref"] = "parent-test";
	params["root_class"] = "Node2D";
	params["scenario"] = "disconnect";
	REQUIRE_FALSE(client.request("fake_propose_scene_patch", params).is_empty());
	for (int i = 0; i < 500 && client.get_state() != "disconnected"; i++) {
		Vector<Dictionary> responses;
		client.poll(responses);
		OS::get_singleton()->delay_usec(2000);
	}
	CHECK(client.get_state() == "disconnected");
}

TEST_CASE("[SlimeAI][Inspect] live unsaved snapshot and stale reference") {
	Node2D *root = memnew(Node2D);
	root->set_name("Root");
	const Dictionary before = SlimeAI::SceneInspector::inspect(root, true);
	REQUIRE_FALSE(before.has("error"));
	CHECK(before["source"] == "unsaved_scene");
	CHECK(root->get_child_count() == 0);
	Node2D *child = memnew(Node2D);
	child->set_name("Human");
	root->add_child(child);
	child->set_owner(root);
	const String ref = SlimeAI::SceneInspector::object_ref(root, child);
	CHECK(SlimeAI::SceneInspector::resolve(root, ref) == child);
	child->set_position(Vector2(3, 7));
	const Dictionary after = SlimeAI::SceneInspector::inspect(root, true);
	CHECK(before["revision"] != after["revision"]);
	root->remove_child(child);
	memdelete(child);
	CHECK(SlimeAI::SceneInspector::resolve(root, ref) == nullptr);
	memdelete(root);
}

TEST_CASE("[SlimeAI][TX] grant, apply, dedup, undo, redo, and save/reopen") {
	const String journal = _temp_path("journal.json");
	const String scene = _temp_path("tscn");
	Node2D *root = memnew(Node2D);
	root->set_name("Root");
	{
		SlimeAI::SceneTransaction tx(journal);
		const Dictionary inspection = SlimeAI::SceneInspector::inspect(root, true);
		REQUIRE_FALSE(inspection.has("error"));
		const Dictionary proposal = _proposal(root, inspection);
		const Dictionary preview = tx.preview(root, "native-test-1", proposal, true);
		REQUIRE(preview["status"] == "preview");
		CHECK(root->get_child_count() == 0);
		const String preview_id = preview["preview_id"];
		CHECK(_error_code(tx.apply(root, preview_id, true)) == "PERMISSION_REQUIRED");
		CHECK(tx.grant(preview_id)["status"] == "granted");
		CHECK(tx.apply(root, preview_id, true)["status"] == "applied");
		CHECK(root->get_child_count() == 1);
		CHECK(bool(tx.preview(root, "native-test-1", proposal, true)["duplicate"]));
		CHECK(root->get_child_count() == 1);
		CHECK(tx.undo(root)["status"] == "undone");
		CHECK(root->get_child_count() == 0);
		CHECK(tx.redo(root)["status"] == "redone");
		CHECK(root->get_child_count() == 1);
		Ref<PackedScene> packed;
		packed.instantiate();
		REQUIRE(packed->pack(root) == OK);
		REQUIRE(ResourceSaver::save(packed, scene) == OK);
	}
	memdelete(root);
	Ref<PackedScene> reopened = ResourceLoader::load(scene);
	REQUIRE(reopened.is_valid());
	Node *reloaded = reopened->instantiate();
	REQUIRE(reloaded != nullptr);
	CHECK(reloaded->get_child_count() == 1);
	CHECK(reloaded->get_child(0)->get_owner() == reloaded);
	memdelete(reloaded);
	DirAccess::remove_absolute(journal);
	DirAccess::remove_absolute(scene);
}

TEST_CASE("[SlimeAI][TX] stale preview and unsupported proposal leave human change") {
	const String journal = _temp_path("journal.json");
	Node2D *root = memnew(Node2D);
	root->set_name("Root");
	{
		SlimeAI::SceneTransaction tx(journal);
		const Dictionary inspection = SlimeAI::SceneInspector::inspect(root, true);
		Dictionary proposal = _proposal(root, inspection);
		Dictionary preview = tx.preview(root, "native-test-2", proposal, true);
		REQUIRE(preview["status"] == "preview");
		const String preview_id = preview["preview_id"];
		CHECK(tx.grant(preview_id)["status"] == "granted");
		root->set_position(Vector2(100, 0));
		CHECK(_error_code(tx.apply(root, preview_id, true)) == "REVISION_CONFLICT");
		CHECK(root->get_position() == Vector2(100, 0));
		CHECK(root->get_child_count() == 0);
		proposal = _proposal(root, SlimeAI::SceneInspector::inspect(root, true));
		Dictionary operation = Array(proposal["operations"])[0];
		operation["class_name"] = "ScriptedNode";
		Array operations;
		operations.push_back(operation);
		proposal["operations"] = operations;
		CHECK(_error_code(tx.preview(root, "native-test-3", proposal, true)) == "UNSUPPORTED_OPERATION");
	}
	memdelete(root);
	DirAccess::remove_absolute(journal);
}

TEST_CASE("[SlimeAI][TX] cancel and restart after interrupted effect") {
	const String journal = _temp_path("journal.json");
	Node2D *root = memnew(Node2D);
	root->set_name("Root");
	const Dictionary proposal = _proposal(root, SlimeAI::SceneInspector::inspect(root, true));
	{
		SlimeAI::SceneTransaction tx(journal);
		Dictionary preview = tx.preview(root, "native-cancel", proposal, true);
		REQUIRE(preview["status"] == "preview");
		CHECK(tx.cancel(preview["preview_id"])["status"] == "cancelled");
		CHECK(_error_code(tx.apply(root, preview["preview_id"], true)) == "STALE_REFERENCE");
		preview = tx.preview(root, "native-interrupted", proposal, true);
		REQUIRE(preview["status"] == "preview");
		CHECK(tx.grant(preview["preview_id"])["status"] == "granted");
		tx.set_inject_failure_after_effect(true);
		CHECK(_error_code(tx.apply(root, preview["preview_id"], true)) == "APPLY_FAILED_RECOVERY_REQUIRED");
		CHECK(root->get_child_count() == 1);
	}
	{
		SlimeAI::SceneTransaction restarted(journal);
		Dictionary status = restarted.status(root, "native-interrupted");
		CHECK(status["status"] == "prepared");
		CHECK(status["effect"] == "present_unconfirmed");
		CHECK(bool(restarted.preview(root, "native-interrupted", proposal, true)["duplicate"]));
		CHECK(_error_code(restarted.preview(root, "fresh-after-interruption", _proposal(root, SlimeAI::SceneInspector::inspect(root, true)), true)) == "RECONCILIATION_REQUIRED");
		const String current_revision = SlimeAI::SceneInspector::inspect(root, true)["revision"];
		CHECK(_error_code(restarted.resolve(root, "native-interrupted", "stale-revision", "present_unconfirmed", true)) == "REVISION_CONFLICT");
		CHECK(restarted.resolve(root, "native-interrupted", current_revision, "present_unconfirmed", true)["status"] == "resolved_without_replay");
		CHECK(root->get_child_count() == 1);
		Dictionary next_proposal = _proposal(root, SlimeAI::SceneInspector::inspect(root, true));
		Dictionary next_operation = Array(next_proposal["operations"])[0];
		next_operation["name"] = "AfterResolution";
		Array next_operations;
		next_operations.push_back(next_operation);
		next_proposal["operations"] = next_operations;
		CHECK(restarted.preview(root, "fresh-after-resolution", next_proposal, true)["status"] == "preview");
	}
	memdelete(root);
	DirAccess::remove_absolute(journal);
}

TEST_CASE("[SlimeAI][Inspect] saved scene inspection preserves file and reports unsaved live edits") {
	const String scene = _temp_path("tscn");
	Node2D *root = memnew(Node2D);
	root->set_name("Root");
	Ref<PackedScene> packed;
	packed.instantiate();
	REQUIRE(packed->pack(root) == OK);
	REQUIRE(ResourceSaver::save(packed, scene) == OK);
	root->set_scene_file_path(scene);
	const String disk_before = FileAccess::get_sha256(scene);
	const Dictionary clean = SlimeAI::SceneInspector::inspect(root, false);
	REQUIRE_FALSE(clean.has("error"));
	CHECK(clean["source"] == "saved_scene_with_live_editor_state");
	CHECK_FALSE(bool(clean["editor_unsaved"]));
	CHECK(String(clean["disk_revision"]) == disk_before);
	CHECK(FileAccess::get_sha256(scene) == disk_before);
	root->set_position(Vector2(5, 9));
	const Dictionary dirty = SlimeAI::SceneInspector::inspect(root, true);
	CHECK(bool(dirty["editor_unsaved"]));
	CHECK(clean["revision"] != dirty["revision"]);
	CHECK(FileAccess::get_sha256(scene) == disk_before);
	memdelete(root);
	DirAccess::remove_absolute(scene);
}

TEST_CASE("[SlimeAI][TX] native property, rename, remove subtree and undo") {
	const String journal = _temp_path("journal.json");
	Node2D *root = memnew(Node2D);
	root->set_name("Root");
	Node2D *child = memnew(Node2D);
	child->set_name("Marker");
	root->add_child(child);
	child->set_owner(root);
	Node2D *grandchild = memnew(Node2D);
	grandchild->set_name("Nested");
	child->add_child(grandchild);
	grandchild->set_owner(root);
	{
		SlimeAI::SceneTransaction tx(journal);
		Dictionary typed;
		typed["type"] = "Vector2";
		Array coordinates;
		coordinates.push_back(12);
		coordinates.push_back(34);
		typed["value"] = coordinates;
		Dictionary op;
		op["op"] = "set_property";
		op["node_ref"] = SlimeAI::SceneInspector::object_ref(root, child);
		op["property"] = "position";
		op["value"] = typed;
		Dictionary preview = tx.preview(root, "property-1", _proposal_for(root, op), true);
		REQUIRE(preview["status"] == "preview");
		CHECK(tx.grant(preview["preview_id"])["status"] == "granted");
		CHECK(tx.apply(root, preview["preview_id"], true)["status"] == "applied");
		CHECK(child->get_position() == Vector2(12, 34));
		CHECK(tx.undo(root)["status"] == "undone");
		CHECK(child->get_position() == Vector2());
		CHECK(tx.redo(root)["status"] == "redone");
		CHECK(child->get_position() == Vector2(12, 34));

		op.clear();
		op["op"] = "rename_node";
		op["node_ref"] = SlimeAI::SceneInspector::object_ref(root, child);
		op["name"] = "Renamed";
		preview = tx.preview(root, "rename-1", _proposal_for(root, op), true);
		REQUIRE(preview["status"] == "preview");
		CHECK(tx.grant(preview["preview_id"])["status"] == "granted");
		CHECK(tx.apply(root, preview["preview_id"], true)["status"] == "applied");
		CHECK(child->get_name() == "Renamed");
		CHECK(tx.undo(root)["status"] == "undone");
		CHECK(child->get_name() == "Marker");
		CHECK(tx.redo(root)["status"] == "redone");

		op.clear();
		op["op"] = "remove_node";
		op["node_ref"] = SlimeAI::SceneInspector::object_ref(root, child);
		preview = tx.preview(root, "remove-1", _proposal_for(root, op), true);
		REQUIRE(preview["status"] == "preview");
		CHECK(Array(preview["before"]).size() == 2);
		CHECK(tx.grant(preview["preview_id"])["status"] == "granted");
		CHECK(tx.apply(root, preview["preview_id"], true)["status"] == "applied");
		CHECK(root->get_child_count() == 0);
		CHECK(tx.undo(root)["status"] == "undone");
		REQUIRE(root->get_child_count() == 1);
		CHECK(root->get_child(0) == child);
		CHECK(child->get_owner() == root);
		CHECK(grandchild->get_owner() == root);
		CHECK(grandchild->get_parent() == child);
	}
	memdelete(root);
	DirAccess::remove_absolute(journal);
}

TEST_CASE("[SlimeAI][TX] unsupported property and differing duplicate payload") {
	const String journal = _temp_path("journal.json");
	Node2D *root = memnew(Node2D);
	root->set_name("Root");
	{
		SlimeAI::SceneTransaction tx(journal);
		Dictionary proposal = _proposal(root, SlimeAI::SceneInspector::inspect(root, true));
		Dictionary preview = tx.preview(root, "same-id", proposal, true);
		REQUIRE(preview["status"] == "preview");
		CHECK(tx.grant(preview["preview_id"])["status"] == "granted");
		CHECK(tx.apply(root, preview["preview_id"], true)["status"] == "applied");
		Dictionary altered = proposal.duplicate(true);
		Dictionary op = Array(altered["operations"])[0];
		op["name"] = "Different";
		Array altered_operations;
		altered_operations.push_back(op);
		altered["operations"] = altered_operations;
		CHECK(_error_code(tx.preview(root, "same-id", altered, true)) == "REQUEST_ALREADY_RECORDED");
		CHECK(root->get_child_count() == 1);
		Dictionary invalid;
		invalid["op"] = "set_property";
		invalid["node_ref"] = SlimeAI::SceneInspector::object_ref(root, root);
		invalid["property"] = "script";
		Dictionary typed;
		typed["type"] = "bool";
		typed["value"] = true;
		invalid["value"] = typed;
		CHECK(_error_code(tx.preview(root, "invalid-prop", _proposal_for(root, invalid), true)) == "UNSUPPORTED_OPERATION");
	}
	memdelete(root);
	DirAccess::remove_absolute(journal);
}

TEST_CASE("[SlimeAI][TX] interrupted saved effect reconciles after scene reopen") {
	const String journal = _temp_path("journal.json");
	const String scene = _temp_path("tscn");
	Node2D *root = memnew(Node2D);
	root->set_name("Root");
	root->set_scene_file_path(scene);
	Ref<PackedScene> initial;
	initial.instantiate();
	REQUIRE(initial->pack(root) == OK);
	REQUIRE(ResourceSaver::save(initial, scene) == OK);
	const Dictionary proposal = _proposal(root, SlimeAI::SceneInspector::inspect(root, false));
	{
		SlimeAI::SceneTransaction tx(journal);
		const Dictionary preview = tx.preview(root, "crash-save-1", proposal, false);
		REQUIRE(preview["status"] == "preview");
		CHECK(tx.grant(preview["preview_id"])["status"] == "granted");
		tx.set_inject_failure_after_effect(true);
		CHECK(_error_code(tx.apply(root, preview["preview_id"], false)) == "APPLY_FAILED_RECOVERY_REQUIRED");
		Ref<PackedScene> changed;
		changed.instantiate();
		REQUIRE(changed->pack(root) == OK);
		REQUIRE(ResourceSaver::save(changed, scene) == OK);
	}
	memdelete(root);
	Ref<PackedScene> reopened = ResourceLoader::load(scene);
	REQUIRE(reopened.is_valid());
	Node *reloaded = reopened->instantiate();
	REQUIRE(reloaded != nullptr);
	{
		SlimeAI::SceneTransaction restarted(journal);
		const Dictionary status = restarted.status(reloaded, "crash-save-1");
		CHECK(status["status"] == "prepared");
		CHECK(status["effect"] == "present_unconfirmed");
		CHECK(bool(restarted.preview(reloaded, "crash-save-1", proposal, false)["duplicate"]));
		CHECK(reloaded->get_child_count() == 1);
	}
	memdelete(reloaded);
	DirAccess::remove_absolute(journal);
	DirAccess::remove_absolute(scene);
}

TEST_CASE("[SlimeAI][TX] unwritable journal prevents scene effect") {
	const String journal = _temp_path("journal-directory");
	REQUIRE(DirAccess::make_dir_recursive_absolute(journal) == OK);
	Node2D *root = memnew(Node2D);
	root->set_name("Root");
	{
		SlimeAI::SceneTransaction tx(journal);
		const Dictionary preview = tx.preview(root, "journal-blocked", _proposal(root, SlimeAI::SceneInspector::inspect(root, true)), true);
		REQUIRE(preview["status"] == "preview");
		CHECK(tx.grant(preview["preview_id"])["status"] == "granted");
		CHECK(_error_code(tx.apply(root, preview["preview_id"], true)) == "APPLY_FAILED_RECOVERY_REQUIRED");
		CHECK(root->get_child_count() == 0);
	}
	memdelete(root);
	DirAccess::remove_absolute(journal);
}

TEST_CASE("[SlimeAI][TX] unowned node cannot be renamed") {
	const String journal = _temp_path("journal.json");
	Node2D *root = memnew(Node2D);
	root->set_name("Root");
	Node2D *child = memnew(Node2D);
	child->set_name("Unowned");
	root->add_child(child);
	SlimeAI::SceneTransaction tx(journal);
	Dictionary operation;
	operation["op"] = "rename_node";
	operation["node_ref"] = SlimeAI::SceneInspector::object_ref(root, child);
	operation["name"] = "Renamed";
	CHECK(_error_code(tx.preview(root, "invalid-owner", _proposal_for(root, operation), true)) == "READ_ONLY_RESOURCE");
	CHECK(child->get_name() == StringName("Unowned"));
	memdelete(root);
	DirAccess::remove_absolute(journal);
}

TEST_CASE("[SlimeAI][Policy] protected removal needs a native grant") {
	const String journal = _temp_path("journal.json");
	Node2D *root = memnew(Node2D);
	root->set_name("Root");
	Node2D *child = memnew(Node2D);
	child->set_name("Marker");
	root->add_child(child);
	child->set_owner(root);
	{
		SlimeAI::SceneTransaction tx(journal);
		CHECK(tx.set_mode(SlimeAI::SceneTransaction::PROTECTED, root)["status"] == "mode_set");
		Dictionary operation;
		operation["op"] = "remove_node";
		operation["node_ref"] = SlimeAI::SceneInspector::object_ref(root, child);
		const Dictionary preview = tx.preview(root, "protected-remove", _proposal_for(root, operation), true);
		REQUIRE(preview["status"] == "preview");
		CHECK(bool(preview["requires_native_grant"]));
		CHECK(_error_code(tx.apply(root, preview["preview_id"], true)) == "PERMISSION_REQUIRED");
		CHECK(root->get_child_count() == 1);
		CHECK(tx.grant(preview["preview_id"])["status"] == "granted");
		CHECK(tx.apply(root, preview["preview_id"], true)["status"] == "applied");
		CHECK(root->get_child_count() == 0);
	}
	memdelete(root);
	DirAccess::remove_absolute(journal);
}

TEST_CASE("[SlimeAI][TX] recovery status preserves a later human edit") {
	const String journal = _temp_path("journal.json");
	Node2D *root = memnew(Node2D);
	root->set_name("Root");
	const Dictionary proposal = _proposal(root, SlimeAI::SceneInspector::inspect(root, true));
	{
		SlimeAI::SceneTransaction tx(journal);
		const Dictionary preview = tx.preview(root, "later-human-edit", proposal, true);
		REQUIRE(preview["status"] == "preview");
		CHECK(tx.grant(preview["preview_id"])["status"] == "granted");
		tx.set_inject_failure_after_effect(true);
		CHECK(_error_code(tx.apply(root, preview["preview_id"], true)) == "APPLY_FAILED_RECOVERY_REQUIRED");
	}
	Node2D *human = memnew(Node2D);
	human->set_name("HumanLater");
	root->add_child(human);
	human->set_owner(root);
	{
		SlimeAI::SceneTransaction restarted(journal);
		const Dictionary status = restarted.status(root, "later-human-edit");
		CHECK(status["status"] == "prepared");
		CHECK(status["effect"] == "present_unconfirmed");
		CHECK(bool(restarted.preview(root, "later-human-edit", proposal, true)["duplicate"]));
		CHECK(_error_code(restarted.preview(root, "fresh-after-human", _proposal(root, SlimeAI::SceneInspector::inspect(root, true)), true)) == "RECONCILIATION_REQUIRED");
		const String reviewed_revision = SlimeAI::SceneInspector::inspect(root, true)["revision"];
		root->set_position(Vector2(9, 12));
		CHECK(_error_code(restarted.resolve(root, "later-human-edit", reviewed_revision, "present_unconfirmed", true)) == "REVISION_CONFLICT");
		const String current_revision = SlimeAI::SceneInspector::inspect(root, true)["revision"];
		CHECK(restarted.resolve(root, "later-human-edit", current_revision, "present_unconfirmed", true)["status"] == "resolved_without_replay");
		CHECK(root->get_child_count() == 2);
		CHECK(root->get_node_or_null(NodePath("HumanLater")) == human);
	}
	memdelete(root);
	DirAccess::remove_absolute(journal);
}

TEST_CASE("[SlimeAI][TX] closing scene before apply leaves it unchanged") {
	const String journal = _temp_path("journal.json");
	Node2D *root = memnew(Node2D);
	root->set_name("Root");
	{
		SlimeAI::SceneTransaction tx(journal);
		const Dictionary preview = tx.preview(root, "scene-closed", _proposal(root, SlimeAI::SceneInspector::inspect(root, true)), true);
		REQUIRE(preview["status"] == "preview");
		CHECK(tx.grant(preview["preview_id"])["status"] == "granted");
		CHECK(_error_code(tx.apply(nullptr, preview["preview_id"], true)) == "REVISION_CONFLICT");
		CHECK(root->get_child_count() == 0);
	}
	memdelete(root);
	DirAccess::remove_absolute(journal);
}

TEST_CASE("[SlimeAI][Policy] mode change invalidates grant and Freedom stays in scene scope") {
	const String journal = _temp_path("journal.json");
	Node2D *root = memnew(Node2D);
	root->set_name("Root");
	Node2D *other = memnew(Node2D);
	other->set_name("Other");
	{
		SlimeAI::SceneTransaction tx(journal);
		Dictionary proposal = _proposal(root, SlimeAI::SceneInspector::inspect(root, true));
		Dictionary preview = tx.preview(root, "mode-1", proposal, true);
		REQUIRE(preview["status"] == "preview");
		CHECK(tx.grant(preview["preview_id"])["status"] == "granted");
		CHECK(tx.set_mode(SlimeAI::SceneTransaction::FREEDOM, root)["status"] == "mode_set");
		CHECK(_error_code(tx.apply(root, preview["preview_id"], true)) == "STALE_REFERENCE");
		preview = tx.preview(root, "mode-2", proposal, true);
		REQUIRE(preview["status"] == "preview");
		CHECK_FALSE(bool(preview["requires_native_grant"]));
		CHECK(tx.apply(root, preview["preview_id"], true)["status"] == "applied");
		proposal = _proposal(other, SlimeAI::SceneInspector::inspect(other, true));
		preview = tx.preview(other, "mode-3", proposal, true);
		REQUIRE(preview["status"] == "preview");
		CHECK(bool(preview["requires_native_grant"]));
		CHECK(tx.grant(preview["preview_id"])["status"] == "granted");
		CHECK(_error_code(tx.apply(other, preview["preview_id"], true)) == "PERMISSION_DENIED");
		CHECK(other->get_child_count() == 0);
	}
	memdelete(root);
	memdelete(other);
	DirAccess::remove_absolute(journal);
}

TEST_CASE("[SlimeAI][TX] visible bool property round trips through undo") {
	const String journal = _temp_path("journal.json");
	Node2D *root = memnew(Node2D);
	root->set_name("Root");
	{
		SlimeAI::SceneTransaction tx(journal);
		Dictionary typed;
		typed["type"] = "bool";
		typed["value"] = false;
		Dictionary op;
		op["op"] = "set_property";
		op["node_ref"] = SlimeAI::SceneInspector::object_ref(root, root);
		op["property"] = "visible";
		op["value"] = typed;
		Dictionary preview = tx.preview(root, "visible-1", _proposal_for(root, op), true);
		REQUIRE(preview["status"] == "preview");
		CHECK(tx.grant(preview["preview_id"])["status"] == "granted");
		CHECK(tx.apply(root, preview["preview_id"], true)["status"] == "applied");
		CHECK_FALSE(root->is_visible());
		CHECK(tx.undo(root)["status"] == "undone");
		CHECK(root->is_visible());
		CHECK(tx.redo(root)["status"] == "redone");
		CHECK_FALSE(root->is_visible());
	}
	memdelete(root);
	DirAccess::remove_absolute(journal);
}

} // namespace TestSlimeAI
