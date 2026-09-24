#include "slime_ai_save_failure_probe.h"

#include "core/io/file_access.h"
#include "core/io/json.h"
#include "core/os/os.h"
#include "editor/editor_interface.h"
#include "editor/editor_node.h"
#include "editor/slime_ai/slime_ai_scene_inspector.h"
#include "editor/slime_ai/slime_ai_scene_transaction.h"
#include "scene/2d/node_2d.h"

namespace SlimeAI {

SaveFailureProbe::SaveFailureProbe() {
	if (OS::get_singleton()->get_environment("SLIME_AI_TEST_SAVE_PROBE") != "1") {
		return;
	}
	probe_dir = OS::get_singleton()->get_environment("SLIME_AI_SAVE_PROBE_DIR");
	expected_scene = OS::get_singleton()->get_environment("SLIME_AI_SAVE_PROBE_SCENE");
	if (probe_dir.is_empty() || expected_scene.is_empty()) {
		probe_dir.clear();
	}
}

void SaveFailureProbe::write_result(const String &p_name, const Dictionary &p_value) const {
	Ref<FileAccess> file = FileAccess::open(probe_dir.path_join(p_name), FileAccess::WRITE);
	if (file.is_valid()) {
		file->store_string(JSON::stringify(p_value, "  "));
		file->flush();
	}
}

void SaveFailureProbe::tick(SceneTransaction &p_transaction, Node *p_root, bool p_editor_unsaved) {
	if (!enabled() || !p_root || p_root->get_scene_file_path() != expected_scene) {
		return;
	}
	if (stage == 0) {
		Node2D *root_2d = Object::cast_to<Node2D>(p_root);
		if (!root_2d) {
			return;
		}
		const Dictionary inspection = SceneInspector::inspect(p_root, p_editor_unsaved);
		if (inspection.has("error")) {
			return;
		}
		previous_disk_hash = FileAccess::get_sha256(expected_scene);
		initial_child_count = p_root->get_child_count();
		operation_id = vformat("save-denial-probe-%d", OS::get_singleton()->get_ticks_usec());
		Dictionary position;
		position["type"] = "Vector2";
		Array coordinates;
		coordinates.push_back(48);
		coordinates.push_back(24);
		position["value"] = coordinates;
		Dictionary properties;
		properties["position"] = position;
		Dictionary operation;
		operation["op"] = "create_child";
		operation["parent_ref"] = inspection["root_ref"];
		operation["class_name"] = "Node2D";
		operation["name"] = "SaveDenialMarker";
		operation["properties"] = properties;
		Array operations;
		operations.push_back(operation);
		Dictionary proposal;
		proposal["scene_ref"] = inspection["scene_ref"];
		proposal["base_revision"] = inspection["revision"];
		proposal["operations"] = operations;
		original_proposal = proposal;
		const Dictionary preview = p_transaction.preview(p_root, operation_id, proposal, p_editor_unsaved);
		const String preview_id = preview.get("preview_id", "");
		if (preview_id.is_empty() || String(p_transaction.grant(preview_id).get("status", "")) != "granted") {
			Dictionary failed;
			failed["status"] = "probe_preview_failed";
			failed["preview"] = preview;
			write_result("probe_error.json", failed);
			stage = 4;
			return;
		}
		const Dictionary applied = p_transaction.apply(p_root, preview_id, p_editor_unsaved);
		Dictionary ready;
		ready["status"] = applied.get("status", "unknown");
		ready["operation_id"] = operation_id;
		ready["preview_id"] = preview_id;
		ready["disk_before"] = previous_disk_hash;
		ready["memory_node_count"] = p_root->get_child_count();
		ready["editor_unsaved"] = EditorNode::get_singleton()->is_scene_unsaved(EditorNode::get_editor_data().get_edited_scene());
		ready["execution"] = applied;
		write_result("ready.json", ready);
		stage = 1;
		return;
	}
	if (stage == 1 && FileAccess::exists(probe_dir.path_join("deny_go.flag"))) {
		const bool injected = OS::get_singleton()->get_environment("SLIME_AI_TEST_FORCE_SAVE_FAILURE") == "1";
		const Error save_error = injected ? ERR_FILE_CANT_WRITE : EditorInterface::get_singleton()->save_scene();
		Dictionary denied;
		denied["status"] = save_error == OK ? "unexpected_save_success" : "save_failed";
		denied["save_error"] = save_error;
		denied["disk_before"] = previous_disk_hash;
		denied["disk_after"] = injected ? FileAccess::get_sha256(expected_scene) : "unreadable_while_exclusively_locked";
		denied["injected"] = injected;
		denied["disk_verification"] = "parent_harness_checks_after_lock_release_before_retry";
		denied["memory_node_count"] = p_root->get_child_count();
		denied["marker_present"] = p_root->get_node_or_null(NodePath("SaveDenialMarker")) != nullptr;
		denied["editor_unsaved"] = EditorNode::get_singleton()->is_scene_unsaved(EditorNode::get_editor_data().get_edited_scene());
		denied["execution"] = p_transaction.status(p_root, operation_id);
		denied["persistence"] = save_error == OK ? "confirmed" : "failed";
		write_result("denied.json", denied);
		stage = 2;
		return;
	}
	if (stage == 2 && FileAccess::exists(probe_dir.path_join("retry_go.flag"))) {
		const Dictionary duplicate = p_transaction.preview(p_root, operation_id, original_proposal, true);
		const Error save_error = EditorInterface::get_singleton()->save_scene();
		Dictionary retried;
		retried["status"] = save_error == OK ? "save_confirmed" : "save_failed";
		retried["save_error"] = save_error;
		retried["disk_before"] = previous_disk_hash;
		retried["disk_after"] = FileAccess::get_sha256(expected_scene);
		retried["memory_node_count"] = p_root->get_child_count();
		retried["marker_present"] = p_root->get_node_or_null(NodePath("SaveDenialMarker")) != nullptr;
		retried["editor_unsaved"] = EditorNode::get_singleton()->is_scene_unsaved(EditorNode::get_editor_data().get_edited_scene());
		retried["operation_status"] = duplicate;
		retried["duplicate_request"] = duplicate.get("duplicate", false);
		retried["no_second_apply"] = p_root->get_child_count() == initial_child_count + 1;
		write_result("retry.json", retried);
		stage = 3;
	}
}

} // namespace SlimeAI
