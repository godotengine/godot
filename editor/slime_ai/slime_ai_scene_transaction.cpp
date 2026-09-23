/**************************************************************************/
/*  slime_ai_scene_transaction.cpp                                      */
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

#include "slime_ai_scene_transaction.h"

#include "core/io/json.h"
#include "core/os/os.h"
#include "editor/editor_data.h"
#include "editor/editor_node.h"
#include "editor/editor_undo_redo_manager.h"
#include "editor/slime_ai/slime_ai_protocol.h"
#include "editor/slime_ai/slime_ai_scene_inspector.h"
#include "editor/slime_ai/slime_ai_scene_commands.h"
#include "editor/slime_ai/slime_ai_scene_command_executor.h"
#include "scene/2d/node_2d.h"
#include "scene/3d/node_3d.h"

#include <cmath>

namespace SlimeAI {

static Dictionary _failed(const String &p_code, const String &p_message, const String &p_recovery) {
	Dictionary result;
	result["status"] = "error";
	result["error"] = error(p_code, p_message, p_recovery);
	return result;
}

static Dictionary _ok(const String &p_status) {
	Dictionary result;
	result["status"] = p_status;
	return result;
}

SceneTransaction::SceneTransaction(const String &p_journal_path) :
		journal(p_journal_path) {
}

Dictionary SceneTransaction::set_mode(ApprovalMode p_mode, Node *p_selected_root) {
	if (p_mode != MANUAL && p_mode != PROTECTED && p_mode != FREEDOM) {
		return _failed("INVALID_ARGUMENT", "Unknown approval mode.", "Choose a supported mode.");
	}
	if (p_mode == FREEDOM && !p_selected_root) {
		return _failed("PERMISSION_DENIED", "Freedom requires an explicitly selected scene.", "Open and select the scene before choosing Freedom.");
	}
	approval_mode = p_mode;
	approved_scene_scope = p_selected_root ? SceneInspector::scene_ref(p_selected_root) : String();
	preview_id.clear();
	preview_data.clear();
	granted = false;
	Dictionary result = _ok("mode_set");
	result["mode"] = p_mode == MANUAL ? "Manual" : (p_mode == PROTECTED ? "Protected" : "Freedom");
	result["scene_scope"] = approved_scene_scope;
	return result;
}

Dictionary SceneTransaction::preview(Node *p_root, const String &p_operation_id, const Dictionary &p_proposal, bool p_editor_unsaved) {
	if (!journal.valid()) {
		return _failed("APPLY_FAILED_RECOVERY_REQUIRED", journal.get_load_error(), "Inspect the journal before editing.");
	}
	if (!p_root || p_operation_id.is_empty() || p_operation_id.length() > 128) {
		return _failed("INVALID_ARGUMENT", "A live scene and native operation ID are required.", "Inspect a scene and retry.");
	}
	const String payload_hash = JSON::stringify(p_proposal).sha256_text();
	const Dictionary existing = journal.get(p_operation_id);
	if (!existing.is_empty()) {
		if (String(existing.get("payload_hash", "")) != payload_hash) {
			return _failed("REQUEST_ALREADY_RECORDED", "Operation ID already belongs to another payload.", "Use a new native operation ID.");
		}
		Dictionary result = status(p_root, p_operation_id);
		result["duplicate"] = true;
		return result;
	}
	Dictionary inspection = SceneInspector::inspect(p_root, p_editor_unsaved);
	if (inspection.has("error")) {
		return _failed(inspection["error"], "Scene snapshot failed.", "Save or repair the scene and inspect again.");
	}
	Dictionary op;
	String code;
	if (!validate_scene_operation(p_proposal, p_root, op, code)) {
		return _failed(code, "Proposal is outside the supported one-scene command set.", "Request a new supported proposal.");
	}
	if (String(p_proposal["scene_ref"]) != String(inspection["scene_ref"]) || String(p_proposal["base_revision"]) != String(inspection["revision"])) {
		return _failed("REVISION_CONFLICT", "Proposal was made against a different scene revision.", "Inspect and preview again.");
	}
	sequence++;
	const Dictionary delta = describe_scene_operation(p_root, op);
	Dictionary immutable_contents;
	immutable_contents["operation_id"] = p_operation_id;
	immutable_contents["proposal"] = p_proposal;
	immutable_contents["delta"] = delta;
	immutable_contents["scene_scope"] = inspection["scene_ref"];
	immutable_contents["disk_revision"] = inspection["disk_revision"];
	const String preview_hash = JSON::stringify(immutable_contents).sha256_text();
	preview_id = (preview_hash + ":" + String::num_uint64(sequence)).sha256_text();
	preview_data = p_proposal.duplicate(true);
	preview_data["operation_id"] = p_operation_id;
	preview_data["payload_hash"] = payload_hash;
	preview_data["preview_hash"] = preview_hash;
	granted = false;
	Dictionary result = _ok("preview");
	result["preview_id"] = preview_id;
	result["preview_hash"] = preview_hash;
	result["operation_id"] = p_operation_id;
	result["base_revision"] = inspection["revision"];
	result["before"] = delta["before"];
	result["after"] = delta["after"];
	result["risk"] = delta["risk"];
	result["scene_scope"] = inspection["scene_path"];
	result["disk_revision"] = inspection["disk_revision"];
	const bool requires_grant = approval_mode == MANUAL || (approval_mode == PROTECTED && String(op["op"]) == "remove_node") || (approval_mode == FREEDOM && approved_scene_scope != String(inspection["scene_ref"]));
	result["requires_native_grant"] = requires_grant;
	result["mode"] = approval_mode == MANUAL ? "Manual" : (approval_mode == PROTECTED ? "Protected" : "Freedom");
	preview_data["requires_native_grant"] = requires_grant;
	return result;
}

Dictionary SceneTransaction::grant(const String &p_preview_id) {
	if (preview_id.is_empty() || p_preview_id != preview_id) {
		return _failed("STALE_REFERENCE", "Preview is unavailable.", "Create a fresh preview.");
	}
	granted = true;
	return _ok("granted");
}

Dictionary SceneTransaction::cancel(const String &p_preview_id) {
	if (preview_id.is_empty() || p_preview_id != preview_id) {
		return _failed("STALE_REFERENCE", "Preview is unavailable.", "Create a fresh preview.");
	}
	preview_id.clear();
	preview_data.clear();
	granted = false;
	return _ok("cancelled");
}

Dictionary SceneTransaction::apply(Node *p_root, const String &p_preview_id, bool p_editor_unsaved) {
	if (preview_id.is_empty() || p_preview_id != preview_id) {
		return _failed("STALE_REFERENCE", "Preview is unavailable.", "Create a fresh preview.");
	}
	if (bool(preview_data.get("requires_native_grant", true)) && !granted) {
		return _failed("PERMISSION_REQUIRED", "A native user grant is required.", "Authorize the visible preview in the dock.");
	}
	if (approval_mode == FREEDOM && approved_scene_scope != SceneInspector::scene_ref(p_root)) {
		return _failed("PERMISSION_DENIED", "Scene is outside the selected Freedom scope.", "Select the intended scene and choose mode again.");
	}
	const String operation_id = preview_data["operation_id"];
	if (!journal.get(operation_id).is_empty()) {
		return status(p_root, operation_id);
	}
	Dictionary inspection = SceneInspector::inspect(p_root, p_editor_unsaved);
	if (inspection.has("error") || String(inspection.get("revision", "")) != String(preview_data["base_revision"])) {
		return _failed("REVISION_CONFLICT", "The scene changed after preview.", "Inspect and preview again; the human edit is preserved.");
	}
	Dictionary op;
	String code;
	Dictionary proposal = preview_data.duplicate(true);
	proposal.erase("operation_id");
	proposal.erase("payload_hash");
	proposal.erase("preview_hash");
	proposal.erase("requires_native_grant");
	if (!validate_scene_operation(proposal, p_root, op, code)) {
		return _failed(code, "Preview target is no longer valid.", "Inspect and preview again.");
	}
	Node *target = operation_target(p_root, op);
	const String kind = op["op"];
	Dictionary record;
	record["payload_hash"] = preview_data["payload_hash"];
	record["scene_path"] = p_root->get_scene_file_path();
	record["target_path"] = String(p_root->get_path_to(target));
	record["kind"] = kind;
	record["before"] = describe_scene_operation(p_root, op)["before"];
	record["after"] = describe_scene_operation(p_root, op)["after"];
	if (kind == "create_child") {
		record["parent_path"] = String(p_root->get_path_to(target));
		record["class_name"] = op["class_name"];
		record["name"] = op["name"];
	}
	record["base_revision"] = inspection["revision"];
	record["base_disk_revision"] = inspection["disk_revision"];
	record["state"] = "prepared";
	if (!journal.put(operation_id, record)) {
		return _failed("APPLY_FAILED_RECOVERY_REQUIRED", "Could not persist the transaction intent.", "Check writable project user data and retry with a fresh ID.");
	}
	Node *created = nullptr;
	if (!execute_scene_operation(p_root, op, operation_id, &test_undo_redo, created)) {
		return _failed("APPLY_FAILED_RECOVERY_REQUIRED", "The prepared command could not execute.", "Query status and inspect the scene before retrying.");
	}
	preview_id.clear();
	preview_data.clear();
	granted = false;
	if (inject_failure_after_effect) {
		inject_failure_after_effect = false;
		return _failed("APPLY_FAILED_RECOVERY_REQUIRED", "Injected interruption after the scene effect and before journal confirmation.", "Save if desired, restart, and query status before any retry.");
	}
	record["state"] = "applied";
	if (!journal.put(operation_id, record)) {
		return _failed("APPLY_FAILED_RECOVERY_REQUIRED", "Scene effect exists but confirmation could not be written.", "Query status; do not retry blindly.");
	}
	Dictionary result = _ok("applied");
	result["operation_id"] = operation_id;
	if (created) {
		result["node_ref"] = SceneInspector::object_ref(p_root, created);
	}
	return result;
}

Dictionary SceneTransaction::status(Node *p_root, const String &p_operation_id) const {
	const Dictionary record = journal.get(p_operation_id);
	if (record.is_empty()) {
		return _failed("STALE_REFERENCE", "No such operation is recorded.", "Inspect the current scene.");
	}
	Dictionary result = _ok(record.get("state", "unknown"));
	result["operation_id"] = p_operation_id;
	if (String(record.get("state", "")) == "prepared") {
		String effect = "unknown";
		if (p_root && String(record.get("scene_path", "")) == p_root->get_scene_file_path()) {
			Node *parent = p_root->get_node_or_null(NodePath(record.get("parent_path", ".")));
			if (parent) {
				Node *candidate = parent->get_node_or_null(NodePath(record.get("name", "AI_Marker")));
				effect = candidate && candidate->get_meta("slime_ai_operation_id", "") == p_operation_id && String(candidate->get_class()) == String(record.get("class_name", "")) ? "present_unconfirmed" : "absent_or_changed";
			}
		}
		result["effect"] = effect;
		result["recovery"] = "Do not retry this ID. Inspect the scene and reconcile the uncertain effect manually.";
	}
	return result;
}

Dictionary SceneTransaction::undo(Node *p_root) {
	if (!p_root) {
		return _failed("STALE_REFERENCE", "No edited scene.", "Open the scene.");
	}
	bool success = EditorUndoRedoManager::get_singleton() && EditorNode::get_singleton() ? EditorUndoRedoManager::get_singleton()->undo_history(EditorNode::get_editor_data().get_current_edited_scene_history_id()) : test_undo_redo.undo();
	return success ? _ok("undone") : _failed("UNSUPPORTED_OPERATION", "No undo action is available.", "Apply a change first.");
}

Dictionary SceneTransaction::redo(Node *p_root) {
	if (!p_root) {
		return _failed("STALE_REFERENCE", "No edited scene.", "Open the scene.");
	}
	bool success = EditorUndoRedoManager::get_singleton() && EditorNode::get_singleton() ? EditorUndoRedoManager::get_singleton()->redo_history(EditorNode::get_editor_data().get_current_edited_scene_history_id()) : test_undo_redo.redo();
	return success ? _ok("redone") : _failed("UNSUPPORTED_OPERATION", "No redo action is available.", "Undo the change first.");
}

} // namespace SlimeAI
