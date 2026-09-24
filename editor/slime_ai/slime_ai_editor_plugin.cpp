/**************************************************************************/
/*  slime_ai_editor_plugin.cpp                                          */
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

#include "slime_ai_editor_plugin.h"

#include "core/config/project_settings.h"
#include "core/io/file_access.h"
#include "core/io/json.h"
#include "core/object/callable_mp.h"
#include "core/os/os.h"
#include "editor/docks/editor_dock.h"
#include "editor/editor_data.h"
#include "editor/editor_interface.h"
#include "editor/editor_node.h"
#include "editor/slime_ai/slime_ai_api_describer.h"
#include "editor/slime_ai/slime_ai_object_inspector.h"
#include "editor/slime_ai/slime_ai_project_inspector.h"
#include "editor/slime_ai/slime_ai_protocol.h"
#include "editor/slime_ai/slime_ai_scene_inspector.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/label.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/scroll_container.h"
#include "scene/gui/text_edit.h"

static Button *_button(VBoxContainer *p_parent, const String &p_text, const Callable &p_action) {
	Button *button = memnew(Button);
	button->set_text(p_text);
	p_parent->add_child(button);
	button->connect("pressed", p_action);
	return button;
}

SlimeAIEditorPlugin::SlimeAIEditorPlugin() :
		transaction(ProjectSettings::get_singleton()->globalize_path("user://slime_ai/" + ProjectSettings::get_singleton()->get_resource_path().sha256_text() + "/journal.json")),
		run_controller(service, transaction) {
	dock = memnew(EditorDock);
	dock->set_title("Slime AI");
	dock->set_default_slot(EditorDock::DOCK_SLOT_RIGHT_BR);
	dock->set_layout_key("SlimeAI");
	VBoxContainer *column = memnew(VBoxContainer);
	ScrollContainer *scroll = memnew(ScrollContainer);
	scroll->set_horizontal_scroll_mode(ScrollContainer::SCROLL_MODE_DISABLED);
	scroll->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	dock->add_child(scroll);
	scroll->add_child(column);
	Label *banner = memnew(Label);
	banner->set_text("LOCAL SERVICE · native grants only · live runs require one-run authorization");
	column->add_child(banner);
	context = memnew(Label);
	context->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
	column->add_child(context);
	_button(column, "Inspect project", callable_mp(this, &SlimeAIEditorPlugin::_inspect_project));
	scene_context_button = _button(column, "Scene: none (inspect)", callable_mp(this, &SlimeAIEditorPlugin::_inspect));
	object_context_button = _button(column, "Node: none (inspect)", callable_mp(this, &SlimeAIEditorPlugin::_inspect_object));
	_button(column, "Describe selected native position", callable_mp(this, &SlimeAIEditorPlugin::_describe_api));
	_button(column, "Mode: Manual", callable_mp(this, &SlimeAIEditorPlugin::_mode_manual));
	_button(column, "Mode: Protected", callable_mp(this, &SlimeAIEditorPlugin::_mode_protected));
	_button(column, "Mode: Freedom for selected scene", callable_mp(this, &SlimeAIEditorPlugin::_mode_freedom));
	_button(column, "Connect local service", callable_mp(this, &SlimeAIEditorPlugin::_connect_service));
	_button(column, "Disconnect local service", callable_mp(this, &SlimeAIEditorPlugin::_disconnect_service));
	_button(column, "Request and preview marker", callable_mp(this, &SlimeAIEditorPlugin::_request_patch));
	_button(column, "Authorize this preview", callable_mp(this, &SlimeAIEditorPlugin::_grant));
	_button(column, "Apply authorized preview", callable_mp(this, &SlimeAIEditorPlugin::_apply));
	_button(column, "Cancel preview", callable_mp(this, &SlimeAIEditorPlugin::_cancel));
	_button(column, "Undo", callable_mp(this, &SlimeAIEditorPlugin::_undo));
	_button(column, "Redo", callable_mp(this, &SlimeAIEditorPlugin::_redo));
	_button(column, "Save scene", callable_mp(this, &SlimeAIEditorPlugin::_save));
	_button(column, "Recorded operation status", callable_mp(this, &SlimeAIEditorPlugin::_status));
	_button(column, "Resolve reviewed uncertain operation (no replay)", callable_mp(this, &SlimeAIEditorPlugin::_resolve_reviewed_operation));
	Label *run_label = memnew(Label);
	run_label->set_text("Bounded provider task · 3 attempts / 4 tools / 1024 output tokens / 120 s");
	run_label->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
	column->add_child(run_label);
	_button(column, "Provider: OpenAI Responses", callable_mp(this, &SlimeAIEditorPlugin::_provider_openai));
	_button(column, "Provider: offline fake", callable_mp(this, &SlimeAIEditorPlugin::_provider_fake));
	_button(column, "Check credential connection", callable_mp(this, &SlimeAIEditorPlugin::_provider_status));
	model_input = memnew(LineEdit);
	model_input->set_placeholder("Explicit OpenAI model ID (required for live run)");
	column->add_child(model_input);
	_button(column, "Intent: Discuss", callable_mp(this, &SlimeAIEditorPlugin::_intent_discuss));
	_button(column, "Intent: Propose", callable_mp(this, &SlimeAIEditorPlugin::_intent_propose));
	_button(column, "Intent: Execute", callable_mp(this, &SlimeAIEditorPlugin::_intent_execute));
	task_input = memnew(TextEdit);
	task_input->set_custom_minimum_size(Size2(300, 60));
	task_input->set_placeholder("Ask about the selected scene or request one supported preview");
	column->add_child(task_input);
	_button(column, "Authorize one bounded live run", callable_mp(this, &SlimeAIEditorPlugin::_authorize_live_once));
	_button(column, "Start selected task", callable_mp(this, &SlimeAIEditorPlugin::_start_run));
	_button(column, "Pause before next tool", callable_mp(this, &SlimeAIEditorPlugin::_pause_run));
	_button(column, "Resume paused tool", callable_mp(this, &SlimeAIEditorPlugin::_resume_run));
	_button(column, "Cancel provider run", callable_mp(this, &SlimeAIEditorPlugin::_cancel_run));
	_button(column, "Show task/change status", callable_mp(this, &SlimeAIEditorPlugin::_run_status));
	_button(column, "Review proposal for Execute", callable_mp(this, &SlimeAIEditorPlugin::_transition_execute));
	activity = memnew(TextEdit);
	activity->set_editable(false);
	activity->set_custom_minimum_size(Size2(300, 75));
	column->add_child(activity);
	display = memnew(TextEdit);
	display->set_editable(false);
	display->set_custom_minimum_size(Size2(300, 160));
	display->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	column->add_child(display);
	add_dock(dock);
	_update_context();
	set_process(true);
}

SlimeAIEditorPlugin::~SlimeAIEditorPlugin() {
	service.stop();
	if (dock) {
		remove_dock(dock);
		memdelete(dock);
	}
}

void SlimeAIEditorPlugin::_show(const Dictionary &p_data) {
	display->set_text(JSON::stringify(p_data, "  "));
	String event = p_data.get("status", "inspection");
	if (p_data.has("error")) {
		Variant error = p_data["error"];
		event = error.get_type() == Variant::DICTIONARY ? String(Dictionary(error).get("code", "error")) : String(error);
	}
	PackedStringArray lines = activity->get_text().split("\n", false);
	while (lines.size() >= 8) {
		lines.remove_at(0);
	}
	lines.push_back(event);
	activity->set_text(String("\n").join(lines));
	_update_context();
}

void SlimeAIEditorPlugin::_update_context() {
	Node *root = EditorNode::get_singleton()->get_edited_scene();
	String selected = "none";
	if (root) {
		const List<Node *> nodes = EditorNode::get_singleton()->get_editor_selection()->get_full_selected_node_list();
		selected = nodes.is_empty() ? String(root->get_name()) : String(nodes.front()->get()->get_name());
	}
	context->set_text(vformat("Service: %s | Provider: %s | Intent: %s | Project: %s\nScene: %s | Selection: %s", service.get_state(), selected_provider, selected_intent, ProjectSettings::get_singleton()->get_setting("application/config/name", "Unnamed Project"), root ? root->get_scene_file_path() : String("none"), selected));
	scene_context_button->set_text(vformat("Scene: %s (inspect)", root ? root->get_scene_file_path().get_file() : String("none")));
	object_context_button->set_text(vformat("Node: %s (inspect)", selected));
}

void SlimeAIEditorPlugin::_inspect_project() {
	_show(SlimeAI::ProjectInspector::inspect());
}

void SlimeAIEditorPlugin::_inspect() {
	Node *root = EditorNode::get_singleton()->get_edited_scene();
	if (!root) {
		_show(SlimeAI::error("STALE_REFERENCE", "No scene is open.", "Open a scene."));
		return;
	}
	last_inspection = SlimeAI::SceneInspector::inspect(root, EditorNode::get_singleton()->is_scene_unsaved(EditorNode::get_editor_data().get_edited_scene()));
	_show(last_inspection);
}

void SlimeAIEditorPlugin::_inspect_object() {
	Node *root = EditorNode::get_singleton()->get_edited_scene();
	if (!root) {
		_show(SlimeAI::error("STALE_REFERENCE", "No scene is open.", "Open a scene."));
		return;
	}
	const List<Node *> nodes = EditorNode::get_singleton()->get_editor_selection()->get_full_selected_node_list();
	Node *node = nodes.is_empty() ? root : nodes.front()->get();
	_show(SlimeAI::ObjectInspector::inspect(root, SlimeAI::SceneInspector::object_ref(root, node), EditorNode::get_singleton()->is_scene_unsaved(EditorNode::get_editor_data().get_edited_scene())));
}

void SlimeAIEditorPlugin::_describe_api() {
	Node *root = EditorNode::get_singleton()->get_edited_scene();
	if (!root) {
		_show(SlimeAI::error("STALE_REFERENCE", "No scene is open.", "Open a scene."));
		return;
	}
	const List<Node *> nodes = EditorNode::get_singleton()->get_editor_selection()->get_full_selected_node_list();
	Node *node = nodes.is_empty() ? root : nodes.front()->get();
	_show(SlimeAI::ApiDescriber::describe(node->get_class(), SNAME("position")));
}

void SlimeAIEditorPlugin::_connect_service() {
	const String script_path = OS::get_singleton()->get_executable_path().get_base_dir().get_base_dir().path_join("tools/slime_ai/agent_service/src/main.ts");
	Dictionary result;
	result["script_path"] = script_path;
	result["started"] = service.start(script_path);
	result["service_state"] = service.get_state();
	result["error"] = service.get_last_error();
	_show(result);
}

void SlimeAIEditorPlugin::_disconnect_service() {
	service.stop();
	Dictionary result;
	result["service_state"] = service.get_state();
	_show(result);
}

void SlimeAIEditorPlugin::_request_patch() {
	Node *root = EditorNode::get_singleton()->get_edited_scene();
	if (!root || !service.is_ready()) {
		_show(SlimeAI::error("CAPABILITY_UNAVAILABLE", "Scene or fake service unavailable.", "Open a scene and connect the service."));
		return;
	}
	_inspect();
	if (last_inspection.has("error")) {
		return;
	}
	Node *parent = root;
	const List<Node *> selected = EditorNode::get_singleton()->get_editor_selection()->get_full_selected_node_list();
	if (!selected.is_empty()) {
		parent = selected.front()->get();
	}
	const String parent_ref = SlimeAI::SceneInspector::object_ref(root, parent);
	if (parent_ref.is_empty()) {
		_show(SlimeAI::error("STALE_REFERENCE", "Selection does not belong to the edited scene.", "Select a node in the open scene."));
		return;
	}
	Dictionary params;
	params["scene_ref"] = last_inspection["scene_ref"];
	params["base_revision"] = last_inspection["revision"];
	params["parent_ref"] = parent_ref;
	params["root_class"] = last_inspection["root_class"];
	params["scenario"] = "normal";
	context_manifest.clear();
	context_manifest["fields_sent"] = params.keys();
	context_manifest["scene_ref"] = params["scene_ref"];
	context_manifest["selected_node_ref"] = parent_ref;
	context_manifest["unsaved_content_sent"] = false;
	context_manifest["editor_unsaved"] = last_inspection["editor_unsaved"];
	const String request_id = service.request("fake_propose_scene_patch", params);
	Dictionary result;
	result["service_request_id"] = request_id;
	result["state"] = request_id.is_empty() ? "request_failed" : "awaiting_fake_proposal";
	result["outgoing_context_manifest"] = context_manifest;
	_show(result);
}

void SlimeAIEditorPlugin::_on_response(const Dictionary &p_frame) {
	if (String(p_frame.get("protocol_version", "")) == "1.1") {
		if (p_frame.has("event")) {
			Node *root = EditorNode::get_singleton()->get_edited_scene();
			_show(run_controller.on_frame(root, root && EditorNode::get_singleton()->is_scene_unsaved(EditorNode::get_editor_data().get_edited_scene()), p_frame));
		} else {
			_show(p_frame);
		}
		return;
	}
	if (String(p_frame["status"]) != "ok") {
		_show(p_frame);
		return;
	}
	const Dictionary result = p_frame["result"];
	if (!result.has("operations")) {
		Dictionary state;
		state["service_state"] = service.get_state();
		state["handshake"] = result;
		_show(state);
		return;
	}
	Node *root = EditorNode::get_singleton()->get_edited_scene();
	if (!root) {
		_show(SlimeAI::error("STALE_REFERENCE", "Scene closed while awaiting proposal.", "Open and inspect again."));
		return;
	}
	last_proposal = result;
	operation_sequence++;
	last_operation_id = vformat("native-%d-%d", OS::get_singleton()->get_ticks_usec(), operation_sequence);
	Dictionary preview = transaction.preview(root, last_operation_id, last_proposal, EditorNode::get_singleton()->is_scene_unsaved(EditorNode::get_editor_data().get_edited_scene()));
	last_preview_id = preview.get("preview_id", "");
	preview["outgoing_context_manifest"] = context_manifest;
	_show(preview);
}

void SlimeAIEditorPlugin::_grant() {
	_show(run_controller.get_preview_id().is_empty() ? transaction.grant(last_preview_id) : run_controller.grant());
}

void SlimeAIEditorPlugin::_apply() {
	Node *root = EditorNode::get_singleton()->get_edited_scene();
	if (!run_controller.get_preview_id().is_empty()) {
		_show(run_controller.apply(root, root && EditorNode::get_singleton()->is_scene_unsaved(EditorNode::get_editor_data().get_edited_scene())));
		return;
	}
	if (OS::get_singleton()->get_environment("SLIME_AI_TEST_CRASH_AFTER_SAVE") == "1") {
		transaction.set_inject_failure_after_effect(true);
	}
	Dictionary result = transaction.apply(root, last_preview_id, root && EditorNode::get_singleton()->is_scene_unsaved(EditorNode::get_editor_data().get_edited_scene()));
	_show(result);
	if (OS::get_singleton()->get_environment("SLIME_AI_TEST_CRASH_AFTER_SAVE") == "1" && result.has("error") && Dictionary(result["error"]).get("code", "") == "APPLY_FAILED_RECOVERY_REQUIRED") {
		if (EditorInterface::get_singleton()->save_scene() == OK) {
			OS::get_singleton()->kill(OS::get_singleton()->get_process_id());
		}
	}
}

void SlimeAIEditorPlugin::_cancel() {
	_show(transaction.cancel(last_preview_id));
	last_preview_id.clear();
}

void SlimeAIEditorPlugin::_undo() {
	_show(transaction.undo(EditorNode::get_singleton()->get_edited_scene()));
}

void SlimeAIEditorPlugin::_redo() {
	_show(transaction.redo(EditorNode::get_singleton()->get_edited_scene()));
}

void SlimeAIEditorPlugin::_save() {
	Dictionary result;
	Node *root = EditorNode::get_singleton()->get_edited_scene();
	const String path = root ? root->get_scene_file_path() : String();
	const String old_hash = path.is_empty() ? String() : FileAccess::get_sha256(path);
	const bool injected = OS::get_singleton()->get_environment("SLIME_AI_TEST_FORCE_SAVE_FAILURE") == "1";
	const Error save_error = injected ? ERR_FILE_CANT_WRITE : EditorInterface::get_singleton()->save_scene();
	const String new_hash = path.is_empty() ? String() : FileAccess::get_sha256(path);
	result["status"] = save_error == OK ? "save_confirmed" : "save_failed";
	result["save_error"] = save_error;
	result["injected_failure"] = injected;
	result["disk_before"] = old_hash;
	result["disk_after"] = new_hash;
	result["previous_disk_preserved"] = old_hash == new_hash;
	result["editor_unsaved"] = root && EditorNode::get_singleton()->is_scene_unsaved(EditorNode::get_editor_data().get_edited_scene());
	result["edit"] = last_operation_id.is_empty() ? Variant("unknown") : Variant(transaction.status(root, last_operation_id));
	result["persistence"] = save_error == OK ? "confirmed_by_editor_save" : "failed_previous_disk_preserved_if_hash_equal";
	run_controller.note_save(save_error, old_hash, new_hash, bool(result["editor_unsaved"]), injected);
	_show(result);
}

void SlimeAIEditorPlugin::_status() {
	reconcile_operation_id.clear();
	reconcile_expected_revision.clear();
	reconcile_observed_effect.clear();
	Node *root = EditorNode::get_singleton()->get_edited_scene();
	String pending_id;
	int pending_count = 0;
	const Array all_ids = transaction.recorded_operation_ids();
	for (int i = 0; i < all_ids.size(); i++) {
		const Dictionary candidate = transaction.status(root, all_ids[i]);
		if (String(candidate.get("status", "")) == "prepared" && String(candidate.get("effect", "")) != "unknown") {
			pending_id = all_ids[i];
			pending_count++;
		}
	}
	if (pending_count == 1 && root) {
		const Dictionary inspection = SlimeAI::SceneInspector::inspect(root, EditorNode::get_singleton()->is_scene_unsaved(EditorNode::get_editor_data().get_edited_scene()));
		if (!inspection.has("error")) {
			reconcile_operation_id = pending_id;
			reconcile_expected_revision = inspection["revision"];
			reconcile_observed_effect = transaction.status(root, pending_id).get("effect", "unknown");
		}
	}
	if (last_operation_id.is_empty()) {
		Dictionary result;
		const Array ids = transaction.recorded_operation_ids();
		Dictionary records;
		for (int i = 0; i < ids.size(); i++) {
			records[ids[i]] = transaction.status(EditorNode::get_singleton()->get_edited_scene(), ids[i]);
		}
		result["recorded_operations"] = records;
		result["recovery"] = "After restart, inspect this list and query each pending operation before retrying.";
		_show(result);
	} else {
		_show(transaction.status(EditorNode::get_singleton()->get_edited_scene(), last_operation_id));
	}
}

void SlimeAIEditorPlugin::_resolve_reviewed_operation() {
	Node *root = EditorNode::get_singleton()->get_edited_scene();
	if (!root || reconcile_operation_id.is_empty()) {
		_show(SlimeAI::error("RECONCILIATION_REQUIRED", "Inspect the recorded operation status first; exactly one unresolved operation in this scene is required.", "Review the observed effect and current scene, then resolve explicitly."));
		return;
	}
	_show(transaction.resolve(root, reconcile_operation_id, reconcile_expected_revision, reconcile_observed_effect, EditorNode::get_singleton()->is_scene_unsaved(EditorNode::get_editor_data().get_edited_scene())));
	reconcile_operation_id.clear();
	reconcile_expected_revision.clear();
	reconcile_observed_effect.clear();
}

void SlimeAIEditorPlugin::_provider_openai() {
	selected_provider = "openai_responses";
	live_authorized_once = false;
	_show(run_controller.status(EditorNode::get_singleton()->get_edited_scene()));
}

void SlimeAIEditorPlugin::_provider_fake() {
	selected_provider = "fake";
	live_authorized_once = false;
	_show(run_controller.status(EditorNode::get_singleton()->get_edited_scene()));
}

void SlimeAIEditorPlugin::_provider_status() {
	Dictionary result;
	result["status"] = "credential_status_requested";
	result["service_request_id"] = service.request("provider_status", Dictionary());
	if (String(result["service_request_id"]).is_empty()) {
		result["status"] = "service_unavailable";
	}
	_show(result);
}

void SlimeAIEditorPlugin::_intent_discuss() {
	selected_intent = "discuss";
	_show(run_controller.status(EditorNode::get_singleton()->get_edited_scene()));
}

void SlimeAIEditorPlugin::_intent_propose() {
	selected_intent = "propose";
	_show(run_controller.status(EditorNode::get_singleton()->get_edited_scene()));
}

void SlimeAIEditorPlugin::_intent_execute() {
	selected_intent = "execute";
	_show(run_controller.status(EditorNode::get_singleton()->get_edited_scene()));
}

void SlimeAIEditorPlugin::_authorize_live_once() {
	if (selected_provider != "openai_responses" || model_input->get_text().strip_edges().is_empty()) {
		_show(SlimeAI::error("MODEL_REQUIRED", "Select OpenAI and enter the exact model ID first.", "Enter a model ID, then authorize one bounded run."));
		return;
	}
	live_authorized_once = true;
	live_authorized_model = model_input->get_text().strip_edges();
	Dictionary result;
	result["status"] = "one_run_authorized";
	result["provider"] = selected_provider;
	result["model"] = live_authorized_model;
	result["limits"] = "3 attempts, 4 tool calls, 1024 output tokens, 120-second deadline";
	result["endpoint"] = "https://api.openai.com/v1/responses";
	_show(result);
}

void SlimeAIEditorPlugin::_start_run() {
	Node *root = EditorNode::get_singleton()->get_edited_scene();
	const String model = selected_provider == "fake" ? String() : model_input->get_text().strip_edges();
	const bool authorized = selected_provider == "openai_responses" && live_authorized_once && live_authorized_model == model;
	live_authorized_once = false;
	_show(run_controller.start(root, root && EditorNode::get_singleton()->is_scene_unsaved(EditorNode::get_editor_data().get_edited_scene()), selected_provider, model, selected_intent, task_input->get_text(), authorized));
}

void SlimeAIEditorPlugin::_cancel_run() {
	_show(run_controller.cancel());
}

void SlimeAIEditorPlugin::_pause_run() {
	_show(run_controller.pause());
}

void SlimeAIEditorPlugin::_resume_run() {
	Node *root = EditorNode::get_singleton()->get_edited_scene();
	_show(run_controller.resume(root, root && EditorNode::get_singleton()->is_scene_unsaved(EditorNode::get_editor_data().get_edited_scene())));
}

void SlimeAIEditorPlugin::_run_status() {
	_show(run_controller.status(EditorNode::get_singleton()->get_edited_scene()));
}

void SlimeAIEditorPlugin::_transition_execute() {
	Node *root = EditorNode::get_singleton()->get_edited_scene();
	_show(run_controller.transition_to_execute(root, root && EditorNode::get_singleton()->is_scene_unsaved(EditorNode::get_editor_data().get_edited_scene())));
}

void SlimeAIEditorPlugin::_mode_manual() {
	last_preview_id.clear();
	_show(transaction.set_mode(SlimeAI::SceneTransaction::MANUAL, EditorNode::get_singleton()->get_edited_scene()));
}

void SlimeAIEditorPlugin::_mode_protected() {
	last_preview_id.clear();
	_show(transaction.set_mode(SlimeAI::SceneTransaction::PROTECTED, EditorNode::get_singleton()->get_edited_scene()));
}

void SlimeAIEditorPlugin::_mode_freedom() {
	last_preview_id.clear();
	_show(transaction.set_mode(SlimeAI::SceneTransaction::FREEDOM, EditorNode::get_singleton()->get_edited_scene()));
}

void SlimeAIEditorPlugin::_notification(int p_what) {
	if (p_what == NOTIFICATION_PROCESS) {
		if (p04_editor_probe.enabled()) {
			Node *root = EditorNode::get_singleton()->get_edited_scene();
			p04_editor_probe.tick(service, run_controller, transaction, root, root && EditorNode::get_singleton()->is_scene_unsaved(EditorNode::get_editor_data().get_edited_scene()));
			return;
		}
		if (save_failure_probe.enabled()) {
			Node *root = EditorNode::get_singleton()->get_edited_scene();
			save_failure_probe.tick(transaction, root, root && EditorNode::get_singleton()->is_scene_unsaved(EditorNode::get_editor_data().get_edited_scene()));
		}
		const uint64_t now = OS::get_singleton()->get_ticks_msec();
		if (now - last_context_update >= 500) {
			last_context_update = now;
			_update_context();
		}
		Vector<Dictionary> frames;
		service.poll(frames);
		for (const Dictionary &frame : frames) {
			_on_response(frame);
		}
	}
}
