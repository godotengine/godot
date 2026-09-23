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
#include "core/io/json.h"
#include "core/object/callable_mp.h"
#include "core/os/os.h"
#include "editor/docks/editor_dock.h"
#include "editor/editor_data.h"
#include "editor/editor_interface.h"
#include "editor/editor_node.h"
#include "editor/slime_ai/slime_ai_protocol.h"
#include "editor/slime_ai/slime_ai_scene_inspector.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/label.h"
#include "scene/gui/text_edit.h"

static Button *_button(VBoxContainer *p_parent, const String &p_text, const Callable &p_action) {
	Button *button = memnew(Button);
	button->set_text(p_text);
	p_parent->add_child(button);
	button->connect("pressed", p_action);
	return button;
}

SlimeAIEditorPlugin::SlimeAIEditorPlugin() :
		transaction(ProjectSettings::get_singleton()->globalize_path("user://slime_ai/" + ProjectSettings::get_singleton()->get_resource_path().sha256_text() + "/journal.json")) {
	dock = memnew(EditorDock);
	dock->set_title("Slime AI — Fake/Test Mode");
	dock->set_default_slot(EditorDock::DOCK_SLOT_RIGHT_BR);
	dock->set_layout_key("SlimeAI");
	VBoxContainer *column = memnew(VBoxContainer);
	dock->add_child(column);
	Label *banner = memnew(Label);
	banner->set_text("LOCAL FAKE SERVICE · native grants only");
	column->add_child(banner);
	_button(column, "Inspect selected scene", callable_mp(this, &SlimeAIEditorPlugin::_inspect));
	_button(column, "Mode: Manual", callable_mp(this, &SlimeAIEditorPlugin::_mode_manual));
	_button(column, "Mode: Protected", callable_mp(this, &SlimeAIEditorPlugin::_mode_protected));
	_button(column, "Mode: Freedom for selected scene", callable_mp(this, &SlimeAIEditorPlugin::_mode_freedom));
	_button(column, "Connect fake service", callable_mp(this, &SlimeAIEditorPlugin::_connect_service));
	_button(column, "Request and preview marker", callable_mp(this, &SlimeAIEditorPlugin::_request_patch));
	_button(column, "Authorize this preview", callable_mp(this, &SlimeAIEditorPlugin::_grant));
	_button(column, "Apply authorized preview", callable_mp(this, &SlimeAIEditorPlugin::_apply));
	_button(column, "Cancel preview", callable_mp(this, &SlimeAIEditorPlugin::_cancel));
	_button(column, "Undo", callable_mp(this, &SlimeAIEditorPlugin::_undo));
	_button(column, "Redo", callable_mp(this, &SlimeAIEditorPlugin::_redo));
	_button(column, "Save scene", callable_mp(this, &SlimeAIEditorPlugin::_save));
	_button(column, "Recorded operation status", callable_mp(this, &SlimeAIEditorPlugin::_status));
	display = memnew(TextEdit);
	display->set_editable(false);
	display->set_custom_minimum_size(Size2(300, 240));
	display->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	column->add_child(display);
	add_dock(dock);
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

void SlimeAIEditorPlugin::_connect_service() {
	const String script_path = OS::get_singleton()->get_executable_path().get_base_dir().get_base_dir().path_join("tools/slime_ai/agent_service/src/main.ts");
	Dictionary result;
	result["script_path"] = script_path;
	result["started"] = service.start(script_path);
	result["service_state"] = service.get_state();
	result["error"] = service.get_last_error();
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
	const String request_id = service.request("fake_propose_scene_patch", params);
	Dictionary result;
	result["service_request_id"] = request_id;
	result["state"] = request_id.is_empty() ? "request_failed" : "awaiting_fake_proposal";
	_show(result);
}

void SlimeAIEditorPlugin::_on_response(const Dictionary &p_frame) {
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
	_show(preview);
}

void SlimeAIEditorPlugin::_grant() {
	_show(transaction.grant(last_preview_id));
}

void SlimeAIEditorPlugin::_apply() {
	Node *root = EditorNode::get_singleton()->get_edited_scene();
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
	result["save_error"] = EditorInterface::get_singleton()->save_scene();
	_show(result);
}

void SlimeAIEditorPlugin::_status() {
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
		Vector<Dictionary> frames;
		service.poll(frames);
		for (const Dictionary &frame : frames) {
			_on_response(frame);
		}
	}
}
