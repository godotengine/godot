/**************************************************************************/
/*  slime_ai_editor_plugin.h                                            */
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

#pragma once

#include "editor/plugins/editor_plugin.h"
#include "editor/slime_ai/slime_ai_scene_transaction.h"
#include "editor/slime_ai/slime_ai_service_client.h"
#include "editor/slime_ai/slime_ai_run_controller.h"
#include "editor/slime_ai/slime_ai_save_failure_probe.h"
#include "editor/slime_ai/slime_ai_p04_editor_probe.h"

class EditorDock;
class Button;
class Label;
class TextEdit;
class LineEdit;

class SlimeAIEditorPlugin : public EditorPlugin {
	GDCLASS(SlimeAIEditorPlugin, EditorPlugin);

	EditorDock *dock = nullptr;
	TextEdit *display = nullptr;
	TextEdit *activity = nullptr;
	Label *context = nullptr;
	Button *scene_context_button = nullptr;
	Button *object_context_button = nullptr;
	SlimeAI::ServiceClient service;
	SlimeAI::SceneTransaction transaction;
	SlimeAI::RunController run_controller;
	SlimeAI::SaveFailureProbe save_failure_probe;
	SlimeAI::P04EditorProbe p04_editor_probe;
	LineEdit *model_input = nullptr;
	TextEdit *task_input = nullptr;
	String selected_provider = "openai_responses";
	String selected_intent = "discuss";
	bool live_authorized_once = false;
	String live_authorized_model;
	Dictionary last_inspection;
	Dictionary last_proposal;
	Dictionary context_manifest;
	String last_preview_id;
	String last_operation_id;
	String reconcile_operation_id;
	String reconcile_expected_revision;
	String reconcile_observed_effect;
	uint64_t operation_sequence = 0;
	uint64_t last_context_update = 0;

	void _show(const Dictionary &p_data);
	void _update_context();
	void _inspect_project();
	void _inspect();
	void _inspect_object();
	void _describe_api();
	void _connect_service();
	void _disconnect_service();
	void _request_patch();
	void _grant();
	void _apply();
	void _cancel();
	void _undo();
	void _redo();
	void _save();
	void _status();
	void _resolve_reviewed_operation();
	void _provider_openai();
	void _provider_fake();
	void _provider_status();
	void _intent_discuss();
	void _intent_propose();
	void _intent_execute();
	void _authorize_live_once();
	void _start_run();
	void _cancel_run();
	void _pause_run();
	void _resume_run();
	void _run_status();
	void _transition_execute();
	void _mode_manual();
	void _mode_protected();
	void _mode_freedom();
	void _on_response(const Dictionary &p_frame);
	void _notification(int p_what);

public:
	virtual String get_plugin_name() const override { return "Slime AI"; }
	SlimeAIEditorPlugin();
	~SlimeAIEditorPlugin();
};
