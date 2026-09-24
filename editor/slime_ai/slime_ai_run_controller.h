#pragma once

#include "core/variant/dictionary.h"
#include "core/string/ustring.h"

class Node;

namespace SlimeAI {

class SceneTransaction;
class ServiceClient;

// Serial host authority for a single provider run. Provider text and tool
// arguments never grant a scene change or choose a project root.
class RunController {
	ServiceClient &service;
	SceneTransaction &transaction;
	String run_id;
	String scene_ref;
	String scene_path;
	String intent = "discuss";
	String provider = "openai_responses";
	String model;
	String state = "idle";
	String preview_id;
	String preview_base_revision;
	String operation_id;
	String model_text;
	Dictionary usage;
	Dictionary preview_record;
	Dictionary save_fact;
	String check_fact = "not_run";
	Dictionary pending_tool_frame;
	uint64_t sequence = 0;
	int tool_calls = 0;
	bool cancelled = false;
	bool cancel_pending = false;
	bool paused = false;
	bool completed_turn_has_call = false;

	Dictionary dispatch_tool(Node *p_root, bool p_editor_unsaved, const String &p_name, const Dictionary &p_args);

public:
	RunController(ServiceClient &p_service, SceneTransaction &p_transaction);
	Dictionary start(Node *p_root, bool p_editor_unsaved, const String &p_provider, const String &p_model, const String &p_intent, const String &p_prompt, bool p_live_authorized);
	Dictionary on_frame(Node *p_root, bool p_editor_unsaved, const Dictionary &p_frame);
	Dictionary cancel();
	Dictionary pause();
	Dictionary resume(Node *p_root, bool p_editor_unsaved);
	Dictionary transition_to_execute(Node *p_root, bool p_editor_unsaved);
	Dictionary grant();
	Dictionary apply(Node *p_root, bool p_editor_unsaved);
	Dictionary status(Node *p_root) const;
	void note_save(int p_save_error, const String &p_disk_before, const String &p_disk_after, bool p_editor_unsaved, bool p_injected);
	String get_preview_id() const { return preview_id; }
	String get_operation_id() const { return operation_id; }
	String get_intent() const { return intent; }
};

} // namespace SlimeAI
