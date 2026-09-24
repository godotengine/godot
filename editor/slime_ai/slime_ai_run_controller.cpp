#include "slime_ai_run_controller.h"

#include "core/config/project_settings.h"
#include "core/io/json.h"
#include "core/os/os.h"
#include "editor/slime_ai/slime_ai_api_describer.h"
#include "editor/slime_ai/slime_ai_code_reader.h"
#include "editor/slime_ai/slime_ai_object_inspector.h"
#include "editor/slime_ai/slime_ai_project_inspector.h"
#include "editor/slime_ai/slime_ai_project_search.h"
#include "editor/slime_ai/slime_ai_protocol.h"
#include "editor/slime_ai/slime_ai_scene_inspector.h"
#include "editor/slime_ai/slime_ai_scene_transaction.h"
#include "editor/slime_ai/slime_ai_service_client.h"

namespace SlimeAI {

static Dictionary _failure(const String &p_code, const String &p_message) {
	Dictionary result;
	result["status"] = "error";
	result["error"] = error(p_code, p_message, "Inspect the current scope and retry only after a safe boundary.");
	return result;
}

static bool _keys(const Dictionary &p_args, const Vector<String> &p_expected) {
	if (p_args.size() != p_expected.size()) {
		return false;
	}
	for (const String &key : p_expected) {
		if (!p_args.has(key)) {
			return false;
		}
	}
	return true;
}

RunController::RunController(ServiceClient &p_service, SceneTransaction &p_transaction) :
		service(p_service), transaction(p_transaction) {
}

Dictionary RunController::start(Node *p_root, bool p_editor_unsaved, const String &p_provider, const String &p_model, const String &p_intent, const String &p_prompt, bool p_live_authorized) {
	if (!service.is_ready() || !p_root || (state != "idle" && state != "completed" && state != "failed" && state != "cancelled")) {
		return _failure("RUN_BUSY", "A ready service and one loaded scene are required; finish the current run first.");
	}
	if ((p_provider != "fake" && p_provider != "openai_responses") || (p_intent != "discuss" && p_intent != "propose" && p_intent != "execute") || p_prompt.strip_edges().is_empty()) {
		return _failure("INVALID_ARGUMENT", "Choose a supported provider, intent, and nonempty task.");
	}
	if ((p_provider == "fake" && (p_live_authorized || !p_model.is_empty())) || (p_provider == "openai_responses" && (!p_live_authorized || p_model.strip_edges().is_empty()))) {
		return _failure("LIVE_AUTHORIZATION_REQUIRED", "Live OpenAI requires an explicit model and one-run authorization.");
	}
	const Dictionary inspection = SceneInspector::inspect(p_root, p_editor_unsaved);
	if (inspection.has("error")) {
		return _failure(String(inspection["error"]), "The selected scene could not be inspected.");
	}
	const String unresolved = transaction.unresolved_operation(p_root);
	if (!unresolved.is_empty() && p_intent != "discuss") {
		return _failure("RECONCILIATION_REQUIRED", "Resolve operation " + unresolved + " before another proposal or edit.");
	}
	Dictionary context;
	context["scene_ref"] = inspection["scene_ref"];
	context["base_revision"] = inspection["revision"];
	context["parent_ref"] = inspection["root_ref"];
	context["root_class"] = inspection["root_class"];
	context["scene_path"] = inspection["scene_path"];
	context["editor_unsaved"] = inspection["editor_unsaved"];
	context["selected_scene_summary"] = inspection["nodes"];
	const String serialized_context = JSON::stringify(context);
	if (serialized_context.utf8().length() > 32768 || p_prompt.utf8().length() > 16384) {
		return _failure("INVALID_ARGUMENT", "Task or selected scene context exceeds the bounded run limit.");
	}
	sequence++;
	run_id = vformat("native-run-%d-%d", OS::get_singleton()->get_ticks_usec(), sequence);
	Dictionary limits;
	limits["max_attempts"] = 3;
	limits["max_tool_calls"] = 4;
	limits["max_output_tokens"] = 1024;
	limits["request_timeout_ms"] = 30000;
	limits["deadline_ms"] = 120000;
	Dictionary params;
	params["run_id"] = run_id;
	params["provider"] = p_provider;
	params["model"] = p_provider == "fake" ? Variant() : Variant(p_model);
	params["intent"] = p_intent;
	params["prompt"] = p_prompt;
	params["context"] = serialized_context;
	params["limits"] = limits;
	params["live_authorized"] = p_live_authorized;
	const String request_id = service.request("run_start", params);
	if (request_id.is_empty()) {
		return _failure("CAPABILITY_UNAVAILABLE", "Could not start the bounded service run.");
	}
	provider = p_provider;
	model = p_model;
	intent = p_intent;
	scene_ref = inspection["scene_ref"];
	scene_path = inspection["scene_path"];
	state = "waiting_for_provider";
	preview_id.clear();
	preview_base_revision.clear();
	operation_id.clear();
	model_text.clear();
	usage.clear();
	preview_record.clear();
	save_fact.clear();
	check_fact = "not_run";
	tool_calls = 0;
	cancelled = false;
	cancel_pending = false;
	paused = false;
	completed_turn_has_call = false;
	pending_tool_frame.clear();
	Dictionary result = status(p_root);
	result["service_request_id"] = request_id;
	Array context_manifest;
	context_manifest.push_back("selected loaded scene summary");
	context_manifest.push_back("scene revision");
	context_manifest.push_back("root reference");
	result["context_manifest"] = context_manifest;
	return result;
}

Dictionary RunController::dispatch_tool(Node *p_root, bool p_editor_unsaved, const String &p_name, const Dictionary &p_args) {
	if (!p_root || SceneInspector::scene_ref(p_root) != scene_ref || p_root->get_scene_file_path() != scene_path) {
		return _failure("STALE_REFERENCE", "The loaded scene changed during the run.");
	}
	if (p_name == "project_inspect" && _keys(p_args, {})) {
		return ProjectInspector::inspect();
	}
	if (p_name == "scene_inspect" && _keys(p_args, {})) {
		return SceneInspector::inspect(p_root, p_editor_unsaved);
	}
	if (p_name == "object_inspect" && _keys(p_args, { "node_ref" })) {
		return ObjectInspector::inspect(p_root, String(p_args["node_ref"]), p_editor_unsaved);
	}
	if (p_name == "api_describe" && _keys(p_args, { "class_name", "property" })) {
		return ApiDescriber::describe(StringName(String(p_args["class_name"])), StringName(String(p_args["property"])));
	}
	if (p_name == "project_search" && _keys(p_args, { "query", "page" })) {
		return ProjectSearch::search(String(p_args["query"]), int(p_args["page"]) * 20, 20);
	}
	if (p_name == "code_read" && _keys(p_args, { "path", "start_line", "line_count" })) {
		return CodeReader::read(String(p_args["path"]), int(p_args["start_line"]), int(p_args["line_count"]));
	}
	if (p_name == "changeset_status" && _keys(p_args, { "operation_id" })) {
		if (operation_id.is_empty() || String(p_args["operation_id"]) != operation_id) {
			return _failure("PERMISSION_DENIED", "This run may inspect only its own operation.");
		}
		return transaction.status(p_root, operation_id);
	}
	if (p_name == "scene_patch_preview" && _keys(p_args, { "scene_ref", "base_revision", "operations" })) {
		if (intent == "discuss") {
			return _failure("PERMISSION_DENIED", "Discuss cannot preview a scene mutation.");
		}
		if (p_args["operations"].get_type() != Variant::ARRAY || Array(p_args["operations"]).size() != 1) {
			return _failure("UNSUPPORTED_OPERATION", "One scene operation is allowed per transaction.");
		}
		sequence++;
		operation_id = vformat("native-run-op-%d-%d", OS::get_singleton()->get_ticks_usec(), sequence);
		Dictionary preview = transaction.preview(p_root, operation_id, p_args, p_editor_unsaved);
		preview_id = preview.get("preview_id", "");
		preview_base_revision = preview.get("base_revision", "");
		preview_record = preview;
		return preview;
	}
	return _failure("UNKNOWN_TOOL", "Unsupported or malformed tool request.");
}

Dictionary RunController::on_frame(Node *p_root, bool p_editor_unsaved, const Dictionary &p_frame) {
	if (p_frame.has("status")) {
		if (cancel_pending && String(p_frame.get("status", "")) == "ok") {
			Dictionary params;
			params["run_id"] = run_id;
			cancel_pending = service.request("run_cancel", params).is_empty();
		}
		return p_frame;
	}
	if (!p_frame.has("event") || !p_frame.has("data") || p_frame["data"].get_type() != Variant::DICTIONARY || String(p_frame.get("run_id", "")) != run_id) {
		return _failure("PROVIDER_PROTOCOL_ERROR", "Run event identity mismatch.");
	}
	const String event = p_frame["event"];
	const Dictionary data = p_frame["data"];
	if (cancelled) {
		return status(p_root);
	}
	if (event == "text_delta") {
		const String delta = data.get("text", "");
		if (model_text.length() + delta.length() <= 16384) {
			model_text += delta;
		}
	} else if (event == "usage_update") {
		usage = data;
	} else if (event == "run_state") {
		state = data.get("state", "unknown");
		if (state == "failed" || state == "cancelled" || state == "completed") {
			completed_turn_has_call = false;
		}
	} else if (event == "turn_completed") {
		completed_turn_has_call = bool(data.get("provider_turn_complete", false)) && bool(data.get("has_tool_call", false));
	} else if (event == "turn_failed") {
		state = "failed";
		completed_turn_has_call = false;
	} else if (event == "tool_call_ready") {
		if (!completed_turn_has_call) {
			return _failure("PROVIDER_PROTOCOL_ERROR", "Tool call arrived without a complete successful provider turn.");
		}
		if (paused) {
			pending_tool_frame = p_frame;
			Dictionary result = status(p_root);
			result["status"] = "paused_before_tool";
			return result;
		}
		completed_turn_has_call = false;
		if (tool_calls >= 4 || state == "failed" || state == "cancelled" || !_keys(data, { "call_id", "tool_name", "arguments", "provider_response_id" }) || data["arguments"].get_type() != Variant::DICTIONARY) {
			return _failure("PROVIDER_PROTOCOL_ERROR", "Unexpected tool call event.");
		}
		tool_calls++;
		const Dictionary tool_result = dispatch_tool(p_root, p_editor_unsaved, data["tool_name"], data["arguments"]);
		Dictionary continuation;
		continuation["status"] = String(tool_result.get("status", "")) == "error" || tool_result.has("error") ? "error" : "ok";
		continuation["result"] = tool_result;
		Dictionary params;
		params["run_id"] = run_id;
		params["call_id"] = data["call_id"];
		params["tool_result"] = continuation;
		if (service.request("run_continue", params).is_empty()) {
			state = "reconciliation_required";
			return _failure("APPLY_FAILED_RECOVERY_REQUIRED", "Tool result could not be returned to the service; inspect native status before retrying.");
		}
		Dictionary result = status(p_root);
		result["tool_result"] = tool_result;
		return result;
	}
	return status(p_root);
}

Dictionary RunController::cancel() {
	if (run_id.is_empty() || state == "completed" || state == "failed" || state == "cancelled") {
		return _failure("STALE_REFERENCE", "No active run to cancel.");
	}
	cancelled = true;
	state = "cancelled";
	completed_turn_has_call = false;
	pending_tool_frame.clear();
	Dictionary params;
	params["run_id"] = run_id;
	const String request_id = service.request("run_cancel", params);
	cancel_pending = request_id.is_empty();
	Dictionary result = status(nullptr);
	result["service_cancel_sent"] = !request_id.is_empty();
	return result;
}

Dictionary RunController::pause() {
	if (run_id.is_empty() || state == "completed" || state == "failed" || state == "cancelled") {
		return _failure("STALE_REFERENCE", "No active run to pause.");
	}
	paused = true;
	Dictionary result = status(nullptr);
	result["status"] = "pause_requested";
	return result;
}

Dictionary RunController::resume(Node *p_root, bool p_editor_unsaved) {
	if (!paused || cancelled) {
		return _failure("STALE_REFERENCE", "No paused tool is available.");
	}
	paused = false;
	if (!pending_tool_frame.is_empty()) {
		const Dictionary frame = pending_tool_frame.duplicate(true);
		pending_tool_frame.clear();
		return on_frame(p_root, p_editor_unsaved, frame);
	}
	return status(p_root);
}

Dictionary RunController::transition_to_execute(Node *p_root, bool p_editor_unsaved) {
	if (intent != "propose" || preview_id.is_empty() || !p_root || SceneInspector::scene_ref(p_root) != scene_ref) {
		return _failure("PERMISSION_DENIED", "No reviewed proposal is available for execution.");
	}
	const Dictionary inspection = SceneInspector::inspect(p_root, p_editor_unsaved);
	const Dictionary preview_status = transaction.status(p_root, operation_id);
	if (inspection.has("error") || String(inspection.get("revision", "")) != preview_base_revision || (!preview_status.has("error") && String(preview_status.get("status", "")) == "prepared")) {
		return _failure("REVISION_CONFLICT", "The scene or operation status requires a new review.");
	}
	intent = "execute";
	Dictionary result = status(p_root);
	result["status"] = "execute_transition_ready";
	result["revalidate_on_apply"] = true;
	return result;
}

Dictionary RunController::grant() {
	if (intent != "execute" || preview_id.is_empty()) {
		return _failure("PERMISSION_DENIED", "Execution intent and a visible preview are required.");
	}
	return transaction.grant(preview_id);
}

Dictionary RunController::apply(Node *p_root, bool p_editor_unsaved) {
	if (intent != "execute" || preview_id.is_empty() || !p_root || SceneInspector::scene_ref(p_root) != scene_ref || p_root->get_scene_file_path() != scene_path) {
		return _failure("PERMISSION_DENIED", "Execution intent and the original loaded scene are required.");
	}
	Dictionary result = transaction.apply(p_root, preview_id, p_editor_unsaved);
	if (String(result.get("status", "")) == "applied") {
		const String node_ref = result.get("node_ref", "");
		check_fact = !node_ref.is_empty() && SceneInspector::resolve(p_root, node_ref) ? "structural_inspection_passed" : "not_run";
		preview_id.clear();
	}
	return result;
}

void RunController::note_save(int p_save_error, const String &p_disk_before, const String &p_disk_after, bool p_editor_unsaved, bool p_injected) {
	save_fact["status"] = p_save_error == OK && !p_editor_unsaved ? "confirmed" : "failed";
	save_fact["error"] = p_save_error;
	save_fact["disk_before"] = p_disk_before;
	save_fact["disk_after"] = p_disk_after;
	save_fact["editor_unsaved"] = p_editor_unsaved;
	save_fact["injected_failure"] = p_injected;
}

Dictionary RunController::status(Node *p_root) const {
	Dictionary result;
	result["status"] = state;
	result["run_id"] = run_id;
	result["provider"] = provider;
	result["model"] = model.is_empty() ? Variant() : Variant(model);
	result["intent"] = intent;
	result["scene_path"] = scene_path;
	result["scene_ref"] = scene_ref;
	result["model_text_untrusted"] = model_text;
	result["usage"] = usage.is_empty() ? Variant() : Variant(usage);
	result["tool_calls"] = tool_calls;
	result["paused"] = paused;
	result["limits"] = "3 attempts, 4 tool calls, 1024 output tokens, 120-second deadline";
	result["preview_id"] = preview_id;
	result["preview"] = preview_record;
	result["operation_id"] = operation_id;
	result["edit"] = operation_id.is_empty() ? Variant("not_requested") : (!preview_id.is_empty() ? Variant("preview_only") : Variant(transaction.status(p_root, operation_id)));
	result["save"] = save_fact.is_empty() ? Variant("not_requested") : Variant(save_fact);
	result["check"] = check_fact;
	result["gameplay_verification"] = "not_run";
	return result;
}

} // namespace SlimeAI
