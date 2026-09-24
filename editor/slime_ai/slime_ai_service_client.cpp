/**************************************************************************/
/*  slime_ai_service_client.cpp                                         */
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

#include "slime_ai_service_client.h"

#include "core/io/json.h"
#include "core/config/project_settings.h"
#include "core/os/os.h"
#include "core/version.h"

namespace SlimeAI {

bool ServiceClient::start(const String &p_script_path) {
	stop();
	if (!OS::get_singleton()->get_environment("SLIME_AI_OPENAI_API_KEY").is_empty()) {
		state = "unavailable";
		last_error = "The editor cannot launch the service with a provider key in its environment. Use the user credential store instead.";
		return false;
	}
	if (!FileAccess::exists(p_script_path)) {
		state = "unavailable";
		last_error = "Fake service script is missing.";
		return false;
	}
	List<String> arguments;
	arguments.push_back("--disable-warning=ExperimentalWarning");
	arguments.push_back("--experimental-strip-types");
	arguments.push_back(p_script_path);
	Dictionary child = OS::get_singleton()->execute_with_pipe("node", arguments, false);
	if (!child.has("stdio") || !child.has("pid")) {
		state = "unavailable";
		last_error = "Node 24 could not start the fake service.";
		return false;
	}
	stdio = child["stdio"];
	stderr_pipe = child["stderr"];
	pid = ProcessID(int64_t(child["pid"]));
	state = "handshaking";
	Dictionary params;
	params["engine_revision"] = String(GODOT_VERSION_FULL_BUILD) + ":" + String(GODOT_VERSION_HASH);
	params["workspace_id"] = ProjectSettings::get_singleton()->get_resource_path().sha256_text();
	Array capabilities;
	capabilities.push_back("scene_inspection");
	params["client_capabilities"] = capabilities;
	return !request("hello", params).is_empty();
}

String ServiceClient::request(const String &p_method, const Dictionary &p_params) {
	if (stdio.is_null() || !pending_id.is_empty() || (p_method != "hello" && state != "ready")) {
		return String();
	}
	request_number++;
	const String id = String::num_uint64(request_number);
	Dictionary envelope;
	envelope["protocol_version"] = p_method.begins_with("run_") || p_method == "provider_status" ? "1.1" : "1.0";
	envelope["request_id"] = id;
	envelope["method"] = p_method;
	envelope["params"] = p_params;
	const CharString wire = (JSON::stringify(envelope) + "\n").utf8();
	if (wire.length() > MAX_FRAME_BYTES || !stdio->store_buffer((const uint8_t *)wire.get_data(), wire.length())) {
		state = "disconnected";
		last_error = "Could not send request to fake service.";
		return String();
	}
	pending_id = id;
	pending_method = p_method;
	return id;
}

void ServiceClient::poll(Vector<Dictionary> &r_responses) {
	if (stdio.is_null()) {
		return;
	}
	const bool exited = pid != 0 && !OS::get_singleton()->is_process_running(pid);
	if (!exited && stderr_pipe.is_valid()) {
		const uint64_t diagnostic_bytes = MIN(uint64_t(4096), stderr_pipe->get_length());
		if (diagnostic_bytes > 0) {
			Vector<uint8_t> diagnostic;
			diagnostic.resize(diagnostic_bytes);
			stderr_pipe->get_buffer(diagnostic.ptrw(), diagnostic_bytes);
			// Stderr remains diagnostic; never treat it as protocol or authorization.
		}
	}
	// An exited child's pipe can be read to EOF without PeekNamedPipe, whose
	// Windows implementation emits an error when the writer has already closed.
	const uint64_t available = exited ? 4096 : MIN(uint64_t(MAX_FRAME_BYTES), stdio->get_length());
	if (available == 0) {
		if (exited) {
			state = "disconnected";
			last_error = "Fake service exited before a complete response.";
			stdio.unref();
			stderr_pipe.unref();
		}
		return;
	}
	Vector<uint8_t> data;
	data.resize(available);
	const uint64_t count = stdio->get_buffer(data.ptrw(), available);
	if (exited && count == 0) {
		state = "disconnected";
		last_error = "Fake service exited before a complete response.";
		stdio.unref();
		stderr_pipe.unref();
		return;
	}
	Vector<Dictionary> frames;
	if (!decoder.feed(data.ptr(), count, frames)) {
		state = "protocol_error";
		last_error = decoder.get_failure();
		stop();
		return;
	}
	for (const Dictionary &frame : frames) {
		String validation_error;
		if (frame.has("event")) {
			if (!validate_run_event(frame, active_run_request_id, active_run_id, validation_error)) {
				state = "protocol_error";
				last_error = validation_error;
				stop();
				return;
			}
			r_responses.push_back(frame);
			if (String(frame["event"]) == "turn_failed") {
				active_run_id.clear();
				active_run_request_id.clear();
			}
			if (String(frame["event"]) == "run_state") {
				const String run_state = Dictionary(frame["data"]).get("state", "");
				if (run_state == "completed" || run_state == "failed" || run_state == "cancelled") {
					active_run_id.clear();
					active_run_request_id.clear();
				}
			}
			continue;
		}
		if (pending_id.is_empty() || !validate_envelope(frame, pending_id, validation_error)) {
			state = "protocol_error";
			last_error = validation_error;
			stop();
			return;
		}
		if (pending_method == "hello") {
			if (String(frame["status"]) != "ok") {
				state = "unavailable";
				last_error = "Fake service handshake failed.";
			} else {
				Dictionary result = frame["result"];
				if (!result.has("protocol_version") || String(result["protocol_version"]) != "1.0" || !result.has("capabilities") || result["capabilities"].get_type() != Variant::ARRAY || !Array(result["capabilities"]).has("fake_propose_scene_patch")) {
					state = "protocol_error";
					last_error = "Fake service capabilities are invalid.";
				} else {
					state = "ready";
				}
			}
		}
		if (pending_method.begins_with("run_") && String(frame["status"]) == "ok") {
			const Dictionary result = frame["result"];
			const String run_id = result.get("run_id", "");
			if (run_id.is_empty() || (!active_run_id.is_empty() && active_run_id != run_id)) {
				state = "protocol_error";
				last_error = "Run acknowledgement identity mismatch.";
				stop();
				return;
			}
			active_run_id = run_id;
			active_run_request_id = pending_id;
		}
		pending_id.clear();
		pending_method.clear();
		r_responses.push_back(frame);
	}
	if (exited && count < available) {
		state = "disconnected";
		last_error = "Fake service exited.";
		stdio.unref();
		stderr_pipe.unref();
	}
}

void ServiceClient::stop() {
	stdio.unref();
	stderr_pipe.unref();
	if (pid != 0 && OS::get_singleton()->is_process_running(pid)) {
		OS::get_singleton()->kill(pid);
	}
	pid = 0;
	pending_id.clear();
	pending_method.clear();
	active_run_request_id.clear();
	active_run_id.clear();
	if (state != "protocol_error") {
		state = "disconnected";
	}
	decoder.reset();
}

ServiceClient::~ServiceClient() {
	stop();
}

} // namespace SlimeAI
