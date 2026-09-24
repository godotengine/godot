/**************************************************************************/
/*  slime_ai_protocol.cpp                                               */
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

#include "slime_ai_protocol.h"

#include "core/io/json.h"
#include "core/templates/hash_set.h"

namespace SlimeAI {

Dictionary error(const String &p_code, const String &p_message, const String &p_recovery) {
	Dictionary result;
	result["code"] = p_code;
	result["message"] = p_message;
	result["recovery"] = p_recovery;
	return result;
}

static bool _has_only(const Dictionary &p_dict, const Vector<String> &p_keys) {
	if (p_dict.size() != p_keys.size()) {
		return false;
	}
	for (const String &key : p_keys) {
		if (!p_dict.has(key)) {
			return false;
		}
	}
	return true;
}

bool validate_envelope(const Dictionary &p_frame, const String &p_request_id, String &r_error) {
	if (!p_frame.has("protocol_version") || p_frame["protocol_version"].get_type() != Variant::STRING || (String(p_frame["protocol_version"]) != "1.0" && String(p_frame["protocol_version"]) != "1.1")) {
		r_error = "Unsupported protocol version.";
		return false;
	}
	if (!p_frame.has("request_id") || p_frame["request_id"].get_type() != Variant::STRING || String(p_frame["request_id"]) != p_request_id) {
		r_error = "Response request ID mismatch.";
		return false;
	}
	if (!p_frame.has("status") || p_frame["status"].get_type() != Variant::STRING) {
		r_error = "Missing response status.";
		return false;
	}
	const String status = p_frame["status"];
	if (status == "ok") {
		if (!_has_only(p_frame, { "protocol_version", "request_id", "status", "result" }) || p_frame["result"].get_type() != Variant::DICTIONARY) {
			r_error = "Invalid success envelope.";
			return false;
		}
	} else if (status == "error") {
		if (!_has_only(p_frame, { "protocol_version", "request_id", "status", "error" }) || p_frame["error"].get_type() != Variant::DICTIONARY) {
			r_error = "Invalid failure envelope.";
			return false;
		}
		const Dictionary detail = p_frame["error"];
		if (!_has_only(detail, { "code", "message", "recovery" })) {
			r_error = "Invalid error detail.";
			return false;
		}
		for (const String &field : { "code", "message", "recovery" }) {
			if (detail[field].get_type() != Variant::STRING || String(detail[field]).length() > 4096) {
				r_error = "Invalid error field.";
				return false;
			}
		}
	} else {
		r_error = "Unknown response status.";
		return false;
	}
	return true;
}

bool validate_run_event(const Dictionary &p_frame, const String &p_request_id, const String &p_run_id, String &r_error) {
	if (!_has_only(p_frame, { "protocol_version", "request_id", "run_id", "event", "data" }) ||
			String(p_frame.get("protocol_version", "")) != "1.1" ||
			String(p_frame.get("request_id", "")) != p_request_id ||
			String(p_frame.get("run_id", "")) != p_run_id ||
			p_frame.get("data", Variant()).get_type() != Variant::DICTIONARY) {
		r_error = "Invalid run event envelope or identity.";
		return false;
	}
	const String event = p_frame["event"];
	if (event != "run_state" && event != "text_delta" && event != "usage_update" && event != "turn_completed" && event != "turn_failed" && event != "tool_call_ready") {
		r_error = "Unknown run event.";
		return false;
	}
	return true;
}

// Godot's JSON parser accepts duplicate object members. Reject them before parsing.
static bool _has_duplicate_keys(const String &p_json) {
	Vector<HashSet<String>> objects;
	int depth = 0;
	for (int i = 0; i < p_json.length(); i++) {
		const char32_t c = p_json[i];
		if (c == '"') {
			const int start = i++;
			for (; i < p_json.length(); i++) {
				if (p_json[i] == '\\') {
					i++;
				} else if (p_json[i] == '"') {
					break;
				}
			}
			if (i >= p_json.length()) {
				return true;
			}
			int next = i + 1;
			while (next < p_json.length() && (p_json[next] == ' ' || p_json[next] == '\t' || p_json[next] == '\r')) {
				next++;
			}
			if (next < p_json.length() && p_json[next] == ':' && !objects.is_empty()) {
				const Variant key = JSON::parse_string(p_json.substr(start, i - start + 1));
				if (key.get_type() != Variant::STRING || objects.write[objects.size() - 1].has(String(key))) {
					return true;
				}
				objects.write[objects.size() - 1].insert(String(key));
			}
		} else if (c == '{') {
			objects.push_back(HashSet<String>());
			depth++;
		} else if (c == '}') {
			if (depth > 0) {
				objects.remove_at(objects.size() - 1);
				depth--;
			}
		}
	}
	return false;
}

bool FrameDecoder::feed(const uint8_t *p_bytes, int p_count, Vector<Dictionary> &r_frames) {
	if (failed || p_count < 0 || (p_count > 0 && p_bytes == nullptr)) {
		return false;
	}
	for (int i = 0; i < p_count; i++) {
		pending.push_back(p_bytes[i]);
		if (pending.size() > MAX_FRAME_BYTES) {
			failed = true;
			failure = "Frame exceeds 1 MiB.";
			return false;
		}
		if (p_bytes[i] == '\n') {
			String json;
			const Error utf8_error = json.append_utf8((const char *)pending.ptr(), pending.size() - 1);
			pending.clear();
			if (utf8_error != OK || _has_duplicate_keys(json)) {
				failed = true;
				failure = "Frame has invalid UTF-8 or duplicate keys.";
				return false;
			}
			Ref<JSON> parser;
			parser.instantiate();
			if (parser->parse(json) != OK || parser->get_data().get_type() != Variant::DICTIONARY) {
				failed = true;
				failure = "Frame is not a JSON object.";
				return false;
			}
			r_frames.push_back(parser->get_data());
		}
	}
	return true;
}

void FrameDecoder::reset() {
	pending.clear();
	failed = false;
	failure.clear();
}

} // namespace SlimeAI
