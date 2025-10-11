/**************************************************************************/
/*  jsonrpc.cpp                                                           */
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

#include "jsonrpc.h"
#include "jsonrpc.compat.inc"

#include "core/io/json.h"

JSONRPC::JSONRPC() {
}

JSONRPC::~JSONRPC() {
}

void JSONRPC::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_method", "name", "callback"), &JSONRPC::set_method);
	ClassDB::bind_method(D_METHOD("process_action", "action", "recurse"), &JSONRPC::process_action, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("process_string", "action"), &JSONRPC::process_string);

	ClassDB::bind_method(D_METHOD("make_request", "method", "params", "id"), &JSONRPC::make_request);
	ClassDB::bind_method(D_METHOD("make_response", "result", "id"), &JSONRPC::make_response);
	ClassDB::bind_method(D_METHOD("make_notification", "method", "params"), &JSONRPC::make_notification);
	ClassDB::bind_method(D_METHOD("make_response_error", "code", "message", "id"), &JSONRPC::make_response_error, DEFVAL(Variant()));

	BIND_ENUM_CONSTANT(PARSE_ERROR);
	BIND_ENUM_CONSTANT(INVALID_REQUEST);
	BIND_ENUM_CONSTANT(METHOD_NOT_FOUND);
	BIND_ENUM_CONSTANT(INVALID_PARAMS);
	BIND_ENUM_CONSTANT(INTERNAL_ERROR);
}

Dictionary JSONRPC::make_response_error(int p_code, const String &p_message, const Variant &p_id) const {
	Dictionary dict;
	dict["jsonrpc"] = "2.0";

	Dictionary err;
	err["code"] = p_code;
	err["message"] = p_message;

	dict["error"] = err;
	dict["id"] = p_id;

	return dict;
}

Dictionary JSONRPC::make_response(const Variant &p_value, const Variant &p_id) {
	Dictionary dict;
	dict["jsonrpc"] = "2.0";
	dict["id"] = p_id;
	dict["result"] = p_value;
	return dict;
}

Dictionary JSONRPC::make_notification(const String &p_method, const Variant &p_params) {
	Dictionary dict;
	dict["jsonrpc"] = "2.0";
	dict["method"] = p_method;
	dict["params"] = p_params;
	return dict;
}

Dictionary JSONRPC::make_request(const String &p_method, const Variant &p_params, const Variant &p_id) {
	Dictionary dict;
	dict["jsonrpc"] = "2.0";
	dict["method"] = p_method;
	dict["params"] = p_params;
	dict["id"] = p_id;
	return dict;
}

inline bool is_response(const Variant &p_obj) {
	return p_obj.get_type() == Variant::DICTIONARY && !p_obj.operator Dictionary().has("method");
}

Variant JSONRPC::process_action(const Variant &p_action, bool p_process_arr_elements) {
	if (p_action.get_type() == Variant::DICTIONARY) {
		if (!is_response(p_action)) {
			return process_request(p_action);
		}
		process_response(p_action);
	} else if (p_action.get_type() == Variant::ARRAY && p_process_arr_elements) {
		Array arr = p_action;
		if (!arr.is_empty()) {
			Array arr_ret;

			bool response_mode = is_response(arr[0]);

			for (const Variant &var : arr) {
				if (is_response(var) != response_mode) {
					return make_response_error(JSONRPC::INVALID_REQUEST, "Batch Request with mixed response and request objects.");
				}

				Variant res = process_action(var);

				if (res.get_type() != Variant::NIL) {
					arr_ret.push_back(res);
				}
			}

			/// If there are no Response objects contained within the Response array as it is to be sent to the client, the server MUST NOT return an empty Array and should return nothing at all.
			if (!arr_ret.is_empty()) {
				return arr_ret;
			}
		} else {
			/// If the batch rpc call itself fails to be recognized ... as an Array with at least one value, the response from the Server MUST be a single Response object.
			return make_response_error(JSONRPC::INVALID_REQUEST, "Invalid Request");
		}
	} else {
		return make_response_error(JSONRPC::INVALID_REQUEST, "Invalid Request");
	}
	return Variant();
}

Variant JSONRPC::process_request(const Dictionary &p_request_object) {
	ERR_FAIL_COND_V(!p_request_object.has("method"), Variant());

	Variant ret;
	String method = p_request_object["method"];

	Array args;
	if (p_request_object.has("params")) {
		Variant params = p_request_object["params"];
		if (params.get_type() == Variant::ARRAY) {
			args = params;
		} else {
			args.push_back(params);
		}
	}

	/// A Notification is a Request object without an "id" member.
	bool is_notification = !p_request_object.has("id");

	Variant id;
	if (!is_notification) {
		id = p_request_object["id"];

		// Account for implementations that discern between int and float on the json serialization level, by using an int if there is a .0 fraction. See #100914
		if (id.get_type() == Variant::FLOAT && id.operator float() == (float)(id.operator int())) {
			id = id.operator int();
		}
	}

	if (methods.has(method)) {
		Variant call_ret = methods[method].callv(args);
		ret = make_response(call_ret, id);
	} else {
		ret = make_response_error(JSONRPC::METHOD_NOT_FOUND, "Method not found: " + method, id);
	}

	/// The Server MUST NOT reply to a Notification
	if (is_notification) {
		ret = Variant();
	}
	return ret;
}

void JSONRPC::process_response(const Dictionary &p_response_object) {
	ERR_FAIL_COND_MSG(!p_response_object.has("id"), "JSONRPC: Received response object without id.");

	const Variant id_var = p_response_object["id"];
	ERR_FAIL_COND_MSG(!id_var.is_num(), "JSONRPC: No response handler registered for id: " + id_var.operator String());
	const int id = id_var.operator int();
	ERR_FAIL_COND_MSG(!response_handlers.has(id), "JSONRPC: No response handler registered for id: " + id_var.operator String());

	if (p_response_object.has("result")) {
		response_handlers[id].call(p_response_object["result"]);
	} else if (p_response_object.has("error")) {
		ERR_PRINT("JSONRPC: Received error response object: " + p_response_object["error"].to_json_string());
	} else {
		ERR_PRINT("JSONRPC: Received response object without 'result' or 'error' field.");
	}

	response_handlers.erase(id);
}

String JSONRPC::process_string(const String &p_input) {
	if (p_input.is_empty()) {
		return String();
	}

	Variant ret;
	JSON json;
	if (json.parse(p_input) == OK) {
		ret = process_action(json.get_data(), true);
	} else {
		ret = make_response_error(JSONRPC::PARSE_ERROR, "Parse error");
	}

	if (ret.get_type() == Variant::NIL) {
		return "";
	}
	return ret.to_json_string();
}

void JSONRPC::set_method(const String &p_name, const Callable &p_callback) {
	methods[p_name] = p_callback;
}

void JSONRPC::set_response_handler(int p_id, const Callable &p_callback) {
	response_handlers[p_id] = p_callback;
}
