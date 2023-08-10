/**************************************************************************/
/*  test_jsonrpc.h                                                        */
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

#ifndef TEST_JSONRPC_H
#define TEST_JSONRPC_H

#include "modules/jsonrpc/jsonrpc.h"

#include "tests/test_macros.h"

namespace TestJSONRPC {

TEST_CASE("[JSONRPC] Make Response Error") {
	JSONRPC jsonrpc;

	// Create an error response dictionary.
	Dictionary error_dict = jsonrpc.make_response_error(JSONRPC::PARSE_ERROR, "Parse error");

	// Check the structure and contents of the error dictionary.
	CHECK(error_dict.has("jsonrpc"));
	CHECK(error_dict.has("error"));
	CHECK(error_dict.has("id"));
	CHECK(error_dict["jsonrpc"] == "2.0");
	Dictionary error = error_dict["error"];
	CHECK(error.has("code"));
	CHECK(error.has("message"));
	CHECK(error["code"] == Variant((int)JSONRPC::PARSE_ERROR));
	CHECK(error["message"] == "Parse error");
}

TEST_CASE("[JSONRPC] Make Response") {
	JSONRPC jsonrpc;

	// Create a response dictionary with sample values.
	Variant response_value = "my value";
	Variant id = 12345;
	Dictionary response_dict = jsonrpc.make_response(response_value, id);

	// Check the structure and contents of the response dictionary.
	CHECK(response_dict.has("jsonrpc"));
	CHECK(response_dict.has("result"));
	CHECK(response_dict.has("id"));
	CHECK(response_dict["jsonrpc"] == "2.0");
	CHECK(response_dict["result"] == response_value);
	CHECK(response_dict["id"] == id);
}

TEST_CASE("[JSONRPC] Make Notification") {
	JSONRPC jsonrpc;

	// Create a notification dictionary with sample values.
	String method = "my_method";
	Variant params = "my params";
	Dictionary notification_dict = jsonrpc.make_notification(method, params);

	// Check the structure and contents of the notification dictionary.
	CHECK_FALSE(notification_dict.has("id"));
	CHECK(notification_dict.has("jsonrpc"));
	CHECK(notification_dict.has("method"));
	CHECK(notification_dict.has("params"));
	CHECK(notification_dict["jsonrpc"] == "2.0");
	CHECK(notification_dict["method"] == method);
	CHECK(notification_dict["params"] == params);
}

TEST_CASE("[JSONRPC] Make Request") {
	JSONRPC jsonrpc;

	// Create a request dictionary with sample values.
	String method = "my_method";
	Variant params = "my params";
	Variant id = 54321;
	Dictionary request_dict = jsonrpc.make_request(method, params, id);

	// Check the structure and contents of the request dictionary.
	CHECK(request_dict.has("jsonrpc"));
	CHECK(request_dict.has("method"));
	CHECK(request_dict.has("params"));
	CHECK(request_dict.has("id"));
	CHECK(request_dict["jsonrpc"] == "2.0");
	CHECK(request_dict["method"] == method);
	CHECK(request_dict["params"] == params);
	CHECK(request_dict["id"] == id);
}

} // namespace TestJSONRPC

#endif // TEST_JSONRPC_H
