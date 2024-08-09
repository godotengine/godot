/**************************************************************************/
/*  test_http_request.h                                                   */
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

#ifndef TEST_HTTP_REQUEST_H
#define TEST_HTTP_REQUEST_H

#include "scene/main/http_request.h"
#include "tests/core/io/test_http_client_mock.h"
#include "tests/test_macros.h"

#include "thirdparty/fakeit/fakeit.hpp"

namespace TestHTTPRequest {

static HTTPClientMock *http_client = nullptr;
HTTPClient *_create_func() {
	http_client = memnew(HTTPClientMock);
	return http_client;
}
}; //namespace TestHTTPRequest

HTTPClient *(*HTTPClient::_create)() = TestHTTPRequest::_create_func;

namespace TestHTTPRequest {

TEST_CASE("[Network][HTTPRequest] Download chunk size") {
	HTTPRequest *http_request = memnew(HTTPRequest);

	auto mock = new fakeit::Mock<HTTPClient>(*http_client);

	int expected_value = 42;

	SUBCASE("is set when HTTP client is disconnected") {
		mock->Reset();

		fakeit::When(Method(*mock, get_status)).AlwaysReturn(HTTPClient::STATUS_DISCONNECTED);
		fakeit::Spy(Method(*mock, set_read_chunk_size));

		http_request->set_download_chunk_size(expected_value);

		fakeit::Verify(Method(*mock, set_read_chunk_size).Using(expected_value));
	}

	SUBCASE("is not set when HTTP client is not disconnected") {
		mock->Reset();

		fakeit::When(Method(*mock, get_status)).AlwaysReturn(HTTPClient::STATUS_CONNECTED);
		fakeit::Spy(Method(*mock, set_read_chunk_size));

		ERR_PRINT_OFF;
		http_request->set_download_chunk_size(expected_value);
		ERR_PRINT_ON;

		fakeit::Verify(Method(*mock, set_read_chunk_size).Using(expected_value)).Never();
	}

	delete mock;
	memdelete(http_request);
}

} // namespace TestHTTPRequest

#endif // TEST_HTTP_REQUEST_H
