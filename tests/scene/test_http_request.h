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

TEST_CASE("[Network][HTTPRequest] Set download chunk size") {
	HTTPClientMock *http_client = memnew(HTTPClientMock);
	HTTPRequest *http_request = memnew(HTTPRequest(Ref<HTTPClient>(http_client)));

	fakeit::Mock<HTTPClient> *spy = new fakeit::Mock<HTTPClient>(*http_client);
	fakeit::When(Method(*spy, get_status)).AlwaysReturn(HTTPClient::STATUS_DISCONNECTED);
	fakeit::Spy(Method(*spy, set_read_chunk_size));

	int expected_value = 42;
	http_request->set_download_chunk_size(expected_value);

	fakeit::Verify(Method(*spy, set_read_chunk_size).Using(expected_value));

	delete spy;
	memdelete(http_request);
}

} // namespace TestHTTPRequest

#endif // TEST_HTTP_REQUEST_H
