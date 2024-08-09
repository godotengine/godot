/**************************************************************************/
/*  test_http_client_mock.h                                               */
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

#ifndef TEST_HTTP_CLIENT_MOCK_H
#define TEST_HTTP_CLIENT_MOCK_H

#include "core/io/http_client.h"

class HTTPClientMock : public HTTPClient {
public:
	Error request(Method p_method, const String &p_url, const Vector<String> &p_headers, const uint8_t *p_body, int p_body_size) override {
		return Error();
	}
	Error connect_to_host(const String &p_host, int p_port = -1, Ref<TLSOptions> p_tls_options = Ref<TLSOptions>()) override {
		return Error();
	}

	void set_connection(const Ref<StreamPeer> &p_connection) override {
	}
	Ref<StreamPeer> get_connection() const override {
		return nullptr;
	}

	void close() override {
	}

	Status get_status() const override {
		return Status();
	}

	bool has_response() const override {
		return bool();
	}
	bool is_response_chunked() const override {
		return bool();
	}
	int get_response_code() const override {
		return int();
	}
	Error get_response_headers(List<String> *r_response) override {
		return Error();
	}
	int64_t get_response_body_length() const override {
		return int64_t();
	}

	PackedByteArray read_response_body_chunk() override {
		return PackedByteArray();
	}

	void set_blocking_mode(bool p_enable) override {
	}
	bool is_blocking_mode_enabled() const override {
		return bool();
	}

	void set_read_chunk_size(int p_size) override {
	}
	int get_read_chunk_size() const override {
		return int();
	}

	Error poll() override {
		return Error();
	}
};

#endif // TEST_HTTP_CLIENT_MOCK_H
