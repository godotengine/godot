/**************************************************************************/
/*  http_request.hpp                                                      */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#pragma once

#include <godot_cpp/classes/global_constants.hpp>
#include <godot_cpp/classes/http_client.hpp>
#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/variant/packed_byte_array.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>
#include <godot_cpp/variant/string.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class TLSOptions;

class HTTPRequest : public Node {
	GDEXTENSION_CLASS(HTTPRequest, Node)

public:
	enum Result {
		RESULT_SUCCESS = 0,
		RESULT_CHUNKED_BODY_SIZE_MISMATCH = 1,
		RESULT_CANT_CONNECT = 2,
		RESULT_CANT_RESOLVE = 3,
		RESULT_CONNECTION_ERROR = 4,
		RESULT_TLS_HANDSHAKE_ERROR = 5,
		RESULT_NO_RESPONSE = 6,
		RESULT_BODY_SIZE_LIMIT_EXCEEDED = 7,
		RESULT_BODY_DECOMPRESS_FAILED = 8,
		RESULT_REQUEST_FAILED = 9,
		RESULT_DOWNLOAD_FILE_CANT_OPEN = 10,
		RESULT_DOWNLOAD_FILE_WRITE_ERROR = 11,
		RESULT_REDIRECT_LIMIT_REACHED = 12,
		RESULT_TIMEOUT = 13,
	};

	Error request(const String &p_url, const PackedStringArray &p_custom_headers = PackedStringArray(), HTTPClient::Method p_method = (HTTPClient::Method)0, const String &p_request_data = String());
	Error request_raw(const String &p_url, const PackedStringArray &p_custom_headers = PackedStringArray(), HTTPClient::Method p_method = (HTTPClient::Method)0, const PackedByteArray &p_request_data_raw = PackedByteArray());
	void cancel_request();
	void set_tls_options(const Ref<TLSOptions> &p_client_options);
	HTTPClient::Status get_http_client_status() const;
	void set_use_threads(bool p_enable);
	bool is_using_threads() const;
	void set_accept_gzip(bool p_enable);
	bool is_accepting_gzip() const;
	void set_body_size_limit(int32_t p_bytes);
	int32_t get_body_size_limit() const;
	void set_max_redirects(int32_t p_amount);
	int32_t get_max_redirects() const;
	void set_download_file(const String &p_path);
	String get_download_file() const;
	int32_t get_downloaded_bytes() const;
	int32_t get_body_size() const;
	void set_timeout(double p_timeout);
	double get_timeout();
	void set_download_chunk_size(int32_t p_chunk_size);
	int32_t get_download_chunk_size() const;
	void set_http_proxy(const String &p_host, int32_t p_port);
	void set_https_proxy(const String &p_host, int32_t p_port);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Node::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(HTTPRequest::Result);

