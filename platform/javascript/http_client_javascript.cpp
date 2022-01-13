/*************************************************************************/
/*  http_client_javascript.cpp                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "core/io/http_client.h"

#ifdef __cplusplus
extern "C" {
#endif

#include "stddef.h"

typedef enum {
	GODOT_JS_FETCH_STATE_REQUESTING = 0,
	GODOT_JS_FETCH_STATE_BODY = 1,
	GODOT_JS_FETCH_STATE_DONE = 2,
	GODOT_JS_FETCH_STATE_ERROR = -1,
} godot_js_fetch_state_t;

extern int godot_js_fetch_create(const char *p_method, const char *p_url, const char **p_headers, int p_headers_len, const uint8_t *p_body, int p_body_len);
extern int godot_js_fetch_read_headers(int p_id, void (*parse_callback)(int p_size, const char **p_headers, void *p_ref), void *p_ref);
extern int godot_js_fetch_read_chunk(int p_id, uint8_t *p_buf, int p_buf_size);
extern void godot_js_fetch_free(int p_id);
extern godot_js_fetch_state_t godot_js_fetch_state_get(int p_id);
extern int godot_js_fetch_body_length_get(int p_id);
extern int godot_js_fetch_http_status_get(int p_id);
extern int godot_js_fetch_is_chunked(int p_id);

#ifdef __cplusplus
}
#endif

void HTTPClient::_parse_headers(int p_len, const char **p_headers, void *p_ref) {
	HTTPClient *client = static_cast<HTTPClient *>(p_ref);
	for (int i = 0; i < p_len; i++) {
		client->response_headers.push_back(String::utf8(p_headers[i]));
	}
}

Error HTTPClient::connect_to_host(const String &p_host, int p_port, bool p_ssl, bool p_verify_host) {
	close();
	if (p_ssl && !p_verify_host) {
		WARN_PRINT("Disabling HTTPClient's host verification is not supported for the HTML5 platform, host will be verified");
	}

	port = p_port;
	use_tls = p_ssl;

	host = p_host;

	String host_lower = host.to_lower();
	if (host_lower.begins_with("http://")) {
		host = host.substr(7, host.length() - 7);
	} else if (host_lower.begins_with("https://")) {
		use_tls = true;
		host = host.substr(8, host.length() - 8);
	}

	ERR_FAIL_COND_V(host.length() < HOST_MIN_LEN, ERR_INVALID_PARAMETER);

	if (port < 0) {
		if (use_tls) {
			port = PORT_HTTPS;
		} else {
			port = PORT_HTTP;
		}
	}

	status = host.is_valid_ip_address() ? STATUS_CONNECTING : STATUS_RESOLVING;

	return OK;
}

void HTTPClient::set_connection(const Ref<StreamPeer> &p_connection) {
	ERR_FAIL_MSG("Accessing an HTTPClient's StreamPeer is not supported for the HTML5 platform.");
}

Ref<StreamPeer> HTTPClient::get_connection() const {
	ERR_FAIL_V_MSG(REF(), "Accessing an HTTPClient's StreamPeer is not supported for the HTML5 platform.");
}

Error HTTPClient::make_request(Method p_method, const String &p_url, const Vector<String> &p_headers, const uint8_t *p_body, int p_body_len) {
	ERR_FAIL_INDEX_V(p_method, METHOD_MAX, ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V_MSG(p_method == METHOD_TRACE || p_method == METHOD_CONNECT, ERR_UNAVAILABLE, "HTTP methods TRACE and CONNECT are not supported for the HTML5 platform.");
	ERR_FAIL_COND_V(status != STATUS_CONNECTED, ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(host.empty(), ERR_UNCONFIGURED);
	ERR_FAIL_COND_V(port < 0, ERR_UNCONFIGURED);
	ERR_FAIL_COND_V(!p_url.begins_with("/"), ERR_INVALID_PARAMETER);

	String url = (use_tls ? "https://" : "http://") + host + ":" + itos(port) + p_url;
	Vector<CharString> keeper;
	Vector<const char *> c_strings;
	for (int i = 0; i < p_headers.size(); i++) {
		keeper.push_back(p_headers[i].utf8());
		c_strings.push_back(keeper[i].get_data());
	}
	if (js_id) {
		godot_js_fetch_free(js_id);
	}
	js_id = godot_js_fetch_create(_methods[p_method], url.utf8().get_data(), c_strings.ptrw(), c_strings.size(), p_body, p_body_len);
	status = STATUS_REQUESTING;
	return OK;
}

Error HTTPClient::request_raw(Method p_method, const String &p_url, const Vector<String> &p_headers, const PoolVector<uint8_t> &p_body) {
	if (p_body.empty()) {
		return make_request(p_method, p_url, p_headers, nullptr, 0);
	}
	PoolByteArray::Read read = p_body.read();
	return make_request(p_method, p_url, p_headers, read.ptr(), p_body.size());
}

Error HTTPClient::request(Method p_method, const String &p_url, const Vector<String> &p_headers, const String &p_body) {
	if (p_body.empty()) {
		return make_request(p_method, p_url, p_headers, nullptr, 0);
	}
	const CharString cs = p_body.utf8();
	return make_request(p_method, p_url, p_headers, (const uint8_t *)cs.get_data(), cs.size() - 1);
}

void HTTPClient::close() {
	host = "";
	port = -1;
	use_tls = false;
	status = STATUS_DISCONNECTED;
	polled_response_code = 0;
	response_headers.resize(0);
	response_buffer.resize(0);
	if (js_id) {
		godot_js_fetch_free(js_id);
		js_id = 0;
	}
}

HTTPClient::Status HTTPClient::get_status() const {
	return status;
}

bool HTTPClient::has_response() const {
	return response_headers.size() > 0;
}

bool HTTPClient::is_response_chunked() const {
	return godot_js_fetch_is_chunked(js_id);
}

int HTTPClient::get_response_code() const {
	return polled_response_code;
}

Error HTTPClient::get_response_headers(List<String> *r_response) {
	if (!response_headers.size()) {
		return ERR_INVALID_PARAMETER;
	}
	for (int i = 0; i < response_headers.size(); i++) {
		r_response->push_back(response_headers[i]);
	}
	response_headers.clear();
	return OK;
}

int HTTPClient::get_response_body_length() const {
	return godot_js_fetch_body_length_get(js_id);
}

PoolByteArray HTTPClient::read_response_body_chunk() {
	ERR_FAIL_COND_V(status != STATUS_BODY, PoolByteArray());

	if (response_buffer.size() != read_limit) {
		response_buffer.resize(read_limit);
	}
	int read = godot_js_fetch_read_chunk(js_id, response_buffer.ptrw(), read_limit);

	// Check if the stream is over.
	godot_js_fetch_state_t state = godot_js_fetch_state_get(js_id);
	if (state == GODOT_JS_FETCH_STATE_DONE) {
		status = STATUS_DISCONNECTED;
	} else if (state != GODOT_JS_FETCH_STATE_BODY) {
		status = STATUS_CONNECTION_ERROR;
	}

	PoolByteArray chunk;
	if (!read) {
		return chunk;
	}
	chunk.resize(read);
	PoolByteArray::Write w = chunk.write();
	memcpy(&w[0], response_buffer.ptr(), read);
	return chunk;
}

void HTTPClient::set_blocking_mode(bool p_enable) {
	ERR_FAIL_COND_MSG(p_enable, "HTTPClient blocking mode is not supported for the HTML5 platform.");
}

bool HTTPClient::is_blocking_mode_enabled() const {
	return false;
}

void HTTPClient::set_read_chunk_size(int p_size) {
	read_limit = p_size;
}

int HTTPClient::get_read_chunk_size() const {
	return read_limit;
}

Error HTTPClient::poll() {
	switch (status) {
		case STATUS_DISCONNECTED:
			return ERR_UNCONFIGURED;

		case STATUS_RESOLVING:
			status = STATUS_CONNECTING;
			return OK;

		case STATUS_CONNECTING:
			status = STATUS_CONNECTED;
			return OK;

		case STATUS_CONNECTED:
			return OK;

		case STATUS_BODY: {
			godot_js_fetch_state_t state = godot_js_fetch_state_get(js_id);
			if (state == GODOT_JS_FETCH_STATE_DONE) {
				status = STATUS_DISCONNECTED;
			} else if (state != GODOT_JS_FETCH_STATE_BODY) {
				status = STATUS_CONNECTION_ERROR;
				return ERR_CONNECTION_ERROR;
			}
			return OK;
		}

		case STATUS_CONNECTION_ERROR:
			return ERR_CONNECTION_ERROR;

		case STATUS_REQUESTING: {
#ifdef DEBUG_ENABLED
			// forcing synchronous requests is not possible on the web
			if (last_polling_frame == Engine::get_singleton()->get_idle_frames()) {
				WARN_PRINT("HTTPClient polled multiple times in one frame, "
						   "but request cannot progress more than once per "
						   "frame on the HTML5 platform.");
			}
			last_polling_frame = Engine::get_singleton()->get_idle_frames();
#endif

			polled_response_code = godot_js_fetch_http_status_get(js_id);
			godot_js_fetch_state_t js_state = godot_js_fetch_state_get(js_id);
			if (js_state == GODOT_JS_FETCH_STATE_REQUESTING) {
				return OK;
			} else if (js_state == GODOT_JS_FETCH_STATE_ERROR) {
				// Fetch is in error state.
				status = STATUS_CONNECTION_ERROR;
				return ERR_CONNECTION_ERROR;
			}
			if (godot_js_fetch_read_headers(js_id, &_parse_headers, this)) {
				// Failed to parse headers.
				status = STATUS_CONNECTION_ERROR;
				return ERR_CONNECTION_ERROR;
			}
			status = STATUS_BODY;
			break;
		}

		default:
			ERR_FAIL_V(ERR_BUG);
	}
	return OK;
}

HTTPClient::HTTPClient() {
}

HTTPClient::~HTTPClient() {
	close();
}
