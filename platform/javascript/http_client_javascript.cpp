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

#include "http_client_javascript.h"

void HTTPClientJavaScript::_parse_headers(int p_len, const char **p_headers, void *p_ref) {
	HTTPClientJavaScript *client = static_cast<HTTPClientJavaScript *>(p_ref);
	for (int i = 0; i < p_len; i++) {
		client->response_headers.push_back(String::utf8(p_headers[i]));
	}
}

Error HTTPClientJavaScript::connect_to_host(const String &p_host, int p_port, bool p_ssl, bool p_verify_host) {
	close();
	if (p_ssl && !p_verify_host) {
		WARN_PRINT("Disabling HTTPClientJavaScript's host verification is not supported for the HTML5 platform, host will be verified");
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

void HTTPClientJavaScript::set_connection(const Ref<StreamPeer> &p_connection) {
	ERR_FAIL_MSG("Accessing an HTTPClientJavaScript's StreamPeer is not supported for the HTML5 platform.");
}

Ref<StreamPeer> HTTPClientJavaScript::get_connection() const {
	ERR_FAIL_V_MSG(REF(), "Accessing an HTTPClientJavaScript's StreamPeer is not supported for the HTML5 platform.");
}

Error HTTPClientJavaScript::request(Method p_method, const String &p_url, const Vector<String> &p_headers, const uint8_t *p_body, int p_body_len) {
	ERR_FAIL_INDEX_V(p_method, METHOD_MAX, ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V_MSG(p_method == METHOD_TRACE || p_method == METHOD_CONNECT, ERR_UNAVAILABLE, "HTTP methods TRACE and CONNECT are not supported for the HTML5 platform.");
	ERR_FAIL_COND_V(status != STATUS_CONNECTED, ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(host.is_empty(), ERR_UNCONFIGURED);
	ERR_FAIL_COND_V(port < 0, ERR_UNCONFIGURED);
	ERR_FAIL_COND_V(!p_url.begins_with("/"), ERR_INVALID_PARAMETER);

	Error err = verify_headers(p_headers);
	if (err) {
		return err;
	}

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

void HTTPClientJavaScript::close() {
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

HTTPClientJavaScript::Status HTTPClientJavaScript::get_status() const {
	return status;
}

bool HTTPClientJavaScript::has_response() const {
	return response_headers.size() > 0;
}

bool HTTPClientJavaScript::is_response_chunked() const {
	return godot_js_fetch_is_chunked(js_id);
}

int HTTPClientJavaScript::get_response_code() const {
	return polled_response_code;
}

Error HTTPClientJavaScript::get_response_headers(List<String> *r_response) {
	if (!response_headers.size()) {
		return ERR_INVALID_PARAMETER;
	}
	for (int i = 0; i < response_headers.size(); i++) {
		r_response->push_back(response_headers[i]);
	}
	response_headers.clear();
	return OK;
}

int64_t HTTPClientJavaScript::get_response_body_length() const {
	return godot_js_fetch_body_length_get(js_id);
}

PackedByteArray HTTPClientJavaScript::read_response_body_chunk() {
	ERR_FAIL_COND_V(status != STATUS_BODY, PackedByteArray());

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

	PackedByteArray chunk;
	if (!read) {
		return chunk;
	}
	chunk.resize(read);
	memcpy(chunk.ptrw(), response_buffer.ptr(), read);
	return chunk;
}

void HTTPClientJavaScript::set_blocking_mode(bool p_enable) {
	ERR_FAIL_COND_MSG(p_enable, "HTTPClientJavaScript blocking mode is not supported for the HTML5 platform.");
}

bool HTTPClientJavaScript::is_blocking_mode_enabled() const {
	return false;
}

void HTTPClientJavaScript::set_read_chunk_size(int p_size) {
	read_limit = p_size;
}

int HTTPClientJavaScript::get_read_chunk_size() const {
	return read_limit;
}

Error HTTPClientJavaScript::poll() {
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
			if (last_polling_frame == Engine::get_singleton()->get_process_frames()) {
				WARN_PRINT("HTTPClientJavaScript polled multiple times in one frame, "
						   "but request cannot progress more than once per "
						   "frame on the HTML5 platform.");
			}
			last_polling_frame = Engine::get_singleton()->get_process_frames();
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

HTTPClient *HTTPClientJavaScript::_create_func() {
	return memnew(HTTPClientJavaScript);
}

HTTPClient *(*HTTPClient::_create)() = HTTPClientJavaScript::_create_func;

HTTPClientJavaScript::HTTPClientJavaScript() {
}

HTTPClientJavaScript::~HTTPClientJavaScript() {
	close();
}
