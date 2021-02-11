/*************************************************************************/
/*  http_client_javascript.cpp                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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
#include "http_request.h"

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

Error HTTPClient::prepare_request(Method p_method, const String &p_url, const Vector<String> &p_headers) {

	ERR_FAIL_INDEX_V(p_method, METHOD_MAX, ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V_MSG(p_method == METHOD_TRACE || p_method == METHOD_CONNECT, ERR_UNAVAILABLE, "HTTP methods TRACE and CONNECT are not supported for the HTML5 platform.");
	ERR_FAIL_COND_V(status != STATUS_CONNECTED, ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(host.empty(), ERR_UNCONFIGURED);
	ERR_FAIL_COND_V(port < 0, ERR_UNCONFIGURED);
	ERR_FAIL_COND_V(!p_url.begins_with("/"), ERR_INVALID_PARAMETER);

	String url = (use_tls ? "https://" : "http://") + host + ":" + itos(port) + p_url;
	godot_xhr_reset(xhr_id);
	godot_xhr_open(xhr_id, _methods[p_method], url.utf8().get_data(),
			username.empty() ? NULL : username.utf8().get_data(),
			password.empty() ? NULL : password.utf8().get_data());

	for (int i = 0; i < p_headers.size(); i++) {
		int header_separator = p_headers[i].find(": ");
		ERR_FAIL_COND_V(header_separator < 0, ERR_INVALID_PARAMETER);
		godot_xhr_set_request_header(xhr_id,
				p_headers[i].left(header_separator).utf8().get_data(),
				p_headers[i].right(header_separator + 2).utf8().get_data());
	}
	response_read_offset = 0;
	status = STATUS_REQUESTING;
	return OK;
}

Error HTTPClient::request_raw(Method p_method, const String &p_url, const Vector<String> &p_headers, const PoolVector<uint8_t> &p_body) {

	Error err = prepare_request(p_method, p_url, p_headers);
	if (err != OK)
		return err;
	if (p_body.empty()) {
		godot_xhr_send(xhr_id, nullptr, 0);
	} else {
		PoolByteArray::Read read = p_body.read();
		godot_xhr_send(xhr_id, read.ptr(), p_body.size());
	}
	return OK;
}

Error HTTPClient::request(Method p_method, const String &p_url, const Vector<String> &p_headers, const String &p_body) {

	Error err = prepare_request(p_method, p_url, p_headers);
	if (err != OK)
		return err;
	if (p_body.empty()) {
		godot_xhr_send(xhr_id, nullptr, 0);
	} else {
		const CharString cs = p_body.utf8();
		godot_xhr_send(xhr_id, cs.get_data(), cs.length());
	}
	return OK;
}

void HTTPClient::close() {

	host = "";
	port = -1;
	use_tls = false;
	status = STATUS_DISCONNECTED;
	polled_response.resize(0);
	polled_response_code = 0;
	polled_response_header = String();
	godot_xhr_reset(xhr_id);
}

HTTPClient::Status HTTPClient::get_status() const {

	return status;
}

bool HTTPClient::has_response() const {

	return !polled_response_header.empty();
}

bool HTTPClient::is_response_chunked() const {

	// TODO evaluate using moz-chunked-arraybuffer, fetch & ReadableStream
	return false;
}

int HTTPClient::get_response_code() const {

	return polled_response_code;
}

Error HTTPClient::get_response_headers(List<String> *r_response) {

	if (polled_response_header.empty())
		return ERR_INVALID_PARAMETER;

	Vector<String> header_lines = polled_response_header.split("\r\n", false);
	for (int i = 0; i < header_lines.size(); ++i) {
		r_response->push_back(header_lines[i]);
	}
	polled_response_header = String();
	return OK;
}

int HTTPClient::get_response_body_length() const {

	return polled_response.size();
}

PoolByteArray HTTPClient::read_response_body_chunk() {

	ERR_FAIL_COND_V(status != STATUS_BODY, PoolByteArray());

	int to_read = MIN(read_limit, polled_response.size() - response_read_offset);
	PoolByteArray chunk;
	chunk.resize(to_read);
	PoolByteArray::Write write = chunk.write();
	PoolByteArray::Read read = polled_response.read();
	memcpy(write.ptr(), read.ptr() + response_read_offset, to_read);
	write = PoolByteArray::Write();
	read = PoolByteArray::Read();
	response_read_offset += to_read;

	if (response_read_offset == polled_response.size()) {
		status = STATUS_CONNECTED;
		polled_response.resize(0);
		godot_xhr_reset(xhr_id);
	}

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
		case STATUS_BODY:
			return OK;

		case STATUS_CONNECTION_ERROR:
			return ERR_CONNECTION_ERROR;

		case STATUS_REQUESTING: {

#ifdef DEBUG_ENABLED
			if (!has_polled) {
				has_polled = true;
			} else {
				// forcing synchronous requests is not possible on the web
				if (last_polling_frame == Engine::get_singleton()->get_idle_frames()) {
					WARN_PRINT("HTTPClient polled multiple times in one frame, "
							   "but request cannot progress more than once per "
							   "frame on the HTML5 platform.");
				}
			}
			last_polling_frame = Engine::get_singleton()->get_idle_frames();
#endif

			polled_response_code = godot_xhr_get_status(xhr_id);
			if (godot_xhr_get_ready_state(xhr_id) != XHR_READY_STATE_DONE) {
				return OK;
			} else if (!polled_response_code) {
				status = STATUS_CONNECTION_ERROR;
				return ERR_CONNECTION_ERROR;
			}

			status = STATUS_BODY;

			PoolByteArray bytes;
			int len = godot_xhr_get_response_headers_length(xhr_id);
			bytes.resize(len + 1);

			PoolByteArray::Write write = bytes.write();
			godot_xhr_get_response_headers(xhr_id, reinterpret_cast<char *>(write.ptr()), len);
			write[len] = 0;
			write = PoolByteArray::Write();

			PoolByteArray::Read read = bytes.read();
			polled_response_header = String::utf8(reinterpret_cast<const char *>(read.ptr()));
			read = PoolByteArray::Read();

			polled_response.resize(godot_xhr_get_response_length(xhr_id));
			write = polled_response.write();
			godot_xhr_get_response(xhr_id, write.ptr(), polled_response.size());
			write = PoolByteArray::Write();
			break;
		}

		default:
			ERR_FAIL_V(ERR_BUG);
	}
	return OK;
}

HTTPClient::HTTPClient() {

	xhr_id = godot_xhr_new();
	read_limit = 4096;
	status = STATUS_DISCONNECTED;
	port = -1;
	use_tls = false;
	polled_response_code = 0;
#ifdef DEBUG_ENABLED
	has_polled = false;
	last_polling_frame = 0;
#endif
}

HTTPClient::~HTTPClient() {

	godot_xhr_free(xhr_id);
}
