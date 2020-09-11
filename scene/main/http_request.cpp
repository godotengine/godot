/*************************************************************************/
/*  http_request.cpp                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "http_request.h"
#include "core/io/compression.h"
#include "core/ustring.h"

void HTTPRequest::_redirect_request(const String &p_new_url) {
}

Error HTTPRequest::_request() {
	return client->connect_to_host(url, port, use_ssl, validate_ssl);
}

Error HTTPRequest::_parse_url(const String &p_url) {
	url = p_url;
	use_ssl = false;

	request_string = "";
	port = 80;
	request_sent = false;
	got_response = false;
	body_len = -1;
	body.resize(0);
	downloaded = 0;
	redirections = 0;

	String url_lower = url.to_lower();
	if (url_lower.begins_with("http://")) {
		url = url.substr(7, url.length() - 7);
	} else if (url_lower.begins_with("https://")) {
		url = url.substr(8, url.length() - 8);
		use_ssl = true;
		port = 443;
	} else {
		ERR_FAIL_V_MSG(ERR_INVALID_PARAMETER, "Malformed URL: " + url + ".");
	}

	ERR_FAIL_COND_V_MSG(url.length() < 1, ERR_INVALID_PARAMETER, "URL too short: " + url + ".");

	int slash_pos = url.find("/");

	if (slash_pos != -1) {
		request_string = url.substr(slash_pos, url.length());
		url = url.substr(0, slash_pos);
	} else {
		request_string = "/";
	}

	int colon_pos = url.find(":");
	if (colon_pos != -1) {
		port = url.substr(colon_pos + 1, url.length()).to_int();
		url = url.substr(0, colon_pos);
		ERR_FAIL_COND_V(port < 1 || port > 65535, ERR_INVALID_PARAMETER);
	}

	return OK;
}

bool HTTPRequest::has_header(const PackedStringArray &p_headers, const String &p_header_name) {
	bool exists = false;

	String lower_case_header_name = p_header_name.to_lower();
	for (int i = 0; i < p_headers.size() && !exists; i++) {
		String sanitized = p_headers[i].strip_edges().to_lower();
		if (sanitized.begins_with(lower_case_header_name)) {
			exists = true;
		}
	}

	return exists;
}

String HTTPRequest::get_header_value(const PackedStringArray &p_headers, const String &p_header_name) {
	String value = "";

	String lowwer_case_header_name = p_header_name.to_lower();
	for (int i = 0; i < p_headers.size(); i++) {
		if (p_headers[i].find(":", 0) >= 0) {
			Vector<String> parts = p_headers[i].split(":", false, 1);
			if (parts[0].strip_edges().to_lower() == lowwer_case_header_name) {
				value = parts[1].strip_edges();
				break;
			}
		}
	}

	return value;
}

Error HTTPRequest::request(const String &p_url, const Vector<String> &p_custom_headers, bool p_ssl_validate_domain, HTTPClient::Method p_method, const String &p_request_data) {
	// Copy the string into a raw buffer
	Vector<uint8_t> raw_data;

	CharString charstr = p_request_data.utf8();
	size_t len = charstr.length();
	raw_data.resize(len);
	uint8_t *w = raw_data.ptrw();
	copymem(w, charstr.ptr(), len);

	return request_raw(p_url, p_custom_headers, p_ssl_validate_domain, p_method, raw_data);
}

Error HTTPRequest::request_raw(const String &p_url, const Vector<String> &p_custom_headers, bool p_ssl_validate_domain, HTTPClient::Method p_method, const Vector<uint8_t> &p_request_data_raw) {
	ERR_FAIL_COND_V(!is_inside_tree(), ERR_UNCONFIGURED);
	ERR_FAIL_COND_V_MSG(requesting, ERR_BUSY, "HTTPRequest is processing a request. Wait for completion or cancel it before attempting a new one.");

	if (timeout > 0) {
		timer->stop();
		timer->start(timeout);
	}

	method = p_method;

	Error err = _parse_url(p_url);
	if (err) {
		return err;
	}

	validate_ssl = p_ssl_validate_domain;

	headers = p_custom_headers;

	if (accept_gzip) {
		// If the user has specified a different Accept-Encoding, don't overwrite it
		if (!has_header(headers, "Accept-Encoding")) {
			headers.push_back("Accept-Encoding: gzip, deflate");
		}
	}

	request_data = p_request_data_raw;

	requesting = true;

	if (use_threads) {
		thread_done = false;
		thread_request_quit = false;
		client->set_blocking_mode(true);
		thread = Thread::create(_thread_func, this);
	} else {
		client->set_blocking_mode(false);
		err = _request();
		if (err != OK) {
			call_deferred("_request_done", RESULT_CANT_CONNECT, 0, PackedStringArray(), PackedByteArray());
			return ERR_CANT_CONNECT;
		}

		set_process_internal(true);
	}

	return OK;
}

void HTTPRequest::_thread_func(void *p_userdata) {
	HTTPRequest *hr = (HTTPRequest *)p_userdata;

	Error err = hr->_request();

	if (err != OK) {
		hr->call_deferred("_request_done", RESULT_CANT_CONNECT, 0, PackedStringArray(), PackedByteArray());
	} else {
		while (!hr->thread_request_quit) {
			bool exit = hr->_update_connection();
			if (exit) {
				break;
			}
			OS::get_singleton()->delay_usec(1);
		}
	}

	hr->thread_done = true;
}

void HTTPRequest::cancel_request() {
	timer->stop();

	if (!requesting) {
		return;
	}

	if (!use_threads) {
		set_process_internal(false);
	} else {
		thread_request_quit = true;
		Thread::wait_to_finish(thread);
		memdelete(thread);
		thread = nullptr;
	}

	if (file) {
		memdelete(file);
		file = nullptr;
	}
	client->close();
	body.resize(0);
	got_response = false;
	response_code = -1;
	request_sent = false;
	requesting = false;
}

bool HTTPRequest::_handle_response(bool *ret_value) {
	if (!client->has_response()) {
		call_deferred("_request_done", RESULT_NO_RESPONSE, 0, PackedStringArray(), PackedByteArray());
		*ret_value = true;
		return true;
	}

	got_response = true;
	response_code = client->get_response_code();
	List<String> rheaders;
	client->get_response_headers(&rheaders);
	response_headers.resize(0);
	downloaded = 0;
	for (List<String>::Element *E = rheaders.front(); E; E = E->next()) {
		response_headers.push_back(E->get());
	}

	if (response_code == 301 || response_code == 302) {
		// Handle redirect

		if (max_redirects >= 0 && redirections >= max_redirects) {
			call_deferred("_request_done", RESULT_REDIRECT_LIMIT_REACHED, response_code, response_headers, PackedByteArray());
			*ret_value = true;
			return true;
		}

		String new_request;

		for (List<String>::Element *E = rheaders.front(); E; E = E->next()) {
			if (E->get().findn("Location: ") != -1) {
				new_request = E->get().substr(9, E->get().length()).strip_edges();
			}
		}

		if (new_request != "") {
			// Process redirect
			client->close();
			int new_redirs = redirections + 1; // Because _request() will clear it
			Error err;
			if (new_request.begins_with("http")) {
				// New url, request all again
				_parse_url(new_request);
			} else {
				request_string = new_request;
			}

			err = _request();
			if (err == OK) {
				request_sent = false;
				got_response = false;
				body_len = -1;
				body.resize(0);
				downloaded = 0;
				redirections = new_redirs;
				*ret_value = false;
				return true;
			}
		}
	}

	return false;
}

bool HTTPRequest::_update_connection() {
	switch (client->get_status()) {
		case HTTPClient::STATUS_DISCONNECTED: {
			call_deferred("_request_done", RESULT_CANT_CONNECT, 0, PackedStringArray(), PackedByteArray());
			return true; // End it, since it's doing something
		} break;
		case HTTPClient::STATUS_RESOLVING: {
			client->poll();
			// Must wait
			return false;
		} break;
		case HTTPClient::STATUS_CANT_RESOLVE: {
			call_deferred("_request_done", RESULT_CANT_RESOLVE, 0, PackedStringArray(), PackedByteArray());
			return true;

		} break;
		case HTTPClient::STATUS_CONNECTING: {
			client->poll();
			// Must wait
			return false;
		} break; // Connecting to IP
		case HTTPClient::STATUS_CANT_CONNECT: {
			call_deferred("_request_done", RESULT_CANT_CONNECT, 0, PackedStringArray(), PackedByteArray());
			return true;

		} break;
		case HTTPClient::STATUS_CONNECTED: {
			if (request_sent) {
				if (!got_response) {
					// No body

					bool ret_value;

					if (_handle_response(&ret_value)) {
						return ret_value;
					}

					call_deferred("_request_done", RESULT_SUCCESS, response_code, response_headers, PackedByteArray());
					return true;
				}
				if (body_len < 0) {
					// Chunked transfer is done
					call_deferred("_request_done", RESULT_SUCCESS, response_code, response_headers, body);
					return true;
				}

				call_deferred("_request_done", RESULT_CHUNKED_BODY_SIZE_MISMATCH, response_code, response_headers, PackedByteArray());
				return true;
				// Request migh have been done
			} else {
				// Did not request yet, do request

				Error err = client->request_raw(method, request_string, headers, request_data);
				if (err != OK) {
					call_deferred("_request_done", RESULT_CONNECTION_ERROR, 0, PackedStringArray(), PackedByteArray());
					return true;
				}

				request_sent = true;
				return false;
			}
		} break; // Connected: break requests only accepted here
		case HTTPClient::STATUS_REQUESTING: {
			// Must wait, still requesting
			client->poll();
			return false;

		} break; // Request in progress
		case HTTPClient::STATUS_BODY: {
			if (!got_response) {
				bool ret_value;

				if (_handle_response(&ret_value)) {
					return ret_value;
				}

				if (!client->is_response_chunked() && client->get_response_body_length() == 0) {
					call_deferred("_request_done", RESULT_SUCCESS, response_code, response_headers, PackedByteArray());
					return true;
				}

				// No body len (-1) if chunked or no content-length header was provided.
				// Change your webserver configuration if you want body len.
				body_len = client->get_response_body_length();

				if (body_size_limit >= 0 && body_len > body_size_limit) {
					call_deferred("_request_done", RESULT_BODY_SIZE_LIMIT_EXCEEDED, response_code, response_headers, PackedByteArray());
					return true;
				}

				if (download_to_file != String()) {
					file = FileAccess::open(download_to_file, FileAccess::WRITE);
					if (!file) {
						call_deferred("_request_done", RESULT_DOWNLOAD_FILE_CANT_OPEN, response_code, response_headers, PackedByteArray());
						return true;
					}
				}
			}

			client->poll();

			PackedByteArray chunk = client->read_response_body_chunk();
			downloaded += chunk.size();

			if (file) {
				const uint8_t *r = chunk.ptr();
				file->store_buffer(r, chunk.size());
				if (file->get_error() != OK) {
					call_deferred("_request_done", RESULT_DOWNLOAD_FILE_WRITE_ERROR, response_code, response_headers, PackedByteArray());
					return true;
				}
			} else {
				body.append_array(chunk);
			}

			if (body_size_limit >= 0 && downloaded > body_size_limit) {
				call_deferred("_request_done", RESULT_BODY_SIZE_LIMIT_EXCEEDED, response_code, response_headers, PackedByteArray());
				return true;
			}

			if (body_len >= 0) {
				if (downloaded == body_len) {
					call_deferred("_request_done", RESULT_SUCCESS, response_code, response_headers, body);
					return true;
				}
			} else if (client->get_status() == HTTPClient::STATUS_DISCONNECTED) {
				// We read till EOF, with no errors. Request is done.
				call_deferred("_request_done", RESULT_SUCCESS, response_code, response_headers, body);
			}

			return false;

		} break; // Request resulted in body: break which must be read
		case HTTPClient::STATUS_CONNECTION_ERROR: {
			call_deferred("_request_done", RESULT_CONNECTION_ERROR, 0, PackedStringArray(), PackedByteArray());
			return true;
		} break;
		case HTTPClient::STATUS_SSL_HANDSHAKE_ERROR: {
			call_deferred("_request_done", RESULT_SSL_HANDSHAKE_ERROR, 0, PackedStringArray(), PackedByteArray());
			return true;
		} break;
	}

	ERR_FAIL_V(false);
}

void HTTPRequest::_request_done(int p_status, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_data) {
	cancel_request();

	// Determine if the request body is compressed
	bool is_compressed;
	String content_encoding = get_header_value(p_headers, "Content-Encoding").to_lower();
	Compression::Mode mode;
	if (content_encoding == "gzip") {
		mode = Compression::Mode::MODE_GZIP;
		is_compressed = true;
	} else if (content_encoding == "deflate") {
		mode = Compression::Mode::MODE_DEFLATE;
		is_compressed = true;
	} else {
		is_compressed = false;
	}

	const PackedByteArray *data = NULL;

	if (accept_gzip && is_compressed && p_data.size() > 0) {
		// Decompress request body
		PackedByteArray *decompressed = memnew(PackedByteArray);
		int result = Compression::decompress_dynamic(decompressed, body_size_limit, p_data.ptr(), p_data.size(), mode);
		if (result == OK) {
			data = decompressed;
		} else if (result == -5) {
			WARN_PRINT("Decompressed size of HTTP response body exceeded body_size_limit");
			p_status = RESULT_BODY_SIZE_LIMIT_EXCEEDED;
			// Just return the raw data if we failed to decompress it
			data = &p_data;
		} else {
			WARN_PRINT("Failed to decompress HTTP response body");
			p_status = RESULT_BODY_DECOMPRESS_FAILED;
			// Just return the raw data if we failed to decompress it
			data = &p_data;
		}
	} else {
		data = &p_data;
	}

	emit_signal("request_completed", p_status, p_code, p_headers, *data);
}

void HTTPRequest::_notification(int p_what) {
	if (p_what == NOTIFICATION_INTERNAL_PROCESS) {
		if (use_threads) {
			return;
		}
		bool done = _update_connection();
		if (done) {
			set_process_internal(false);
			// cancel_request(); called from _request done now
		}
	}

	if (p_what == NOTIFICATION_EXIT_TREE) {
		if (requesting) {
			cancel_request();
		}
	}
}

void HTTPRequest::set_use_threads(bool p_use) {
	ERR_FAIL_COND(get_http_client_status() != HTTPClient::STATUS_DISCONNECTED);
	use_threads = p_use;
}

bool HTTPRequest::is_using_threads() const {
	return use_threads;
}

void HTTPRequest::set_accept_gzip(bool p_gzip) {
	accept_gzip = p_gzip;
}

bool HTTPRequest::is_accepting_gzip() const {
	return accept_gzip;
}

void HTTPRequest::set_body_size_limit(int p_bytes) {
	ERR_FAIL_COND(get_http_client_status() != HTTPClient::STATUS_DISCONNECTED);

	body_size_limit = p_bytes;
}

int HTTPRequest::get_body_size_limit() const {
	return body_size_limit;
}

void HTTPRequest::set_download_file(const String &p_file) {
	ERR_FAIL_COND(get_http_client_status() != HTTPClient::STATUS_DISCONNECTED);

	download_to_file = p_file;
}

String HTTPRequest::get_download_file() const {
	return download_to_file;
}

void HTTPRequest::set_download_chunk_size(int p_chunk_size) {
	ERR_FAIL_COND(get_http_client_status() != HTTPClient::STATUS_DISCONNECTED);

	client->set_read_chunk_size(p_chunk_size);
}

int HTTPRequest::get_download_chunk_size() const {
	return client->get_read_chunk_size();
}

HTTPClient::Status HTTPRequest::get_http_client_status() const {
	return client->get_status();
}

void HTTPRequest::set_max_redirects(int p_max) {
	max_redirects = p_max;
}

int HTTPRequest::get_max_redirects() const {
	return max_redirects;
}

int HTTPRequest::get_downloaded_bytes() const {
	return downloaded;
}

int HTTPRequest::get_body_size() const {
	return body_len;
}

void HTTPRequest::set_timeout(int p_timeout) {
	ERR_FAIL_COND(p_timeout < 0);
	timeout = p_timeout;
}

int HTTPRequest::get_timeout() {
	return timeout;
}

void HTTPRequest::_timeout() {
	cancel_request();
	call_deferred("_request_done", RESULT_TIMEOUT, 0, PackedStringArray(), PackedByteArray());
}

void HTTPRequest::_bind_methods() {
	ClassDB::bind_method(D_METHOD("request", "url", "custom_headers", "ssl_validate_domain", "method", "request_data"), &HTTPRequest::request, DEFVAL(PackedStringArray()), DEFVAL(true), DEFVAL(HTTPClient::METHOD_GET), DEFVAL(String()));
	ClassDB::bind_method(D_METHOD("request_raw", "url", "custom_headers", "ssl_validate_domain", "method", "request_data_raw"), &HTTPRequest::request_raw, DEFVAL(PackedStringArray()), DEFVAL(true), DEFVAL(HTTPClient::METHOD_GET), DEFVAL(PackedByteArray()));
	ClassDB::bind_method(D_METHOD("cancel_request"), &HTTPRequest::cancel_request);

	ClassDB::bind_method(D_METHOD("get_http_client_status"), &HTTPRequest::get_http_client_status);

	ClassDB::bind_method(D_METHOD("set_use_threads", "enable"), &HTTPRequest::set_use_threads);
	ClassDB::bind_method(D_METHOD("is_using_threads"), &HTTPRequest::is_using_threads);

	ClassDB::bind_method(D_METHOD("set_accept_gzip", "enable"), &HTTPRequest::set_accept_gzip);
	ClassDB::bind_method(D_METHOD("is_accepting_gzip"), &HTTPRequest::is_accepting_gzip);

	ClassDB::bind_method(D_METHOD("set_body_size_limit", "bytes"), &HTTPRequest::set_body_size_limit);
	ClassDB::bind_method(D_METHOD("get_body_size_limit"), &HTTPRequest::get_body_size_limit);

	ClassDB::bind_method(D_METHOD("set_max_redirects", "amount"), &HTTPRequest::set_max_redirects);
	ClassDB::bind_method(D_METHOD("get_max_redirects"), &HTTPRequest::get_max_redirects);

	ClassDB::bind_method(D_METHOD("set_download_file", "path"), &HTTPRequest::set_download_file);
	ClassDB::bind_method(D_METHOD("get_download_file"), &HTTPRequest::get_download_file);

	ClassDB::bind_method(D_METHOD("get_downloaded_bytes"), &HTTPRequest::get_downloaded_bytes);
	ClassDB::bind_method(D_METHOD("get_body_size"), &HTTPRequest::get_body_size);

	ClassDB::bind_method(D_METHOD("_redirect_request"), &HTTPRequest::_redirect_request);
	ClassDB::bind_method(D_METHOD("_request_done"), &HTTPRequest::_request_done);

	ClassDB::bind_method(D_METHOD("set_timeout", "timeout"), &HTTPRequest::set_timeout);
	ClassDB::bind_method(D_METHOD("get_timeout"), &HTTPRequest::get_timeout);

	ClassDB::bind_method(D_METHOD("set_download_chunk_size"), &HTTPRequest::set_download_chunk_size);
	ClassDB::bind_method(D_METHOD("get_download_chunk_size"), &HTTPRequest::get_download_chunk_size);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "download_file", PROPERTY_HINT_FILE), "set_download_file", "get_download_file");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "download_chunk_size", PROPERTY_HINT_RANGE, "256,16777216"), "set_download_chunk_size", "get_download_chunk_size");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "use_threads"), "set_use_threads", "is_using_threads");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "accept_gzip"), "set_accept_gzip", "is_accepting_gzip");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "body_size_limit", PROPERTY_HINT_RANGE, "-1,2000000000"), "set_body_size_limit", "get_body_size_limit");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_redirects", PROPERTY_HINT_RANGE, "-1,64"), "set_max_redirects", "get_max_redirects");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "timeout", PROPERTY_HINT_RANGE, "0,86400"), "set_timeout", "get_timeout");

	ADD_SIGNAL(MethodInfo("request_completed", PropertyInfo(Variant::INT, "result"), PropertyInfo(Variant::INT, "response_code"), PropertyInfo(Variant::PACKED_STRING_ARRAY, "headers"), PropertyInfo(Variant::PACKED_BYTE_ARRAY, "body")));

	BIND_ENUM_CONSTANT(RESULT_SUCCESS);
	//BIND_ENUM_CONSTANT( RESULT_NO_BODY );
	BIND_ENUM_CONSTANT(RESULT_CHUNKED_BODY_SIZE_MISMATCH);
	BIND_ENUM_CONSTANT(RESULT_CANT_CONNECT);
	BIND_ENUM_CONSTANT(RESULT_CANT_RESOLVE);
	BIND_ENUM_CONSTANT(RESULT_CONNECTION_ERROR);
	BIND_ENUM_CONSTANT(RESULT_SSL_HANDSHAKE_ERROR);
	BIND_ENUM_CONSTANT(RESULT_NO_RESPONSE);
	BIND_ENUM_CONSTANT(RESULT_BODY_SIZE_LIMIT_EXCEEDED);
	BIND_ENUM_CONSTANT(RESULT_BODY_DECOMPRESS_FAILED);
	BIND_ENUM_CONSTANT(RESULT_REQUEST_FAILED);
	BIND_ENUM_CONSTANT(RESULT_DOWNLOAD_FILE_CANT_OPEN);
	BIND_ENUM_CONSTANT(RESULT_DOWNLOAD_FILE_WRITE_ERROR);
	BIND_ENUM_CONSTANT(RESULT_REDIRECT_LIMIT_REACHED);
	BIND_ENUM_CONSTANT(RESULT_TIMEOUT);
}

HTTPRequest::HTTPRequest() {
	thread = nullptr;

	port = 80;
	redirections = 0;
	max_redirects = 8;
	body_len = -1;
	got_response = false;
	validate_ssl = false;
	use_ssl = false;
	accept_gzip = true;
	response_code = 0;
	request_sent = false;
	requesting = false;
	client.instance();
	use_threads = false;
	thread_done = false;
	downloaded = 0;
	body_size_limit = -1;
	file = nullptr;

	timer = memnew(Timer);
	timer->set_one_shot(true);
	timer->connect("timeout", callable_mp(this, &HTTPRequest::_timeout));
	add_child(timer);
	timeout = 0;
}

HTTPRequest::~HTTPRequest() {
	if (file) {
		memdelete(file);
	}
}
