/*************************************************************************/
/*  http_client_curl.cpp                                                 */
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

#include "http_client_curl.h"

#include "core/config/project_settings.h"
#include "core/io/certs_compressed.gen.h"
#include "core/io/compression.h"
#include "core/string/print_string.h"
#include "core/templates/ring_buffer.h"
#include "core/version.h"

char const *HTTPClientCurl::methods[10] = {
	"GET",
	"HEAD",
	"POST",
	"PUT",
	"DELETE",
	"OPTIONS",
	"TRACE",
	"CONNECT",
	"PATCH",
	"MAX",
};

void HTTPClientCurl::make_default() {
	print_verbose("Libcurl HTTP Client enabled.");
	_create = _create_func;
}

HTTPClient *HTTPClientCurl::_create_func() {
	return memnew(HTTPClientCurl);
}

size_t HTTPClientCurl::_header_callback(char *buffer, size_t size, size_t nitems, void *userdata) {
	HTTPClientCurl *client = (HTTPClientCurl *)userdata;
	String s((const char *)buffer);
	s = s.to_lower();

	// Handle status line
	if (s.begins_with("http/")) {
		Vector<String> parts = s.split(" ");
		if (parts.size() >= 2) {
			client->response_code = parts[1].to_int();
		}

		client->response_available = true;

		return size * nitems;
	}

	// All other headers.
	Vector<String> parts = s.split(":");
	if (parts.size() < 2) {
		return size * nitems;
	}

	client->response_headers.push_back(s);

	String header = parts[0];
	String val = parts[1].strip_edges();

	// Handle headers like Date whose val also includes a ':'.
	if (parts.size() > 2) {
		val = String(":").join(parts.slice(1));
	}

	// Use the content length to determine body size.
	if (header == "content-length") {
		client->body_size = val.to_int();
	}

	// If the Connection header is set to "close" then
	// keep-alive isn't enabled.
	if (header == "connection" && val == "close") {
		client->keep_alive = false;
	}

	if (header == "transfer-encoding" && val == "chunked") {
		client->body_size = -1;
		client->chunked = true;
	}

	client->response_available = true;

	return size * nitems;
}

size_t HTTPClientCurl::_read_callback(char *buffer, size_t size, size_t nitems, void *userdata) {
	HTTPClientCurl *client = (HTTPClientCurl *)userdata;
	int len = size * nitems;
	int left = client->request_body.size() - client->request_body_offset;
	if (left == 0) {
		return 0;
	}
	if (left > len) {
		left = len;
	}
	memcpy(buffer, client->request_body.ptr() + client->request_body_offset, left);
	client->request_body_offset += left;
	return left;
}

size_t HTTPClientCurl::_write_callback(char *buffer, size_t size, size_t nitems, void *userdata) {
	HTTPClientCurl *client = (HTTPClientCurl *)userdata;
	PackedByteArray chunk;
	chunk.resize(size * nitems);
	memcpy(chunk.ptrw(), buffer, size * nitems);
	client->response_chunks.append(chunk);
	client->status = STATUS_BODY;
	const int s = size * nitems;
	client->body_read += s;
	return s;
}

curl_slist *HTTPClientCurl::_ip_addr_to_slist(const IPAddress &p_addr) {
	String addr = String(p_addr);
	if (addr.find(":") != -1) {
		addr = "[" + addr + "]";
	}
	String h = host + ":" + String::num_int64(port) + ":" + addr;
	return curl_slist_append(nullptr, h.ascii().get_data());
}

String HTTPClientCurl::_hostname_from_url(const String &p_url) {
	String hostname = p_url.trim_prefix("http://");
	hostname = hostname.trim_prefix("https://");
	return hostname.split("/")[0];
}

Error HTTPClientCurl::_resolve_dns() {
	IP::ResolverStatus rstatus = IP::get_singleton()->get_resolve_item_status(resolver_id);
	switch (rstatus) {
		case IP::RESOLVER_STATUS_WAITING:
			return OK;
		case IP::RESOLVER_STATUS_DONE: {
			addr = IP::get_singleton()->get_resolve_item_address(resolver_id);

			Error err = _request(true);

			IP::get_singleton()->erase_resolve_item(resolver_id);
			resolver_id = IP::RESOLVER_INVALID_ID;

			if (err != OK) {
				status = STATUS_CANT_CONNECT;
				return err;
			}
			return OK;
		} break;
		case IP::RESOLVER_STATUS_NONE:
		case IP::RESOLVER_STATUS_ERROR: {
			IP::get_singleton()->erase_resolve_item(resolver_id);
			resolver_id = IP::RESOLVER_INVALID_ID;
			close();
			status = STATUS_CANT_RESOLVE;
			return ERR_CANT_RESOLVE;
		} break;
	}

	return OK;
}

Error HTTPClientCurl::_poll_curl() {
	CURLMcode rc = curl_multi_perform(curl, &still_running);
	if (still_running) {
		rc = curl_multi_wait(curl, nullptr, 0, 0, nullptr);
	}

	if (rc != CURLM_OK) {
		ERR_PRINT_ONCE("Curl multi error while performing. RC: " + String::num_int64(rc));
		return FAILED;
	}

	if (still_running == 0) {
		int n = 0;
		CURLMsg *msg = curl_multi_info_read(curl, &n);
		if (msg && msg->msg == CURLMSG_DONE) {
			CURLcode return_code = curl_easy_getinfo(msg->easy_handle, CURLINFO_RESPONSE_CODE, &response_code);
			if (return_code != CURLE_OK) {
				ERR_PRINT_ONCE("Couldnt get curl status code. RC:" + String::num_int64(return_code));
				return FAILED;
			}

			curl_multi_remove_handle(curl, msg->easy_handle);
			curl_easy_cleanup(msg->easy_handle);
			in_flight = false;

			// This isn't the ideal place for this check, but I'm not sure
			// where else to put it. Basically, we need to transition to STATUS_BODY
			// somewhere and for requests that have a response body, this naturally happens
			// the first time curl invokes the write callback. However when there is no
			// response body, that callback is never called, so we need to do it somewhere else.
			// This issue is compounded by the fact that we don't know the content-length until
			// all of the headers have been read, and curl doesn't give us an easy way to know when
			// that is. Doing the check here when the request is completed means that we have guaranteed
			// to have read all of the headers and thus the content length can be trusted, however it has
			// the downside that the entire response body will be read into memory while we wait for the
			// request to complete. It would be nice if we could come up with a better place to do this check
			// so that we can make better use of memory while the response body chunks are being downloaded
			// and processed.
			if (body_size == 0) {
				status = STATUS_BODY;
			}
		}
	}

	return OK;
}

void HTTPClientCurl::_init_upload(CURL *p_chandle, Method p_method) {
	// Special cases for POST and PUT to configure uploads.
	switch (p_method) {
		case METHOD_GET:
			break;
		case METHOD_HEAD:
			break;
		case METHOD_POST:
			curl_easy_setopt(p_chandle, CURLOPT_POST, 1L);
			curl_easy_setopt(p_chandle, CURLOPT_POSTFIELDSIZE, (curl_off_t)request_body_size);
			break;
		case METHOD_PUT:
			curl_easy_setopt(p_chandle, CURLOPT_UPLOAD, 1L);
			curl_easy_setopt(p_chandle, CURLOPT_INFILESIZE_LARGE, (curl_off_t)request_body.size());
			break;
		case METHOD_DELETE:
			break;
		case METHOD_OPTIONS:
			break;
		case METHOD_TRACE:
			break;
		case METHOD_CONNECT:
			break;
		case METHOD_PATCH:
			break;
		case METHOD_MAX:
			break;
	}

	// Somewhat counter intuitively, the read function is actually used by libcurl to send data,
	// while the write function (see below) is used by libcurl to write response data to storage
	// (or in our case, memory).
	curl_easy_setopt(p_chandle, CURLOPT_READFUNCTION, _read_callback);
	curl_easy_setopt(p_chandle, CURLOPT_READDATA, this);
}

Error HTTPClientCurl::_init_dns(CURL *p_chandle, IPAddress p_addr) {
	// TODO: Support resolving multiple addresses.
	curl_slist *h = _ip_addr_to_slist(p_addr);
	CURLcode return_code = curl_easy_setopt(p_chandle, CURLOPT_RESOLVE, h);
	if (return_code != CURLE_OK) {
		ERR_PRINT("failed to initialize dns resolver: " + String::num_int64(return_code));
		return FAILED;
	}
	return OK;
}

Error HTTPClientCurl::_init_request_headers(CURL *p_chandler, Vector<String> p_headers) {
	bool add_host = true;
	bool add_clen = request_body_size > 0;
	bool add_uagent = true;
	bool add_accept = true;
	curl_slist *curl_headers = nullptr;
	for (int i = 0; i < p_headers.size(); i++) {
		curl_headers = curl_slist_append(curl_headers, p_headers[i].ascii().get_data());

		if (add_host && p_headers[i].findn("Host:") == 0) {
			add_host = false;
		}
		if (add_clen && p_headers[i].findn("Content-Length:") == 0) {
			add_clen = false;
		}
		if (add_uagent && p_headers[i].findn("User-Agent:") == 0) {
			add_uagent = false;
		}
		if (add_accept && p_headers[i].findn("Accept:") == 0) {
			add_accept = false;
		}
	}

	// Add default headers.
	if (add_host) {
		if ((ssl && port == PORT_HTTPS) || (!ssl && port == PORT_HTTP)) {
			// Don't append the standard ports
			curl_headers = curl_slist_append(curl_headers, ("Host: " + host).ascii().get_data());
		} else {
			curl_headers = curl_slist_append(curl_headers, ("Host: " + host + ":" + itos(port)).ascii().get_data());
		}
	}

	if (add_clen) {
		curl_headers = curl_slist_append(curl_headers, ("Content-Length: " + itos(request_body_size)).ascii().get_data());
	}

	if (add_uagent) {
		const String uagent = "User-Agent: GodotEngine/" + String(VERSION_FULL_BUILD) + " (" + OS::get_singleton()->get_name() + ")";
		curl_headers = curl_slist_append(curl_headers, uagent.ascii().get_data());
	}

	if (add_accept) {
		curl_headers = curl_slist_append(curl_headers, String("Accept: */*").ascii().get_data());
	}

	if (curl_headers) {
		CURLcode return_code = curl_easy_setopt(p_chandler, CURLOPT_HTTPHEADER, curl_headers);
		if (return_code != CURLE_OK) {
			ERR_PRINT("failed to set request headers: " + String::num_uint64(return_code));
			return FAILED;
		}
	}
	return OK;
}

Error HTTPClientCurl::_request(bool p_init_dns) {
	String h = host;
	if (h.is_valid_ip_address() && h.find(":") != -1) {
		h = "[" + h + "]";
	}

	CURL *eh = curl_easy_init();
	curl_easy_setopt(eh, CURLOPT_URL, (scheme + h + ":" + String::num_int64(port) + url).ascii().get_data());
	curl_easy_setopt(eh, CURLOPT_CUSTOMREQUEST, methods[(int)method]);
	if (method == HTTPClient::Method::METHOD_HEAD) {
		curl_easy_setopt(eh, CURLOPT_NOBODY, 1L);
	}
	curl_easy_setopt(eh, CURLOPT_BUFFERSIZE, read_chunk_size);
	Error err;
	if (p_init_dns) {
		err = _init_dns(eh, addr);
		if (err != OK) {
			return err;
		}
	}

	if (request_body_size > 0) {
		_init_upload(eh, method);
	}

	if (ssl) {
		curl_easy_setopt(eh, CURLOPT_USE_SSL, CURLUSESSL_ALL);

		if (ca_path != "") {
			curl_easy_setopt(eh, CURLOPT_CAINFO, ca_path.ascii().get_data());
		} else {
			curl_blob ca_blob;
			ca_blob.data = (uint8_t *)ca_data.ptr();
			ca_blob.len = ca_data.size();
			ca_blob.flags = CURL_BLOB_COPY;

			curl_easy_setopt(eh, CURLOPT_CAINFO_BLOB, &ca_blob);
		}
	}

	if (!verify_host) {
		// When CURLOPT_SSL_VERIFYHOST is 2 (the default), that certificate must indicate
		// that the server is the server to which you meant to connect, or
		// the connection fails. Simply put, it means it has to have the same
		// name in the certificate as is in the URL you operate against.
		// @see https://curl.se/libcurl/c/CURLOPT_SSL_VERIFYHOST.html
		curl_easy_setopt(eh, CURLOPT_SSL_VERIFYHOST, 0L);
	}

	// Initialize callbacks.
	curl_easy_setopt(eh, CURLOPT_HEADERFUNCTION, _header_callback);
	curl_easy_setopt(eh, CURLOPT_HEADERDATA, this);
	curl_easy_setopt(eh, CURLOPT_WRITEFUNCTION, _write_callback);
	curl_easy_setopt(eh, CURLOPT_WRITEDATA, this);

	err = _init_request_headers(eh, request_headers);
	if (err != OK) {
		return err;
	}

	// Set the request context. CURLOPT_PRIVATE is just arbitrary data
	// that can be associated with request handlers. It's used here to
	// keep track of certain data that needs to be manipulated throughout
	// the pipeline.
	// @see https://curl.se/libcurl/c/CURLOPT_PRIVATE.html
	curl_easy_setopt(eh, CURLOPT_PRIVATE, this);

	curl_multi_add_handle(curl, eh);

	in_flight = true;
	status = STATUS_REQUESTING;
	still_running = 0;
	return OK;
}

Error HTTPClientCurl::get_response_headers(List<String> *r_response) {
	*r_response = response_headers;
	return OK;
}

Error HTTPClientCurl::connect_to_host(const String &p_host, int p_port, bool p_ssl, bool p_verify_host) {
	if (curl == nullptr) {
		curl = curl_multi_init();
	}

	response_code = 0;
	body_size = 0;
	request_body_size = 0;
	body_read = 0;
	response_headers.clear();
	request_body.clear();
	request_body_offset = 0;
	response_available = false;
	response_code = 0;
	response_chunks.clear();
	keep_alive = false;
	status = STATUS_CONNECTED;
	ssl = p_ssl;
	scheme = ssl ? "https://" : "http://";
	host = p_host.trim_prefix("http://").trim_prefix("https://");
	port = p_port;

	verify_host = p_verify_host;

	_init_ca_path();

	return OK;
}

HTTPClientCurl::~HTTPClientCurl() {
	if (curl) {
		curl_multi_cleanup(curl);
		curl = nullptr;
	}
	close();
}

void HTTPClientCurl::close() {
}

Error HTTPClientCurl::request(Method p_method, const String &p_url, const Vector<String> &p_headers, const uint8_t *p_body, int p_body_size) {
	// Only one request can be in flight at a time.
	if (in_flight) {
		return ERR_ALREADY_IN_USE;
	}

	method = p_method;
	url = p_url;
	request_headers = p_headers;
	request_body.resize(p_body_size);
	memcpy(request_body.ptrw(), p_body, p_body_size);
	request_body_size = p_body_size;

	if (host.is_valid_ip_address()) {
		_request(false);
	} else {
		resolver_id = IP::get_singleton()->resolve_hostname_queue_item(host, IP::Type::TYPE_ANY);
		status = STATUS_RESOLVING;
	}

	return OK;
}

Error HTTPClientCurl::poll() {
	if (status == STATUS_RESOLVING) {
		ERR_FAIL_COND_V(resolver_id == IP::RESOLVER_INVALID_ID, ERR_BUG);
		return _resolve_dns();
	}

	// Important! Since polling libcurl will greedily read response data from the
	// network we don't want to poll when we are in STATUS_BODY state. The reason
	// for this is that the HTTPClient API is expected to only read from the network
	// when read_response_body_chunk is called. This means that here, in poll, we only
	// poll libcurl when we are not in the STATUS_BODY state and we poll libcurl in
	// read_response_body_chunk instead, when we are in STATUS_BODY state.
	if (status != STATUS_BODY) {
		return _poll_curl();
	}
	return OK;
}

PackedByteArray HTTPClientCurl::read_response_body_chunk() {
	if (status == STATUS_BODY) {
		Error err = _poll_curl();
		if (err != OK) {
			ERR_PRINT_ONCE("Failed when polling curl in STATUS_BODY. RC: " + String::num_int64(err));
			return PackedByteArray();
		}
	}

	if (response_chunks.is_empty()) {
		if ((!is_response_chunked() && body_read == body_size) || (!in_flight && is_response_chunked())) {
			status = keep_alive ? STATUS_CONNECTED : STATUS_DISCONNECTED;
		}
		return PackedByteArray();
	}

	PackedByteArray chunk = response_chunks[0];
	response_chunks.remove_at(0);

	return chunk;
}

void HTTPClientCurl::_init_ca_path() {
	ca_path = _GLOBAL_DEF("network/ssl/certificate_bundle_override", "");
#ifdef BUILTIN_CERTS_ENABLED
	// Use builtin certs only if user did not override it in project settings.
	if (ca_path == "") {
		ca_data.clear();
		ca_data.resize(_certs_uncompressed_size + 1);
		Compression::decompress(ca_data.ptrw(), _certs_uncompressed_size, _certs_compressed, _certs_compressed_size, Compression::MODE_DEFLATE);
		ca_data.ptrw()[_certs_uncompressed_size] = 0; // Make sure it ends with string terminator
	}
#endif
}
