/*************************************************************************/
/*  http_client_curl.h                                                   */
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

#ifndef HTTP_CLIENT_CURL_H
#define HTTP_CLIENT_CURL_H

#include "core/io/http_client.h"
#include "core/templates/ring_buffer.h"

#include <curl/curl.h>
#include <stdio.h>

class HTTPClientCurl : public HTTPClient {
	static const char *methods[10];
	static size_t _header_callback(char *buffer, size_t size, size_t nitems, void *userdata);
	static size_t _read_callback(char *buffer, size_t size, size_t nitems, void *userdata);
	static size_t _write_callback(char *buffer, size_t size, size_t nitems, void *userdata);

	CURLM *curl = nullptr;
	int still_running = 0;
	bool ssl = false;
	bool verify_host = false;
	String ca_path;
	PackedByteArray ca_data;
	bool blocking_mode = false;
	int read_chunk_size = 65536;
	bool in_flight = false;

	String scheme;
	String host;
	int port;

	Status status = STATUS_DISCONNECTED;
	bool response_available = false;
	int response_code = 0;
	Vector<PackedByteArray> response_chunks;
	int body_size = 0;
	int body_read = 0;
	bool chunked = false;
	bool keep_alive = true;
	List<String> response_headers;
	IPAddress addr;
	IP::ResolverID resolver_id = 0;

	HTTPClient::Method method = HTTPClient::METHOD_GET;
	String url;
	Vector<String> request_headers;
	PackedByteArray request_body;
	int request_body_offset = 0;
	int request_body_size = 0;

	curl_slist *_ip_addr_to_slist(const IPAddress &p_addr);
	String _hostname_from_url(const String &p_url);
	Error _poll_curl();
	void _init_upload(CURL *p_chandle, Method p_method);
	Error _init_dns(CURL *p_handle, IPAddress p_addr);
	Error _init_request_headers(CURL *p_chandler, Vector<String> p_headers);
	void _init_ca_path();

protected:
	virtual Error _resolve_dns();
	virtual Error _request(bool p_init_dns);

public:
	virtual ~HTTPClientCurl() override;
	static void make_default();
	static HTTPClient *_create_func();

	virtual Error connect_to_host(const String &p_host, int p_port = -1, bool p_ssl = false, bool p_verify_host = true) override;
	virtual void close() override;
	virtual void set_connection(const Ref<StreamPeer> &p_connection) override { ERR_FAIL_MSG("Accessing an HTTPClientCurl's StreamPeer is not supported."); }
	virtual Ref<StreamPeer> get_connection() const override { ERR_FAIL_V_MSG(REF(), "Accessing an HTTPClientCurl's StreemPeer is not supported."); }

	Status get_status() const override { return status; }
	virtual bool has_response() const override { return response_available; }
	virtual bool is_response_chunked() const override { return chunked; }
	virtual int get_response_code() const override { return response_code; }
	virtual Error get_response_headers(List<String> *r_response) override;
	virtual int64_t get_response_body_length() const override { return is_response_chunked() ? -1 : body_size; }
	virtual PackedByteArray read_response_body_chunk() override;
	virtual void set_blocking_mode(bool p_enabled) override { /* blocking mode is not yet implemented */
	}
	virtual bool is_blocking_mode_enabled() const override { return blocking_mode; }
	virtual void set_read_chunk_size(int p_size) override { read_chunk_size = CLAMP(p_size, 1024, CURL_MAX_READ_SIZE); }
	virtual int get_read_chunk_size() const override { return read_chunk_size; }

	virtual Error request(Method p_method, const String &p_url, const Vector<String> &p_headers, const uint8_t *p_body, int p_body_size) override;
	virtual Error poll() override;
};

#endif // #define HTTP_CLIENT_CURL_H
