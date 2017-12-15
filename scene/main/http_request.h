/*************************************************************************/
/*  http_request.h                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifndef HTTPREQUEST_H
#define HTTPREQUEST_H

#include "io/http_client.h"
#include "node.h"
#include "os/file_access.h"
#include "os/thread.h"

class HTTPRequest : public Node {

	GDCLASS(HTTPRequest, Node);

public:
	enum Result {
		RESULT_SUCCESS,
		RESULT_CHUNKED_BODY_SIZE_MISMATCH,
		RESULT_CANT_CONNECT,
		RESULT_CANT_RESOLVE,
		RESULT_CONNECTION_ERROR,
		RESULT_SSL_HANDSHAKE_ERROR,
		RESULT_NO_RESPONSE,
		RESULT_BODY_SIZE_LIMIT_EXCEEDED,
		RESULT_REQUEST_FAILED,
		RESULT_DOWNLOAD_FILE_CANT_OPEN,
		RESULT_DOWNLOAD_FILE_WRITE_ERROR,
		RESULT_REDIRECT_LIMIT_REACHED

	};

private:
	bool requesting;

	String request_string;
	String url;
	int port;
	Vector<String> headers;
	bool validate_ssl;
	bool use_ssl;
	HTTPClient::Method method;
	String request_data;

	bool request_sent;
	Ref<HTTPClient> client;
	PoolByteArray body;
	volatile bool use_threads;

	bool got_response;
	int response_code;
	PoolVector<String> response_headers;

	String download_to_file;

	FileAccess *file;

	int body_len;
	volatile int downloaded;
	int body_size_limit;

	int redirections;

	HTTPClient::Status status;

	bool _update_connection();

	int max_redirects;

	void _redirect_request(const String &p_new_url);

	bool _handle_response(bool *ret_value);

	Error _parse_url(const String &p_url);
	Error _request();

	volatile bool thread_done;
	volatile bool thread_request_quit;

	Thread *thread;

	void _request_done(int p_status, int p_code, const PoolStringArray &headers, const PoolByteArray &p_data);
	static void _thread_func(void *p_userdata);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	Error request(const String &p_url, const Vector<String> &p_custom_headers = Vector<String>(), bool p_ssl_validate_domain = true, HTTPClient::Method p_method = HTTPClient::METHOD_GET, const String &p_request_data = ""); //connects to a full url and perform request
	void cancel_request();
	HTTPClient::Status get_http_client_status() const;

	void set_use_threads(bool p_use);
	bool is_using_threads() const;

	void set_download_file(const String &p_file);
	String get_download_file() const;

	void set_body_size_limit(int p_bytes);
	int get_body_size_limit() const;

	void set_max_redirects(int p_max);
	int get_max_redirects() const;

	int get_downloaded_bytes() const;
	int get_body_size() const;

	HTTPRequest();
	~HTTPRequest();
};

VARIANT_ENUM_CAST(HTTPRequest::Result);

#endif // HTTPREQUEST_H
