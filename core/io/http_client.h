/*************************************************************************/
/*  http_client.h                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
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
#ifndef HTTP_CLIENT_H
#define HTTP_CLIENT_H

#include "io/ip.h"
#include "io/stream_peer.h"
#include "io/stream_peer_tcp.h"
#include "reference.h"

class HTTPClient : public Reference {

	GDCLASS(HTTPClient, Reference);

public:
	enum ResponseCode {

		// 1xx informational
		RESPONSE_CONTINUE = 100,
		RESPONSE_SWITCHING_PROTOCOLS = 101,
		RESPONSE_PROCESSING = 102,

		// 2xx successful
		RESPONSE_OK = 200,
		RESPONSE_CREATED = 201,
		RESPONSE_ACCEPTED = 202,
		RESPONSE_NON_AUTHORITATIVE_INFORMATION = 203,
		RESPONSE_NO_CONTENT = 204,
		RESPONSE_RESET_CONTENT = 205,
		RESPONSE_PARTIAL_CONTENT = 206,
		RESPONSE_MULTI_STATUS = 207,
		RESPONSE_IM_USED = 226,

		// 3xx redirection
		RESPONSE_MULTIPLE_CHOICES = 300,
		RESPONSE_MOVED_PERMANENTLY = 301,
		RESPONSE_FOUND = 302,
		RESPONSE_SEE_OTHER = 303,
		RESPONSE_NOT_MODIFIED = 304,
		RESPONSE_USE_PROXY = 305,
		RESPONSE_TEMPORARY_REDIRECT = 307,

		// 4xx client error
		RESPONSE_BAD_REQUEST = 400,
		RESPONSE_UNAUTHORIZED = 401,
		RESPONSE_PAYMENT_REQUIRED = 402,
		RESPONSE_FORBIDDEN = 403,
		RESPONSE_NOT_FOUND = 404,
		RESPONSE_METHOD_NOT_ALLOWED = 405,
		RESPONSE_NOT_ACCEPTABLE = 406,
		RESPONSE_PROXY_AUTHENTICATION_REQUIRED = 407,
		RESPONSE_REQUEST_TIMEOUT = 408,
		RESPONSE_CONFLICT = 409,
		RESPONSE_GONE = 410,
		RESPONSE_LENGTH_REQUIRED = 411,
		RESPONSE_PRECONDITION_FAILED = 412,
		RESPONSE_REQUEST_ENTITY_TOO_LARGE = 413,
		RESPONSE_REQUEST_URI_TOO_LONG = 414,
		RESPONSE_UNSUPPORTED_MEDIA_TYPE = 415,
		RESPONSE_REQUESTED_RANGE_NOT_SATISFIABLE = 416,
		RESPONSE_EXPECTATION_FAILED = 417,
		RESPONSE_UNPROCESSABLE_ENTITY = 422,
		RESPONSE_LOCKED = 423,
		RESPONSE_FAILED_DEPENDENCY = 424,
		RESPONSE_UPGRADE_REQUIRED = 426,

		// 5xx server error
		RESPONSE_INTERNAL_SERVER_ERROR = 500,
		RESPONSE_NOT_IMPLEMENTED = 501,
		RESPONSE_BAD_GATEWAY = 502,
		RESPONSE_SERVICE_UNAVAILABLE = 503,
		RESPONSE_GATEWAY_TIMEOUT = 504,
		RESPONSE_HTTP_VERSION_NOT_SUPPORTED = 505,
		RESPONSE_INSUFFICIENT_STORAGE = 507,
		RESPONSE_NOT_EXTENDED = 510,

	};

	enum Method {

		METHOD_GET,
		METHOD_HEAD,
		METHOD_POST,
		METHOD_PUT,
		METHOD_DELETE,
		METHOD_OPTIONS,
		METHOD_TRACE,
		METHOD_CONNECT,
		METHOD_MAX
	};

	enum Status {
		STATUS_DISCONNECTED,
		STATUS_RESOLVING, //resolving hostname (if passed a hostname)
		STATUS_CANT_RESOLVE,
		STATUS_CONNECTING, //connecting to ip
		STATUS_CANT_CONNECT,
		STATUS_CONNECTED, //connected, requests only accepted here
		STATUS_REQUESTING, // request in progress
		STATUS_BODY, // request resulted in body, which must be read
		STATUS_CONNECTION_ERROR,
		STATUS_SSL_HANDSHAKE_ERROR,

	};

private:
	Status status;
	IP::ResolverID resolving;
	int conn_port;
	String conn_host;
	bool ssl;
	bool ssl_verify_host;
	bool blocking;

	Vector<uint8_t> response_str;

	bool chunked;
	Vector<uint8_t> chunk;
	int chunk_left;
	int body_size;
	int body_left;

	Ref<StreamPeerTCP> tcp_connection;
	Ref<StreamPeer> connection;

	int response_num;
	Vector<String> response_headers;

	static void _bind_methods();
	PoolStringArray _get_response_headers();
	Dictionary _get_response_headers_as_dictionary();
	int read_chunk_size;

	Error _get_http_data(uint8_t *p_buffer, int p_bytes, int &r_received);

public:
	//Error connect_and_get(const String& p_url,bool p_verify_host=true); //connects to a full url and perform request
	Error connect_to_host(const String &p_host, int p_port, bool p_ssl = false, bool p_verify_host = true);

	void set_connection(const Ref<StreamPeer> &p_connection);
	Ref<StreamPeer> get_connection() const;

	Error request_raw(Method p_method, const String &p_url, const Vector<String> &p_headers, const PoolVector<uint8_t> &p_body);
	Error request(Method p_method, const String &p_url, const Vector<String> &p_headers, const String &p_body = String());
	Error send_body_text(const String &p_body);
	Error send_body_data(const PoolByteArray &p_body);

	void close();

	Status get_status() const;

	bool has_response() const;
	bool is_response_chunked() const;
	int get_response_code() const;
	Error get_response_headers(List<String> *r_response);
	int get_response_body_length() const;

	PoolByteArray read_response_body_chunk(); // can't get body as partial text because of most encodings UTF8, gzip, etc.

	void set_blocking_mode(bool p_enable); //useful mostly if running in a thread
	bool is_blocking_mode_enabled() const;

	void set_read_chunk_size(int p_size);

	Error poll();

	String query_string_from_dict(const Dictionary &p_dict);

	HTTPClient();
	~HTTPClient();
};

VARIANT_ENUM_CAST(HTTPClient::Method);
VARIANT_ENUM_CAST(HTTPClient::Status);

#endif // HTTP_CLIENT_H
