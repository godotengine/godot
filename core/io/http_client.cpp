/*************************************************************************/
/*  http_client.cpp                                                      */
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
#include "http_client.h"
#include "io/stream_peer_ssl.h"

Error HTTPClient::connect(const String &p_host, int p_port, bool p_ssl, bool p_verify_host) {

	close();
	conn_port = p_port;
	conn_host = p_host;

	if (conn_host.begins_with("http://")) {

		conn_host = conn_host.replace_first("http://", "");
	} else if (conn_host.begins_with("https://")) {
		//use https
		conn_host = conn_host.replace_first("https://", "");
	}

	ssl = p_ssl;
	ssl_verify_host = p_verify_host;
	connection = tcp_connection;

	if (conn_host.is_valid_ip_address()) {
		//is ip
		Error err = tcp_connection->connect(IP_Address(conn_host), p_port);
		if (err) {
			status = STATUS_CANT_CONNECT;
			return err;
		}

		status = STATUS_CONNECTING;
	} else {
		//is hostname
		resolving = IP::get_singleton()->resolve_hostname_queue_item(conn_host);
		status = STATUS_RESOLVING;
	}

	return OK;
}

void HTTPClient::set_connection(const Ref<StreamPeer> &p_connection) {

	close();
	connection = p_connection;
	status = STATUS_CONNECTED;
}

Ref<StreamPeer> HTTPClient::get_connection() const {

	return connection;
}

Error HTTPClient::request_raw(Method p_method, const String &p_url, const Vector<String> &p_headers, const DVector<uint8_t> &p_body) {

	ERR_FAIL_INDEX_V(p_method, METHOD_MAX, ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(status != STATUS_CONNECTED, ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(connection.is_null(), ERR_INVALID_DATA);

	static const char *_methods[METHOD_MAX] = {
		"GET",
		"HEAD",
		"POST",
		"PUT",
		"DELETE",
		"OPTIONS",
		"TRACE",
		"CONNECT"
	};

	String request = String(_methods[p_method]) + " " + p_url + " HTTP/1.1\r\n";
	if ((ssl && conn_port == 443) || (!ssl && conn_port == 80)) {
		// don't append the standard ports
		request += "Host: " + conn_host + "\r\n";
	} else {
		request += "Host: " + conn_host + ":" + itos(conn_port) + "\r\n";
	}
	bool add_clen = p_body.size() > 0;
	for (int i = 0; i < p_headers.size(); i++) {
		request += p_headers[i] + "\r\n";
		if (add_clen && p_headers[i].find("Content-Length:") == 0) {
			add_clen = false;
		}
	}
	if (add_clen) {
		request += "Content-Length: " + itos(p_body.size()) + "\r\n";
		//should it add utf8 encoding? not sure
	}
	request += "\r\n";
	CharString cs = request.utf8();

	DVector<uint8_t> data;

	//Maybe this goes faster somehow?
	for (int i = 0; i < cs.length(); i++) {
		data.append(cs[i]);
	}
	data.append_array(p_body);

	DVector<uint8_t>::Read r = data.read();
	Error err = connection->put_data(&r[0], data.size());

	if (err) {
		close();
		status = STATUS_CONNECTION_ERROR;
		return err;
	}

	status = STATUS_REQUESTING;

	return OK;
}

Error HTTPClient::request(Method p_method, const String &p_url, const Vector<String> &p_headers, const String &p_body) {

	ERR_FAIL_INDEX_V(p_method, METHOD_MAX, ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(status != STATUS_CONNECTED, ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(connection.is_null(), ERR_INVALID_DATA);

	static const char *_methods[METHOD_MAX] = {
		"GET",
		"HEAD",
		"POST",
		"PUT",
		"DELETE",
		"OPTIONS",
		"TRACE",
		"CONNECT"
	};

	String request = String(_methods[p_method]) + " " + p_url + " HTTP/1.1\r\n";
	if ((ssl && conn_port == 443) || (!ssl && conn_port == 80)) {
		// don't append the standard ports
		request += "Host: " + conn_host + "\r\n";
	} else {
		request += "Host: " + conn_host + ":" + itos(conn_port) + "\r\n";
	}
	bool add_clen = p_body.length() > 0;
	for (int i = 0; i < p_headers.size(); i++) {
		request += p_headers[i] + "\r\n";
		if (add_clen && p_headers[i].find("Content-Length:") == 0) {
			add_clen = false;
		}
	}
	if (add_clen) {
		request += "Content-Length: " + itos(p_body.utf8().length()) + "\r\n";
		//should it add utf8 encoding? not sure
	}
	request += "\r\n";
	request += p_body;

	CharString cs = request.utf8();
	Error err = connection->put_data((const uint8_t *)cs.ptr(), cs.length());
	if (err) {
		close();
		status = STATUS_CONNECTION_ERROR;
		return err;
	}

	status = STATUS_REQUESTING;

	return OK;
}

Error HTTPClient::send_body_text(const String &p_body) {

	return OK;
}

Error HTTPClient::send_body_data(const ByteArray &p_body) {

	return OK;
}

bool HTTPClient::has_response() const {

	return response_headers.size() != 0;
}

bool HTTPClient::is_response_chunked() const {

	return chunked;
}

int HTTPClient::get_response_code() const {

	return response_num;
}

Error HTTPClient::get_response_headers(List<String> *r_response) {

	if (!response_headers.size())
		return ERR_INVALID_PARAMETER;

	for (int i = 0; i < response_headers.size(); i++) {

		r_response->push_back(response_headers[i]);
	}

	response_headers.clear();

	return OK;
}

void HTTPClient::close() {

	if (tcp_connection->get_status() != StreamPeerTCP::STATUS_NONE)
		tcp_connection->disconnect();

	connection.unref();
	status = STATUS_DISCONNECTED;
	if (resolving != IP::RESOLVER_INVALID_ID) {

		IP::get_singleton()->erase_resolve_item(resolving);
		resolving = IP::RESOLVER_INVALID_ID;
	}

	response_headers.clear();
	response_str.clear();
	body_size = 0;
	body_left = 0;
	chunk_left = 0;
	response_num = 0;
}

Error HTTPClient::poll() {

	switch (status) {

		case STATUS_RESOLVING: {
			ERR_FAIL_COND_V(resolving == IP::RESOLVER_INVALID_ID, ERR_BUG);

			IP::ResolverStatus rstatus = IP::get_singleton()->get_resolve_item_status(resolving);
			switch (rstatus) {
				case IP::RESOLVER_STATUS_WAITING:
					return OK; //still resolving

				case IP::RESOLVER_STATUS_DONE: {

					IP_Address host = IP::get_singleton()->get_resolve_item_address(resolving);
					Error err = tcp_connection->connect(host, conn_port);
					IP::get_singleton()->erase_resolve_item(resolving);
					resolving = IP::RESOLVER_INVALID_ID;
					if (err) {
						status = STATUS_CANT_CONNECT;
						return err;
					}

					status = STATUS_CONNECTING;
				} break;
				case IP::RESOLVER_STATUS_NONE:
				case IP::RESOLVER_STATUS_ERROR: {

					IP::get_singleton()->erase_resolve_item(resolving);
					resolving = IP::RESOLVER_INVALID_ID;
					close();
					status = STATUS_CANT_RESOLVE;
					return ERR_CANT_RESOLVE;
				} break;
			}
		} break;
		case STATUS_CONNECTING: {

			StreamPeerTCP::Status s = tcp_connection->get_status();
			switch (s) {

				case StreamPeerTCP::STATUS_CONNECTING: {
					return OK; //do none
				} break;
				case StreamPeerTCP::STATUS_CONNECTED: {
					if (ssl) {
						Ref<StreamPeerSSL> ssl = StreamPeerSSL::create();
						Error err = ssl->connect(tcp_connection, true, ssl_verify_host ? conn_host : String());
						if (err != OK) {
							close();
							status = STATUS_SSL_HANDSHAKE_ERROR;
							return ERR_CANT_CONNECT;
						}
						//print_line("SSL! TURNED ON!");
						connection = ssl;
					}
					status = STATUS_CONNECTED;
					return OK;
				} break;
				case StreamPeerTCP::STATUS_ERROR:
				case StreamPeerTCP::STATUS_NONE: {

					close();
					status = STATUS_CANT_CONNECT;
					return ERR_CANT_CONNECT;
				} break;
			}
		} break;
		case STATUS_CONNECTED: {
			//request something please
			return OK;
		} break;
		case STATUS_REQUESTING: {

			while (true) {
				uint8_t byte;
				int rec = 0;
				Error err = _get_http_data(&byte, 1, rec);
				if (err != OK) {
					close();
					status = STATUS_CONNECTION_ERROR;
					return ERR_CONNECTION_ERROR;
				}

				if (rec == 0)
					return OK; //keep trying!

				response_str.push_back(byte);
				int rs = response_str.size();
				if (
						(rs >= 2 && response_str[rs - 2] == '\n' && response_str[rs - 1] == '\n') ||
						(rs >= 4 && response_str[rs - 4] == '\r' && response_str[rs - 3] == '\n' && rs >= 4 && response_str[rs - 2] == '\r' && response_str[rs - 1] == '\n')) {

					//end of response, parse.
					response_str.push_back(0);
					String response;
					response.parse_utf8((const char *)response_str.ptr());
					//print_line("END OF RESPONSE? :\n"+response+"\n------");
					Vector<String> responses = response.split("\n");
					body_size = 0;
					chunked = false;
					body_left = 0;
					chunk_left = 0;
					response_str.clear();
					response_headers.clear();
					response_num = RESPONSE_OK;

					for (int i = 0; i < responses.size(); i++) {

						String header = responses[i].strip_edges();
						String s = header.to_lower();
						if (s.length() == 0)
							continue;
						if (s.begins_with("content-length:")) {
							body_size = s.substr(s.find(":") + 1, s.length()).strip_edges().to_int();
							body_left = body_size;
						}

						if (s.begins_with("transfer-encoding:")) {
							String encoding = header.substr(header.find(":") + 1, header.length()).strip_edges();
							//print_line("TRANSFER ENCODING: "+encoding);
							if (encoding == "chunked") {
								chunked = true;
							}
						}

						if (i == 0 && responses[i].begins_with("HTTP")) {

							String num = responses[i].get_slicec(' ', 1);
							response_num = num.to_int();
						} else {

							response_headers.push_back(header);
						}
					}

					if (body_size == 0 && !chunked) {

						status = STATUS_CONNECTED; //ask for something again?
					} else {
						status = STATUS_BODY;
					}
					return OK;
				}
			}
			//wait for response
			return OK;
		} break;
		case STATUS_DISCONNECTED: {
			return ERR_UNCONFIGURED;
		} break;
		case STATUS_CONNECTION_ERROR: {
			return ERR_CONNECTION_ERROR;
		} break;
		case STATUS_CANT_CONNECT: {
			return ERR_CANT_CONNECT;
		} break;
		case STATUS_CANT_RESOLVE: {
			return ERR_CANT_RESOLVE;
		} break;
	}

	return OK;
}

Dictionary HTTPClient::_get_response_headers_as_dictionary() {

	List<String> rh;
	get_response_headers(&rh);
	Dictionary ret;
	for (const List<String>::Element *E = rh.front(); E; E = E->next()) {
		String s = E->get();
		int sp = s.find(":");
		if (sp == -1)
			continue;
		String key = s.substr(0, sp).strip_edges();
		String value = s.substr(sp + 1, s.length()).strip_edges();
		ret[key] = value;
	}

	return ret;
}

StringArray HTTPClient::_get_response_headers() {

	List<String> rh;
	get_response_headers(&rh);
	StringArray ret;
	ret.resize(rh.size());
	int idx = 0;
	for (const List<String>::Element *E = rh.front(); E; E = E->next()) {
		ret.set(idx++, E->get());
	}

	return ret;
}

int HTTPClient::get_response_body_length() const {

	return body_size;
}

ByteArray HTTPClient::read_response_body_chunk() {

	ERR_FAIL_COND_V(status != STATUS_BODY, ByteArray());

	Error err = OK;

	if (chunked) {

		while (true) {

			if (chunk_left == 0) {
				//reading len
				uint8_t b;
				int rec = 0;
				err = _get_http_data(&b, 1, rec);

				if (rec == 0)
					break;

				chunk.push_back(b);

				if (chunk.size() > 32) {
					ERR_PRINT("HTTP Invalid chunk hex len");
					status = STATUS_CONNECTION_ERROR;
					return ByteArray();
				}

				if (chunk.size() > 2 && chunk[chunk.size() - 2] == '\r' && chunk[chunk.size() - 1] == '\n') {

					int len = 0;
					for (int i = 0; i < chunk.size() - 2; i++) {
						char c = chunk[i];
						int v = 0;
						if (c >= '0' && c <= '9')
							v = c - '0';
						else if (c >= 'a' && c <= 'f')
							v = c - 'a' + 10;
						else if (c >= 'A' && c <= 'F')
							v = c - 'A' + 10;
						else {
							ERR_PRINT("HTTP Chunk len not in hex!!");
							status = STATUS_CONNECTION_ERROR;
							return ByteArray();
						}
						len <<= 4;
						len |= v;
						if (len > (1 << 24)) {
							ERR_PRINT("HTTP Chunk too big!! >16mb");
							status = STATUS_CONNECTION_ERROR;
							return ByteArray();
						}
					}

					if (len == 0) {
						//end!
						status = STATUS_CONNECTED;
						chunk.clear();
						return ByteArray();
					}

					chunk_left = len + 2;
					chunk.resize(chunk_left);
				}
			} else {

				int rec = 0;
				err = _get_http_data(&chunk[chunk.size() - chunk_left], chunk_left, rec);
				if (rec == 0) {
					break;
				}
				chunk_left -= rec;

				if (chunk_left == 0) {

					if (chunk[chunk.size() - 2] != '\r' || chunk[chunk.size() - 1] != '\n') {
						ERR_PRINT("HTTP Invalid chunk terminator (not \\r\\n)");
						status = STATUS_CONNECTION_ERROR;
						return ByteArray();
					}

					ByteArray ret;
					ret.resize(chunk.size() - 2);
					{
						ByteArray::Write w = ret.write();
						copymem(w.ptr(), chunk.ptr(), chunk.size() - 2);
					}
					chunk.clear();

					return ret;
				}

				break;
			}
		}

	} else {

		int to_read = MIN(body_left, read_chunk_size);
		ByteArray ret;
		ret.resize(to_read);
		int _offset = 0;
		while (to_read > 0) {
			int rec = 0;
			{
				ByteArray::Write w = ret.write();
				err = _get_http_data(w.ptr() + _offset, to_read, rec);
			}
			if (rec > 0) {
				body_left -= rec;
				to_read -= rec;
				_offset += rec;
			} else {
				if (to_read > 0) //ended up reading less
					ret.resize(_offset);
				break;
			}
		}
		if (body_left == 0) {
			status = STATUS_CONNECTED;
		}
		return ret;
	}

	if (err != OK) {
		close();
		if (err == ERR_FILE_EOF) {

			status = STATUS_DISCONNECTED; //server disconnected
		} else {

			status = STATUS_CONNECTION_ERROR;
		}
	} else if (body_left == 0 && !chunked) {

		status = STATUS_CONNECTED;
	}

	return ByteArray();
}

HTTPClient::Status HTTPClient::get_status() const {

	return status;
}

void HTTPClient::set_blocking_mode(bool p_enable) {

	blocking = p_enable;
}

bool HTTPClient::is_blocking_mode_enabled() const {

	return blocking;
}

Error HTTPClient::_get_http_data(uint8_t *p_buffer, int p_bytes, int &r_received) {

	if (blocking) {

		Error err = connection->get_data(p_buffer, p_bytes);
		if (err == OK)
			r_received = p_bytes;
		else
			r_received = 0;
		return err;
	} else {
		return connection->get_partial_data(p_buffer, p_bytes, r_received);
	}
}

void HTTPClient::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("connect:Error", "host", "port", "use_ssl", "verify_host"), &HTTPClient::connect, DEFVAL(false), DEFVAL(true));
	ObjectTypeDB::bind_method(_MD("set_connection", "connection:StreamPeer"), &HTTPClient::set_connection);
	ObjectTypeDB::bind_method(_MD("get_connection:StreamPeer"), &HTTPClient::get_connection);
	ObjectTypeDB::bind_method(_MD("request_raw", "method", "url", "headers", "body"), &HTTPClient::request_raw);
	ObjectTypeDB::bind_method(_MD("request", "method", "url", "headers", "body"), &HTTPClient::request, DEFVAL(String()));
	ObjectTypeDB::bind_method(_MD("send_body_text", "body"), &HTTPClient::send_body_text);
	ObjectTypeDB::bind_method(_MD("send_body_data", "body"), &HTTPClient::send_body_data);
	ObjectTypeDB::bind_method(_MD("close"), &HTTPClient::close);

	ObjectTypeDB::bind_method(_MD("has_response"), &HTTPClient::has_response);
	ObjectTypeDB::bind_method(_MD("is_response_chunked"), &HTTPClient::is_response_chunked);
	ObjectTypeDB::bind_method(_MD("get_response_code"), &HTTPClient::get_response_code);
	ObjectTypeDB::bind_method(_MD("get_response_headers"), &HTTPClient::_get_response_headers);
	ObjectTypeDB::bind_method(_MD("get_response_headers_as_dictionary"), &HTTPClient::_get_response_headers_as_dictionary);
	ObjectTypeDB::bind_method(_MD("get_response_body_length"), &HTTPClient::get_response_body_length);
	ObjectTypeDB::bind_method(_MD("read_response_body_chunk"), &HTTPClient::read_response_body_chunk);
	ObjectTypeDB::bind_method(_MD("set_read_chunk_size", "bytes"), &HTTPClient::set_read_chunk_size);

	ObjectTypeDB::bind_method(_MD("set_blocking_mode", "enabled"), &HTTPClient::set_blocking_mode);
	ObjectTypeDB::bind_method(_MD("is_blocking_mode_enabled"), &HTTPClient::is_blocking_mode_enabled);

	ObjectTypeDB::bind_method(_MD("get_status"), &HTTPClient::get_status);
	ObjectTypeDB::bind_method(_MD("poll:Error"), &HTTPClient::poll);

	ObjectTypeDB::bind_method(_MD("query_string_from_dict:String", "fields"), &HTTPClient::query_string_from_dict);

	BIND_CONSTANT(METHOD_GET);
	BIND_CONSTANT(METHOD_HEAD);
	BIND_CONSTANT(METHOD_POST);
	BIND_CONSTANT(METHOD_PUT);
	BIND_CONSTANT(METHOD_DELETE);
	BIND_CONSTANT(METHOD_OPTIONS);
	BIND_CONSTANT(METHOD_TRACE);
	BIND_CONSTANT(METHOD_CONNECT);
	BIND_CONSTANT(METHOD_MAX);

	BIND_CONSTANT(STATUS_DISCONNECTED);
	BIND_CONSTANT(STATUS_RESOLVING); //resolving hostname (if passed a hostname)
	BIND_CONSTANT(STATUS_CANT_RESOLVE);
	BIND_CONSTANT(STATUS_CONNECTING); //connecting to ip
	BIND_CONSTANT(STATUS_CANT_CONNECT);
	BIND_CONSTANT(STATUS_CONNECTED); //connected );  requests only accepted here
	BIND_CONSTANT(STATUS_REQUESTING); // request in progress
	BIND_CONSTANT(STATUS_BODY); // request resulted in body );  which must be read
	BIND_CONSTANT(STATUS_CONNECTION_ERROR);
	BIND_CONSTANT(STATUS_SSL_HANDSHAKE_ERROR);

	BIND_CONSTANT(RESPONSE_CONTINUE);
	BIND_CONSTANT(RESPONSE_SWITCHING_PROTOCOLS);
	BIND_CONSTANT(RESPONSE_PROCESSING);

	// 2xx successful
	BIND_CONSTANT(RESPONSE_OK);
	BIND_CONSTANT(RESPONSE_CREATED);
	BIND_CONSTANT(RESPONSE_ACCEPTED);
	BIND_CONSTANT(RESPONSE_NON_AUTHORITATIVE_INFORMATION);
	BIND_CONSTANT(RESPONSE_NO_CONTENT);
	BIND_CONSTANT(RESPONSE_RESET_CONTENT);
	BIND_CONSTANT(RESPONSE_PARTIAL_CONTENT);
	BIND_CONSTANT(RESPONSE_MULTI_STATUS);
	BIND_CONSTANT(RESPONSE_IM_USED);

	// 3xx redirection
	BIND_CONSTANT(RESPONSE_MULTIPLE_CHOICES);
	BIND_CONSTANT(RESPONSE_MOVED_PERMANENTLY);
	BIND_CONSTANT(RESPONSE_FOUND);
	BIND_CONSTANT(RESPONSE_SEE_OTHER);
	BIND_CONSTANT(RESPONSE_NOT_MODIFIED);
	BIND_CONSTANT(RESPONSE_USE_PROXY);
	BIND_CONSTANT(RESPONSE_TEMPORARY_REDIRECT);

	// 4xx client error
	BIND_CONSTANT(RESPONSE_BAD_REQUEST);
	BIND_CONSTANT(RESPONSE_UNAUTHORIZED);
	BIND_CONSTANT(RESPONSE_PAYMENT_REQUIRED);
	BIND_CONSTANT(RESPONSE_FORBIDDEN);
	BIND_CONSTANT(RESPONSE_NOT_FOUND);
	BIND_CONSTANT(RESPONSE_METHOD_NOT_ALLOWED);
	BIND_CONSTANT(RESPONSE_NOT_ACCEPTABLE);
	BIND_CONSTANT(RESPONSE_PROXY_AUTHENTICATION_REQUIRED);
	BIND_CONSTANT(RESPONSE_REQUEST_TIMEOUT);
	BIND_CONSTANT(RESPONSE_CONFLICT);
	BIND_CONSTANT(RESPONSE_GONE);
	BIND_CONSTANT(RESPONSE_LENGTH_REQUIRED);
	BIND_CONSTANT(RESPONSE_PRECONDITION_FAILED);
	BIND_CONSTANT(RESPONSE_REQUEST_ENTITY_TOO_LARGE);
	BIND_CONSTANT(RESPONSE_REQUEST_URI_TOO_LONG);
	BIND_CONSTANT(RESPONSE_UNSUPPORTED_MEDIA_TYPE);
	BIND_CONSTANT(RESPONSE_REQUESTED_RANGE_NOT_SATISFIABLE);
	BIND_CONSTANT(RESPONSE_EXPECTATION_FAILED);
	BIND_CONSTANT(RESPONSE_UNPROCESSABLE_ENTITY);
	BIND_CONSTANT(RESPONSE_LOCKED);
	BIND_CONSTANT(RESPONSE_FAILED_DEPENDENCY);
	BIND_CONSTANT(RESPONSE_UPGRADE_REQUIRED);

	// 5xx server error
	BIND_CONSTANT(RESPONSE_INTERNAL_SERVER_ERROR);
	BIND_CONSTANT(RESPONSE_NOT_IMPLEMENTED);
	BIND_CONSTANT(RESPONSE_BAD_GATEWAY);
	BIND_CONSTANT(RESPONSE_SERVICE_UNAVAILABLE);
	BIND_CONSTANT(RESPONSE_GATEWAY_TIMEOUT);
	BIND_CONSTANT(RESPONSE_HTTP_VERSION_NOT_SUPPORTED);
	BIND_CONSTANT(RESPONSE_INSUFFICIENT_STORAGE);
	BIND_CONSTANT(RESPONSE_NOT_EXTENDED);
}

void HTTPClient::set_read_chunk_size(int p_size) {
	ERR_FAIL_COND(p_size < 256 || p_size > (1 << 24));
	read_chunk_size = p_size;
}

String HTTPClient::query_string_from_dict(const Dictionary &p_dict) {
	String query = "";
	Array keys = p_dict.keys();
	for (int i = 0; i < keys.size(); ++i) {
		query += "&" + String(keys[i]).http_escape() + "=" + String(p_dict[keys[i]]).http_escape();
	}
	query.erase(0, 1);
	return query;
}

HTTPClient::HTTPClient() {

	tcp_connection = StreamPeerTCP::create_ref();
	resolving = IP::RESOLVER_INVALID_ID;
	status = STATUS_DISCONNECTED;
	conn_port = 80;
	body_size = 0;
	chunked = false;
	body_left = 0;
	chunk_left = 0;
	response_num = 0;
	ssl = false;
	blocking = false;
	read_chunk_size = 4096;
}

HTTPClient::~HTTPClient() {
}
