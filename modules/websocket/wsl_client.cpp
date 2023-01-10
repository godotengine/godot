/**************************************************************************/
/*  wsl_client.cpp                                                        */
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

#ifndef JAVASCRIPT_ENABLED

#include "wsl_client.h"
#include "core/io/ip.h"
#include "core/project_settings.h"

void WSLClient::_do_handshake() {
	if (_requested < _request.size() - 1) {
		int sent = 0;
		Error err = _connection->put_partial_data(((const uint8_t *)_request.get_data() + _requested), _request.size() - _requested - 1, sent);
		// Sending handshake failed
		if (err != OK) {
			disconnect_from_host();
			_on_error();
			return;
		}
		_requested += sent;

	} else {
		int read = 0;
		while (true) {
			if (_resp_pos >= WSL_MAX_HEADER_SIZE) {
				// Header is too big
				disconnect_from_host();
				_on_error();
				ERR_FAIL_MSG("Response headers too big.");
			}
			Error err = _connection->get_partial_data(&_resp_buf[_resp_pos], 1, read);
			if (err == ERR_FILE_EOF) {
				// We got a disconnect.
				disconnect_from_host();
				_on_error();
				return;
			} else if (err != OK) {
				// Got some error.
				disconnect_from_host();
				_on_error();
				return;
			} else if (read != 1) {
				// Busy, wait next poll.
				break;
			}
			// Check "\r\n\r\n" header terminator
			char *r = (char *)_resp_buf;
			int l = _resp_pos;
			if (l > 3 && r[l] == '\n' && r[l - 1] == '\r' && r[l - 2] == '\n' && r[l - 3] == '\r') {
				r[l - 3] = '\0';
				String protocol;
				// Response is over, verify headers and create peer.
				if (!_verify_headers(protocol)) {
					disconnect_from_host();
					_on_error();
					ERR_FAIL_MSG("Invalid response headers.");
				}
				// Create peer.
				WSLPeer::PeerData *data = memnew(struct WSLPeer::PeerData);
				data->obj = this;
				data->conn = _connection;
				data->tcp = _tcp;
				data->is_server = false;
				data->id = 1;
				_peer->make_context(data, _in_buf_size, _in_pkt_size, _out_buf_size, _out_pkt_size);
				_peer->set_no_delay(true);
				_on_connect(protocol);
				break;
			}
			_resp_pos += 1;
		}
	}
}

bool WSLClient::_verify_headers(String &r_protocol) {
	String s = (char *)_resp_buf;
	Vector<String> psa = s.split("\r\n");
	int len = psa.size();
	ERR_FAIL_COND_V_MSG(len < 4, false, "Not enough response headers, got: " + itos(len) + ", expected >= 4.");

	Vector<String> req = psa[0].split(" ", false);
	ERR_FAIL_COND_V_MSG(req.size() < 2, false, "Invalid protocol or status code.");

	// Wrong protocol
	ERR_FAIL_COND_V_MSG(req[0] != "HTTP/1.1" || req[1] != "101", false, "Invalid protocol or status code.");

	Map<String, String> headers;
	for (int i = 1; i < len; i++) {
		Vector<String> header = psa[i].split(":", false, 1);
		ERR_FAIL_COND_V_MSG(header.size() != 2, false, "Invalid header -> " + psa[i] + ".");
		String name = header[0].to_lower();
		String value = header[1].strip_edges();
		if (headers.has(name)) {
			headers[name] += "," + value;
		} else {
			headers[name] = value;
		}
	}

#define _WSL_CHECK(NAME, VALUE)                                                         \
	ERR_FAIL_COND_V_MSG(!headers.has(NAME) || headers[NAME].to_lower() != VALUE, false, \
			"Missing or invalid header '" + String(NAME) + "'. Expected value '" + VALUE + "'.");
#define _WSL_CHECK_NC(NAME, VALUE)                                           \
	ERR_FAIL_COND_V_MSG(!headers.has(NAME) || headers[NAME] != VALUE, false, \
			"Missing or invalid header '" + String(NAME) + "'. Expected value '" + VALUE + "'.");
	_WSL_CHECK("connection", "upgrade");
	_WSL_CHECK("upgrade", "websocket");
	_WSL_CHECK_NC("sec-websocket-accept", WSLPeer::compute_key_response(_key));
#undef _WSL_CHECK_NC
#undef _WSL_CHECK
	if (_protocols.size() == 0) {
		// We didn't request a custom protocol
		ERR_FAIL_COND_V(headers.has("sec-websocket-protocol"), false);
	} else {
		ERR_FAIL_COND_V(!headers.has("sec-websocket-protocol"), false);
		r_protocol = headers["sec-websocket-protocol"];
		bool valid = false;
		for (int i = 0; i < _protocols.size(); i++) {
			if (_protocols[i] != r_protocol) {
				continue;
			}
			valid = true;
			break;
		}
		if (!valid) {
			return false;
		}
	}
	return true;
}

Error WSLClient::connect_to_host(String p_host, String p_path, uint16_t p_port, bool p_ssl, const Vector<String> p_protocols, const Vector<String> p_custom_headers) {
	ERR_FAIL_COND_V(_connection.is_valid(), ERR_ALREADY_IN_USE);
	ERR_FAIL_COND_V(p_path.empty(), ERR_INVALID_PARAMETER);

	_peer = Ref<WSLPeer>(memnew(WSLPeer));

	if (p_host.is_valid_ip_address()) {
		_ip_candidates.push_back(IP_Address(p_host));
	} else {
		// Queue hostname for resolution.
		_resolver_id = IP::get_singleton()->resolve_hostname_queue_item(p_host);
		ERR_FAIL_COND_V(_resolver_id == IP::RESOLVER_INVALID_ID, ERR_INVALID_PARAMETER);
		// Check if it was found in cache.
		IP::ResolverStatus ip_status = IP::get_singleton()->get_resolve_item_status(_resolver_id);
		if (ip_status == IP::RESOLVER_STATUS_DONE) {
			_ip_candidates = IP::get_singleton()->get_resolve_item_addresses(_resolver_id);
			IP::get_singleton()->erase_resolve_item(_resolver_id);
			_resolver_id = IP::RESOLVER_INVALID_ID;
		}
	}

	// We assume OK while hostname resolution is pending.
	Error err = _resolver_id != IP::RESOLVER_INVALID_ID ? OK : FAILED;
	while (_ip_candidates.size()) {
		err = _tcp->connect_to_host(_ip_candidates.pop_front(), p_port);
		if (err == OK) {
			break;
		}
	}
	if (err != OK) {
		_tcp->disconnect_from_host();
		_on_error();
		return err;
	}
	_connection = _tcp;
	_use_ssl = p_ssl;
	_host = p_host;
	_port = p_port;
	// Strip edges from protocols.
	_protocols.resize(p_protocols.size());
	String *pw = _protocols.ptrw();
	for (int i = 0; i < p_protocols.size(); i++) {
		pw[i] = p_protocols[i].strip_edges();
	}

	_key = WSLPeer::generate_key();
	String request = "GET " + p_path + " HTTP/1.1\r\n";
	String port = "";
	if ((p_port != 80 && !p_ssl) || (p_port != 443 && p_ssl)) {
		port = ":" + itos(p_port);
	}
	request += "Host: " + p_host + port + "\r\n";
	request += "Upgrade: websocket\r\n";
	request += "Connection: Upgrade\r\n";
	request += "Sec-WebSocket-Key: " + _key + "\r\n";
	request += "Sec-WebSocket-Version: 13\r\n";
	if (p_protocols.size() > 0) {
		request += "Sec-WebSocket-Protocol: ";
		for (int i = 0; i < p_protocols.size(); i++) {
			if (i != 0) {
				request += ",";
			}
			request += p_protocols[i];
		}
		request += "\r\n";
	}
	for (int i = 0; i < p_custom_headers.size(); i++) {
		request += p_custom_headers[i] + "\r\n";
	}
	request += "\r\n";
	_request = request.utf8();

	return OK;
}

int WSLClient::get_max_packet_size() const {
	return (1 << _out_buf_size) - PROTO_SIZE;
}

void WSLClient::poll() {
	if (_resolver_id != IP::RESOLVER_INVALID_ID) {
		IP::ResolverStatus ip_status = IP::get_singleton()->get_resolve_item_status(_resolver_id);
		if (ip_status == IP::RESOLVER_STATUS_WAITING) {
			return;
		}
		// Anything else is either a candidate or a failure.
		Error err = FAILED;
		if (ip_status == IP::RESOLVER_STATUS_DONE) {
			_ip_candidates = IP::get_singleton()->get_resolve_item_addresses(_resolver_id);
			while (_ip_candidates.size()) {
				err = _tcp->connect_to_host(_ip_candidates.pop_front(), _port);
				if (err == OK) {
					break;
				}
			}
		}
		IP::get_singleton()->erase_resolve_item(_resolver_id);
		_resolver_id = IP::RESOLVER_INVALID_ID;
		if (err != OK) {
			disconnect_from_host();
			_on_error();
			return;
		}
	}
	if (_peer->is_connected_to_host()) {
		_peer->poll();
		if (!_peer->is_connected_to_host()) {
			disconnect_from_host();
			_on_disconnect(_peer->close_code != -1);
		}
		return;
	}

	if (_connection.is_null()) {
		return; // Not connected.
	}

	switch (_tcp->get_status()) {
		case StreamPeerTCP::STATUS_NONE:
			// Clean close
			disconnect_from_host();
			_on_error();
			break;
		case StreamPeerTCP::STATUS_CONNECTED: {
			_ip_candidates.clear();
			Ref<StreamPeerSSL> ssl;
			if (_use_ssl) {
				if (_connection == _tcp) {
					// Start SSL handshake
					ssl = Ref<StreamPeerSSL>(StreamPeerSSL::create());
					ERR_FAIL_COND_MSG(ssl.is_null(), "SSL is not available in this build.");
					ssl->set_blocking_handshake_enabled(false);
					if (ssl->connect_to_stream(_tcp, verify_ssl, _host, ssl_cert) != OK) {
						disconnect_from_host();
						_on_error();
						return;
					}
					_connection = ssl;
				} else {
					ssl = static_cast<Ref<StreamPeerSSL>>(_connection);
					ERR_FAIL_COND(ssl.is_null()); // Bug?
					ssl->poll();
				}
				if (ssl->get_status() == StreamPeerSSL::STATUS_HANDSHAKING) {
					return; // Need more polling.
				} else if (ssl->get_status() != StreamPeerSSL::STATUS_CONNECTED) {
					disconnect_from_host();
					_on_error();
					return; // Error.
				}
			}
			// Do websocket handshake.
			_do_handshake();
		} break;
		case StreamPeerTCP::STATUS_ERROR:
			while (_ip_candidates.size() > 0) {
				_tcp->disconnect_from_host();
				if (_tcp->connect_to_host(_ip_candidates.pop_front(), _port) == OK) {
					return;
				}
			}
			disconnect_from_host();
			_on_error();
			break;
		case StreamPeerTCP::STATUS_CONNECTING:
			break; // Wait for connection
	}
}

Ref<WebSocketPeer> WSLClient::get_peer(int p_peer_id) const {
	ERR_FAIL_COND_V(p_peer_id != 1, nullptr);

	return _peer;
}

NetworkedMultiplayerPeer::ConnectionStatus WSLClient::get_connection_status() const {
	if (_peer->is_connected_to_host()) {
		return CONNECTION_CONNECTED;
	}

	if (_tcp->is_connected_to_host() || _resolver_id != IP::RESOLVER_INVALID_ID) {
		return CONNECTION_CONNECTING;
	}

	return CONNECTION_DISCONNECTED;
}

void WSLClient::disconnect_from_host(int p_code, String p_reason) {
	_peer->close(p_code, p_reason);
	_connection = Ref<StreamPeer>(nullptr);
	_tcp = Ref<StreamPeerTCP>(memnew(StreamPeerTCP));

	_key = "";
	_host = "";
	_protocols.clear();
	_use_ssl = false;

	_request = "";
	_requested = 0;

	memset(_resp_buf, 0, sizeof(_resp_buf));
	_resp_pos = 0;

	if (_resolver_id != IP::RESOLVER_INVALID_ID) {
		IP::get_singleton()->erase_resolve_item(_resolver_id);
		_resolver_id = IP::RESOLVER_INVALID_ID;
	}

	_ip_candidates.clear();
}

IP_Address WSLClient::get_connected_host() const {
	ERR_FAIL_COND_V(!_peer->is_connected_to_host(), IP_Address());
	return _peer->get_connected_host();
}

uint16_t WSLClient::get_connected_port() const {
	ERR_FAIL_COND_V(!_peer->is_connected_to_host(), 0);
	return _peer->get_connected_port();
}

Error WSLClient::set_buffers(int p_in_buffer, int p_in_packets, int p_out_buffer, int p_out_packets) {
	ERR_FAIL_COND_V_MSG(_connection.is_valid(), FAILED, "Buffers sizes can only be set before listening or connecting.");

	_in_buf_size = nearest_shift(p_in_buffer - 1) + 10;
	_in_pkt_size = nearest_shift(p_in_packets - 1);
	_out_buf_size = nearest_shift(p_out_buffer - 1) + 10;
	_out_pkt_size = nearest_shift(p_out_packets - 1);
	return OK;
}

WSLClient::WSLClient() {
	_in_buf_size = nearest_shift((int)GLOBAL_GET(WSC_IN_BUF) - 1) + 10;
	_in_pkt_size = nearest_shift((int)GLOBAL_GET(WSC_IN_PKT) - 1);
	_out_buf_size = nearest_shift((int)GLOBAL_GET(WSC_OUT_BUF) - 1) + 10;
	_out_pkt_size = nearest_shift((int)GLOBAL_GET(WSC_OUT_PKT) - 1);

	_peer.instance();
	_tcp.instance();
	disconnect_from_host();
}

WSLClient::~WSLClient() {
	_peer->close_now();
	_peer->invalidate();
	disconnect_from_host();
}

#endif // JAVASCRIPT_ENABLED
