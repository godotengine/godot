/*************************************************************************/
/*  stream_peer_ssl.cpp                                                  */
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
#include "stream_peer_ssl.h"

StreamPeerSSL *(*StreamPeerSSL::_create)() = NULL;

StreamPeerSSL *StreamPeerSSL::create() {

	return _create();
}

StreamPeerSSL::LoadCertsFromMemory StreamPeerSSL::load_certs_func = NULL;
bool StreamPeerSSL::available = false;
bool StreamPeerSSL::initialize_certs = true;

void StreamPeerSSL::load_certs_from_memory(const PoolByteArray &p_memory) {
	if (load_certs_func)
		load_certs_func(p_memory);
}

bool StreamPeerSSL::is_available() {
	return available;
}

void StreamPeerSSL::_bind_methods() {

	ClassDB::bind_method(D_METHOD("accept_stream:Error", "stream:StreamPeer"), &StreamPeerSSL::accept_stream);
	ClassDB::bind_method(D_METHOD("connect_to_stream:Error", "stream:StreamPeer", "validate_certs", "for_hostname"), &StreamPeerSSL::connect_to_stream, DEFVAL(false), DEFVAL(String()));
	ClassDB::bind_method(D_METHOD("get_status"), &StreamPeerSSL::get_status);
	ClassDB::bind_method(D_METHOD("disconnect_from_stream"), &StreamPeerSSL::disconnect_from_stream);
	BIND_CONSTANT(STATUS_DISCONNECTED);
	BIND_CONSTANT(STATUS_CONNECTED);
	BIND_CONSTANT(STATUS_ERROR_NO_CERTIFICATE);
	BIND_CONSTANT(STATUS_ERROR_HOSTNAME_MISMATCH);
}

StreamPeerSSL::StreamPeerSSL() {
}
