/*************************************************************************/
/*  stream_peer_tcp.cpp                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "stream_peer_tcp.h"

StreamPeerTCP *(*StreamPeerTCP::_create)() = NULL;

Error StreamPeerTCP::_connect(const String &p_address, int p_port) {

	IP_Address ip;
	if (p_address.is_valid_ip_address()) {
		ip = p_address;
	} else {
		ip = IP::get_singleton()->resolve_hostname(p_address);
		if (!ip.is_valid())
			return ERR_CANT_RESOLVE;
	}

	connect(ip, p_port);
	return OK;
}

void StreamPeerTCP::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("connect", "host", "port"), &StreamPeerTCP::_connect);
	ObjectTypeDB::bind_method(_MD("is_connected"), &StreamPeerTCP::is_connected);
	ObjectTypeDB::bind_method(_MD("get_status"), &StreamPeerTCP::get_status);
	ObjectTypeDB::bind_method(_MD("get_connected_host"), &StreamPeerTCP::get_connected_host);
	ObjectTypeDB::bind_method(_MD("get_connected_port"), &StreamPeerTCP::get_connected_port);
	ObjectTypeDB::bind_method(_MD("disconnect"), &StreamPeerTCP::disconnect);

	BIND_CONSTANT(STATUS_NONE);
	BIND_CONSTANT(STATUS_CONNECTING);
	BIND_CONSTANT(STATUS_CONNECTED);
	BIND_CONSTANT(STATUS_ERROR);
}

Ref<StreamPeerTCP> StreamPeerTCP::create_ref() {

	if (!_create)
		return Ref<StreamPeerTCP>();
	return Ref<StreamPeerTCP>(_create());
}

StreamPeerTCP *StreamPeerTCP::create() {

	if (!_create)
		return NULL;
	return _create();
}

StreamPeerTCP::StreamPeerTCP() {
}

StreamPeerTCP::~StreamPeerTCP(){

};
