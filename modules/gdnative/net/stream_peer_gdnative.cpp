/*************************************************************************/
/*  stream_peer_gdnative.cpp                                             */
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

#include "stream_peer_gdnative.h"

StreamPeerGDNative::StreamPeerGDNative() {
	interface = nullptr;
}

StreamPeerGDNative::~StreamPeerGDNative() {
}

void StreamPeerGDNative::set_native_stream_peer(const godot_net_stream_peer *p_interface) {
	interface = p_interface;
}

void StreamPeerGDNative::_bind_methods() {
}

Error StreamPeerGDNative::put_data(const uint8_t *p_data, int p_bytes) {
	ERR_FAIL_COND_V(interface == nullptr, ERR_UNCONFIGURED);
	return (Error)(interface->put_data(interface->data, p_data, p_bytes));
}

Error StreamPeerGDNative::put_partial_data(const uint8_t *p_data, int p_bytes, int &r_sent) {
	ERR_FAIL_COND_V(interface == nullptr, ERR_UNCONFIGURED);
	return (Error)(interface->put_partial_data(interface->data, p_data, p_bytes, &r_sent));
}

Error StreamPeerGDNative::get_data(uint8_t *p_buffer, int p_bytes) {
	ERR_FAIL_COND_V(interface == nullptr, ERR_UNCONFIGURED);
	return (Error)(interface->get_data(interface->data, p_buffer, p_bytes));
}

Error StreamPeerGDNative::get_partial_data(uint8_t *p_buffer, int p_bytes, int &r_received) {
	ERR_FAIL_COND_V(interface == nullptr, ERR_UNCONFIGURED);
	return (Error)(interface->get_partial_data(interface->data, p_buffer, p_bytes, &r_received));
}

int StreamPeerGDNative::get_available_bytes() const {
	ERR_FAIL_COND_V(interface == nullptr, 0);
	return interface->get_available_bytes(interface->data);
}

extern "C" {

void GDAPI godot_net_bind_stream_peer(godot_object *p_obj, const godot_net_stream_peer *p_interface) {
	((StreamPeerGDNative *)p_obj)->set_native_stream_peer(p_interface);
}
}
