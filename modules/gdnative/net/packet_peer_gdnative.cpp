/*************************************************************************/
/*  packet_peer_gdnative.cpp                                             */
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

#include "packet_peer_gdnative.h"

PacketPeerGDNative::PacketPeerGDNative() {
	interface = nullptr;
}

PacketPeerGDNative::~PacketPeerGDNative() {
}

void PacketPeerGDNative::set_native_packet_peer(const godot_net_packet_peer *p_impl) {
	interface = p_impl;
}

void PacketPeerGDNative::_bind_methods() {
}

Error PacketPeerGDNative::get_packet(const uint8_t **r_buffer, int &r_buffer_size) {
	ERR_FAIL_COND_V(interface == nullptr, ERR_UNCONFIGURED);
	return (Error)interface->get_packet(interface->data, r_buffer, &r_buffer_size);
}

Error PacketPeerGDNative::put_packet(const uint8_t *p_buffer, int p_buffer_size) {
	ERR_FAIL_COND_V(interface == nullptr, ERR_UNCONFIGURED);
	return (Error)interface->put_packet(interface->data, p_buffer, p_buffer_size);
}

int PacketPeerGDNative::get_max_packet_size() const {
	ERR_FAIL_COND_V(interface == nullptr, 0);
	return interface->get_max_packet_size(interface->data);
}

int PacketPeerGDNative::get_available_packet_count() const {
	ERR_FAIL_COND_V(interface == nullptr, 0);
	return interface->get_available_packet_count(interface->data);
}

extern "C" {

void GDAPI godot_net_bind_packet_peer(godot_object *p_obj, const godot_net_packet_peer *p_impl) {
	((PacketPeerGDNative *)p_obj)->set_native_packet_peer(p_impl);
}
}
