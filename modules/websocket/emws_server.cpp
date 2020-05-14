/*************************************************************************/
/*  emws_server.cpp                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifdef JAVASCRIPT_ENABLED

#include "emws_server.h"
#include "core/os/os.h"

Error EMWSServer::listen(int p_port, Vector<String> p_protocols, bool gd_mp_api) {
	return FAILED;
}

bool EMWSServer::is_listening() const {
	return false;
}

void EMWSServer::stop() {
}

bool EMWSServer::has_peer(int p_id) const {
	return false;
}

Ref<WebSocketPeer> EMWSServer::get_peer(int p_id) const {
	return nullptr;
}

Vector<String> EMWSServer::get_protocols() const {
	Vector<String> out;

	return out;
}

IP_Address EMWSServer::get_peer_address(int p_peer_id) const {
	return IP_Address();
}

int EMWSServer::get_peer_port(int p_peer_id) const {
	return 0;
}

void EMWSServer::disconnect_peer(int p_peer_id, int p_code, String p_reason) {
}

void EMWSServer::poll() {
}

int EMWSServer::get_max_packet_size() const {
	return 0;
}

Error EMWSServer::set_buffers(int p_in_buffer, int p_in_packets, int p_out_buffer, int p_out_packets) {
	return OK;
}

EMWSServer::EMWSServer() {
}

EMWSServer::~EMWSServer() {
}

#endif // JAVASCRIPT_ENABLED
