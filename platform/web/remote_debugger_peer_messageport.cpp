/**************************************************************************/
/*  remote_debugger_peer_messageport.cpp                                  */
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

#include "remote_debugger_peer_messageport.h"

#include "core/config/project_settings.h"
#include "core/io/marshalls.h"

extern "C" {
int godot_js_debugger_bind(int p_id, void (*p_callback)(void *p_ref, const uint8_t *p_buffer, size_t p_len), void *p_ref);
void godot_js_debugger_post(int p_id, const uint8_t *p_buffer, size_t p_len);
int godot_js_debugger_active(int p_id);
void godot_js_debugger_unbind(int p_id);
}

void RemoteDebuggerPeerMessagePort::on_debug_message(void *p_ref, const uint8_t *p_buffer, size_t p_len) {
	RemoteDebuggerPeerMessagePort *peer = static_cast<RemoteDebuggerPeerMessagePort *>(p_ref);
	ERR_FAIL_NULL(peer);
	ERR_FAIL_COND(peer->in_queue.size() >= peer->max_queued_messages);
	Variant var;
	Error err = decode_variant(var, p_buffer, p_len);
	ERR_FAIL_COND(err != OK);
	ERR_FAIL_COND(var.get_type() != Variant::ARRAY);
	peer->in_queue.push_back(var);
}

Error RemoteDebuggerPeerMessagePort::connect_to_host(const String &p_uri) {
	_js_id = godot_js_debugger_bind(-1, &on_debug_message, this);
	return _js_id != -1 ? OK : FAILED;
}

bool RemoteDebuggerPeerMessagePort::is_peer_connected() {
	if (_js_id == -1) {
		return false;
	}
	return godot_js_debugger_active(_js_id);
}

void RemoteDebuggerPeerMessagePort::poll() {
}

int RemoteDebuggerPeerMessagePort::get_max_message_size() const {
	return out_buffer.size();
}

bool RemoteDebuggerPeerMessagePort::has_message() {
	return in_queue.size() > 0;
}

Array RemoteDebuggerPeerMessagePort::get_message() {
	ERR_FAIL_COND_V(in_queue.is_empty(), Array());
	const Array msg = in_queue.front()->get();
	in_queue.pop_front();
	return msg;
}

Error RemoteDebuggerPeerMessagePort::put_message(const Array &p_arr) {
	if (_js_id == -1) {
		return ERR_UNCONFIGURED;
	}
	Variant v = p_arr;
	int size = 0;
	Error err = encode_variant(v, nullptr, size);
	ERR_FAIL_COND_V(err != OK || size < 1 || size > out_buffer.size(), FAILED);
	encode_variant(v, out_buffer.ptrw(), size);
	godot_js_debugger_post(_js_id, out_buffer.ptr(), size);
	return OK;
}

void RemoteDebuggerPeerMessagePort::close() {
	if (_js_id == -1) {
		return;
	}
	godot_js_debugger_unbind(_js_id);
	_js_id = -1;
}

bool RemoteDebuggerPeerMessagePort::can_block() const {
	return false;
}

RemoteDebuggerPeerMessagePort::RemoteDebuggerPeerMessagePort(int p_id) {
	max_queued_messages = (int)GLOBAL_GET("network/limits/debugger/max_queued_messages");
	out_buffer.resize(1 << 18);
	if (p_id != -1) {
		_js_id = godot_js_debugger_bind(p_id, &on_debug_message, this);
		ERR_FAIL_COND(_js_id == -1);
	}
}

RemoteDebuggerPeerMessagePort::~RemoteDebuggerPeerMessagePort() {
	close();
}

Ref<RemoteDebuggerPeer> RemoteDebuggerPeerMessagePort::create(const String &p_uri) {
	ERR_FAIL_COND_V(!p_uri.begins_with("messageport://"), nullptr);
	Ref<RemoteDebuggerPeerMessagePort> peer;
	peer.instantiate();
	Error err = peer->connect_to_host(p_uri);
	if (err != OK) {
		return nullptr;
	}
	return peer;
}
