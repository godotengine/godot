/**************************************************************************/
/*  remote_debugger_peer_messageport.h                                    */
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

#pragma once

#include "godot_js.h"

#include "core/debugger/remote_debugger_peer.h"

class RemoteDebuggerPeerMessagePort : public RemoteDebuggerPeer {
	GDSOFTCLASS(RemoteDebuggerPeerMessagePort, RemoteDebuggerPeer);

	List<Array> in_queue;
	Vector<uint8_t> out_buffer;

	int max_queued_messages;

	int _js_id = -1;

	WASM_EXPORT static void on_debug_message(void *p_ref, const uint8_t *p_buffer, size_t p_len);

public:
	static Ref<RemoteDebuggerPeer> create(const String &p_uri);

	Error connect_to_host(const String &p_uri);

	virtual bool is_peer_connected() override;
	virtual int get_max_message_size() const override;
	virtual bool has_message() override;
	virtual Error put_message(const Array &p_arr) override;
	virtual Array get_message() override;
	virtual void close() override;
	virtual void poll() override;
	virtual bool can_block() const override;

	RemoteDebuggerPeerMessagePort(int p_id = -1);
	virtual ~RemoteDebuggerPeerMessagePort();
};
