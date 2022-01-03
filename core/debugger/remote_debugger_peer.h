/*************************************************************************/
/*  remote_debugger_peer.h                                               */
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

#ifndef REMOTE_DEBUGGER_PEER_H
#define REMOTE_DEBUGGER_PEER_H

#include "core/io/stream_peer_tcp.h"
#include "core/object/ref_counted.h"
#include "core/os/mutex.h"
#include "core/os/thread.h"
#include "core/string/ustring.h"

class RemoteDebuggerPeer : public RefCounted {
protected:
	int max_queued_messages = 4096;

public:
	virtual bool is_peer_connected() = 0;
	virtual bool has_message() = 0;
	virtual Error put_message(const Array &p_arr) = 0;
	virtual Array get_message() = 0;
	virtual void close() = 0;
	virtual void poll() = 0;
	virtual int get_max_message_size() const = 0;
	virtual bool can_block() const { return true; } // If blocking io is allowed on main thread (debug).

	RemoteDebuggerPeer();
};

class RemoteDebuggerPeerTCP : public RemoteDebuggerPeer {
private:
	Ref<StreamPeerTCP> tcp_client;
	Mutex mutex;
	Thread thread;
	List<Array> in_queue;
	List<Array> out_queue;
	int out_left = 0;
	int out_pos = 0;
	Vector<uint8_t> out_buf;
	int in_left = 0;
	int in_pos = 0;
	Vector<uint8_t> in_buf;
	bool connected = false;
	bool running = false;

	static void _thread_func(void *p_ud);

	void _poll();
	void _write_out();
	void _read_in();

public:
	static RemoteDebuggerPeer *create(const String &p_uri);

	Error connect_to_host(const String &p_host, uint16_t p_port);

	void poll();
	bool is_peer_connected();
	bool has_message();
	Array get_message();
	Error put_message(const Array &p_arr);
	int get_max_message_size() const;
	void close();

	RemoteDebuggerPeerTCP(Ref<StreamPeerTCP> p_stream = Ref<StreamPeerTCP>());
	~RemoteDebuggerPeerTCP();
};

#endif // REMOTE_DEBUGGER_PEER_H
