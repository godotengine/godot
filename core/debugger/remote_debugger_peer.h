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
#include "core/string/ustring.h"

class RemoteDebuggerPeer : public RefCounted {
protected:
	int max_queued_messages = 4096;

public:
	enum {
		CHANNEL_MAIN_THREAD = 0,
		CHANNEL_OTHER,
		CHANNEL_DISCARDABLE,
		NUM_CHANNELS
	};

	virtual bool is_peer_connected() = 0;
	virtual bool has_message(int p_channel) = 0;
	virtual Error put_message(int p_channel, const Array &p_arr) = 0;
	virtual Array get_message(int p_channel) = 0;
	virtual void close() = 0;
	virtual void poll(int p_channel) = 0;
	virtual int get_max_message_size() const = 0;

	// If blocking io is allowed on main thread (debug).
	virtual bool can_block(int p_channel) const {
		(void)p_channel;
		return true;
	}

	RemoteDebuggerPeer();
};

class RemoteDebuggerPeerTCP : public RemoteDebuggerPeer {
	Ref<StreamPeerTCP> tcp_client;
	Thread thread;
	bool connected = false;
	bool running = false;

	// Each of these is a separately locked in and out queue, which are conceptually
	// like SCTP streams that don't block each other.
	struct Channel {
		// Accessed only by TCP thread.
		int out_left = 0;
		int out_pos = 0;
		Vector<uint8_t> out_buf;
		int in_left = 0;
		int in_pos = 0;
		Vector<uint8_t> in_buf;

		// Shared with clients under lock.
		struct Shared {
			Mutex mutex;
			List<Array> in_queue;
			List<Array> out_queue;
		};
		Shared shared;
	};

	Channel channels[NUM_CHANNELS];

	// If >= 0, this channel has a partial read in progress.
	int current_read_channel = -1;

	// If >= 0, this channel has a partial write in progress.
	int current_write_channel = -1;

	static void _thread_func(void *p_ud);

	void _poll(int p_channel);
	void _write_out();
	void _read_in();

	// REVISIT This means remote debugger takes 16 MiB just because it exists (in/out buffers).
	enum {
		MAX_MESSAGE_SIZE = 8 << 20
	};

protected:
	// Descendants must call this manually from _close, because the virtual
	// close() is used in the destructor and won't actually call up.
	void _close();

public:
	static RemoteDebuggerPeer *create(const String &p_uri);

	Error connect_to_host(const String &p_host, uint16_t p_port);

	void poll(int p_channel) override;
	bool is_peer_connected() override;
	bool has_message(int p_channel) override;
	Array get_message(int p_channel) override;
	Error put_message(int p_channel, const Array &p_arr) override;
	int get_max_message_size() const override;
	void close() override;

	explicit RemoteDebuggerPeerTCP(Ref<StreamPeerTCP> p_tcp = Ref<StreamPeerTCP>());
	~RemoteDebuggerPeerTCP();
};

#endif // REMOTE_DEBUGGER_PEER_H
