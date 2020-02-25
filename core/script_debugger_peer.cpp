/*************************************************************************/
/*  script_debugger_peer.cpp                                             */
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

#include "script_debugger_peer.h"

#include "core/io/packet_peer.h"
#include "core/io/stream_peer_tcp.h"
#include "core/os/mutex.h"
#include "core/os/os.h"
#include "core/os/thread.h"

class ScriptDebuggerPeerTCP : public ScriptDebuggerPeer {
private:
	enum {
		QUEUE_MAX = 2048,
		POLL_USEC_MAX = 100,
	};

	Ref<StreamPeerTCP> tcp_client = Ref<StreamPeerTCP>(memnew(StreamPeerTCP));
	Ref<PacketPeerStream> packet_peer = Ref<PacketPeerStream>(memnew(PacketPeerStream));
	Mutex mutex;
	Thread *thread = NULL;
	List<Array> in_queue;
	List<Array> out_queue;
	bool connected = false;
	bool running = false;

	static void _thread_func(void *p_ud);

	void _poll();

public:
	void poll();
	Error connect_to_host(const String &p_host, uint16_t p_port);

	bool is_peer_connected() {
		return connected;
	}

	bool has_message() {
		return in_queue.size() > 0;
	}

	Array get_message() {
		MutexLock lock(mutex);
		ERR_FAIL_COND_V(!has_message(), Array());
		Array out = in_queue[0];
		in_queue.pop_front();
		return out;
	}

	Error put_message(const Array &p_arr) {
		MutexLock lock(mutex);
		if (out_queue.size() >= 2048) // XXX Should we keep track of size instead?
			return ERR_OUT_OF_MEMORY;

		out_queue.push_back(p_arr);
		return OK;
	}

	void close() {
		if (thread) {
			running = false;
			Thread::wait_to_finish(thread);
			memdelete(thread);
			thread = NULL;
		}
		MutexLock lock(mutex);
		tcp_client->disconnect_from_host();
		packet_peer->set_stream_peer(Ref<StreamPeer>());
	}

	ScriptDebuggerPeerTCP() {
		packet_peer->set_output_buffer_max_size((1024 * 1024 * 8) - 4); // 8 MiB should be way more than enough, minus 4 bytes for separator.
	}

	~ScriptDebuggerPeerTCP() {
		close();
	}
};

Error ScriptDebuggerPeerTCP::connect_to_host(const String &p_host, uint16_t p_port) {

	IP_Address ip;
	if (p_host.is_valid_ip_address())
		ip = p_host;
	else
		ip = IP::get_singleton()->resolve_hostname(p_host);

	int port = p_port;

	const int tries = 6;
	int waits[tries] = { 1, 10, 100, 1000, 1000, 1000 };

	tcp_client->connect_to_host(ip, port);

	for (int i = 0; i < tries; i++) {

		if (tcp_client->get_status() == StreamPeerTCP::STATUS_CONNECTED) {
			print_verbose("Remote Debugger: Connected!");
			break;
		} else {

			const int ms = waits[i];
			OS::get_singleton()->delay_usec(ms * 1000);
			print_verbose("Remote Debugger: Connection failed with status: '" + String::num(tcp_client->get_status()) + "', retrying in " + String::num(ms) + " msec.");
		};
	};

	if (tcp_client->get_status() != StreamPeerTCP::STATUS_CONNECTED) {

		ERR_PRINT("Remote Debugger: Unable to connect. Status: " + String::num(tcp_client->get_status()) + ".");
		return FAILED;
	};
	packet_peer->set_stream_peer(tcp_client);
	connected = true;
#ifndef NO_THREADS
	running = true;
	thread = Thread::create(_thread_func, this);
#endif
	return OK;
}

void ScriptDebuggerPeerTCP::_thread_func(void *p_ud) {
	ScriptDebuggerPeerTCP *peer = (ScriptDebuggerPeerTCP *)p_ud;
	while (peer->running && peer->is_peer_connected()) {
		peer->_poll();
		if (!peer->is_peer_connected())
			break;
		OS::get_singleton()->delay_usec(100);
	}
}

void ScriptDebuggerPeerTCP::poll() {
#ifdef NO_THREADS
	_poll();
#endif
}

void ScriptDebuggerPeerTCP::_poll() {
	MutexLock lock(mutex);
	// Poll in
	uint64_t ticks = OS::get_singleton()->get_ticks_usec();
	while (connected && packet_peer->get_available_packet_count() > 0 && in_queue.size() < QUEUE_MAX && OS::get_singleton()->get_ticks_usec() - ticks < POLL_USEC_MAX) {
		Variant var;
		const Error err = packet_peer->get_var(var);
		connected = tcp_client->get_status() == StreamPeerTCP::STATUS_CONNECTED;
		if (err != OK) {
			ERR_PRINT("Error reading variant from peer");
			break;
		}
		ERR_CONTINUE_MSG(var.get_type() != Variant::ARRAY, "Malformed packet received, not an Array.");
		in_queue.push_back(var);
	}
	// Poll out
	ticks = OS::get_singleton()->get_ticks_usec();
	while (connected && out_queue.size() > 0 && OS::get_singleton()->get_ticks_usec() - ticks < POLL_USEC_MAX) {
		Array arr = out_queue[0];
		out_queue.pop_front();
		const Error err = packet_peer->put_var(arr);
		connected = tcp_client->get_status() == StreamPeerTCP::STATUS_CONNECTED;
		if (err != OK) {
			ERR_PRINT("Error writing variant to peer");
			break;
		}
	}
}

Ref<ScriptDebuggerPeer> ScriptDebuggerPeer::create_from_uri(const String p_uri) {
	String debug_host = p_uri;
	uint16_t debug_port = 6007;
	if (debug_host.find(":") != -1) {
		int sep_pos = debug_host.find_last(":");
		debug_port = debug_host.substr(sep_pos + 1, debug_host.length()).to_int();
		debug_host = debug_host.substr(0, sep_pos);
	}
	Ref<ScriptDebuggerPeerTCP> peer = Ref<ScriptDebuggerPeer>(memnew(ScriptDebuggerPeerTCP));
	Error err = peer->connect_to_host(debug_host, debug_port);
	if (err != OK)
		return Ref<ScriptDebuggerPeer>();
	return peer;
}
