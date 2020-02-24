/*************************************************************************/
/*  editor_debugger_server.cpp                                           */
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

#include "editor_debugger_server.h"

#include "core/io/packet_peer.h"
#include "core/io/tcp_server.h"
#include "core/os/mutex.h"
#include "core/os/thread.h"
#include "editor/editor_log.h"
#include "editor/editor_node.h"
#include "editor/editor_settings.h"

class EditorDebuggerPeerTCP : public EditorDebuggerPeer {

private:
	enum {
		QUEUE_MAX = 2048,
		POLL_USEC_MAX = 100,
	};
	Ref<StreamPeerTCP> tcp;
	Ref<PacketPeerStream> packet_peer;
	List<Array> out_queue;
	List<Array> in_queue;
	Mutex mutex;
	bool connected = false;

public:
	Error poll() {
		MutexLock lock(mutex);
		connected = tcp->get_status() == StreamPeerTCP::STATUS_CONNECTED;
		Error err = OK;
		uint64_t ticks = OS::get_singleton()->get_ticks_usec();
		while (connected && packet_peer->get_available_packet_count() > 0 && in_queue.size() < QUEUE_MAX && OS::get_singleton()->get_ticks_usec() - ticks < POLL_USEC_MAX) {
			Variant var;
			err = packet_peer->get_var(var);
			connected = tcp->get_status() == StreamPeerTCP::STATUS_CONNECTED;
			if (err != OK) {
				ERR_PRINT("Error reading variant from peer");
				break;
			}
			ERR_CONTINUE_MSG(var.get_type() != Variant::ARRAY, "Malformed packet received, not an Array.");
			in_queue.push_back(var);
		}
		ticks = OS::get_singleton()->get_ticks_usec();
		while (connected && out_queue.size() > 0 && OS::get_singleton()->get_ticks_usec() - ticks < POLL_USEC_MAX) {
			Array arr = out_queue[0];
			out_queue.pop_front();
			packet_peer->put_var(arr);
			connected = tcp->get_status() == StreamPeerTCP::STATUS_CONNECTED;
		}
		return err;
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

	Error put_message(const Array p_arr) {
		MutexLock lock(mutex);
		if (out_queue.size() > QUEUE_MAX) {
			return ERR_OUT_OF_MEMORY;
		}
		out_queue.push_back(p_arr);
		return OK;
	}

	int get_max_message_size() const {
		return 8 << 20; // 8 MiB
	}

	bool is_peer_connected() {
		return connected;
	}

	void close() {
		MutexLock lock(mutex);
		connected = false;
		tcp->disconnect_from_host();
	}

	EditorDebuggerPeerTCP(Ref<StreamPeerTCP> p_stream) {
		packet_peer.instance();
		tcp = p_stream;
		if (tcp.is_null()) {
			tcp.instance(); // Bug?
		}
		packet_peer->set_stream_peer(tcp);
	}

	~EditorDebuggerPeerTCP() {
		close();
		packet_peer->set_stream_peer(Ref<StreamPeer>());
	}
};

class EditorDebuggerServerTCP : public EditorDebuggerServer {

private:
	Ref<TCP_Server> server;
	List<Ref<EditorDebuggerPeer> > peers;
	Thread *thread = NULL;
	Mutex mutex;
	bool running = false;

	static void _poll_func(void *p_ud);

public:
	virtual Error start();
	virtual void stop();
	virtual bool is_active() const;
	virtual bool is_connection_available() const;
	virtual Ref<EditorDebuggerPeer> take_connection();

	EditorDebuggerServerTCP();
};

EditorDebuggerServerTCP::EditorDebuggerServerTCP() {
	server.instance();
}

Error EditorDebuggerServerTCP::start() {
	int remote_port = (int)EditorSettings::get_singleton()->get("network/debug/remote_port");
	const Error err = server->listen(remote_port);
	if (err != OK) {
		EditorNode::get_log()->add_message(String("Error listening on port ") + itos(remote_port), EditorLog::MSG_TYPE_ERROR);
		return err;
	}
	running = true;
	thread = Thread::create(_poll_func, this);
	return err;
}

void EditorDebuggerServerTCP::stop() {
	server->stop();
	if (thread != NULL) {
		running = false;
		Thread::wait_to_finish(thread);
		memdelete(thread);
		thread = NULL;
	}
}

bool EditorDebuggerServerTCP::is_active() const {
	return server->is_listening();
}

bool EditorDebuggerServerTCP::is_connection_available() const {
	return server->is_listening() && server->is_connection_available();
}

Ref<EditorDebuggerPeer> EditorDebuggerServerTCP::take_connection() {
	ERR_FAIL_COND_V(!is_connection_available(), Ref<EditorDebuggerPeer>());
	MutexLock lock(mutex);
	Ref<EditorDebuggerPeerTCP> peer = memnew(EditorDebuggerPeerTCP(server->take_connection()));
	peers.push_back(peer);
	return peer;
}

void EditorDebuggerServerTCP::_poll_func(void *p_ud) {
	EditorDebuggerServerTCP *me = (EditorDebuggerServerTCP *)p_ud;
	while (me->running) {
		me->mutex.lock();
		List<Ref<EditorDebuggerPeer> > remove;
		for (int i = 0; i < me->peers.size(); i++) {
			Ref<EditorDebuggerPeer> peer = me->peers[i];
			Error err = ((EditorDebuggerPeerTCP *)peer.ptr())->poll();
			if (err != OK || !peer->is_peer_connected())
				remove.push_back(peer);
		}
		for (List<Ref<EditorDebuggerPeer> >::Element *E = remove.front(); E; E = E->next()) {
			me->peers.erase(E->get());
		}
		me->mutex.unlock();
		OS::get_singleton()->delay_usec(50);
	}
}

EditorDebuggerServer *EditorDebuggerServer::create_default() {
	return memnew(EditorDebuggerServerTCP);
}
