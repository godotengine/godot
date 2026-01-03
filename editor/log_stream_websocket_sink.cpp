// Editor-only WebSocket sink for LogStream.
#include "log_stream_websocket_sink.h"

#include "core/io/json.h"
#include "core/os/os.h"

static Dictionary _entry_to_dict(const LogStreamEntry &p_entry) {
	Dictionary d;
	d["seq"] = (int64_t)p_entry.seq;
	d["ts_usec"] = (int64_t)p_entry.timestamp_usec;
	d["level"] = LogStreamEntry::level_to_string(p_entry.level);
	d["message"] = p_entry.message;
	d["file"] = p_entry.file;
	d["line"] = p_entry.line;
	d["function"] = p_entry.function;
	d["category"] = p_entry.category;
	d["stack"] = p_entry.stack;
	d["project"] = p_entry.project;
	d["session_id"] = p_entry.session_id;
	return d;
}

EditorLogStreamWebSocketSink::EditorLogStreamWebSocketSink(const Config &p_config) :
		config(p_config) {
}

void EditorLogStreamWebSocketSink::configure(const Config &p_config) {
	MutexLock lock(mutex);
	config = p_config;
	pending.clear();

	// Close existing connections
	for (KeyValue<int, PeerData> &E : peers) {
		if (E.value.ws.is_valid()) {
			E.value.ws->close();
		}
	}
	peers.clear();
	pending_peers.clear();

	// Stop server
	if (tcp_server.is_valid()) {
		tcp_server->stop();
		tcp_server.unref();
	}
}

void EditorLogStreamWebSocketSink::ensure_server() {
	if (tcp_server.is_valid() && tcp_server->is_listening()) {
		return;
	}

	tcp_server.instantiate();
	Error err = tcp_server->listen(config.port, IPAddress("*"));
	if (err == OK) {
		print_line(vformat("LogStream WebSocket server started on port %d", config.port));
	} else {
		ERR_PRINT(vformat("LogStream WebSocket server failed to start on port %d: %d", config.port, err));
		tcp_server.unref();
	}
}

void EditorLogStreamWebSocketSink::accept_new_connections() {
	if (tcp_server.is_null() || !tcp_server->is_listening()) {
		return;
	}

	while (tcp_server->is_connection_available()) {
		Ref<StreamPeerTCP> tcp = tcp_server->take_connection();
		if (tcp.is_null()) {
			print_line("WebSocket sink: took null TCP connection");
			continue;
		}

		print_line(vformat("WebSocket sink: new TCP connection from %s:%d", tcp->get_connected_host(), tcp->get_connected_port()));

		Ref<WebSocketPeer> ws = Ref<WebSocketPeer>(WebSocketPeer::create());
		if (ws.is_null()) {
			print_line("WebSocket sink: failed to create WebSocketPeer");
			continue;
		}

		Error err = ws->accept_stream(tcp);
		if (err != OK) {
			print_line(vformat("WebSocket sink: accept_stream failed: %d", err));
			continue;
		}

		int peer_id = next_peer_id++;
		PendingPeer pp;
		pp.ws = ws;
		pp.connect_time = OS::get_singleton()->get_ticks_msec();
		pending_peers[peer_id] = pp;
		print_line(vformat("WebSocket sink: added pending peer %d, total pending=%d", peer_id, pending_peers.size()));
	}
}

void EditorLogStreamWebSocketSink::poll_pending_peers() {
	Vector<int> to_remove;
	Vector<int> to_promote;
	uint64_t now = OS::get_singleton()->get_ticks_msec();

	// First, collect all pending peer IDs to avoid iterator corruption
	Vector<int> peer_ids;
	for (const KeyValue<int, PendingPeer> &E : pending_peers) {
		peer_ids.push_back(E.key);
	}

	// Now process each peer safely
	for (int peer_id : peer_ids) {
		// Check if peer still exists (might have been removed by another operation)
		if (!pending_peers.has(peer_id)) {
			continue;
		}

		PendingPeer &pp = pending_peers[peer_id];
		Ref<WebSocketPeer> ws = pp.ws;
		if (ws.is_null()) {
			print_line(vformat("WebSocket sink: pending peer %d has null ws", peer_id));
			to_remove.push_back(peer_id);
			continue;
		}

		// Poll to advance handshake
		ws->poll();
		WebSocketPeer::State state = ws->get_ready_state();
		uint64_t elapsed = now - pp.connect_time;

		print_line(vformat("WebSocket sink: polling pending peer %d, state=%d, elapsed=%dms", peer_id, state, elapsed));

		if (state == WebSocketPeer::STATE_OPEN) {
			// Handshake complete, promote to full peer
			print_line(vformat("WebSocket sink: promoting peer %d to full peer", peer_id));
			to_promote.push_back(peer_id);
		} else if (state == WebSocketPeer::STATE_CLOSED || state == WebSocketPeer::STATE_CLOSING) {
			// Failed
			print_line(vformat("WebSocket sink: pending peer %d closed/closing, removing", peer_id));
			to_remove.push_back(peer_id);
		} else if (state == WebSocketPeer::STATE_CONNECTING) {
			// Still connecting, check timeout
			if (now - pp.connect_time > 3000) { // 3 second timeout
				print_line(vformat("WebSocket sink: pending peer %d handshake timeout", peer_id));
				ws->close();
				to_remove.push_back(peer_id);
			}
		}
	}

	// Promote successful connections
	for (int peer_id : to_promote) {
		if (!pending_peers.has(peer_id)) {
			print_line(vformat("WebSocket sink: ERROR - peer %d not in pending_peers during promotion", peer_id));
			continue;
		}
		PendingPeer &pp = pending_peers[peer_id];
		if (pp.ws.is_null()) {
			print_line(vformat("WebSocket sink: ERROR - peer %d has null ws during promotion", peer_id));
			pending_peers.erase(peer_id);
			continue;
		}
		PeerData pd;
		pd.ws = pp.ws;
		pd.authed = config.auth_token.is_empty(); // Auto-auth if no token required
		peers[peer_id] = pd;
		pending_peers.erase(peer_id);
		print_line(vformat("WebSocket sink: peer %d promoted! authed=%s, total peers=%d", peer_id, pd.authed ? "true" : "false", peers.size()));
	}

	// Remove failed connections
	for (int peer_id : to_remove) {
		if (pending_peers.has(peer_id)) {
			pending_peers.erase(peer_id);
			print_line(vformat("WebSocket sink: removed pending peer %d", peer_id));
		}
	}
}

void EditorLogStreamWebSocketSink::poll_peers() {
	// Remove disconnected peers
	Vector<int> to_remove;

	for (KeyValue<int, PeerData> &E : peers) {
		Ref<WebSocketPeer> ws = E.value.ws;
		if (ws.is_null()) {
			to_remove.push_back(E.key);
			continue;
		}

		ws->poll();
		WebSocketPeer::State state = ws->get_ready_state();

		if (state == WebSocketPeer::STATE_CLOSED || state == WebSocketPeer::STATE_CLOSING) {
			to_remove.push_back(E.key);
			continue;
		}

		// Handle incoming messages (for auth)
		if (!E.value.authed && !config.auth_token.is_empty()) {
			while (ws->get_available_packet_count() > 0) {
				const uint8_t *buf = nullptr;
				int len = 0;
				Error err = ws->get_packet(&buf, len);
				if (err != OK || buf == nullptr || len == 0) {
					continue;
				}

				String msg = String::utf8((const char *)buf, len);
				if (msg == config.auth_token) {
					E.value.authed = true;
				} else {
					// Wrong token, disconnect
					ws->close();
					to_remove.push_back(E.key);
					break;
				}
			}
		}
	}

	// Remove disconnected/rejected peers
	for (int peer_id : to_remove) {
		peers.erase(peer_id);
	}
}

void EditorLogStreamWebSocketSink::poll() {
	accept_new_connections();
	poll_pending_peers();
	poll_peers();
}

void EditorLogStreamWebSocketSink::poll_server() {
	MutexLock lock(mutex);
	if (!config.enabled) {
		return;
	}
	ensure_server();
	poll();

	// Flush pending logs if time threshold met (even if no new logs arrived)
	if (!pending.is_empty()) {
		uint64_t now = OS::get_singleton()->get_ticks_usec();
		flush_if_ready(now);
	}
}

void EditorLogStreamWebSocketSink::handle_entry(const LogStreamEntry &p_entry) {
	MutexLock lock(mutex);
	if (!config.enabled) {
		return;
	}
	ensure_server();
	poll();

	pending.push_back(p_entry);
	print_line(vformat("WebSocket sink: received log, pending=%d, peers=%d", pending.size(), peers.size()));
	uint64_t now = OS::get_singleton()->get_ticks_usec();
	flush_if_ready(now);
}

void EditorLogStreamWebSocketSink::flush() {
	MutexLock lock(mutex);
	if (!config.enabled || pending.is_empty()) {
		return;
	}
	ensure_server();
	poll();
	send_batch(pending);
	pending.clear();
	last_flush_usec = OS::get_singleton()->get_ticks_usec();
}

void EditorLogStreamWebSocketSink::flush_if_ready(uint64_t p_now) {
	const bool count_ready = config.batch_size > 0 && pending.size() >= config.batch_size;
	const bool time_ready = config.batch_msec > 0 && (p_now - last_flush_usec) >= (uint64_t)config.batch_msec * 1000;
	if (!count_ready && !time_ready) {
		return;
	}
	if (pending.is_empty()) {
		return;
	}
	send_batch(pending);
	pending.clear();
	last_flush_usec = p_now;
}

void EditorLogStreamWebSocketSink::send_batch(const Vector<LogStreamEntry> &p_batch) {
	if (peers.is_empty()) {
		print_line(vformat("WebSocket sink: send_batch called but no peers connected"));
		return;
	}

	print_line(vformat("WebSocket sink: sending batch of %d entries to %d peers", p_batch.size(), peers.size()));

	// Build JSON array
	Array arr;
	for (const LogStreamEntry &e : p_batch) {
		arr.push_back(_entry_to_dict(e));
	}
	String payload = JSON::stringify(arr);
	Vector<uint8_t> bytes = payload.to_utf8_buffer();

	// Send to all authenticated peers
	int sent_count = 0;
	for (KeyValue<int, PeerData> &E : peers) {
		if (E.value.authed && E.value.ws.is_valid()) {
			Ref<WebSocketPeer> ws = E.value.ws;
			if (ws->get_ready_state() == WebSocketPeer::STATE_OPEN) {
				ws->send_text(payload);
				sent_count++;
				print_line(vformat("WebSocket sink: sent batch to peer %d", E.key));
			} else {
				print_line(vformat("WebSocket sink: peer %d not open (state=%d)", E.key, ws->get_ready_state()));
			}
		} else {
			print_line(vformat("WebSocket sink: peer %d not authed or ws invalid (authed=%d, valid=%d)", E.key, E.value.authed, E.value.ws.is_valid()));
		}
	}
	print_line(vformat("WebSocket sink: sent to %d/%d peers", sent_count, peers.size()));
}
