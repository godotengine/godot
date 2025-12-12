// Editor-only WebSocket sink for LogStream.
#pragma once

#include "core/io/stream_peer_tcp.h"
#include "core/io/tcp_server.h"
#include "core/logstream/log_stream.h"
#include "core/os/mutex.h"
#include "core/templates/hash_map.h"
#include "core/templates/hash_set.h"
#include "modules/websocket/websocket_peer.h"

class EditorLogStreamWebSocketSink : public LogStreamSink {
public:
	struct Config {
		bool enabled = false;
		int port;
		int batch_size;
		int batch_msec;
		String auth_token;

		Config() :
				port(17865),
				batch_size(50),
				batch_msec(100) {}
	};

	explicit EditorLogStreamWebSocketSink(const Config &p_config = Config());
	~EditorLogStreamWebSocketSink() override = default;

	void handle_entry(const LogStreamEntry &p_entry) override;
	void flush() override;
	void configure(const Config &p_config);
	void poll_server(); // Call regularly to accept connections/handle messages

private:
	struct PendingPeer {
		Ref<WebSocketPeer> ws;
		uint64_t connect_time = 0;
	};

	struct PeerData {
		Ref<WebSocketPeer> ws;
		bool authed = false;
	};

	mutable Mutex mutex;
	Config config;
	Ref<TCPServer> tcp_server;
	HashMap<int, PendingPeer> pending_peers;
	HashMap<int, PeerData> peers;
	int next_peer_id = 1;
	Vector<LogStreamEntry> pending;
	uint64_t last_flush_usec = 0;

	void ensure_server();
	void poll();
	void accept_new_connections();
	void poll_pending_peers();
	void poll_peers();
	void flush_if_ready(uint64_t p_now);
	void send_batch(const Vector<LogStreamEntry> &p_batch);
};

