// Placeholder WebSocket sink for LogStream entries (batched buffer, no network yet).
#include "log_stream_websocket_sink.h"

#include "core/os/os.h"

LogStreamWebSocketSink::LogStreamWebSocketSink(const Config &p_config) :
		config(p_config) {
}

void LogStreamWebSocketSink::configure(const Config &p_config) {
	MutexLock lock(mutex);
	config = p_config;
	pending.clear();
}

void LogStreamWebSocketSink::handle_entry(const LogStreamEntry &p_entry) {
	MutexLock lock(mutex);
	if (!config.enabled) {
		return;
	}
	pending.push_back(p_entry);
	flush_if_needed_locked();
}

void LogStreamWebSocketSink::flush() {
	MutexLock lock(mutex);
	pending.clear();
	last_flush_usec = OS::get_singleton()->get_ticks_usec();
}

void LogStreamWebSocketSink::flush_if_needed_locked() {
	const bool count_ready = config.batch_size > 0 && pending.size() >= config.batch_size;
	const uint64_t now = OS::get_singleton()->get_ticks_usec();
	const bool time_ready = config.batch_msec > 0 && (now - last_flush_usec) >= (uint64_t)config.batch_msec * 1000;

	if (count_ready || time_ready) {
		// TODO: send over WebSocket when implemented.
		pending.clear();
		last_flush_usec = now;
	}
}

