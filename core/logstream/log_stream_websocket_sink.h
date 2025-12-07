// Placeholder WebSocket sink for LogStream entries.
#pragma once

#include "core/logstream/log_stream.h"
#include "core/os/mutex.h"

class LogStreamWebSocketSink : public LogStreamSink {
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

	explicit LogStreamWebSocketSink(const Config &p_config);
	~LogStreamWebSocketSink() override = default;

	void handle_entry(const LogStreamEntry &p_entry) override;
	void flush() override;
	void configure(const Config &p_config);

private:
	Mutex mutex;
	Config config;
	Vector<LogStreamEntry> pending;
	uint64_t last_flush_usec = 0;

	void flush_if_needed_locked();
};

