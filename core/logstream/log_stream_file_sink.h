// File sink for LogStream entries (JSONL or plain text with optional rotation).
#pragma once

#include "core/io/file_access.h"
#include "core/io/dir_access.h"
#include "core/io/json.h"
#include "core/logstream/log_stream.h"
#include "core/os/mutex.h"
#include "core/os/time.h"

class LogStreamFileSink : public LogStreamSink {
public:
	struct Config {
		bool enabled = true;
		String path;
		bool jsonl = true;
		int max_size_mb = 0; // 0 means no rotation.

		Config() :
				path("user://logs/editor.log") {}
	};

	explicit LogStreamFileSink(const Config &p_config);
	~LogStreamFileSink() override;

	void handle_entry(const LogStreamEntry &p_entry) override;
	void flush() override;

	void configure(const Config &p_config);

private:
	Mutex mutex;
	Config config;
	Ref<FileAccess> file;

	void ensure_file();
	void rotate_if_needed();
	String serialize_jsonl(const LogStreamEntry &p_entry) const;
	String serialize_plain(const LogStreamEntry &p_entry) const;
};

