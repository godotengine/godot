// Godot LogStream core types and router scaffolding.
#pragma once

#include "core/os/mutex.h"
#include "core/string/ustring.h"
#include "core/templates/vector.h"

class LogStreamSink;

struct LogStreamEntry {
	enum Level {
		LEVEL_ERROR,
		LEVEL_WARNING,
		LEVEL_INFO,
		LEVEL_DEBUG,
		LEVEL_VERBOSE,
	};

	uint64_t seq = 0;
	uint64_t timestamp_usec = 0;
	Level level = LEVEL_INFO;
	String message;
	String file;
	int line = -1;
	String function;
	String category;
	Vector<String> stack;
	String project;
	String session_id;

	static const char *level_to_string(Level p_level);
};

class LogStreamSink {
public:
	virtual void handle_entry(const LogStreamEntry &p_entry) = 0;
	virtual void flush() {}
	virtual ~LogStreamSink() {}
};

class LogStreamRouter {
public:
	struct Config {
		bool enabled = true;
		int max_entries = 5000;
	};

	static LogStreamRouter *get_singleton();
	static void create_singleton(const Config &p_config);
	static void create_singleton(); // Uses default config.
	static void free_singleton();

	explicit LogStreamRouter(const Config &p_config);
	~LogStreamRouter();

	void configure(const Config &p_config);
	Config get_config() const;

	void add_sink(LogStreamSink *p_sink);
	void clear_sinks();

	void push_entry(const LogStreamEntry &p_entry);
	Vector<LogStreamEntry> get_entries_snapshot() const;
	void clear_entries();

private:
	static LogStreamRouter *singleton;

	mutable Mutex mutex;
	Config config;
	Vector<LogStreamEntry> buffer;
	Vector<LogStreamSink *> sinks;
	uint64_t seq_counter = 0;

	void prune_if_needed();
};

