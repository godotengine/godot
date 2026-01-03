// Godot LogStream core router implementation (scaffolding).
#include "log_stream.h"

#include "core/error/error_macros.h"
#include "core/os/memory.h"
#include "core/os/mutex.h"
#include "core/templates/vector.h"

LogStreamRouter *LogStreamRouter::singleton = nullptr;

const char *LogStreamEntry::level_to_string(LogStreamEntry::Level p_level) {
	switch (p_level) {
		case LEVEL_ERROR:
			return "error";
		case LEVEL_WARNING:
			return "warning";
		case LEVEL_INFO:
			return "info";
		case LEVEL_DEBUG:
			return "debug";
		case LEVEL_VERBOSE:
			return "verbose";
	}
	return "unknown";
}

LogStreamRouter *LogStreamRouter::get_singleton() {
	return singleton;
}

void LogStreamRouter::create_singleton(const Config &p_config) {
	if (singleton) {
		return;
	}
	singleton = memnew(LogStreamRouter(p_config));
}

void LogStreamRouter::create_singleton() {
	create_singleton(Config());
}

void LogStreamRouter::free_singleton() {
	if (!singleton) {
		return;
	}
	memdelete(singleton);
	singleton = nullptr;
}

LogStreamRouter::LogStreamRouter(const Config &p_config) :
		config(p_config) {
	buffer.reserve(config.max_entries > 0 ? config.max_entries : 0);
}

LogStreamRouter::~LogStreamRouter() {
	clear_sinks();
}

void LogStreamRouter::configure(const Config &p_config) {
	MutexLock lock(mutex);
	config = p_config;
	prune_if_needed();
}

LogStreamRouter::Config LogStreamRouter::get_config() const {
	MutexLock lock(mutex);
	return config;
}

void LogStreamRouter::add_sink(LogStreamSink *p_sink) {
	ERR_FAIL_NULL(p_sink);
	MutexLock lock(mutex);
	sinks.push_back(p_sink);
}

void LogStreamRouter::clear_sinks() {
	MutexLock lock(mutex);
	for (LogStreamSink *sink : sinks) {
		memdelete(sink);
	}
	sinks.clear();
}

void LogStreamRouter::push_entry(const LogStreamEntry &p_entry) {
	MutexLock lock(mutex);
	if (!config.enabled) {
		return;
	}

	LogStreamEntry entry = p_entry;
	if (entry.seq == 0) {
		entry.seq = ++seq_counter;
	}

	buffer.push_back(entry);
	prune_if_needed();

	for (LogStreamSink *sink : sinks) {
		if (sink) {
			sink->handle_entry(entry);
		}
	}
}

Vector<LogStreamEntry> LogStreamRouter::get_entries_snapshot() const {
	MutexLock lock(mutex);
	return buffer;
}

void LogStreamRouter::clear_entries() {
	MutexLock lock(mutex);
	buffer.clear();
}

void LogStreamRouter::prune_if_needed() {
	if (config.max_entries <= 0) {
		return;
	}
	const int max_entries = config.max_entries;
	while (buffer.size() > max_entries) {
		buffer.remove_at(0);
	}
}

