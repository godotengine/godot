// File sink for LogStream entries (JSONL or plain text with optional rotation).
#include "log_stream_file_sink.h"

#include "core/os/os.h"

LogStreamFileSink::LogStreamFileSink(const Config &p_config) :
		config(p_config) {
}

LogStreamFileSink::~LogStreamFileSink() {
	flush();
	file.unref();
}

void LogStreamFileSink::configure(const Config &p_config) {
	MutexLock lock(mutex);
	config = p_config;
}

void LogStreamFileSink::handle_entry(const LogStreamEntry &p_entry) {
	MutexLock lock(mutex);
	if (!config.enabled) {
		return;
	}

	ensure_file();
	if (file.is_null()) {
		return;
	}

	String line = config.jsonl ? serialize_jsonl(p_entry) : serialize_plain(p_entry);
	file->store_string(line);
	if (!line.ends_with("\n")) {
		file->store_string("\n");
	}
	file->flush();
	rotate_if_needed();
}

void LogStreamFileSink::flush() {
	MutexLock lock(mutex);
	if (file.is_valid()) {
		file->flush();
	}
}

void LogStreamFileSink::ensure_file() {
	if (file.is_valid()) {
		return;
	}

	Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_USERDATA);
	if (da.is_valid()) {
		da->make_dir_recursive(config.path.get_base_dir());
	}

	file = FileAccess::open(config.path, FileAccess::WRITE_READ);
	if (file.is_valid()) {
		file->seek_end();
	}
}

void LogStreamFileSink::rotate_if_needed() {
	if (!file.is_valid() || config.max_size_mb <= 0) {
		return;
	}

	const uint64_t max_bytes = (uint64_t)config.max_size_mb * 1024 * 1024;
	const uint64_t len = file->get_position();
	if (len < max_bytes) {
		return;
	}

	String base_dir = config.path.get_base_dir();
	String base_name = config.path.get_file().get_basename();
	String extension = config.path.get_extension();
	String timestamp = Time::get_singleton()->get_datetime_string_from_system().replace_char(':', '.');
	String rotated = base_dir.path_join(vformat("%s.%s", base_name, timestamp));
	if (!extension.is_empty()) {
		rotated += "." + extension;
	}

	file.unref();
	Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_USERDATA);
	if (da.is_valid()) {
		da->rename(config.path, rotated);
	}

	file = FileAccess::open(config.path, FileAccess::WRITE);
	if (file.is_valid()) {
		file->seek_end();
	}
}

String LogStreamFileSink::serialize_jsonl(const LogStreamEntry &p_entry) const {
	Dictionary dict;
	dict["seq"] = (int64_t)p_entry.seq;
	dict["ts_usec"] = (int64_t)p_entry.timestamp_usec;
	dict["level"] = LogStreamEntry::level_to_string(p_entry.level);
	dict["message"] = p_entry.message;
	dict["file"] = p_entry.file;
	dict["line"] = p_entry.line;
	dict["function"] = p_entry.function;
	dict["category"] = p_entry.category;
	dict["stack"] = p_entry.stack;
	dict["project"] = p_entry.project;
	dict["session_id"] = p_entry.session_id;
	return JSON::stringify(dict);
}

String LogStreamFileSink::serialize_plain(const LogStreamEntry &p_entry) const {
	String level = LogStreamEntry::level_to_string(p_entry.level);
	String location;
	if (!p_entry.file.is_empty() && p_entry.line > 0) {
		location = vformat(" (%s:%d)", p_entry.file, p_entry.line);
	}
	return vformat("[%s] %s%s\n", level, p_entry.message, location);
}

