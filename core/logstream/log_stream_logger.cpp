// Logger adapter that forwards Godot log output into the LogStreamRouter.
#include "log_stream_logger.h"

#include <cstdio>

#include "core/object/script_backtrace.h"
#include "core/os/memory.h"
#include "core/os/os.h"
#include "core/string/print_string.h"

static LogStreamEntry::Level _map_error_type(Logger::ErrorType p_type) {
	switch (p_type) {
		case Logger::ERR_WARNING:
			return LogStreamEntry::LEVEL_WARNING;
		case Logger::ERR_ERROR:
		case Logger::ERR_SCRIPT:
		case Logger::ERR_SHADER:
		default:
			return LogStreamEntry::LEVEL_ERROR;
	}
}

static String _format_va(const char *p_format, va_list p_list) {
	const int static_buf_size = 1024;
	char static_buf[static_buf_size];
	va_list list_copy;
	va_copy(list_copy, p_list);
	int len = vsnprintf(static_buf, static_buf_size, p_format, p_list);
	String output;
	if (len >= static_buf_size) {
		char *buf = (char *)memalloc(len + 1);
		if (buf) {
			vsnprintf(buf, len + 1, p_format, list_copy);
			output = String::utf8(buf, len);
			memfree(buf);
		}
	} else {
		output = String::utf8(static_buf, len);
	}
	va_end(list_copy);
	return output;
}

LogStreamLogger::LogStreamLogger() {
	// Don't create singleton here - EditorNode will create it with proper config
}

void LogStreamLogger::logv(const char *p_format, va_list p_list, bool p_err) {
	if (!LogStreamRouter::get_singleton()) {
		return;
	}

	LogStreamEntry entry;
	entry.level = p_err ? LogStreamEntry::LEVEL_ERROR : LogStreamEntry::LEVEL_INFO;
	entry.timestamp_usec = OS::get_singleton()->get_ticks_usec();
	entry.message = _format_va(p_format, p_list);

	LogStreamRouter::get_singleton()->push_entry(entry);
}

void LogStreamLogger::log_error(const char *p_function, const char *p_file, int p_line, const char *p_code, const char *p_rationale, bool p_editor_notify, ErrorType p_type, const Vector<Ref<ScriptBacktrace>> &p_script_backtraces) {
	if (!LogStreamRouter::get_singleton()) {
		return;
	}

	LogStreamEntry entry;
	entry.level = _map_error_type(p_type);
	entry.timestamp_usec = OS::get_singleton()->get_ticks_usec();
	entry.message = p_rationale && *p_rationale ? String::utf8(p_rationale) : String::utf8(p_code);
	entry.file = p_file ? String::utf8(p_file) : String();
	entry.line = p_line;
	entry.function = p_function ? String::utf8(p_function) : String();
	entry.category = String(LogStreamEntry::level_to_string(entry.level));
	entry.stack.resize(p_script_backtraces.size());
	for (int i = 0; i < p_script_backtraces.size(); i++) {
		if (p_script_backtraces[i].is_valid()) {
			entry.stack.write[i] = p_script_backtraces[i]->format(0);
		}
	}

	LogStreamRouter::get_singleton()->push_entry(entry);
}

