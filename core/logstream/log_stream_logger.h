// Logger adapter that forwards Godot log output into the LogStreamRouter.
#pragma once

#include "core/io/logger.h"
#include "core/logstream/log_stream.h"

class LogStreamLogger : public Logger {
public:
	LogStreamLogger();
	~LogStreamLogger() override {}

	void logv(const char *p_format, va_list p_list, bool p_err) override _PRINTF_FORMAT_ATTRIBUTE_2_0;
	void log_error(const char *p_function, const char *p_file, int p_line, const char *p_code, const char *p_rationale, bool p_editor_notify = false, ErrorType p_type = ERR_ERROR, const Vector<Ref<ScriptBacktrace>> &p_script_backtraces = {}) override;
};

