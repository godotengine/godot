/**************************************************************************/
/*  logger.cpp                                                            */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "logger.h"

#include "core/core_globals.h"
#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/object/script_backtrace.h"
#include "core/os/os.h"
#include "core/os/time.h"
#include "core/string/regex.h"
#include "core/templates/rb_set.h"

#include <cstdio>

#if defined(MINGW_ENABLED) || defined(_MSC_VER)
#define sprintf sprintf_s
#endif

bool Logger::should_log(bool p_err) {
	return (!p_err || CoreGlobals::print_error_enabled) && (p_err || CoreGlobals::print_line_enabled);
}

void Logger::set_flush_stdout_on_print(bool p_value) {
	_flush_stdout_on_print = p_value;
}

void Logger::log_error(const char *p_function, const char *p_file, int p_line, const char *p_code, const char *p_rationale, bool p_editor_notify, ErrorType p_type, const Vector<Ref<ScriptBacktrace>> &p_script_backtraces) {
	if (!should_log(true)) {
		return;
	}

	const char *err_details = p_rationale && *p_rationale ? p_rationale : p_code;

	logf_error("%s: %s\n", error_type_string(p_type), err_details);
	logf_error("%sat: %s (%s:%i)\n", error_type_indent(p_type), p_function, p_file, p_line);

	for (const Ref<ScriptBacktrace> &backtrace : p_script_backtraces) {
		if (!backtrace->is_empty()) {
			logf_error("%s\n", backtrace->format(strlen(error_type_indent(p_type))).utf8().get_data());
		}
	}
}

void Logger::logf(const char *p_format, ...) {
	if (!should_log(false)) {
		return;
	}

	va_list argp;
	va_start(argp, p_format);

	logv(p_format, argp, false);

	va_end(argp);
}

void Logger::logf_error(const char *p_format, ...) {
	if (!should_log(true)) {
		return;
	}

	va_list argp;
	va_start(argp, p_format);

	logv(p_format, argp, true);

	va_end(argp);
}

void RotatedFileLogger::clear_old_backups() {
	int max_backups = max_files - 1; // -1 for the current file

	String basename = base_path.get_file().get_basename();
	String extension = base_path.get_extension();

	Ref<DirAccess> da = DirAccess::open(base_path.get_base_dir());
	if (da.is_null()) {
		return;
	}

	da->list_dir_begin();
	String f = da->get_next();
	// backups is a RBSet because it guarantees that iterating on it is done in sorted order.
	// RotatedFileLogger depends on this behavior to delete the oldest log file first.
	RBSet<String> backups;
	while (!f.is_empty()) {
		if (!da->current_is_dir() && f.begins_with(basename) && f.get_extension() == extension && f != base_path.get_file()) {
			backups.insert(f);
		}
		f = da->get_next();
	}
	da->list_dir_end();

	if (backups.size() > max_backups) {
		// since backups are appended with timestamp and Set iterates them in sorted order,
		// first backups are the oldest
		int to_delete = backups.size() - max_backups;
		for (RBSet<String>::Element *E = backups.front(); E && to_delete > 0; E = E->next(), --to_delete) {
			da->remove(E->get());
		}
	}
}

void RotatedFileLogger::rotate_file() {
	file.unref();

	if (FileAccess::exists(base_path)) {
		if (max_files > 1) {
			String timestamp = Time::get_singleton()->get_datetime_string_from_system().replace_char(':', '.');
			String backup_name = base_path.get_basename() + timestamp;
			if (!base_path.get_extension().is_empty()) {
				backup_name += "." + base_path.get_extension();
			}

			Ref<DirAccess> da = DirAccess::open(base_path.get_base_dir());
			if (da.is_valid()) {
				da->copy(base_path, backup_name);
			}
			clear_old_backups();
		}
	} else {
		Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_USERDATA);
		if (da.is_valid()) {
			da->make_dir_recursive(base_path.get_base_dir());
		}
	}

	file = FileAccess::open(base_path, FileAccess::WRITE);
	ERR_FAIL_COND_MSG(file.is_null(), "Failed to open log file for writing: " + base_path);
	file->detach_from_objectdb(); // Note: This FileAccess instance will exist longer than ObjectDB, therefore can't be registered in ObjectDB.
}

RotatedFileLogger::RotatedFileLogger(const String &p_base_path, int p_max_files) :
		base_path(p_base_path.simplify_path()),
		max_files(p_max_files > 0 ? p_max_files : 1) {
	rotate_file();

	strip_ansi_regex.instantiate();
	strip_ansi_regex->detach_from_objectdb(); // Note: This RegEx instance will exist longer than ObjectDB, therefore can't be registered in ObjectDB.
	strip_ansi_regex->compile("\u001b\\[((?:\\d|;)*)([a-zA-Z])");
}

void RotatedFileLogger::logv(const char *p_format, va_list p_list, bool p_err) {
	if (!should_log(p_err)) {
		return;
	}

	if (file.is_valid()) {
		const int static_buf_size = 512;
		char static_buf[static_buf_size];
		char *buf = static_buf;
		va_list list_copy;
		va_copy(list_copy, p_list);
		int len = vsnprintf(buf, static_buf_size, p_format, p_list);
		if (len >= static_buf_size) {
			buf = (char *)Memory::alloc_static(len + 1);
			vsnprintf(buf, len + 1, p_format, list_copy);
		}
		va_end(list_copy);

		// Strip ANSI escape codes (such as those inserted by `print_rich()`)
		// before writing to file, as text editors cannot display those
		// correctly.
		file->store_string(strip_ansi_regex->sub(String::utf8(buf), "", true));

		if (len >= static_buf_size) {
			Memory::free_static(buf);
		}

		if (p_err || _flush_stdout_on_print) {
			// Don't always flush when printing stdout to avoid performance
			// issues when `print()` is spammed in release builds.
			file->flush();
		}
	}
}

void StdLogger::logv(const char *p_format, va_list p_list, bool p_err) {
	if (!should_log(p_err)) {
		return;
	}

	if (p_err) {
		vfprintf(stderr, p_format, p_list);
	} else {
		vprintf(p_format, p_list);
		if (_flush_stdout_on_print) {
			// Don't always flush when printing stdout to avoid performance
			// issues when `print()` is spammed in release builds.
			fflush(stdout);
		}
	}
}

void StdLogger::log_error(const char *p_function, const char *p_file, int p_line, const char *p_code, const char *p_rationale, bool p_editor_notify, ErrorType p_type, const Vector<Ref<ScriptBacktrace>> &p_script_backtraces) {
	if (!should_log(true)) {
		return;
	}

	constexpr const char GRAY[] = "\u001b[0;90m";
	constexpr const char RED[] = "\u001b[0;31m";
	constexpr const char RED_BOLD[] = "\u001b[1;31m";
	constexpr const char YELLOW[] = "\u001b[0;33m";
	constexpr const char YELLOW_BOLD[] = "\u001b[1;33m";
	constexpr const char MAGENTA[] = "\u001b[0;35m";
	constexpr const char MAGENTA_BOLD[] = "\u001b[1;35m";
	constexpr const char CYAN[] = "\u001b[0;36m";
	constexpr const char CYAN_BOLD[] = "\u001b[1;36m";
	constexpr const char RESET[] = "\u001b[0m";

	const char *bold_color = "";
	const char *normal_color = "";
	const char *gray_color = "";
	const char *reset_color = "";

	if (OS::get_singleton() && OS::get_singleton()->is_stderr_color()) {
		gray_color = GRAY;
		reset_color = RESET;
		switch (p_type) {
			case ERR_WARNING:
				bold_color = YELLOW_BOLD;
				normal_color = YELLOW;
				break;
			case ERR_SCRIPT:
				bold_color = MAGENTA_BOLD;
				normal_color = MAGENTA;
				break;
			case ERR_SHADER:
				bold_color = CYAN_BOLD;
				normal_color = CYAN;
				break;
			case ERR_ERROR:
				bold_color = RED_BOLD;
				normal_color = RED;
				break;
		}
	}

	const char *err_details = p_rationale && *p_rationale ? p_rationale : p_code;

	logf_error("%s%s:%s %s\n", bold_color, error_type_string(p_type), normal_color, err_details);
	logf_error("%s%sat: %s (%s:%i)%s\n", gray_color, error_type_indent(p_type), p_function, p_file, p_line, reset_color);

	for (const Ref<ScriptBacktrace> &backtrace : p_script_backtraces) {
		if (!backtrace->is_empty()) {
			logf_error("%s%s%s\n", gray_color, backtrace->format(strlen(error_type_indent(p_type))).utf8().get_data(), reset_color);
		}
	}
}

CompositeLogger::CompositeLogger(const Vector<Logger *> &p_loggers) :
		loggers(p_loggers) {
}

void CompositeLogger::logv(const char *p_format, va_list p_list, bool p_err) {
	if (!should_log(p_err)) {
		return;
	}

	for (int i = 0; i < loggers.size(); ++i) {
		va_list list_copy;
		va_copy(list_copy, p_list);
		loggers[i]->logv(p_format, list_copy, p_err);
		va_end(list_copy);
	}
}

void CompositeLogger::log_error(const char *p_function, const char *p_file, int p_line, const char *p_code, const char *p_rationale, bool p_editor_notify, ErrorType p_type, const Vector<Ref<ScriptBacktrace>> &p_script_backtraces) {
	if (!should_log(true)) {
		return;
	}

	for (int i = 0; i < loggers.size(); ++i) {
		loggers[i]->log_error(p_function, p_file, p_line, p_code, p_rationale, p_editor_notify, p_type, p_script_backtraces);
	}
}

void CompositeLogger::add_logger(Logger *p_logger) {
	loggers.push_back(p_logger);
}

CompositeLogger::~CompositeLogger() {
	for (int i = 0; i < loggers.size(); ++i) {
		memdelete(loggers[i]);
	}
}
