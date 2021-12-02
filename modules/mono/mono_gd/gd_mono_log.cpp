/*************************************************************************/
/*  gd_mono_log.cpp                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "gd_mono_log.h"

#include <stdlib.h> // abort

#include "core/io/dir_access.h"
#include "core/os/os.h"

#include "../godotsharp_dirs.h"
#include "../utils/string_utils.h"

static CharString get_default_log_level() {
#ifdef DEBUG_ENABLED
	return String("info").utf8();
#else
	return String("warning").utf8();
#endif
}

GDMonoLog *GDMonoLog::singleton = nullptr;

#ifdef GD_MONO_LOG_ENABLED

static int get_log_level_id(const char *p_log_level) {
	const char *valid_log_levels[] = { "error", "critical", "warning", "message", "info", "debug", nullptr };

	int i = 0;
	while (valid_log_levels[i]) {
		if (!strcmp(valid_log_levels[i], p_log_level)) {
			return i;
		}
		i++;
	}

	return -1;
}

static String make_text(const char *log_domain, const char *log_level, const char *message) {
	String text(message);
	text += " (in domain ";
	text += log_domain;
	if (log_level) {
		text += ", ";
		text += log_level;
	}
	text += ")";
	return text;
}

void GDMonoLog::mono_log_callback(const char *log_domain, const char *log_level, const char *message, mono_bool fatal, void *) {
	FileAccess *f = GDMonoLog::get_singleton()->log_file;

	if (GDMonoLog::get_singleton()->log_level_id >= get_log_level_id(log_level)) {
		String text = make_text(log_domain, log_level, message);
		text += "\n";

		f->seek_end();
		f->store_string(text);
	}

	if (fatal) {
		String text = make_text(log_domain, log_level, message);
		ERR_PRINT("Mono: FATAL ERROR '" + text + "', ABORTING! Logfile: '" + GDMonoLog::get_singleton()->log_file_path + "'.");
		// Make sure to flush before aborting
		f->flush();
		f->close();
		memdelete(f);

		abort();
	}
}

bool GDMonoLog::_try_create_logs_dir(const String &p_logs_dir) {
	if (!DirAccess::exists(p_logs_dir)) {
		DirAccessRef diraccess = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
		ERR_FAIL_COND_V(!diraccess, false);
		Error logs_mkdir_err = diraccess->make_dir_recursive(p_logs_dir);
		ERR_FAIL_COND_V_MSG(logs_mkdir_err != OK, false, "Failed to create mono logs directory.");
	}

	return true;
}

void GDMonoLog::_delete_old_log_files(const String &p_logs_dir) {
	static const uint64_t MAX_SECS = 5 * 86400; // 5 days

	DirAccessRef da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	ERR_FAIL_COND(!da);

	Error err = da->change_dir(p_logs_dir);
	ERR_FAIL_COND_MSG(err != OK, "Cannot change directory to '" + p_logs_dir + "'.");

	ERR_FAIL_COND(da->list_dir_begin() != OK);

	String current = da->get_next();
	while (!current.is_empty()) {
		if (da->current_is_dir() || !current.ends_with(".txt")) {
			current = da->get_next();
			continue;
		}

		uint64_t modified_time = FileAccess::get_modified_time(da->get_current_dir().plus_file(current));

		if (OS::get_singleton()->get_unix_time() - modified_time > MAX_SECS) {
			da->remove(current);
		}
		current = da->get_next();
	}

	da->list_dir_end();
}

void GDMonoLog::initialize() {
	CharString log_level = OS::get_singleton()->get_environment("GODOT_MONO_LOG_LEVEL").utf8();

	if (log_level.length() != 0 && get_log_level_id(log_level.get_data()) == -1) {
		ERR_PRINT(String() + "Mono: Ignoring invalid log level (GODOT_MONO_LOG_LEVEL): '" + log_level.get_data() + "'.");
		log_level = CharString();
	}

	if (log_level.length() == 0) {
		log_level = get_default_log_level();
	}

	String logs_dir = GodotSharpDirs::get_mono_logs_dir();

	if (_try_create_logs_dir(logs_dir)) {
		_delete_old_log_files(logs_dir);

		OS::Date date_now = OS::get_singleton()->get_date();
		OS::Time time_now = OS::get_singleton()->get_time();

		String log_file_name = str_format("%04d-%02d-%02d_%02d.%02d.%02d",
				(int)date_now.year, (int)date_now.month, (int)date_now.day,
				(int)time_now.hour, (int)time_now.minute, (int)time_now.second);

		log_file_name += str_format("_%d", OS::get_singleton()->get_process_id());

		log_file_name += ".log";

		log_file_path = logs_dir.plus_file(log_file_name);

		log_file = FileAccess::open(log_file_path, FileAccess::WRITE);
		if (!log_file) {
			ERR_PRINT("Mono: Cannot create log file at: " + log_file_path);
		}
	}

	mono_trace_set_level_string(log_level.get_data());
	log_level_id = get_log_level_id(log_level.get_data());

	if (log_file) {
		OS::get_singleton()->print("Mono: Log file is: '%s'\n", log_file_path.utf8().get_data());
		mono_trace_set_log_handler(mono_log_callback, this);
	} else {
		OS::get_singleton()->printerr("Mono: No log file, using default log handler\n");
	}
}

GDMonoLog::GDMonoLog() {
	singleton = this;
}

GDMonoLog::~GDMonoLog() {
	singleton = nullptr;

	if (log_file) {
		log_file->close();
		memdelete(log_file);
	}
}

#else

void GDMonoLog::initialize() {
	CharString log_level = get_default_log_level();
	mono_trace_set_level_string(log_level.get_data());
}

GDMonoLog::GDMonoLog() {
	singleton = this;
}

GDMonoLog::~GDMonoLog() {
	singleton = nullptr;
}

#endif // !defined(JAVASCRIPT_ENABLED)
