/*************************************************************************/
/*  gd_mono_log.cpp                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include <mono/utils/mono-logger.h>
#include <stdlib.h> // abort

#include "os/dir_access.h"
#include "os/os.h"

#include "../godotsharp_dirs.h"

static int log_level_get_id(const char *p_log_level) {

	const char *valid_log_levels[] = { "error", "critical", "warning", "message", "info", "debug", NULL };

	int i = 0;
	while (valid_log_levels[i]) {
		if (!strcmp(valid_log_levels[i], p_log_level))
			return i;
		i++;
	}

	return -1;
}

void gdmono_MonoLogCallback(const char *log_domain, const char *log_level, const char *message, mono_bool fatal, void *user_data) {

	FileAccess *f = GDMonoLog::get_singleton()->get_log_file();

	if (GDMonoLog::get_singleton()->get_log_level_id() >= log_level_get_id(log_level)) {
		String text(message);
		text += " (in domain ";
		text += log_domain;
		if (log_level) {
			text += ", ";
			text += log_level;
		}
		text += ")\n";

		f->seek_end();
		f->store_string(text);
	}

	if (fatal) {
		ERR_PRINTS("Mono: FALTAL ERROR, ABORTING! Logfile: " + GDMonoLog::get_singleton()->get_log_file_path() + "\n");
		abort();
	}
}

GDMonoLog *GDMonoLog::singleton = NULL;

bool GDMonoLog::_try_create_logs_dir(const String &p_logs_dir) {

	if (!DirAccess::exists(p_logs_dir)) {
		DirAccessRef diraccess = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
		ERR_FAIL_COND_V(!diraccess, false);
		Error logs_mkdir_err = diraccess->make_dir_recursive(p_logs_dir);
		ERR_EXPLAIN("Failed to create mono logs directory");
		ERR_FAIL_COND_V(logs_mkdir_err != OK, false);
	}

	return true;
}

void GDMonoLog::_open_log_file(const String &p_file_path) {

	log_file = FileAccess::open(p_file_path, FileAccess::WRITE);

	ERR_EXPLAIN("Failed to create log file");
	ERR_FAIL_COND(!log_file);
}

void GDMonoLog::_delete_old_log_files(const String &p_logs_dir) {

	static const uint64_t MAX_SECS = 5 * 86400;

	DirAccessRef da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	ERR_FAIL_COND(!da);

	Error err = da->change_dir(p_logs_dir);
	ERR_FAIL_COND(err != OK);

	ERR_FAIL_COND(da->list_dir_begin() != OK);

	String current;
	while ((current = da->get_next()).length()) {
		if (da->current_is_dir())
			continue;
		if (!current.ends_with(".txt"))
			continue;

		String name = current.get_basename();
		uint64_t unixtime = (uint64_t)name.to_int64();

		if (OS::get_singleton()->get_unix_time() - unixtime > MAX_SECS) {
			da->remove(current);
		}
	}

	da->list_dir_end();
}

void GDMonoLog::initialize() {

#ifdef DEBUG_ENABLED
	const char *log_level = "debug";
#else
	const char *log_level = "warning";
#endif

	String logs_dir = GodotSharpDirs::get_mono_logs_dir();

	if (_try_create_logs_dir(logs_dir)) {
		_delete_old_log_files(logs_dir);

		log_file_path = logs_dir.plus_file(String::num_int64(OS::get_singleton()->get_unix_time()) + ".txt");
		_open_log_file(log_file_path);
	}

	mono_trace_set_level_string(log_level);
	log_level_id = log_level_get_id(log_level);

	if (log_file) {
		if (OS::get_singleton()->is_stdout_verbose())
			OS::get_singleton()->print(String("Mono: Logfile is " + log_file_path + "\n").utf8());
		mono_trace_set_log_handler(gdmono_MonoLogCallback, this);
	} else {
		OS::get_singleton()->printerr("Mono: No log file, using default log handler\n");
	}
}

GDMonoLog::GDMonoLog() {

	singleton = this;

	log_level_id = -1;
}

GDMonoLog::~GDMonoLog() {

	singleton = NULL;

	if (log_file) {
		log_file->close();
		memdelete(log_file);
	}
}
