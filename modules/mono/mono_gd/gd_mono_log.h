/*************************************************************************/
/*  gd_mono_log.h                                                        */
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
#ifndef GD_MONO_LOG_H
#define GD_MONO_LOG_H

#include "os/file_access.h"

class GDMonoLog {

	int log_level_id;

	FileAccess *log_file;
	String log_file_path;

	bool _try_create_logs_dir(const String &p_logs_dir);
	void _open_log_file(const String &p_file_path);
	void _delete_old_log_files(const String &p_logs_dir);

	static GDMonoLog *singleton;

public:
	_FORCE_INLINE_ static GDMonoLog *get_singleton() { return singleton; }

	void initialize();

	_FORCE_INLINE_ FileAccess *get_log_file() { return log_file; }
	_FORCE_INLINE_ String get_log_file_path() { return log_file_path; }
	_FORCE_INLINE_ int get_log_level_id() { return log_level_id; }

	GDMonoLog();
	~GDMonoLog();
};

#endif // GD_MONO_LOG_H
