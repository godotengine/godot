/*************************************************************************/
/*  gd_mono_log.h                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include <mono/utils/mono-logger.h>

#include "core/typedefs.h"

#if !defined(JAVASCRIPT_ENABLED) && !defined(IPHONE_ENABLED)
// We have custom mono log callbacks for WASM and iOS
#define GD_MONO_LOG_ENABLED
#endif

#ifdef GD_MONO_LOG_ENABLED
#include "core/os/file_access.h"
#endif

class GDMonoLog {
#ifdef GD_MONO_LOG_ENABLED
	int log_level_id;

	FileAccess *log_file;
	String log_file_path;

	bool _try_create_logs_dir(const String &p_logs_dir);
	void _delete_old_log_files(const String &p_logs_dir);

	static void mono_log_callback(const char *log_domain, const char *log_level, const char *message, mono_bool fatal, void *user_data);
#endif

	static GDMonoLog *singleton;

public:
	_FORCE_INLINE_ static GDMonoLog *get_singleton() { return singleton; }

	void initialize();

	GDMonoLog();
	~GDMonoLog();
};

#endif // GD_MONO_LOG_H
