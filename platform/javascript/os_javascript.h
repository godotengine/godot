/*************************************************************************/
/*  os_javascript.h                                                      */
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

#ifndef OS_JAVASCRIPT_H
#define OS_JAVASCRIPT_H

#include "audio_driver_javascript.h"
#include "core/input/input.h"
#include "drivers/unix/os_unix.h"
#include "servers/audio_server.h"

#include <emscripten/html5.h>

class OS_JavaScript : public OS_Unix {
	MainLoop *main_loop = nullptr;
	List<AudioDriverJavaScript *> audio_drivers;

	bool idb_is_syncing = false;
	bool idb_available = false;
	bool idb_needs_sync = false;

	static void main_loop_callback();

	static void file_access_close_callback(const String &p_file, int p_flags);
	static void fs_sync_callback();

protected:
	void initialize() override;

	void set_main_loop(MainLoop *p_main_loop) override;
	void delete_main_loop() override;

	void finalize() override;

	bool _check_internal_feature_support(const String &p_feature) override;

public:
	// Override return type to make writing static callbacks less tedious.
	static OS_JavaScript *get_singleton();

	void initialize_joypads() override;

	MainLoop *get_main_loop() const override;
	bool main_loop_iterate();

	Error execute(const String &p_path, const List<String> &p_arguments, String *r_pipe = nullptr, int *r_exitcode = nullptr, bool read_stderr = false, Mutex *p_pipe_mutex = nullptr, bool p_open_console = false) override;
	Error create_process(const String &p_path, const List<String> &p_arguments, ProcessID *r_child_id = nullptr, bool p_open_console = false) override;
	Error kill(const ProcessID &p_pid) override;
	int get_process_id() const override;
	int get_processor_count() const override;
	int get_default_thread_pool_size() const override { return 1; }

	String get_executable_path() const override;
	Error shell_open(String p_uri) override;
	String get_name() const override;
	// Override default OS implementation which would block the main thread with delay_usec.
	// Implemented in javascript_main.cpp loop callback instead.
	void add_frame_delay(bool p_can_draw) override {}

	String get_cache_path() const override;
	String get_config_path() const override;
	String get_data_path() const override;
	String get_user_data_dir() const override;

	bool is_userfs_persistent() const override;
	bool is_single_window() const override { return true; }

	void alert(const String &p_alert, const String &p_title = "ALERT!") override;

	Error open_dynamic_library(const String p_path, void *&p_library_handle, bool p_also_set_library_path) override;

	void resume_audio();

	OS_JavaScript();
};

#endif
