/*************************************************************************/
/*  os_unix.h                                                            */
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

#ifndef OS_UNIX_H
#define OS_UNIX_H

#ifdef UNIX_ENABLED

#include "core/os/os.h"
#include "drivers/unix/ip_unix.h"

class OS_Unix : public OS {
protected:
	// UNIX only handles the core functions.
	// inheriting platforms under unix (eg. X11) should handle the rest

	virtual void initialize_core();
	virtual int unix_initialize_audio(int p_audio_driver);
	//virtual Error initialize(int p_video_driver,int p_audio_driver);

	virtual void finalize_core() override;

	String stdin_buf;

public:
	OS_Unix();

	virtual String get_stdin_string(bool p_block) override;

	//virtual void set_mouse_show(bool p_show);
	//virtual void set_mouse_grab(bool p_grab);
	//virtual bool is_mouse_grab_enabled() const = 0;
	//virtual void get_mouse_position(int &x, int &y) const;
	//virtual void set_window_title(const String& p_title);

	//virtual void set_video_mode(const VideoMode& p_video_mode);
	//virtual VideoMode get_video_mode() const;
	//virtual void get_fullscreen_mode_list(List<VideoMode> *p_list) const;

	virtual Error open_dynamic_library(const String p_path, void *&p_library_handle, bool p_also_set_library_path = false) override;
	virtual Error close_dynamic_library(void *p_library_handle) override;
	virtual Error get_dynamic_library_symbol_handle(void *p_library_handle, const String p_name, void *&p_symbol_handle, bool p_optional = false) override;

	virtual Error set_cwd(const String &p_cwd) override;

	virtual String get_name() const override;

	virtual Date get_date(bool utc) const override;
	virtual Time get_time(bool utc) const override;
	virtual TimeZoneInfo get_time_zone_info() const override;

	virtual double get_unix_time() const override;

	virtual void delay_usec(uint32_t p_usec) const override;
	virtual uint64_t get_ticks_usec() const override;

	virtual Error execute(const String &p_path, const List<String> &p_arguments, String *r_pipe = nullptr, int *r_exitcode = nullptr, bool read_stderr = false, Mutex *p_pipe_mutex = nullptr) override;
	virtual Error create_process(const String &p_path, const List<String> &p_arguments, ProcessID *r_child_id = nullptr) override;
	virtual Error kill(const ProcessID &p_pid) override;
	virtual int get_process_id() const override;

	virtual bool has_environment(const String &p_var) const override;
	virtual String get_environment(const String &p_var) const override;
	virtual bool set_environment(const String &p_var, const String &p_value) const override;
	virtual String get_locale() const override;

	virtual int get_processor_count() const override;

	virtual void debug_break() override;
	virtual void initialize_debugging() override;

	virtual String get_executable_path() const override;
	virtual String get_user_data_dir() const override;
};

class UnixTerminalLogger : public StdLogger {
public:
	virtual void log_error(const char *p_function, const char *p_file, int p_line, const char *p_code, const char *p_rationale, bool p_editor_notify, ErrorType p_type = ERR_ERROR);
	virtual ~UnixTerminalLogger();
};

#endif

#endif
