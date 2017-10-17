/*************************************************************************/
/*  os_unix.h                                                            */
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
#ifndef OS_UNIX_H
#define OS_UNIX_H

/**
	@author Juan Linietsky <reduzio@gmail.com>
*/

#ifdef UNIX_ENABLED

#include "drivers/unix/ip_unix.h"
#include "os/os.h"

class OS_Unix : public OS {

	uint64_t ticks_start;

protected:
	// UNIX only handles the core functions.
	// inheriting platforms under unix (eg. X11) should handle the rest

	//virtual int get_video_driver_count() const;
	//virtual const char * get_video_driver_name(int p_driver) const;
	//virtual VideoMode get_default_video_mode() const;

	virtual int get_audio_driver_count() const;
	virtual const char *get_audio_driver_name(int p_driver) const;

	virtual void initialize_logger();
	virtual void initialize_core();
	virtual int unix_initialize_audio(int p_audio_driver);
	//virtual void initialize(int p_video_driver,int p_audio_driver);

	virtual void finalize_core();

	String stdin_buf;

	String get_global_settings_path() const;

public:
	virtual void alert(const String &p_alert, const String &p_title = "ALERT!");
	virtual String get_stdin_string(bool p_block);

	//virtual void set_mouse_show(bool p_show);
	//virtual void set_mouse_grab(bool p_grab);
	//virtual bool is_mouse_grab_enabled() const = 0;
	//virtual void get_mouse_position(int &x, int &y) const;
	//virtual void set_window_title(const String& p_title);

	//virtual void set_video_mode(const VideoMode& p_video_mode);
	//virtual VideoMode get_video_mode() const;
	//virtual void get_fullscreen_mode_list(List<VideoMode> *p_list) const;

	virtual Error open_dynamic_library(const String p_path, void *&p_library_handle);
	virtual Error close_dynamic_library(void *p_library_handle);
	virtual Error get_dynamic_library_symbol_handle(void *p_library_handle, const String p_name, void *&p_symbol_handle, bool p_optional = false);

	virtual Error set_cwd(const String &p_cwd);

	virtual String get_name();

	virtual Date get_date(bool utc) const;
	virtual Time get_time(bool utc) const;
	virtual TimeZoneInfo get_time_zone_info() const;

	virtual uint64_t get_unix_time() const;
	virtual uint64_t get_system_time_secs() const;

	virtual void delay_usec(uint32_t p_usec) const;
	virtual uint64_t get_ticks_usec() const;

	virtual Error execute(const String &p_path, const List<String> &p_arguments, bool p_blocking, ProcessID *r_child_id = NULL, String *r_pipe = NULL, int *r_exitcode = NULL, bool read_stderr = false);
	virtual Error kill(const ProcessID &p_pid);
	virtual int get_process_id() const;

	virtual bool has_environment(const String &p_var) const;
	virtual String get_environment(const String &p_var) const;
	virtual String get_locale() const;

	virtual int get_processor_count() const;

	virtual void debug_break();

	virtual String get_installed_templates_path() const;
	virtual String get_executable_path() const;
	virtual String get_data_dir() const;

	//virtual void run( MainLoop * p_main_loop );
};

class UnixTerminalLogger : public StdLogger {
public:
	virtual void log_error(const char *p_function, const char *p_file, int p_line, const char *p_code, const char *p_rationale, ErrorType p_type = ERR_ERROR);
	virtual ~UnixTerminalLogger();
};

#endif

#endif
