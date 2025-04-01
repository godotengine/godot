/**************************************************************************/
/*  os_unix.h                                                             */
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

#pragma once

#ifdef UNIX_ENABLED

#include "core/os/os.h"
#include "drivers/unix/ip_unix.h"

#ifdef __GLIBC__
#include <iconv.h>
#include <langinfo.h>
#define gd_iconv_t iconv_t
#define gd_iconv_open iconv_open
#define gd_iconv iconv
#define gd_iconv_close iconv_close
#else
typedef void *gd_iconv_t;
typedef gd_iconv_t (*PIConvOpen)(const char *, const char *);
typedef size_t (*PIConv)(gd_iconv_t, char **, size_t *, char **, size_t *);
typedef int (*PIConvClose)(gd_iconv_t);
typedef const char *(*PIConvLocaleCharset)(void);
#endif

class OS_Unix : public OS {
	struct ProcessInfo {
		mutable bool is_running = true;
		mutable int exit_code = -1;
	};
	HashMap<ProcessID, ProcessInfo> *process_map = nullptr;
	Mutex process_map_mutex;

#ifdef __GLIBC__
	bool _iconv_ok = true;
#else
	bool _iconv_ok = false;

	PIConvOpen gd_iconv_open = nullptr;
	PIConv gd_iconv = nullptr;
	PIConvClose gd_iconv_close = nullptr;
	PIConvLocaleCharset gd_locale_charset = nullptr;

	void _load_iconv();
#endif

protected:
	// UNIX only handles the core functions.
	// inheriting platforms under unix (eg. X11) should handle the rest

	virtual void initialize_core();
	virtual int unix_initialize_audio(int p_audio_driver);

	virtual void finalize_core() override;

public:
	OS_Unix();

	virtual Vector<String> get_video_adapter_driver_info() const override;

	virtual String get_stdin_string(int64_t p_buffer_size = 1024) override;
	virtual PackedByteArray get_stdin_buffer(int64_t p_buffer_size = 1024) override;
	virtual StdHandleType get_stdin_type() const override;
	virtual StdHandleType get_stdout_type() const override;
	virtual StdHandleType get_stderr_type() const override;

	virtual Error get_entropy(uint8_t *r_buffer, int p_bytes) override;

	virtual Error open_dynamic_library(const String &p_path, void *&p_library_handle, GDExtensionData *p_data = nullptr) override;
	virtual Error close_dynamic_library(void *p_library_handle) override;
	virtual Error get_dynamic_library_symbol_handle(void *p_library_handle, const String &p_name, void *&p_symbol_handle, bool p_optional = false) override;

	virtual Error set_cwd(const String &p_cwd) override;

	virtual String get_name() const override;
	virtual String get_distribution_name() const override;
	virtual String get_version() const override;

	virtual String get_temp_path() const override;

	virtual DateTime get_datetime(bool p_utc) const override;
	virtual TimeZoneInfo get_time_zone_info() const override;

	virtual double get_unix_time() const override;

	virtual void delay_usec(uint32_t p_usec) const override;
	virtual uint64_t get_ticks_usec() const override;

	virtual Dictionary get_memory_info() const override;

	virtual String multibyte_to_string(const String &p_encoding, const PackedByteArray &p_array) const override;
	virtual PackedByteArray string_to_multibyte(const String &p_encoding, const String &p_string) const override;

	virtual Error execute(const String &p_path, const List<String> &p_arguments, String *r_pipe = nullptr, int *r_exitcode = nullptr, bool read_stderr = false, Mutex *p_pipe_mutex = nullptr, bool p_open_console = false) override;
	virtual Dictionary execute_with_pipe(const String &p_path, const List<String> &p_arguments, bool p_blocking = true) override;
	virtual Error create_process(const String &p_path, const List<String> &p_arguments, ProcessID *r_child_id = nullptr, bool p_open_console = false) override;
	virtual Error kill(const ProcessID &p_pid) override;
	virtual int get_process_id() const override;
	virtual bool is_process_running(const ProcessID &p_pid) const override;
	virtual int get_process_exit_code(const ProcessID &p_pid) const override;

	virtual bool has_environment(const String &p_var) const override;
	virtual String get_environment(const String &p_var) const override;
	virtual void set_environment(const String &p_var, const String &p_value) const override;
	virtual void unset_environment(const String &p_var) const override;

	virtual String get_locale() const override;

	virtual void initialize_debugging() override;

	virtual String get_executable_path() const override;
	virtual String get_user_data_dir(const String &p_user_dir) const override;
};

class UnixTerminalLogger : public StdLogger {
public:
	virtual void log_error(const char *p_function, const char *p_file, int p_line, const char *p_code, const char *p_rationale, bool p_editor_notify = false, ErrorType p_type = ERR_ERROR) override;
	virtual ~UnixTerminalLogger();
};

#endif // UNIX_ENABLED
