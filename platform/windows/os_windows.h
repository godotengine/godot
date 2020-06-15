/*************************************************************************/
/*  os_windows.h                                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef OS_WINDOWS_H
#define OS_WINDOWS_H

#include "core/input/input.h"
#include "core/os/os.h"
#include "core/project_settings.h"
#include "crash_handler_windows.h"
#include "drivers/unix/ip_unix.h"
#include "drivers/wasapi/audio_driver_wasapi.h"
#include "drivers/winmidi/midi_driver_winmidi.h"
#include "key_mapping_windows.h"
#include "servers/audio_server.h"
#include "servers/rendering/rasterizer.h"
#include "servers/rendering_server.h"
#ifdef XAUDIO2_ENABLED
#include "drivers/xaudio2/audio_driver_xaudio2.h"
#endif

#if defined(OPENGL_ENABLED)
#include "context_gl_windows.h"
#endif

#if defined(VULKAN_ENABLED)
#include "drivers/vulkan/rendering_device_vulkan.h"
#include "platform/windows/vulkan_context_win.h"
#endif

#include <fcntl.h>
#include <io.h>
#include <stdio.h>
#include <windows.h>
#include <windowsx.h>

class JoypadWindows;
class OS_Windows : public OS {
#ifdef STDOUT_FILE
	FILE *stdo;
#endif

	uint64_t ticks_start;
	uint64_t ticks_per_second;

	HINSTANCE hInstance;
	MainLoop *main_loop;

	String tablet_driver;
	Vector<String> tablet_drivers;

#ifdef WASAPI_ENABLED
	AudioDriverWASAPI driver_wasapi;
#endif
#ifdef XAUDIO2_ENABLED
	AudioDriverXAudio2 driver_xaudio2;
#endif
#ifdef WINMIDI_ENABLED
	MIDIDriverWinMidi driver_midi;
#endif

	CrashHandler crash_handler;

	bool force_quit;
	HWND main_window;

	// functions used by main to initialize/deinitialize the OS
protected:
	virtual void initialize();

	virtual void set_main_loop(MainLoop *p_main_loop);
	virtual void delete_main_loop();

	virtual void finalize();
	virtual void finalize_core();
	virtual String get_stdin_string(bool p_block);

	String _quote_command_line_argument(const String &p_text) const;

	struct ProcessInfo {
		STARTUPINFO si;
		PROCESS_INFORMATION pi;
	};
	Map<ProcessID, ProcessInfo> *process_map;

public:
	virtual Error open_dynamic_library(const String p_path, void *&p_library_handle, bool p_also_set_library_path = false);
	virtual Error close_dynamic_library(void *p_library_handle);
	virtual Error get_dynamic_library_symbol_handle(void *p_library_handle, const String p_name, void *&p_symbol_handle, bool p_optional = false);

	virtual MainLoop *get_main_loop() const;

	virtual String get_name() const;

	virtual int get_tablet_driver_count() const;
	virtual String get_tablet_driver_name(int p_driver) const;
	virtual String get_current_tablet_driver() const;
	virtual void set_current_tablet_driver(const String &p_driver);

	virtual void initialize_joypads() {}

	virtual Date get_date(bool utc) const;
	virtual Time get_time(bool utc) const;
	virtual TimeZoneInfo get_time_zone_info() const;
	virtual double get_unix_time() const;

	virtual Error set_cwd(const String &p_cwd);

	virtual void delay_usec(uint32_t p_usec) const;
	virtual uint64_t get_ticks_usec() const;

	virtual Error execute(const String &p_path, const List<String> &p_arguments, bool p_blocking = true, ProcessID *r_child_id = nullptr, String *r_pipe = nullptr, int *r_exitcode = nullptr, bool read_stderr = false, Mutex *p_pipe_mutex = nullptr);
	virtual Error kill(const ProcessID &p_pid);
	virtual int get_process_id() const;

	virtual bool has_environment(const String &p_var) const;
	virtual String get_environment(const String &p_var) const;
	virtual bool set_environment(const String &p_var, const String &p_value) const;

	virtual String get_executable_path() const;

	virtual String get_locale() const;

	virtual int get_processor_count() const;

	virtual String get_config_path() const;
	virtual String get_data_path() const;
	virtual String get_cache_path() const;
	virtual String get_godot_dir_name() const;

	virtual String get_system_dir(SystemDir p_dir) const;
	virtual String get_user_data_dir() const;

	virtual String get_unique_id() const;

	virtual Error shell_open(String p_uri);

	void run();

	virtual bool _check_internal_feature_support(const String &p_feature);

	void disable_crash_handler();
	bool is_disable_crash_handler() const;
	virtual void initialize_debugging();

	virtual Error move_to_trash(const String &p_path);

	void set_main_window(HWND p_main_window) { main_window = p_main_window; }

	HINSTANCE get_hinstance() { return hInstance; }
	OS_Windows(HINSTANCE _hInstance);
	~OS_Windows();
};

#endif
