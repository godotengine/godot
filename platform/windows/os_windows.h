/*************************************************************************/
/*  os_windows.h                                                         */
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

#ifndef OS_WINDOWS_H
#define OS_WINDOWS_H

#include "core/config/project_settings.h"
#include "core/input/input.h"
#include "core/os/os.h"
#include "crash_handler_windows.h"
#include "drivers/unix/ip_unix.h"
#include "drivers/wasapi/audio_driver_wasapi.h"
#include "drivers/winmidi/midi_driver_winmidi.h"
#include "key_mapping_windows.h"
#include "servers/audio_server.h"
#include "servers/rendering/renderer_compositor.h"
#include "servers/rendering_server.h"
#ifdef XAUDIO2_ENABLED
#include "drivers/xaudio2/audio_driver_xaudio2.h"
#endif

#if defined(VULKAN_ENABLED)
#include "drivers/vulkan/rendering_device_vulkan.h"
#include "platform/windows/vulkan_context_win.h"
#endif

#include <fcntl.h>
#include <io.h>
#include <shellapi.h>
#include <stdio.h>
#define WIN32_LEAN_AND_MEAN
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
	virtual void initialize() override;

	virtual void set_main_loop(MainLoop *p_main_loop) override;
	virtual void delete_main_loop() override;

	virtual void finalize() override;
	virtual void finalize_core() override;
	virtual String get_stdin_string(bool p_block) override;

	String _quote_command_line_argument(const String &p_text) const;

	struct ProcessInfo {
		STARTUPINFO si;
		PROCESS_INFORMATION pi;
	};
	Map<ProcessID, ProcessInfo> *process_map;

public:
	virtual void alert(const String &p_alert, const String &p_title = "ALERT!") override;

	virtual Error open_dynamic_library(const String p_path, void *&p_library_handle, bool p_also_set_library_path = false) override;
	virtual Error close_dynamic_library(void *p_library_handle) override;
	virtual Error get_dynamic_library_symbol_handle(void *p_library_handle, const String p_name, void *&p_symbol_handle, bool p_optional = false) override;

	virtual MainLoop *get_main_loop() const override;

	virtual String get_name() const override;

	virtual void initialize_joypads() override {}

	virtual Date get_date(bool p_utc) const override;
	virtual Time get_time(bool p_utc) const override;
	virtual TimeZoneInfo get_time_zone_info() const override;
	virtual double get_unix_time() const override;

	virtual Error set_cwd(const String &p_cwd) override;

	virtual void delay_usec(uint32_t p_usec) const override;
	virtual uint64_t get_ticks_usec() const override;

	virtual Error execute(const String &p_path, const List<String> &p_arguments, String *r_pipe = nullptr, int *r_exitcode = nullptr, bool read_stderr = false, Mutex *p_pipe_mutex = nullptr, bool p_open_console = false) override;
	virtual Error create_process(const String &p_path, const List<String> &p_arguments, ProcessID *r_child_id = nullptr, bool p_open_console = false) override;
	virtual Error kill(const ProcessID &p_pid) override;
	virtual int get_process_id() const override;

	virtual bool has_environment(const String &p_var) const override;
	virtual String get_environment(const String &p_var) const override;
	virtual bool set_environment(const String &p_var, const String &p_value) const override;

	virtual String get_executable_path() const override;

	virtual String get_locale() const override;

	virtual int get_processor_count() const override;

	virtual String get_config_path() const override;
	virtual String get_data_path() const override;
	virtual String get_cache_path() const override;
	virtual String get_godot_dir_name() const override;

	virtual String get_system_dir(SystemDir p_dir, bool p_shared_storage = true) const override;
	virtual String get_user_data_dir() const override;

	virtual String get_unique_id() const override;

	virtual Error shell_open(String p_uri) override;

	void run();

	virtual bool _check_internal_feature_support(const String &p_feature) override;

	virtual void disable_crash_handler() override;
	virtual bool is_disable_crash_handler() const override;
	virtual void initialize_debugging() override;

	virtual Error move_to_trash(const String &p_path) override;

	void set_main_window(HWND p_main_window) { main_window = p_main_window; }

	HINSTANCE get_hinstance() { return hInstance; }
	OS_Windows(HINSTANCE _hInstance);
	~OS_Windows();
};

#endif
