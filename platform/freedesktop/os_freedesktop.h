/*************************************************************************/
/*  os_freedesktop.h                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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

#ifndef OS_FREEDESKTOP_H
#define OS_FREEDESKTOP_H

#include "crash_handler_x11.h"
#include "drivers/alsa/audio_driver_alsa.h"
#include "drivers/pulseaudio/audio_driver_pulseaudio.h"
#include "drivers/unix/os_unix.h"
#include "os/input.h"
#include "power_x11.h"
#include "servers/audio_server.h"
/**
	@author Juan Linietsky <reduzio@gmail.com>
*/

class OS_Freedesktop : public OS_Unix {

	List<String> args;
	MainLoop *main_loop;
	char *xmbstring;
	int xmblen;
	unsigned long last_timestamp;

	virtual void delete_main_loop();
	IP_Unix *ip_unix;

	bool force_quit;
	bool minimized;
	bool window_has_focus;
	bool do_mouse_warp;

#ifdef ALSA_ENABLED
	AudioDriverALSA driver_alsa;
#endif

#ifdef PULSEAUDIO_ENABLED
	AudioDriverPulseAudio driver_pulseaudio;
#endif

	PowerX11 *power_manager;

	CrashHandler crash_handler;

	int audio_driver_index;
	unsigned int capture_idle;

protected:
	virtual int get_audio_driver_count() const;
	virtual const char *get_audio_driver_name(int p_driver) const;

	virtual void initialize_core();
	virtual Error initialize(int p_audio_driver);
	virtual void finalize();

	virtual void set_main_loop(MainLoop *p_main_loop);

public:
	virtual String get_name();

	virtual MainLoop *get_main_loop() const;

	virtual String get_config_path() const;
	virtual String get_data_path() const;
	virtual String get_cache_path() const;

	virtual String get_system_dir(SystemDir p_dir) const;

	virtual Error shell_open(String p_uri);

	virtual String get_unique_id() const;

	virtual void alert(const String &p_alert, const String &p_title = "ALERT!");

	virtual OS::PowerState get_power_state();
	virtual int get_power_seconds_left();
	virtual int get_power_percent_left();

	virtual bool _check_internal_feature_support(const String &p_feature);

	void run();

	void disable_crash_handler();
	bool is_disable_crash_handler() const;

	virtual Error move_to_trash(const String &p_path);

	OS_Freedesktop();
};

#endif
