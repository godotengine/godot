/*************************************************************************/
/*  os_linuxbsd.h                                                        */
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

#ifndef OS_LINUXBSD_H
#define OS_LINUXBSD_H

#include "core/input/input.h"
#include "crash_handler_linuxbsd.h"
#include "drivers/alsa/audio_driver_alsa.h"
#include "drivers/alsamidi/midi_driver_alsamidi.h"
#include "drivers/pulseaudio/audio_driver_pulseaudio.h"
#include "drivers/unix/os_unix.h"
#include "joypad_linux.h"
#include "servers/audio_server.h"
#include "servers/rendering/rasterizer.h"
#include "servers/rendering_server.h"

class OS_LinuxBSD : public OS_Unix {
	virtual void delete_main_loop();

	bool force_quit;

#ifdef JOYDEV_ENABLED
	JoypadLinux *joypad = nullptr;
#endif

#ifdef ALSA_ENABLED
	AudioDriverALSA driver_alsa;
#endif

#ifdef ALSAMIDI_ENABLED
	MIDIDriverALSAMidi driver_alsamidi;
#endif

#ifdef PULSEAUDIO_ENABLED
	AudioDriverPulseAudio driver_pulseaudio;
#endif

	CrashHandler crash_handler;

	MainLoop *main_loop;

protected:
	virtual void initialize();
	virtual void finalize();

	virtual void initialize_joypads();

	virtual void set_main_loop(MainLoop *p_main_loop);

public:
	virtual String get_name() const;

	virtual MainLoop *get_main_loop() const;

	virtual String get_config_path() const;
	virtual String get_data_path() const;
	virtual String get_cache_path() const;

	virtual String get_system_dir(SystemDir p_dir) const;

	virtual Error shell_open(String p_uri);

	virtual String get_unique_id() const;

	virtual bool _check_internal_feature_support(const String &p_feature);

	void run();

	void disable_crash_handler();
	bool is_disable_crash_handler() const;

	virtual Error move_to_trash(const String &p_path);

	OS_LinuxBSD();
};

#endif
