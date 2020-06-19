/*************************************************************************/
/*  os_osx.h                                                             */
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

#ifndef OS_OSX_H
#define OS_OSX_H

#include "core/input/input.h"
#include "crash_handler_osx.h"
#include "drivers/coreaudio/audio_driver_coreaudio.h"
#include "drivers/coremidi/midi_driver_coremidi.h"
#include "drivers/unix/os_unix.h"
#include "joypad_osx.h"
#include "servers/audio_server.h"

class OS_OSX : public OS_Unix {
	virtual void delete_main_loop();

	bool force_quit;

	JoypadOSX *joypad_osx;

#ifdef COREAUDIO_ENABLED
	AudioDriverCoreAudio audio_driver;
#endif
#ifdef COREMIDI_ENABLED
	MIDIDriverCoreMidi midi_driver;
#endif

	CrashHandler crash_handler;

<<<<<<< HEAD
	MainLoop *main_loop;
=======
	float _mouse_scale(float p_scale) {
		if (_display_scale() > 1.0)
			return p_scale;
		else
			return 1.0;
	}

	float _display_scale() const;
	float _display_scale(id screen) const;

	void _update_window();

	int video_driver_index;
	virtual int get_current_video_driver() const;

	struct GlobalMenuItem {
		String label;
		Variant signal;
		Variant meta;

		GlobalMenuItem() {
			//NOP
		}

		GlobalMenuItem(const String &p_label, const Variant &p_signal, const Variant &p_meta) {
			label = p_label;
			signal = p_signal;
			meta = p_meta;
		}
	};

	Map<String, Vector<GlobalMenuItem> > global_menus;
	List<String> global_menus_order;
>>>>>>> master

public:
	String open_with_filename;

protected:
	virtual void initialize_core();
	virtual void initialize();
	virtual void finalize();

	virtual void initialize_joypads();

	virtual void set_main_loop(MainLoop *p_main_loop);

public:
	virtual String get_name() const;

	virtual Error open_dynamic_library(const String p_path, void *&p_library_handle, bool p_also_set_library_path = false);

	virtual MainLoop *get_main_loop() const;

	virtual String get_config_path() const;
	virtual String get_data_path() const;
	virtual String get_cache_path() const;
	virtual String get_bundle_resource_dir() const;
	virtual String get_godot_dir_name() const;

	virtual String get_system_dir(SystemDir p_dir) const;

	Error shell_open(String p_uri);

	String get_locale() const;

	virtual String get_executable_path() const;

<<<<<<< HEAD
	virtual String get_unique_id() const; //++
=======
	virtual LatinKeyboardVariant get_latin_keyboard_variant() const;
	virtual int keyboard_get_layout_count() const;
	virtual int keyboard_get_current_layout() const;
	virtual void keyboard_set_current_layout(int p_index);
	virtual String keyboard_get_layout_language(int p_index) const;
	virtual String keyboard_get_layout_name(int p_index) const;

	virtual void move_window_to_foreground();

	virtual int get_screen_count() const;
	virtual int get_current_screen() const;
	virtual void set_current_screen(int p_screen);
	virtual Point2 get_screen_position(int p_screen = -1) const;
	virtual Size2 get_screen_size(int p_screen = -1) const;
	virtual int get_screen_dpi(int p_screen = -1) const;

	virtual Point2 get_window_position() const;
	virtual void set_window_position(const Point2 &p_position);
	virtual Size2 get_max_window_size() const;
	virtual Size2 get_min_window_size() const;
	virtual void set_min_window_size(const Size2 p_size);
	virtual void set_max_window_size(const Size2 p_size);
	virtual void set_window_size(const Size2 p_size);
	virtual void set_window_fullscreen(bool p_enabled);
	virtual bool is_window_fullscreen() const;
	virtual void set_window_resizable(bool p_enabled);
	virtual bool is_window_resizable() const;
	virtual void set_window_minimized(bool p_enabled);
	virtual bool is_window_minimized() const;
	virtual void set_window_maximized(bool p_enabled);
	virtual bool is_window_maximized() const;
	virtual void set_window_always_on_top(bool p_enabled);
	virtual bool is_window_always_on_top() const;
	virtual bool is_window_focused() const;
	virtual void request_attention();
	virtual String get_joy_guid(int p_device) const;

	virtual void set_borderless_window(bool p_borderless);
	virtual bool get_borderless_window();

	virtual bool get_window_per_pixel_transparency_enabled() const;
	virtual void set_window_per_pixel_transparency_enabled(bool p_enabled);

	virtual void set_ime_active(const bool p_active);
	virtual void set_ime_position(const Point2 &p_pos);
	virtual Point2 get_ime_selection() const;
	virtual String get_ime_text() const;

	virtual String get_unique_id() const;

	virtual OS::PowerState get_power_state();
	virtual int get_power_seconds_left();
	virtual int get_power_percent_left();
>>>>>>> master

	virtual bool _check_internal_feature_support(const String &p_feature);

	void run();

	void disable_crash_handler();
	bool is_disable_crash_handler() const;

	virtual Error move_to_trash(const String &p_path);

	OS_OSX();
};

#endif
