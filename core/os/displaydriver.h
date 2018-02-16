/*************************************************************************/
/*  displaydriver.h                                                      */
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

#ifndef DISPLAYDRIVER_H
#define DISPLAYDRIVER_H

#include "os.h"

#include "engine.h"
#include "image.h"
#include "list.h"
#include "ustring.h"
#include "vector.h"

/**
	@author Juan Linietsky <reduzio@gmail.com>
*/

class DisplayDriver {

	static DisplayDriver *singleton;
	bool _keep_screen_on;
	String _local_clipboard;
	bool _no_window;
	int _orientation;
	bool _allow_hidpi;
	bool _use_vsync;

public:
	typedef void (*ImeCallback)(void *p_inp, String p_text, Point2 p_selection);

	enum RenderThreadMode {

		RENDER_THREAD_UNSAFE,
		RENDER_THREAD_SAFE,
		RENDER_SEPARATE_THREAD
	};

	struct VideoMode {

		int width, height;
		bool fullscreen;
		bool resizable;
		bool borderless_window;
		bool maximized;
		bool always_on_top;
		bool use_vsync;
		float get_aspect() const { return (float)width / (float)height; }
		VideoMode(int p_width = 1024, int p_height = 600, bool p_fullscreen = false, bool p_resizable = true, bool p_borderless_window = false, bool p_maximized = false, bool p_always_on_top = false, bool p_use_vsync = false) {
			width = p_width;
			height = p_height;
			fullscreen = p_fullscreen;
			resizable = p_resizable;
			borderless_window = p_borderless_window;
			maximized = p_maximized;
			always_on_top = p_always_on_top;
			use_vsync = p_use_vsync;
		}
	};

protected:
	friend class Main;

	RenderThreadMode _render_thread_mode;

	// functions used by main to initialize/deintialize the OS
	virtual int get_video_driver_count() const = 0;
	virtual const char *get_video_driver_name(int p_driver) const = 0;

	virtual Error initialize(const VideoMode &p_desired, int p_video_driver) = 0;
	virtual void finalize() = 0;

public:
	static DisplayDriver *get_singleton();

	enum MouseMode {
		MOUSE_MODE_VISIBLE,
		MOUSE_MODE_HIDDEN,
		MOUSE_MODE_CAPTURED,
		MOUSE_MODE_CONFINED
	};

	virtual void set_main_loop(MainLoop *p_main_loop) = 0;

	virtual void set_mouse_mode(MouseMode p_mode);
	virtual MouseMode get_mouse_mode() const;

	virtual void warp_mouse_position(const Point2 &p_to) {}
	virtual Point2 get_mouse_position() const = 0;
	virtual int get_mouse_button_state() const = 0;
	virtual void set_window_title(const String &p_title) = 0;

	virtual void set_clipboard(const String &p_text);
	virtual String get_clipboard() const;

	virtual void set_video_mode(const VideoMode &p_video_mode, int p_screen = 0) = 0;
	virtual VideoMode get_video_mode(int p_screen = 0) const = 0;
	virtual void get_fullscreen_mode_list(List<VideoMode> *p_list, int p_screen = 0) const = 0;

	virtual int get_screen_count() const { return 1; }
	virtual int get_current_screen() const { return 0; }
	virtual void set_current_screen(int p_screen) {}
	virtual Point2 get_screen_position(int p_screen = -1) const { return Point2(); }
	virtual Size2 get_screen_size(int p_screen = -1) const { return get_window_size(); }
	virtual int get_screen_dpi(int p_screen = -1) const { return 72; }
	virtual Point2 get_window_position() const { return Vector2(); }
	virtual void set_window_position(const Point2 &p_position) {}
	virtual Size2 get_window_size() const = 0;
	virtual Size2 get_real_window_size() const { return get_window_size(); }
	virtual void set_window_size(const Size2 p_size) {}
	virtual void set_window_fullscreen(bool p_enabled) {}
	virtual bool is_window_fullscreen() const { return true; }
	virtual void set_window_resizable(bool p_enabled) {}
	virtual bool is_window_resizable() const { return false; }
	virtual void set_window_minimized(bool p_enabled) {}
	virtual bool is_window_minimized() const { return false; }
	virtual void set_window_maximized(bool p_enabled) {}
	virtual bool is_window_maximized() const { return true; }
	virtual void set_window_always_on_top(bool p_enabled) {}
	virtual bool is_window_always_on_top() const { return false; }
	virtual void request_attention() {}
	virtual void center_window();

	virtual void set_borderless_window(bool p_borderless) {}
	virtual bool get_borderless_window() { return 0; }

	virtual void set_ime_position(const Point2 &p_pos) {}
	virtual void set_ime_intermediate_text_callback(ImeCallback p_callback, void *p_inp) {}

	virtual void set_keep_screen_on(bool p_enabled);
	virtual bool is_keep_screen_on() const;

	virtual String get_name() = 0;
	virtual bool can_draw() const = 0;

	enum CursorShape {
		CURSOR_ARROW,
		CURSOR_IBEAM,
		CURSOR_POINTING_HAND,
		CURSOR_CROSS,
		CURSOR_WAIT,
		CURSOR_BUSY,
		CURSOR_DRAG,
		CURSOR_CAN_DROP,
		CURSOR_FORBIDDEN,
		CURSOR_VSIZE,
		CURSOR_HSIZE,
		CURSOR_BDIAGSIZE,
		CURSOR_FDIAGSIZE,
		CURSOR_MOVE,
		CURSOR_VSPLIT,
		CURSOR_HSPLIT,
		CURSOR_HELP,
		CURSOR_MAX
	};

	virtual bool has_virtual_keyboard() const;
	virtual void show_virtual_keyboard(const String &p_existing_text, const Rect2 &p_screen_rect = Rect2());
	virtual void hide_virtual_keyboard();

	// returns height of the currently shown virtual keyboard (0 if keyboard is hidden)
	virtual int get_virtual_keyboard_height() const;

	virtual void set_cursor_shape(CursorShape p_shape) = 0;
	virtual void set_custom_mouse_cursor(const RES &p_cursor, CursorShape p_shape, const Vector2 &p_hotspot) = 0;

	RenderThreadMode get_render_thread_mode() const { return _render_thread_mode; }

	virtual void set_no_window_mode(bool p_enable);
	virtual bool is_no_window_mode_enabled() const;

	virtual bool has_touchscreen_ui_hint() const;

	enum ScreenOrientation {

		SCREEN_LANDSCAPE,
		SCREEN_PORTRAIT,
		SCREEN_REVERSE_LANDSCAPE,
		SCREEN_REVERSE_PORTRAIT,
		SCREEN_SENSOR_LANDSCAPE,
		SCREEN_SENSOR_PORTRAIT,
		SCREEN_SENSOR,
	};

	virtual void set_screen_orientation(ScreenOrientation p_orientation);
	ScreenOrientation get_screen_orientation() const;

	virtual void enable_for_stealing_focus(OS::ProcessID pid) {}
	virtual void move_window_to_foreground() {}

	virtual void release_rendering_thread();
	virtual void make_rendering_thread();
	virtual void swap_buffers();

	virtual void set_icon(const Ref<Image> &p_icon);

	virtual Error native_video_play(String p_path, float p_volume, String p_audio_track, String p_subtitle_track);
	virtual bool native_video_is_playing() const;
	virtual void native_video_pause();
	virtual void native_video_unpause();
	virtual void native_video_stop();

	virtual Error dialog_show(String p_title, String p_description, Vector<String> p_buttons, Object *p_obj, String p_callback);
	virtual Error dialog_input_text(String p_title, String p_description, String p_partial, Object *p_obj, String p_callback);

	enum LatinKeyboardVariant {
		LATIN_KEYBOARD_QWERTY,
		LATIN_KEYBOARD_QWERTZ,
		LATIN_KEYBOARD_AZERTY,
		LATIN_KEYBOARD_QZERTY,
		LATIN_KEYBOARD_DVORAK,
		LATIN_KEYBOARD_NEO,
		LATIN_KEYBOARD_COLEMAK,
	};

	virtual LatinKeyboardVariant get_latin_keyboard_variant() const;

	virtual bool is_joy_known(int p_device);
	virtual String get_joy_guid(int p_device) const;

	//amazing hack because OpenGL needs this to be set on a separate thread..
	//also core can't access servers, so a callback must be used
	typedef void (*SwitchVSyncCallbackInThread)(bool);

	static SwitchVSyncCallbackInThread switch_vsync_function;
	void set_use_vsync(bool p_enable);
	bool is_vsync_enabled() const;

	//real, actual overridable function to switch vsync, which needs to be called from graphics thread if needed
	virtual void _set_use_vsync(bool p_enable) {}

	virtual void force_process_input(){};
	bool has_feature(const String &p_feature);

	bool is_hidpi_allowed() const { return _allow_hidpi; }

	enum EngineContext {
		CONTEXT_EDITOR,
		CONTEXT_PROJECTMAN,
	};

	virtual void set_context(int p_context);
	virtual void process_events();

	DisplayDriver();
	virtual ~DisplayDriver();
};

#endif
