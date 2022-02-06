/*************************************************************************/
/*  os_iphone.h                                                          */
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

#ifndef OS_IPHONE_H
#define OS_IPHONE_H

#include "platform/uikit/uikit_os.h"

#include "ios.h"

class OSIPhone : public OS_UIKit {
private:
	static HashMap<String, void *> dynamic_symbol_lookup_table;
	friend void register_dynamic_symbol(char *name, void *address);

	iOS *ios;

	int virtual_keyboard_height = 0;

	virtual Error initialize(const VideoMode &p_desired, int p_video_driver, int p_audio_driver);
	virtual void finalize();

public:
	static OSIPhone *get_singleton();

	OSIPhone(String p_data_dir, String p_cache_dir);
	~OSIPhone();

	virtual Error get_dynamic_library_symbol_handle(void *p_library_handle, const String p_name, void *&p_symbol_handle, bool p_optional = false);

	virtual void alert(const String &p_alert, const String &p_title = "ALERT!");

	virtual String get_name() const;
	virtual String get_model_name() const;

	virtual void set_clipboard(const String &p_text);
	virtual String get_clipboard() const;

	virtual String get_processor_name() const;

	virtual void vibrate_handheld(int p_duration_ms = 500);

	virtual bool _check_internal_feature_support(const String &p_feature);

	virtual int get_screen_dpi(int p_screen = -1) const;
	virtual float get_screen_refresh_rate(int p_screen = -1) const;

	void set_virtual_keyboard_height(int p_height);

	virtual bool has_virtual_keyboard() const;
	virtual void show_virtual_keyboard(const String &p_existing_text, const Rect2 &p_screen_rect = Rect2(), bool p_multiline = false, int p_max_input_length = -1, int p_cursor_start = -1, int p_cursor_end = -1);
	virtual void hide_virtual_keyboard();
	virtual int get_virtual_keyboard_height() const;

	virtual Rect2 get_window_safe_area() const;

	virtual bool has_touchscreen_ui_hint() const;

	virtual Error native_video_play(String p_path, float p_volume, String p_audio_track, String p_subtitle_track);
	virtual bool native_video_is_playing() const;
	virtual void native_video_pause();
	virtual void native_video_unpause();
	virtual void native_video_focus_out();
	virtual void native_video_stop();

	void on_focus_out();
	void on_focus_in();
};

#endif // OS_IPHONE_H
