/**************************************************************************/
/*  display_server_macos_base.h                                           */
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

#include "servers/display/display_server.h"

#define FontVariation __FontVariation

#import <AppKit/AppKit.h>
#import <IOKit/pwr_mgt/IOPMLib.h>

#undef FontVariation

class RenderingContextDriver;
class RenderingDevice;

class DisplayServerMacOSBase : public DisplayServer {
	GDSOFTCLASS(DisplayServerMacOSBase, DisplayServer)

	id tts = nullptr;
	IOPMAssertionID screen_keep_on_assertion = kIOPMNullAssertionID;

	struct LayoutInfo {
		String name;
		String code;
	};
	mutable Vector<LayoutInfo> kbd_layouts;
	mutable int current_layout = 0;
	mutable bool keyboard_layout_dirty = true;

	Callable system_theme_changed;

	constexpr static CGFloat HARDWARE_REFERENCE_LUMINANCE_NITS = 100.0f;

public:
	struct HDROutput {
		static constexpr float AUTO_MAX_LUMINANCE = -1.0f;

		bool requested = false;
		float max_luminance = AUTO_MAX_LUMINANCE;

		bool is_auto_max_luminance() const { return max_luminance < 0.0f; }
	};

protected:
	_THREAD_SAFE_CLASS_

	MouseMode mouse_mode = MOUSE_MODE_VISIBLE;
	MouseMode mouse_mode_base = MOUSE_MODE_VISIBLE;
	MouseMode mouse_mode_override = MOUSE_MODE_VISIBLE;
	bool mouse_mode_override_enabled = false;

	void _mouse_update_mode();
	virtual void _mouse_apply_mode(MouseMode p_prev_mode, MouseMode p_new_mode) = 0;

	String im_text;
	Point2i im_selection;

	CursorShape cursor_shape = CURSOR_ARROW;

	void initialize_tts() const;

	static CGDirectDisplayID _get_display_id_for_screen(NSScreen *p_screen);

	void _update_keyboard_layouts() const;
	static void _keyboard_layout_changed(CFNotificationCenterRef center, void *observer, CFStringRef name, const void *object, CFDictionaryRef user_info);

#if defined(RD_ENABLED)
	RenderingContextDriver *rendering_context = nullptr;
	RenderingDevice *rendering_device = nullptr;
#endif

	virtual HDROutput &_get_hdr_output(WindowID p_window) = 0;
	virtual const HDROutput &_get_hdr_output(WindowID p_window) const = 0;

	constexpr float _calculate_current_reference_luminance(CGFloat p_max_potential_edr_value, CGFloat p_max_edr_value) const;
	void _update_hdr_output(WindowID p_window, const HDROutput &p_hdr);

public:
	virtual bool tts_is_speaking() const override;
	virtual bool tts_is_paused() const override;
	virtual TypedArray<Dictionary> tts_get_voices() const override;

	virtual void tts_speak(const String &p_text, const String &p_voice, int p_volume = 50, float p_pitch = 1.f, float p_rate = 1.f, int64_t p_utterance_id = 0, bool p_interrupt = false) override;
	virtual void tts_pause() override;
	virtual void tts_resume() override;
	virtual void tts_stop() override;

	void emit_system_theme_changed();
	virtual bool is_dark_mode_supported() const override;
	virtual bool is_dark_mode() const override;
	virtual Color get_accent_color() const override;
	virtual Color get_base_color() const override;
	virtual void set_system_theme_change_callback(const Callable &p_callable) override;

	virtual void mouse_set_mode(MouseMode p_mode) override;
	virtual MouseMode mouse_get_mode() const override;
	virtual void mouse_set_mode_override(MouseMode p_mode) override;
	virtual MouseMode mouse_get_mode_override() const override;
	virtual void mouse_set_mode_override_enabled(bool p_override_enabled) override;
	virtual bool mouse_is_mode_override_enabled() const override;

	virtual void clipboard_set(const String &p_text) override;
	virtual String clipboard_get() const override;
	virtual Ref<Image> clipboard_get_image() const override;
	virtual bool clipboard_has() const override;
	virtual bool clipboard_has_image() const override;

	virtual int get_primary_screen() const override;
	virtual float screen_get_refresh_rate(int p_screen = SCREEN_OF_MAIN_WINDOW) const override;
	virtual void screen_set_keep_on(bool p_enable) override;
	virtual bool screen_is_kept_on() const override;

	virtual int accessibility_should_increase_contrast() const override;
	virtual int accessibility_should_reduce_animation() const override;
	virtual int accessibility_should_reduce_transparency() const override;
	virtual int accessibility_screen_reader_active() const override;

	void update_im_text(const Point2i &p_selection, const String &p_text);
	virtual Point2i ime_get_selection() const override;
	virtual String ime_get_text() const override;

	virtual CursorShape cursor_get_shape() const override;

	virtual void beep() const override;

	virtual int keyboard_get_layout_count() const override;
	virtual int keyboard_get_current_layout() const override;
	virtual void keyboard_set_current_layout(int p_index) override;
	virtual String keyboard_get_layout_language(int p_index) const override;
	virtual String keyboard_get_layout_name(int p_index) const override;
	virtual Key keyboard_get_keycode_from_physical(Key p_keycode) const override;
	virtual Key keyboard_get_label_from_physical(Key p_keycode) const override;
	virtual void show_emoji_and_symbol_picker() const override;

	virtual void window_get_edr_values(WindowID p_window, CGFloat *r_max_potential_edr_value = nullptr, CGFloat *r_max_edr_value = nullptr) const = 0;
	virtual bool window_is_hdr_output_supported(WindowID p_window = MAIN_WINDOW_ID) const override;
	virtual void window_request_hdr_output(const bool p_enable, WindowID p_window = MAIN_WINDOW_ID) override;
	virtual bool window_is_hdr_output_requested(WindowID p_window = MAIN_WINDOW_ID) const override;
	virtual bool window_is_hdr_output_enabled(WindowID p_window = MAIN_WINDOW_ID) const override;

	virtual void window_set_hdr_output_reference_luminance(const float p_reference_luminance, WindowID p_window = MAIN_WINDOW_ID) override;
	virtual float window_get_hdr_output_reference_luminance(WindowID p_window = MAIN_WINDOW_ID) const override;
	virtual float window_get_hdr_output_current_reference_luminance(WindowID p_window = MAIN_WINDOW_ID) const override;

	virtual void window_set_hdr_output_max_luminance(const float p_max_luminance, WindowID p_window = MAIN_WINDOW_ID) override;
	virtual float window_get_hdr_output_max_luminance(WindowID p_window = MAIN_WINDOW_ID) const override;
	virtual float window_get_hdr_output_current_max_luminance(WindowID p_window = MAIN_WINDOW_ID) const override;

	virtual float window_get_output_max_linear_value(WindowID p_window = MAIN_WINDOW_ID) const override;

	DisplayServerMacOSBase();
	~DisplayServerMacOSBase();
};
