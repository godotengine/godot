/*************************************************************************/
/*  display_server_iphone.h                                              */
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

#ifndef display_server_iphone_h
#define display_server_iphone_h

#include "platform/uikit/uikit_display_server.h"

#include "core/input/input.h"
#include "servers/display_server.h"

#if defined(VULKAN_ENABLED)
#include "drivers/vulkan/rendering_device_vulkan.h"
#include "servers/rendering/renderer_rd/renderer_compositor_rd.h"

#import <QuartzCore/CAMetalLayer.h>
#ifdef USE_VOLK
#include <volk.h>
#else
#include <vulkan/vulkan.h>
#endif
#endif

CALayer *initialize_uikit_rendering_layer(const String &p_driver);

class DisplayServerIPhone : public DisplayServerUIKit {
	GDCLASS(DisplayServerIPhone, DisplayServer)

	_THREAD_SAFE_CLASS_

	DisplayServer::ScreenOrientation screen_orientation;

	int virtual_keyboard_height = 0;

	DisplayServerIPhone(const String &p_rendering_driver, DisplayServer::WindowMode p_mode, DisplayServer::VSyncMode p_vsync_mode, uint32_t p_flags, const Vector2i &p_resolution, Error &r_error);
	~DisplayServerIPhone();

public:
	static DisplayServerIPhone *get_singleton();

	static void register_iphone_driver();
	static DisplayServer *create_func(const String &p_rendering_driver, WindowMode p_mode, DisplayServer::VSyncMode p_vsync_mode, uint32_t p_flags, const Vector2i &p_resolution, Error &r_error);
	static Vector<String> get_rendering_drivers_func();

	// MARK: -

	virtual bool has_feature(Feature p_feature) const override;
	virtual String get_name() const override;

	virtual Size2i screen_get_size(int p_screen = SCREEN_OF_MAIN_WINDOW) const override;
	virtual Rect2i screen_get_usable_rect(int p_screen = SCREEN_OF_MAIN_WINDOW) const override;
	virtual int screen_get_dpi(int p_screen = SCREEN_OF_MAIN_WINDOW) const override;

	virtual int64_t window_get_native_handle(HandleType p_handle_type, WindowID p_window = MAIN_WINDOW_ID) const override;

	virtual void screen_set_orientation(DisplayServer::ScreenOrientation p_orientation, int p_screen) override;
	virtual DisplayServer::ScreenOrientation screen_get_orientation(int p_screen) const override;

	virtual bool screen_is_touchscreen(int p_screen) const override;

	virtual void virtual_keyboard_show(const String &p_existing_text, const Rect2 &p_screen_rect, bool p_multiline, int p_max_length, int p_cursor_start, int p_cursor_end) override;
	virtual void virtual_keyboard_hide() override;

	void virtual_keyboard_set_height(int height);
	virtual int virtual_keyboard_get_height() const override;

	virtual void clipboard_set(const String &p_text) override;
	virtual String clipboard_get() const override;
};

#endif /* display_server_iphone_h */
