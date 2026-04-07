/**************************************************************************/
/*  display_server_tvos.mm                                                */
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

#import "display_server_tvos.h"

#import "drivers/apple_embedded/godot_app_delegate_service_apple_embedded.h"
#import "drivers/apple_embedded/godot_view_apple_embedded.h"
#import "drivers/apple_embedded/godot_view_controller.h"

#import <UIKit/UIKit.h>

DisplayServerTVOS *DisplayServerTVOS::get_singleton() {
	return (DisplayServerTVOS *)DisplayServerAppleEmbedded::get_singleton();
}

DisplayServerTVOS::DisplayServerTVOS(const String &p_rendering_driver, DisplayServerEnums::WindowMode p_mode, DisplayServerEnums::VSyncMode p_vsync_mode, uint32_t p_flags, const Vector2i *p_position, const Vector2i &p_resolution, int p_screen, DisplayServerEnums::Context p_context, int64_t p_parent_window, Error &r_error) :
		DisplayServerAppleEmbedded(p_rendering_driver, p_mode, p_vsync_mode, p_flags, p_position, p_resolution, p_screen, p_context, p_parent_window, r_error) {
}

DisplayServerTVOS::~DisplayServerTVOS() {
}

DisplayServer *DisplayServerTVOS::create_func(const String &p_rendering_driver, DisplayServerEnums::WindowMode p_mode, DisplayServerEnums::VSyncMode p_vsync_mode, uint32_t p_flags, const Vector2i *p_position, const Vector2i &p_resolution, int p_screen, DisplayServerEnums::Context p_context, int64_t p_parent_window, Error &r_error) {
	return memnew(DisplayServerTVOS(p_rendering_driver, p_mode, p_vsync_mode, p_flags, p_position, p_resolution, p_screen, p_context, p_parent_window, r_error));
}

void DisplayServerTVOS::register_tvos_driver() {
	register_create_function("tvOS", create_func, get_rendering_drivers_func);
}

String DisplayServerTVOS::get_name() const {
	return "tvOS";
}

int DisplayServerTVOS::screen_get_dpi(int p_screen) const {
	p_screen = _get_screen_index(p_screen);
	int screen_count = get_screen_count();
	ERR_FAIL_INDEX_V(p_screen, screen_count, 72);

	// No way to reliably determine DPI.
	return 72;
}

UIScreen *DisplayServerTVOS::_get_ui_screen(int p_screen) const {
	p_screen = _get_screen_index(p_screen);
	int screen_count = get_screen_count();
	ERR_FAIL_INDEX_V(p_screen, screen_count, nil);

	return GDTAppDelegateService.viewController.godotView.window.windowScene.screen;
}

float DisplayServerTVOS::screen_get_refresh_rate(int p_screen) const {
	UIScreen *screen = _get_ui_screen(p_screen);
	ERR_FAIL_NULL_V(screen, SCREEN_REFRESH_RATE_FALLBACK);
	float fps = screen.maximumFramesPerSecond;
	if ([NSProcessInfo processInfo].lowPowerModeEnabled) {
		fps = 60;
	}
	return fps;
}

float DisplayServerTVOS::screen_get_scale(int p_screen) const {
	UIScreen *screen = _get_ui_screen(p_screen);
	ERR_FAIL_NULL_V(screen, 1.0f);

	return screen.scale;
}

// TODO: tvOS virtual keyboard support.
// UITextView.editable is unavailable on tvOS and UIKeyInput on custom views
// doesn't trigger the keyboard (rdar://27389949). UITextField works but needs
// further integration with the Godot input system. For now, keyboard input
// is not supported on tvOS.

// MARK: HDR

bool DisplayServerTVOS::_screen_hdr_is_supported() const {
	UIScreen *screen = _get_ui_screen();
	ERR_FAIL_NULL_V(screen, false);
	return screen.potentialEDRHeadroom > 1.0;
}

float DisplayServerTVOS::_screen_potential_edr_headroom() const {
	UIScreen *screen = _get_ui_screen();
	ERR_FAIL_NULL_V(screen, false);
	return screen.potentialEDRHeadroom;
}

float DisplayServerTVOS::_screen_current_edr_headroom() const {
	UIScreen *screen = _get_ui_screen();
	ERR_FAIL_NULL_V(screen, false);
	return screen.currentEDRHeadroom;
}
