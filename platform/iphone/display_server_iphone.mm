/*************************************************************************/
/*  display_server_iphone.mm                                             */
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

#include "display_server_iphone.h"

#import "app_delegate.h"
#include "core/config/project_settings.h"
#include "core/io/file_access_pack.h"
#import "device_metrics.h"
#import "godot_view.h"
#import "godot_view_controller.h"
#include "ios.h"
#import "keyboard_input_view.h"
#include "os_iphone.h"

#import <Foundation/Foundation.h>
#import <sys/utsname.h>

CALayer *initialize_uikit_rendering_layer(const String &p_driver) {
	NSString *driverName = [NSString stringWithUTF8String:p_driver.utf8().get_data()];
	return [AppDelegate.viewController.godotView initializeRenderingForDriver:driverName];
}

DisplayServerIPhone *DisplayServerIPhone::get_singleton() {
	return (DisplayServerIPhone *)DisplayServer::get_singleton();
}

DisplayServerIPhone::DisplayServerIPhone(const String &p_rendering_driver, WindowMode p_mode, DisplayServer::VSyncMode p_vsync_mode, uint32_t p_flags, const Vector2i &p_resolution, Error &r_error) :
		DisplayServerUIKit(p_rendering_driver, p_mode, p_vsync_mode, p_flags, p_resolution, r_error) {
}

DisplayServerIPhone::~DisplayServerIPhone() {
}

DisplayServer *DisplayServerIPhone::create_func(const String &p_rendering_driver, WindowMode p_mode, DisplayServer::VSyncMode p_vsync_mode, uint32_t p_flags, const Vector2i &p_resolution, Error &r_error) {
	return memnew(DisplayServerIPhone(p_rendering_driver, p_mode, p_vsync_mode, p_flags, p_resolution, r_error));
}

Vector<String> DisplayServerIPhone::get_rendering_drivers_func() {
	Vector<String> drivers;

#if defined(VULKAN_ENABLED)
	drivers.push_back("vulkan");
#endif
#if defined(GLES3_ENABLED)
	drivers.push_back("opengl_es");
#endif

	return drivers;
}

void DisplayServerIPhone::register_iphone_driver() {
	register_create_function("iphone", create_func, get_rendering_drivers_func);
}

// MARK: -

bool DisplayServerIPhone::has_feature(Feature p_feature) const {
	switch (p_feature) {
		// case FEATURE_CURSOR_SHAPE:
		// case FEATURE_CUSTOM_CURSOR_SHAPE:
		// case FEATURE_GLOBAL_MENU:
		// case FEATURE_HIDPI:
		// case FEATURE_ICON:
		// case FEATURE_IME:
		// case FEATURE_MOUSE:
		// case FEATURE_MOUSE_WARP:
		// case FEATURE_NATIVE_DIALOG:
		// case FEATURE_NATIVE_ICON:
		// case FEATURE_WINDOW_TRANSPARENCY:
		case FEATURE_CLIPBOARD:
		case FEATURE_KEEP_SCREEN_ON:
		case FEATURE_ORIENTATION:
		case FEATURE_TOUCHSCREEN:
		case FEATURE_VIRTUAL_KEYBOARD:
			return true;
		default:
			return false;
	}
}

String DisplayServerIPhone::get_name() const {
	return "iPhone";
}

Size2i DisplayServerIPhone::screen_get_size(int p_screen) const {
	CALayer *layer = AppDelegate.viewController.godotView.renderingLayer;

	if (!layer) {
		return Size2i();
	}

	return Size2i(layer.bounds.size.width, layer.bounds.size.height) * screen_get_scale(p_screen);
}

Rect2i DisplayServerIPhone::screen_get_usable_rect(int p_screen) const {
	if (@available(iOS 11, *)) {
		UIEdgeInsets insets = UIEdgeInsetsZero;
		UIView *view = AppDelegate.viewController.godotView;

		if ([view respondsToSelector:@selector(safeAreaInsets)]) {
			insets = [view safeAreaInsets];
		}

		float scale = screen_get_scale(p_screen);
		Size2i insets_position = Size2i(insets.left, insets.top) * scale;
		Size2i insets_size = Size2i(insets.left + insets.right, insets.top + insets.bottom) * scale;

		return Rect2i(screen_get_position(p_screen) + insets_position, screen_get_size(p_screen) - insets_size);
	} else {
		return Rect2i(screen_get_position(p_screen), screen_get_size(p_screen));
	}
}

int DisplayServerIPhone::screen_get_dpi(int p_screen) const {
	struct utsname systemInfo;
	uname(&systemInfo);

	NSString *string = [NSString stringWithCString:systemInfo.machine encoding:NSUTF8StringEncoding];

	NSDictionary *iOSModelToDPI = [GodotDeviceMetrics dpiList];

	for (NSArray *keyArray in iOSModelToDPI) {
		if ([keyArray containsObject:string]) {
			NSNumber *value = iOSModelToDPI[keyArray];
			return [value intValue];
		}
	}

	// If device wasn't found in dictionary
	// make a best guess from device metrics.
	CGFloat scale = [UIScreen mainScreen].scale;

	UIUserInterfaceIdiom idiom = [UIDevice currentDevice].userInterfaceIdiom;

	switch (idiom) {
		case UIUserInterfaceIdiomPad:
			return scale == 2 ? 264 : 132;
		case UIUserInterfaceIdiomPhone: {
			if (scale == 3) {
				CGFloat nativeScale = [UIScreen mainScreen].nativeScale;
				return nativeScale == 3 ? 458 : 401;
			}

			return 326;
		}
		default:
			return 72;
	}
}

int64_t DisplayServerIPhone::window_get_native_handle(HandleType p_handle_type, WindowID p_window) const {
	ERR_FAIL_COND_V(p_window != MAIN_WINDOW_ID, 0);
	switch (p_handle_type) {
		case DISPLAY_HANDLE: {
			return 0; // Not supported.
		}
		case WINDOW_HANDLE: {
			return (int64_t)AppDelegate.viewController;
		}
		case WINDOW_VIEW: {
			return (int64_t)AppDelegate.viewController.godotView;
		}
		default: {
			return 0;
		}
	}
}

void DisplayServerIPhone::screen_set_orientation(DisplayServer::ScreenOrientation p_orientation, int p_screen) {
	screen_orientation = p_orientation;
}

DisplayServer::ScreenOrientation DisplayServerIPhone::screen_get_orientation(int p_screen) const {
	return screen_orientation;
}

bool DisplayServerIPhone::screen_is_touchscreen(int p_screen) const {
	return true;
}

void DisplayServerIPhone::virtual_keyboard_show(const String &p_existing_text, const Rect2 &p_screen_rect, bool p_multiline, int p_max_length, int p_cursor_start, int p_cursor_end) {
	NSString *existingString = [[NSString alloc] initWithUTF8String:p_existing_text.utf8().get_data()];

	[AppDelegate.viewController.keyboardView
			becomeFirstResponderWithString:existingString
								 multiline:p_multiline
							   cursorStart:p_cursor_start
								 cursorEnd:p_cursor_end];
}

void DisplayServerIPhone::virtual_keyboard_hide() {
	[AppDelegate.viewController.keyboardView resignFirstResponder];
}

void DisplayServerIPhone::virtual_keyboard_set_height(int height) {
	virtual_keyboard_height = height * screen_get_max_scale();
}

int DisplayServerIPhone::virtual_keyboard_get_height() const {
	return virtual_keyboard_height;
}

void DisplayServerIPhone::clipboard_set(const String &p_text) {
	[UIPasteboard generalPasteboard].string = [NSString stringWithUTF8String:p_text.utf8()];
}

String DisplayServerIPhone::clipboard_get() const {
	NSString *text = [UIPasteboard generalPasteboard].string;

	return String::utf8([text UTF8String]);
}
