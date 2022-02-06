/*************************************************************************/
/*  os_iphone.mm                                                         */
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

#include "os_iphone.h"

#include "drivers/gles2/rasterizer_gles2.h"
#include "drivers/gles3/rasterizer_gles3.h"
#include "servers/visual/visual_server_raster.h"
#include "servers/visual/visual_server_wrap_mt.h"

#include "main/main.h"

#include "core/io/file_access_pack.h"
#include "core/os/dir_access.h"
#include "core/os/file_access.h"
#include "core/project_settings.h"
#include "drivers/unix/syslog_logger.h"

#import "app_delegate.h"
#import "device_metrics.h"
#import "godot_view.h"
#import "godot_view_controller.h"
#import "keyboard_input_view.h"
#import "native_video_view.h"

#import <UIKit/UIKit.h>
#include <dlfcn.h>
#include <sys/sysctl.h>
#import <sys/utsname.h>

// Initialization order between compilation units is not guaranteed,
// so we use this as a hack to ensure certain code is called before
// everything else, but after all units are initialized.
typedef void (*init_callback)();
static init_callback *ios_init_callbacks = NULL;
static int ios_init_callbacks_count = 0;
static int ios_init_callbacks_capacity = 0;
HashMap<String, void *> OSIPhone::dynamic_symbol_lookup_table;

void add_ios_init_callback(init_callback cb) {
	if (ios_init_callbacks_count == ios_init_callbacks_capacity) {
		void *new_ptr = realloc(ios_init_callbacks, sizeof(cb) * 32);
		if (new_ptr) {
			ios_init_callbacks = (init_callback *)(new_ptr);
			ios_init_callbacks_capacity += 32;
		}
	}
	if (ios_init_callbacks_capacity > ios_init_callbacks_count) {
		ios_init_callbacks[ios_init_callbacks_count] = cb;
		++ios_init_callbacks_count;
	}
}

OSIPhone *OSIPhone::get_singleton() {
	return (OSIPhone *)OS::get_singleton();
}

Error OSIPhone::initialize(const VideoMode &p_desired, int p_video_driver, int p_audio_driver) {
	Error result = OS_UIKit::initialize(p_desired, p_video_driver, p_audio_driver);

	if (result != OK) {
		return result;
	}

	ios = memnew(iOS);
	Engine::get_singleton()->add_singleton(Engine::Singleton("iOS", ios));

	return OK;
}

void OSIPhone::finalize() {
	if (ios) {
		memdelete(ios);
	}

	OS_UIKit::finalize();
}

void OSIPhone::alert(const String &p_alert, const String &p_title) {
	const CharString utf8_alert = p_alert.utf8();
	const CharString utf8_title = p_title.utf8();
	iOS::alert(utf8_alert.get_data(), utf8_title.get_data());
}

bool OSIPhone::has_virtual_keyboard() const {
	return true;
};

void OSIPhone::show_virtual_keyboard(const String &p_existing_text, const Rect2 &p_screen_rect, bool p_multiline, int p_max_input_length, int p_cursor_start, int p_cursor_end) {
	NSString *existingString = [[NSString alloc] initWithUTF8String:p_existing_text.utf8().get_data()];

	[AppDelegate.viewController.keyboardView
			becomeFirstResponderWithString:existingString
								 multiline:p_multiline
							   cursorStart:p_cursor_start
								 cursorEnd:p_cursor_end];
};

void OSIPhone::hide_virtual_keyboard() {
	[AppDelegate.viewController.keyboardView resignFirstResponder];
}

void OSIPhone::set_virtual_keyboard_height(int p_height) {
	virtual_keyboard_height = p_height * [UIScreen mainScreen].nativeScale;
}

int OSIPhone::get_virtual_keyboard_height() const {
	return virtual_keyboard_height;
}

String OSIPhone::get_name() const {
	return "iOS";
}

void OSIPhone::set_clipboard(const String &p_text) {
	[UIPasteboard generalPasteboard].string = [NSString stringWithUTF8String:p_text.utf8()];
}

String OSIPhone::get_clipboard() const {
	NSString *text = [UIPasteboard generalPasteboard].string;

	return String::utf8([text UTF8String]);
}

String OSIPhone::get_model_name() const {
	String model = ios->get_model();
	if (model != "") {
		return model;
	}

	return OS_Unix::get_model_name();
}

int OSIPhone::get_screen_dpi(int p_screen) const {
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

float OSIPhone::get_screen_refresh_rate(int p_screen) const {
	return [UIScreen mainScreen].maximumFramesPerSecond;
}

Rect2 OSIPhone::get_window_safe_area() const {
	if (@available(iOS 11, *)) {
		UIEdgeInsets insets = UIEdgeInsetsZero;
		UIView *view = AppDelegate.viewController.godotView;

		if ([view respondsToSelector:@selector(safeAreaInsets)]) {
			insets = [view safeAreaInsets];
		}

		float scale = [UIScreen mainScreen].nativeScale;
		Size2i insets_position = Size2i(insets.left, insets.top) * scale;
		Size2i insets_size = Size2i(insets.left + insets.right, insets.top + insets.bottom) * scale;

		return Rect2i(insets_position, get_window_size() - insets_size);
	} else {
		return Rect2i(Size2i(0, 0), get_window_size());
	}
}

bool OSIPhone::has_touchscreen_ui_hint() const {
	return true;
}

Error OSIPhone::native_video_play(String p_path, float p_volume, String p_audio_track, String p_subtitle_track) {
	FileAccess *f = FileAccess::open(p_path, FileAccess::READ);
	bool exists = f && f->is_open();

	String user_data_dir = OSIPhone::get_singleton()->get_user_data_dir();

	if (!exists) {
		return FAILED;
	}

	String tempFile = OSIPhone::get_singleton()->get_user_data_dir();

	if (p_path.begins_with("res://")) {
		if (PackedData::get_singleton()->has_path(p_path)) {
			printf("Unable to play %s using the native player as it resides in a .pck file\n", p_path.utf8().get_data());
			return ERR_INVALID_PARAMETER;
		} else {
			p_path = p_path.replace("res:/", ProjectSettings::get_singleton()->get_resource_path());
		}
	} else if (p_path.begins_with("user://")) {
		p_path = p_path.replace("user:/", user_data_dir);
	}

	memdelete(f);

	printf("Playing video: %s\n", p_path.utf8().get_data());

	String file_path = ProjectSettings::get_singleton()->globalize_path(p_path);

	NSString *filePath = [[NSString alloc] initWithUTF8String:file_path.utf8().get_data()];
	NSString *audioTrack = [NSString stringWithUTF8String:p_audio_track.utf8()];
	NSString *subtitleTrack = [NSString stringWithUTF8String:p_subtitle_track.utf8()];

	if (![AppDelegate.viewController playVideoAtPath:filePath
											  volume:p_volume
											   audio:audioTrack
											subtitle:subtitleTrack]) {
		return OK;
	}

	return FAILED;
}

bool OSIPhone::native_video_is_playing() const {
	return [AppDelegate.viewController.videoView isVideoPlaying];
}

void OSIPhone::native_video_pause() {
	if (native_video_is_playing()) {
		[AppDelegate.viewController.videoView pauseVideo];
	}
}

void OSIPhone::native_video_unpause() {
	[AppDelegate.viewController.videoView unpauseVideo];
}

void OSIPhone::native_video_focus_out() {
	[AppDelegate.viewController.videoView unfocusVideo];
}

void OSIPhone::native_video_stop() {
	if (native_video_is_playing()) {
		[AppDelegate.viewController.videoView stopVideo];
	}
}

String OSIPhone::get_processor_name() const {
	char buffer[256];
	size_t buffer_len = 256;
	if (sysctlbyname("machdep.cpu.brand_string", &buffer, &buffer_len, NULL, 0) == 0) {
		return String::utf8(buffer, buffer_len);
	}
	ERR_FAIL_V_MSG("", String("Couldn't get the CPU model name. Returning an empty string."));
}

void OSIPhone::vibrate_handheld(int p_duration_ms) {
	// iOS does not support duration for vibration
	AudioServicesPlaySystemSound(kSystemSoundID_Vibrate);
}

bool OSIPhone::_check_internal_feature_support(const String &p_feature) {
	return p_feature == "mobile";
}

void register_dynamic_symbol(char *name, void *address) {
	OSIPhone::dynamic_symbol_lookup_table[String(name)] = address;
}

Error OSIPhone::get_dynamic_library_symbol_handle(void *p_library_handle, const String p_name, void *&p_symbol_handle, bool p_optional) {
	if (p_library_handle == RTLD_SELF) {
		void **ptr = OSIPhone::dynamic_symbol_lookup_table.getptr(p_name);
		if (ptr) {
			p_symbol_handle = *ptr;
			return OK;
		}
	}
	return OS_Unix::get_dynamic_library_symbol_handle(p_library_handle, p_name, p_symbol_handle, p_optional);
}

OSIPhone::OSIPhone(String p_data_dir, String p_cache_dir) :
		OS_UIKit(p_data_dir, p_cache_dir) {
	for (int i = 0; i < ios_init_callbacks_count; ++i) {
		ios_init_callbacks[i]();
	}
	free(ios_init_callbacks);
	ios_init_callbacks = NULL;
	ios_init_callbacks_count = 0;
	ios_init_callbacks_capacity = 0;
};

OSIPhone::~OSIPhone() {
}

void OSIPhone::on_focus_out() {
	if (is_focused) {
		is_focused = false;

		if (get_main_loop()) {
			get_main_loop()->notification(MainLoop::NOTIFICATION_WM_FOCUS_OUT);
		}

		[AppDelegate.viewController.godotView stopRendering];

		if (native_video_is_playing()) {
			native_video_focus_out();
		}

		audio_driver.stop();
	}
}

void OSIPhone::on_focus_in() {
	if (!is_focused) {
		is_focused = true;

		if (get_main_loop()) {
			get_main_loop()->notification(MainLoop::NOTIFICATION_WM_FOCUS_IN);
		}

		[AppDelegate.viewController.godotView startRendering];

		if (native_video_is_playing()) {
			native_video_unpause();
		}

		audio_driver.start();
	}
}
