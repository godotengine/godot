/*************************************************************************/
/*  os_tvos.mm                                                           */
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

#include "os_tvos.h"

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
#import "godot_view.h"
#import "godot_view_controller.h"
#import "keyboard_input_view.h"

#import <UIKit/UIKit.h>
#include <dlfcn.h>
#import <sys/utsname.h>

// Initialization order between compilation units is not guaranteed,
// so we use this as a hack to ensure certain code is called before
// everything else, but after all units are initialized.
typedef void (*init_callback)();
static init_callback *tvos_init_callbacks = NULL;
static int tvos_init_callbacks_count = 0;
static int tvos_init_callbacks_capacity = 0;
HashMap<String, void *> OSAppleTV::dynamic_symbol_lookup_table;

void add_tvos_init_callback(init_callback cb) {
	if (tvos_init_callbacks_count == tvos_init_callbacks_capacity) {
		void *new_ptr = realloc(tvos_init_callbacks, sizeof(cb) * 32);
		if (new_ptr) {
			tvos_init_callbacks = (init_callback *)(new_ptr);
			tvos_init_callbacks_capacity += 32;
		}
	}
	if (tvos_init_callbacks_capacity > tvos_init_callbacks_count) {
		tvos_init_callbacks[tvos_init_callbacks_count] = cb;
		++tvos_init_callbacks_count;
	}
}

OSAppleTV *OSAppleTV::get_singleton() {
	return (OSAppleTV *)OS::get_singleton();
}

Error OSAppleTV::initialize(const VideoMode &p_desired, int p_video_driver, int p_audio_driver) {
	Error result = OS_UIKit::initialize(p_desired, p_video_driver, p_audio_driver);

	if (result != OK) {
		return result;
	}

	tvos = memnew(tvOS);
	Engine::get_singleton()->add_singleton(Engine::Singleton("tvOS", tvos));

	return OK;
}

void OSAppleTV::finalize() {
	if (tvos) {
		memdelete(tvos);
	}

	OS_UIKit::finalize();
}

void OSAppleTV::alert(const String &p_alert, const String &p_title) {
	const CharString utf8_alert = p_alert.utf8();
	const CharString utf8_title = p_title.utf8();
	tvOS::alert(utf8_alert.get_data(), utf8_title.get_data());
}

bool OSAppleTV::has_virtual_keyboard() const {
	return true;
};

void OSAppleTV::show_virtual_keyboard(const String &p_existing_text, const Rect2 &p_screen_rect, bool p_multiline, int p_max_input_length, int p_cursor_start, int p_cursor_end) {
	NSString *existingString = [[NSString alloc] initWithUTF8String:p_existing_text.utf8().get_data()];

	[AppDelegate.viewController.keyboardView
			becomeFirstResponderWithString:existingString
								 multiline:p_multiline
							   cursorStart:p_cursor_start
								 cursorEnd:p_cursor_end];
};

void OSAppleTV::hide_virtual_keyboard() {
	[AppDelegate.viewController.keyboardView resignFirstResponder];
}

int OSAppleTV::get_virtual_keyboard_height() const {
	return 0;
}

String OSAppleTV::get_name() const {
	return "tvOS";
}

String OSAppleTV::get_model_name() const {
	String model = tvos->get_model();
	if (model != "") {
		return model;
	}

	return OS_Unix::get_model_name();
}

int OSAppleTV::get_screen_dpi(int p_screen) const {
	return 96;
}

Rect2 OSAppleTV::get_window_safe_area() const {
	if (@available(tvOS 11, *)) {
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

bool OSAppleTV::has_touchscreen_ui_hint() const {
	return true;
}

bool OSAppleTV::_check_internal_feature_support(const String &p_feature) {
	return p_feature == "mobile";
}

void register_dynamic_symbol(char *name, void *address) {
	OSAppleTV::dynamic_symbol_lookup_table[String(name)] = address;
}

Error OSAppleTV::get_dynamic_library_symbol_handle(void *p_library_handle, const String p_name, void *&p_symbol_handle, bool p_optional) {
	if (p_library_handle == RTLD_SELF) {
		void **ptr = OSAppleTV::dynamic_symbol_lookup_table.getptr(p_name);
		if (ptr) {
			p_symbol_handle = *ptr;
			return OK;
		}
	}
	return OS_Unix::get_dynamic_library_symbol_handle(p_library_handle, p_name, p_symbol_handle, p_optional);
}

OSAppleTV::OSAppleTV(String p_data_dir, String p_cache_dir) :
		OS_UIKit(p_data_dir, p_cache_dir) {
	for (int i = 0; i < tvos_init_callbacks_count; ++i) {
		tvos_init_callbacks[i]();
	}
	free(tvos_init_callbacks);
	tvos_init_callbacks = NULL;
	tvos_init_callbacks_count = 0;
	tvos_init_callbacks_capacity = 0;
};

OSAppleTV::~OSAppleTV() {
}

void OSAppleTV::on_focus_out() {
	if (is_focused) {
		is_focused = false;

		if (get_main_loop()) {
			get_main_loop()->notification(MainLoop::NOTIFICATION_WM_FOCUS_OUT);
		}

		[AppDelegate.viewController.godotView stopRendering];

		audio_driver.stop();
	}
}

void OSAppleTV::on_focus_in() {
	if (!is_focused) {
		is_focused = true;

		if (get_main_loop()) {
			get_main_loop()->notification(MainLoop::NOTIFICATION_WM_FOCUS_IN);
		}

		[AppDelegate.viewController.godotView startRendering];

		audio_driver.start();
	}
}

bool OSAppleTV::get_overrides_menu_button() const {
	return overrides_menu_button;
}

void OSAppleTV::set_overrides_menu_button(bool p_flag) {
	overrides_menu_button = p_flag;
}
