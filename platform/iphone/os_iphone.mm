/*************************************************************************/
/*  os_iphone.mm                                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifdef IPHONE_ENABLED

#include "os_iphone.h"
#import "app_delegate.h"
#include "core/config/project_settings.h"
#include "core/io/file_access_pack.h"
#include "core/os/dir_access.h"
#include "core/os/file_access.h"
#include "display_server_iphone.h"
#include "drivers/unix/syslog_logger.h"
#import "godot_view.h"
#include "main/main.h"
#import "view_controller.h"

#import <AudioToolbox/AudioServices.h>
#import <UIKit/UIKit.h>
#import <dlfcn.h>

#if defined(VULKAN_ENABLED)
#include "servers/rendering/renderer_rd/renderer_compositor_rd.h"
#import <QuartzCore/CAMetalLayer.h>
#include <vulkan/vulkan_metal.h>
#endif

// Initialization order between compilation units is not guaranteed,
// so we use this as a hack to ensure certain code is called before
// everything else, but after all units are initialized.
typedef void (*init_callback)();
static init_callback *ios_init_callbacks = nullptr;
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

void register_dynamic_symbol(char *name, void *address) {
	OSIPhone::dynamic_symbol_lookup_table[String(name)] = address;
}

OSIPhone *OSIPhone::get_singleton() {
	return (OSIPhone *)OS::get_singleton();
}

OSIPhone::OSIPhone(String p_data_dir) {
	for (int i = 0; i < ios_init_callbacks_count; ++i) {
		ios_init_callbacks[i]();
	}
	free(ios_init_callbacks);
	ios_init_callbacks = nullptr;
	ios_init_callbacks_count = 0;
	ios_init_callbacks_capacity = 0;

	main_loop = nullptr;

	// can't call set_data_dir from here, since it requires DirAccess
	// which is initialized in initialize_core
	user_data_dir = p_data_dir;

	Vector<Logger *> loggers;
	loggers.push_back(memnew(SyslogLogger));
#ifdef DEBUG_ENABLED
	// it seems iOS app's stdout/stderr is only obtainable if you launch it from
	// Xcode
	loggers.push_back(memnew(StdLogger));
#endif
	_set_logger(memnew(CompositeLogger(loggers)));

	AudioDriverManager::add_driver(&audio_driver);

	DisplayServerIPhone::register_iphone_driver();
}

OSIPhone::~OSIPhone() {}

void OSIPhone::initialize_core() {
	OS_Unix::initialize_core();

	set_user_data_dir(user_data_dir);
}

void OSIPhone::initialize() {
	initialize_core();
}

void OSIPhone::initialize_modules() {
	ios = memnew(iOS);
	Engine::get_singleton()->add_singleton(Engine::Singleton("iOS", ios));

	joypad_iphone = memnew(JoypadIPhone);
}

void OSIPhone::deinitialize_modules() {
	if (joypad_iphone) {
		memdelete(joypad_iphone);
	}

	if (ios) {
		memdelete(ios);
	}

	godot_ios_plugins_deinitialize();
}

void OSIPhone::set_main_loop(MainLoop *p_main_loop) {
	main_loop = p_main_loop;

	if (main_loop) {
		main_loop->initialize();
	}
}

MainLoop *OSIPhone::get_main_loop() const {
	return main_loop;
}

void OSIPhone::delete_main_loop() {
	if (main_loop) {
		main_loop->finalize();
		memdelete(main_loop);
	};

	main_loop = nullptr;
}

bool OSIPhone::iterate() {
	if (!main_loop) {
		return true;
	}

	if (DisplayServer::get_singleton()) {
		DisplayServer::get_singleton()->process_events();
	}

	return Main::iteration();
}

void OSIPhone::start() {
	godot_ios_plugins_initialize();

	Main::start();

	if (joypad_iphone) {
		joypad_iphone->start_processing();
	}
}

void OSIPhone::finalize() {
	deinitialize_modules();

	// Already gets called
	//    delete_main_loop();
}

// MARK: Dynamic Libraries

Error OSIPhone::open_dynamic_library(const String p_path, void *&p_library_handle, bool p_also_set_library_path) {
	if (p_path.length() == 0) {
		p_library_handle = RTLD_SELF;
		return OK;
	}
	return OS_Unix::open_dynamic_library(p_path, p_library_handle, p_also_set_library_path);
}

Error OSIPhone::close_dynamic_library(void *p_library_handle) {
	if (p_library_handle == RTLD_SELF) {
		return OK;
	}
	return OS_Unix::close_dynamic_library(p_library_handle);
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

void OSIPhone::alert(const String &p_alert, const String &p_title) {
	const CharString utf8_alert = p_alert.utf8();
	const CharString utf8_title = p_title.utf8();
	iOS::alert(utf8_alert.get_data(), utf8_title.get_data());
}

String OSIPhone::get_name() const {
	return "iOS";
};

String OSIPhone::get_model_name() const {
	String model = ios->get_model();
	if (model != "")
		return model;

	return OS_Unix::get_model_name();
}

Error OSIPhone::shell_open(String p_uri) {
	NSString *urlPath = [[NSString alloc] initWithUTF8String:p_uri.utf8().get_data()];
	NSURL *url = [NSURL URLWithString:urlPath];

	if (![[UIApplication sharedApplication] canOpenURL:url]) {
		return ERR_CANT_OPEN;
	}

	printf("opening url %s\n", p_uri.utf8().get_data());

	[[UIApplication sharedApplication] openURL:url options:@{} completionHandler:nil];

	return OK;
};

void OSIPhone::set_user_data_dir(String p_dir) {
	DirAccess *da = DirAccess::open(p_dir);

	user_data_dir = da->get_current_dir();
	printf("setting data dir to %s from %s\n", user_data_dir.utf8().get_data(), p_dir.utf8().get_data());
	memdelete(da);
}

String OSIPhone::get_user_data_dir() const {
	return user_data_dir;
}

String OSIPhone::get_locale() const {
	NSString *preferedLanguage = [NSLocale preferredLanguages].firstObject;

	if (preferedLanguage) {
		return String::utf8([preferedLanguage UTF8String]).replace("-", "_");
	}

	NSString *localeIdentifier = [[NSLocale currentLocale] localeIdentifier];
	return String::utf8([localeIdentifier UTF8String]).replace("-", "_");
}

String OSIPhone::get_unique_id() const {
	NSString *uuid = [UIDevice currentDevice].identifierForVendor.UUIDString;
	return String::utf8([uuid UTF8String]);
}

void OSIPhone::vibrate_handheld(int p_duration_ms) {
	// iOS does not support duration for vibration
	AudioServicesPlaySystemSound(kSystemSoundID_Vibrate);
}

bool OSIPhone::_check_internal_feature_support(const String &p_feature) {
	return p_feature == "mobile";
}

void OSIPhone::on_focus_out() {
	if (is_focused) {
		is_focused = false;

		if (DisplayServerIPhone::get_singleton()) {
			DisplayServerIPhone::get_singleton()->send_window_event(DisplayServer::WINDOW_EVENT_FOCUS_OUT);
		}

		[AppDelegate.viewController.godotView stopRendering];

		audio_driver.stop();
	}
}

void OSIPhone::on_focus_in() {
	if (!is_focused) {
		is_focused = true;

		if (DisplayServerIPhone::get_singleton()) {
			DisplayServerIPhone::get_singleton()->send_window_event(DisplayServer::WINDOW_EVENT_FOCUS_IN);
		}

		[AppDelegate.viewController.godotView startRendering];

		audio_driver.start();
	}
}

#endif // IPHONE_ENABLED
