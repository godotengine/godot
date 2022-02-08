/*************************************************************************/
/*  uikit_os.mm                                                          */
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

#include "uikit_os.h"

#include "core/config/project_settings.h"
#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/io/file_access_pack.h"
#include "drivers/unix/syslog_logger.h"
#include "main/main.h"
#include "uikit_display_server.h"

#import <UIKit/UIKit.h>
#import <dlfcn.h>

#if defined(VULKAN_ENABLED)
#include "servers/rendering/renderer_rd/renderer_compositor_rd.h"
#import <QuartzCore/CAMetalLayer.h>
#ifdef USE_VOLK
#include <volk.h>
#else
#include <vulkan/vulkan.h>
#endif
#endif

OS_UIKit *OS_UIKit::get_singleton() {
	return (OS_UIKit *)OS::get_singleton();
}

OS_UIKit::OS_UIKit(String p_data_dir, String p_cache_dir) {
	main_loop = nullptr;

	// can't call set_data_dir from here, since it requires DirAccess
	// which is initialized in initialize_core
	user_data_dir = p_data_dir;
	cache_dir = p_cache_dir;

	Vector<Logger *> loggers;
	loggers.push_back(memnew(SyslogLogger));
#ifdef DEBUG_ENABLED
	// it seems iOS app's stdout/stderr is only obtainable if you launch it from
	// Xcode
	loggers.push_back(memnew(StdLogger));
#endif
	_set_logger(memnew(CompositeLogger(loggers)));

	AudioDriverManager::add_driver(&audio_driver);
}

OS_UIKit::~OS_UIKit() {}

void OS_UIKit::initialize_core() {
	OS_Unix::initialize_core();

	set_user_data_dir(user_data_dir);

	uikit_joypad = memnew(UIKitJoypad);
}

void OS_UIKit::initialize() {
	initialize_core();
}

void OS_UIKit::set_main_loop(MainLoop *p_main_loop) {
	main_loop = p_main_loop;

	if (main_loop) {
		main_loop->initialize();
	}
}

MainLoop *OS_UIKit::get_main_loop() const {
	return main_loop;
}

void OS_UIKit::delete_main_loop() {
	if (main_loop) {
		main_loop->finalize();
		memdelete(main_loop);
	};

	main_loop = nullptr;
}

bool OS_UIKit::iterate() {
	if (!main_loop) {
		return true;
	}

	if (DisplayServerUIKit::get_singleton()) {
		DisplayServerUIKit::get_singleton()->process_events();
	}

	return Main::iteration();
}

void OS_UIKit::start() {
	Main::start();

	if (uikit_joypad) {
		uikit_joypad->start_processing();
	}
}

void OS_UIKit::finalize() {
	if (uikit_joypad) {
		memdelete(uikit_joypad);
	}

	// Already gets called
	//    delete_main_loop();
}

// MARK: Dynamic Libraries

Error OS_UIKit::open_dynamic_library(const String p_path, void *&p_library_handle, bool p_also_set_library_path) {
	if (p_path.length() == 0) {
		p_library_handle = RTLD_SELF;
		return OK;
	}
	return OS_Unix::open_dynamic_library(p_path, p_library_handle, p_also_set_library_path);
}

Error OS_UIKit::close_dynamic_library(void *p_library_handle) {
	if (p_library_handle == RTLD_SELF) {
		return OK;
	}
	return OS_Unix::close_dynamic_library(p_library_handle);
}

String OS_UIKit::get_name() const {
	return "UIKit";
};

Error OS_UIKit::shell_open(String p_uri) {
	NSString *urlPath = [[NSString alloc] initWithUTF8String:p_uri.utf8().get_data()];
	NSURL *url = [NSURL URLWithString:urlPath];

	if (![[UIApplication sharedApplication] canOpenURL:url]) {
		return ERR_CANT_OPEN;
	}

	printf("opening url %s\n", p_uri.utf8().get_data());

	[[UIApplication sharedApplication] openURL:url options:@{} completionHandler:nil];

	return OK;
};

void OS_UIKit::set_user_data_dir(String p_dir) {
	DirAccessRef da = DirAccess::open(p_dir);
	user_data_dir = da->get_current_dir();
	printf("setting data dir to %s from %s\n", user_data_dir.utf8().get_data(), p_dir.utf8().get_data());
}

String OS_UIKit::get_user_data_dir() const {
	return user_data_dir;
}

String OS_UIKit::get_cache_path() const {
	return cache_dir;
}

String OS_UIKit::get_locale() const {
	NSString *preferedLanguage = [NSLocale preferredLanguages].firstObject;

	if (preferedLanguage) {
		return String::utf8([preferedLanguage UTF8String]).replace("-", "_");
	}

	NSString *localeIdentifier = [[NSLocale currentLocale] localeIdentifier];
	return String::utf8([localeIdentifier UTF8String]).replace("-", "_");
}

String OS_UIKit::get_unique_id() const {
	NSString *uuid = [UIDevice currentDevice].identifierForVendor.UUIDString;
	return String::utf8([uuid UTF8String]);
}

int OS_UIKit::joy_id_for_name(const String &p_name) {
	if (!uikit_joypad) {
		return -1;
	}

	return uikit_joypad->joy_id_for_name(p_name);
}
