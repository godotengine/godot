/*************************************************************************/
/*  os_ios.mm                                                            */
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

#ifdef IOS_ENABLED

#include "os_ios.h"

#import "app_delegate.h"
#include "core/config/project_settings.h"
#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/io/file_access_pack.h"
#include "display_server_ios.h"
#include "drivers/unix/syslog_logger.h"
#import "godot_view.h"
#include "main/main.h"
#import "view_controller.h"

#import <AudioToolbox/AudioServices.h>
#import <CoreText/CoreText.h>
#import <UIKit/UIKit.h>
#import <dlfcn.h>
#include <sys/sysctl.h>

#if defined(VULKAN_ENABLED)
#include "servers/rendering/renderer_rd/renderer_compositor_rd.h"
#import <QuartzCore/CAMetalLayer.h>
#ifdef USE_VOLK
#include <volk.h>
#else
#include <vulkan/vulkan.h>
#endif
#endif

// Initialization order between compilation units is not guaranteed,
// so we use this as a hack to ensure certain code is called before
// everything else, but after all units are initialized.
typedef void (*init_callback)();
static init_callback *ios_init_callbacks = nullptr;
static int ios_init_callbacks_count = 0;
static int ios_init_callbacks_capacity = 0;
HashMap<String, void *> OS_IOS::dynamic_symbol_lookup_table;

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
	OS_IOS::dynamic_symbol_lookup_table[String(name)] = address;
}

OS_IOS *OS_IOS::get_singleton() {
	return (OS_IOS *)OS::get_singleton();
}

OS_IOS::OS_IOS(String p_data_dir, String p_cache_dir) {
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

	DisplayServerIOS::register_ios_driver();
}

OS_IOS::~OS_IOS() {}

void OS_IOS::alert(const String &p_alert, const String &p_title) {
	const CharString utf8_alert = p_alert.utf8();
	const CharString utf8_title = p_title.utf8();
	iOS::alert(utf8_alert.get_data(), utf8_title.get_data());
}

void OS_IOS::initialize_core() {
	OS_Unix::initialize_core();

	set_user_data_dir(user_data_dir);
}

void OS_IOS::initialize() {
	initialize_core();
}

void OS_IOS::initialize_modules() {
	ios = memnew(iOS);
	Engine::get_singleton()->add_singleton(Engine::Singleton("iOS", ios));

	joypad_ios = memnew(JoypadIOS);
}

void OS_IOS::deinitialize_modules() {
	if (joypad_ios) {
		memdelete(joypad_ios);
	}

	if (ios) {
		memdelete(ios);
	}
}

void OS_IOS::set_main_loop(MainLoop *p_main_loop) {
	main_loop = p_main_loop;

	if (main_loop) {
		main_loop->initialize();
	}
}

MainLoop *OS_IOS::get_main_loop() const {
	return main_loop;
}

void OS_IOS::delete_main_loop() {
	if (main_loop) {
		main_loop->finalize();
		memdelete(main_loop);
	}

	main_loop = nullptr;
}

bool OS_IOS::iterate() {
	if (!main_loop) {
		return true;
	}

	if (DisplayServer::get_singleton()) {
		DisplayServer::get_singleton()->process_events();
	}

	return Main::iteration();
}

void OS_IOS::start() {
	Main::start();

	if (joypad_ios) {
		joypad_ios->start_processing();
	}
}

void OS_IOS::finalize() {
	deinitialize_modules();

	// Already gets called
	//delete_main_loop();
}

// MARK: Dynamic Libraries

Error OS_IOS::open_dynamic_library(const String p_path, void *&p_library_handle, bool p_also_set_library_path, String *r_resolved_path) {
	if (p_path.length() == 0) {
		p_library_handle = RTLD_SELF;

		if (r_resolved_path != nullptr) {
			*r_resolved_path = p_path;
		}

		return OK;
	}
	return OS_Unix::open_dynamic_library(p_path, p_library_handle, p_also_set_library_path, r_resolved_path);
}

Error OS_IOS::close_dynamic_library(void *p_library_handle) {
	if (p_library_handle == RTLD_SELF) {
		return OK;
	}
	return OS_Unix::close_dynamic_library(p_library_handle);
}

Error OS_IOS::get_dynamic_library_symbol_handle(void *p_library_handle, const String p_name, void *&p_symbol_handle, bool p_optional) {
	if (p_library_handle == RTLD_SELF) {
		void **ptr = OS_IOS::dynamic_symbol_lookup_table.getptr(p_name);
		if (ptr) {
			p_symbol_handle = *ptr;
			return OK;
		}
	}
	return OS_Unix::get_dynamic_library_symbol_handle(p_library_handle, p_name, p_symbol_handle, p_optional);
}

String OS_IOS::get_name() const {
	return "iOS";
}

String OS_IOS::get_model_name() const {
	String model = ios->get_model();
	if (model != "") {
		return model;
	}

	return OS_Unix::get_model_name();
}

Error OS_IOS::shell_open(String p_uri) {
	NSString *urlPath = [[NSString alloc] initWithUTF8String:p_uri.utf8().get_data()];
	NSURL *url = [NSURL URLWithString:urlPath];

	if (![[UIApplication sharedApplication] canOpenURL:url]) {
		return ERR_CANT_OPEN;
	}

	printf("opening url %s\n", p_uri.utf8().get_data());

	[[UIApplication sharedApplication] openURL:url options:@{} completionHandler:nil];

	return OK;
}

void OS_IOS::set_user_data_dir(String p_dir) {
	Ref<DirAccess> da = DirAccess::open(p_dir);
	user_data_dir = da->get_current_dir();
	printf("setting data dir to %s from %s\n", user_data_dir.utf8().get_data(), p_dir.utf8().get_data());
}

String OS_IOS::get_user_data_dir() const {
	return user_data_dir;
}

String OS_IOS::get_cache_path() const {
	return cache_dir;
}

String OS_IOS::get_locale() const {
	NSString *preferedLanguage = [NSLocale preferredLanguages].firstObject;

	if (preferedLanguage) {
		return String::utf8([preferedLanguage UTF8String]).replace("-", "_");
	}

	NSString *localeIdentifier = [[NSLocale currentLocale] localeIdentifier];
	return String::utf8([localeIdentifier UTF8String]).replace("-", "_");
}

String OS_IOS::get_unique_id() const {
	NSString *uuid = [UIDevice currentDevice].identifierForVendor.UUIDString;
	return String::utf8([uuid UTF8String]);
}

String OS_IOS::get_processor_name() const {
	char buffer[256];
	size_t buffer_len = 256;
	if (sysctlbyname("machdep.cpu.brand_string", &buffer, &buffer_len, NULL, 0) == 0) {
		return String::utf8(buffer, buffer_len);
	}
	ERR_FAIL_V_MSG("", String("Couldn't get the CPU model name. Returning an empty string."));
}

Vector<String> OS_IOS::get_system_fonts() const {
	HashSet<String> font_names;
	CFArrayRef fonts = CTFontManagerCopyAvailableFontFamilyNames();
	if (fonts) {
		for (CFIndex i = 0; i < CFArrayGetCount(fonts); i++) {
			CFStringRef cf_name = (CFStringRef)CFArrayGetValueAtIndex(fonts, i);
			if (cf_name && (CFStringGetLength(cf_name) > 0) && (CFStringCompare(cf_name, CFSTR("LastResort"), kCFCompareCaseInsensitive) != kCFCompareEqualTo) && (CFStringGetCharacterAtIndex(cf_name, 0) != '.')) {
				NSString *ns_name = (__bridge NSString *)cf_name;
				font_names.insert(String::utf8([ns_name UTF8String]));
			}
		}
		CFRelease(fonts);
	}

	Vector<String> ret;
	for (const String &E : font_names) {
		ret.push_back(E);
	}
	return ret;
}

String OS_IOS::get_system_font_path(const String &p_font_name, bool p_bold, bool p_italic) const {
	String ret;

	String font_name = p_font_name;
	if (font_name.to_lower() == "sans-serif") {
		font_name = "Helvetica";
	} else if (font_name.to_lower() == "serif") {
		font_name = "Times";
	} else if (font_name.to_lower() == "monospace") {
		font_name = "Courier";
	} else if (font_name.to_lower() == "fantasy") {
		font_name = "Papyrus";
	} else if (font_name.to_lower() == "cursive") {
		font_name = "Apple Chancery";
	};

	CFStringRef name = CFStringCreateWithCString(kCFAllocatorDefault, font_name.utf8().get_data(), kCFStringEncodingUTF8);

	CTFontSymbolicTraits traits = 0;
	if (p_bold) {
		traits |= kCTFontBoldTrait;
	}
	if (p_italic) {
		traits |= kCTFontItalicTrait;
	}

	CFNumberRef sym_traits = CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt32Type, &traits);
	CFMutableDictionaryRef traits_dict = CFDictionaryCreateMutable(kCFAllocatorDefault, 0, nullptr, nullptr);
	CFDictionaryAddValue(traits_dict, kCTFontSymbolicTrait, sym_traits);

	CFMutableDictionaryRef attributes = CFDictionaryCreateMutable(kCFAllocatorDefault, 0, nullptr, nullptr);
	CFDictionaryAddValue(attributes, kCTFontFamilyNameAttribute, name);
	CFDictionaryAddValue(attributes, kCTFontTraitsAttribute, traits_dict);

	CTFontDescriptorRef font = CTFontDescriptorCreateWithAttributes(attributes);
	if (font) {
		CFURLRef url = (CFURLRef)CTFontDescriptorCopyAttribute(font, kCTFontURLAttribute);
		if (url) {
			NSString *font_path = [NSString stringWithString:[(__bridge NSURL *)url path]];
			ret = String::utf8([font_path UTF8String]);
			CFRelease(url);
		}
		CFRelease(font);
	}

	CFRelease(attributes);
	CFRelease(traits_dict);
	CFRelease(sym_traits);
	CFRelease(name);

	return ret;
}

void OS_IOS::vibrate_handheld(int p_duration_ms) {
	if (ios->supports_haptic_engine()) {
		ios->vibrate_haptic_engine((float)p_duration_ms / 1000.f);
	} else {
		// iOS <13 does not support duration for vibration
		AudioServicesPlaySystemSound(kSystemSoundID_Vibrate);
	}
}

bool OS_IOS::_check_internal_feature_support(const String &p_feature) {
	return p_feature == "mobile";
}

void OS_IOS::on_focus_out() {
	if (is_focused) {
		is_focused = false;

		if (DisplayServerIOS::get_singleton()) {
			DisplayServerIOS::get_singleton()->send_window_event(DisplayServer::WINDOW_EVENT_FOCUS_OUT);
		}

		[AppDelegate.viewController.godotView stopRendering];

		audio_driver.stop();
	}
}

void OS_IOS::on_focus_in() {
	if (!is_focused) {
		is_focused = true;

		if (DisplayServerIOS::get_singleton()) {
			DisplayServerIOS::get_singleton()->send_window_event(DisplayServer::WINDOW_EVENT_FOCUS_IN);
		}

		[AppDelegate.viewController.godotView startRendering];

		audio_driver.start();
	}
}

#endif // IOS_ENABLED
