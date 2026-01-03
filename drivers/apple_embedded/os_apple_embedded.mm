/**************************************************************************/
/*  os_apple_embedded.mm                                                  */
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

#import "os_apple_embedded.h"

#ifdef APPLE_EMBEDDED_ENABLED

#import "display_server_apple_embedded.h"
#import "godot_app_delegate_service_apple_embedded.h"
#import "godot_view_apple_embedded.h"
#import "godot_view_controller.h"

#include "core/config/project_settings.h"
#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/os/main_loop.h"
#include "core/profiling/profiling.h"
#import "drivers/apple/os_log_logger.h"
#include "main/main.h"

#import <AVFoundation/AVFAudio.h>
#import <AudioToolbox/AudioServices.h>
#import <CoreText/CoreText.h>
#import <UIKit/UIKit.h>
#import <dlfcn.h>
#include <sys/sysctl.h>

#if defined(RD_ENABLED)
#include "servers/rendering/renderer_rd/renderer_compositor_rd.h"
#import <QuartzCore/CAMetalLayer.h>

#if defined(VULKAN_ENABLED)
#include "drivers/vulkan/godot_vulkan.h"
#endif // VULKAN_ENABLED
#endif

// Initialization order between compilation units is not guaranteed,
// so we use this as a hack to ensure certain code is called before
// everything else, but after all units are initialized.
typedef void (*init_callback)();
static init_callback *apple_init_callbacks = nullptr;
static int apple_embedded_platform_init_callbacks_count = 0;
static int apple_embedded_platform_init_callbacks_capacity = 0;
HashMap<String, void *> OS_AppleEmbedded::dynamic_symbol_lookup_table;

void add_apple_embedded_platform_init_callback(init_callback cb) {
	if (apple_embedded_platform_init_callbacks_count == apple_embedded_platform_init_callbacks_capacity) {
		void *new_ptr = realloc(apple_init_callbacks, sizeof(cb) * (apple_embedded_platform_init_callbacks_capacity + 32));
		if (new_ptr) {
			apple_init_callbacks = (init_callback *)(new_ptr);
			apple_embedded_platform_init_callbacks_capacity += 32;
		} else {
			ERR_FAIL_MSG("Unable to allocate memory for extension callbacks.");
		}
	}
	apple_init_callbacks[apple_embedded_platform_init_callbacks_count++] = cb;
}

void register_dynamic_symbol(char *name, void *address) {
	OS_AppleEmbedded::dynamic_symbol_lookup_table[String(name)] = address;
}

Rect2 fit_keep_aspect_centered(const Vector2 &p_container, const Vector2 &p_rect) {
	real_t available_ratio = p_container.width / p_container.height;
	real_t fit_ratio = p_rect.width / p_rect.height;
	Rect2 result;
	if (fit_ratio < available_ratio) {
		// Fit height - we'll have horizontal gaps
		result.size.height = p_container.height;
		result.size.width = p_container.height * fit_ratio;
		result.position.y = 0;
		result.position.x = (p_container.width - result.size.width) * 0.5f;
	} else {
		// Fit width - we'll have vertical gaps
		result.size.width = p_container.width;
		result.size.height = p_container.width / fit_ratio;
		result.position.x = 0;
		result.position.y = (p_container.height - result.size.height) * 0.5f;
	}
	return result;
}

Rect2 fit_keep_aspect_covered(const Vector2 &p_container, const Vector2 &p_rect) {
	real_t available_ratio = p_container.width / p_container.height;
	real_t fit_ratio = p_rect.width / p_rect.height;
	Rect2 result;
	if (fit_ratio < available_ratio) {
		// Need to scale up to fit width, and crop height
		result.size.width = p_container.width;
		result.size.height = p_container.width / fit_ratio;
		result.position.x = 0;
		result.position.y = (p_container.height - result.size.height) * 0.5f;
	} else {
		// Need to scale up to fit height, and crop width
		result.size.width = p_container.height * fit_ratio;
		result.size.height = p_container.height;
		result.position.x = (p_container.width - result.size.width) * 0.5f;
		result.position.y = 0;
	}
	return result;
}

OS_AppleEmbedded *OS_AppleEmbedded::get_singleton() {
	return (OS_AppleEmbedded *)OS::get_singleton();
}

OS_AppleEmbedded::OS_AppleEmbedded() {
	for (int i = 0; i < apple_embedded_platform_init_callbacks_count; ++i) {
		apple_init_callbacks[i]();
	}
	free(apple_init_callbacks);
	apple_init_callbacks = nullptr;
	apple_embedded_platform_init_callbacks_count = 0;
	apple_embedded_platform_init_callbacks_capacity = 0;

	main_loop = nullptr;

	Vector<Logger *> loggers;
	loggers.push_back(memnew(OsLogLogger(NSBundle.mainBundle.bundleIdentifier.UTF8String)));
	_set_logger(memnew(CompositeLogger(loggers)));

	AudioDriverManager::add_driver(&audio_driver);
}

OS_AppleEmbedded::~OS_AppleEmbedded() {}

void OS_AppleEmbedded::alert(const String &p_alert, const String &p_title) {
	const CharString utf8_alert = p_alert.utf8();
	const CharString utf8_title = p_title.utf8();
	AppleEmbedded::alert(utf8_alert.get_data(), utf8_title.get_data());
}

void OS_AppleEmbedded::initialize_core() {
	OS_Unix::initialize_core();
}

void OS_AppleEmbedded::initialize() {
	initialize_core();
}

void OS_AppleEmbedded::initialize_joypads() {
	joypad_apple = memnew(JoypadApple);
}

void OS_AppleEmbedded::initialize_modules() {
	apple_embedded = memnew(AppleEmbedded);
	Engine::get_singleton()->add_singleton(Engine::Singleton("AppleEmbedded", apple_embedded));
}

void OS_AppleEmbedded::deinitialize_modules() {
	if (joypad_apple) {
		memdelete(joypad_apple);
	}

	if (apple_embedded) {
		memdelete(apple_embedded);
	}
}

void OS_AppleEmbedded::set_main_loop(MainLoop *p_main_loop) {
	main_loop = p_main_loop;
}

MainLoop *OS_AppleEmbedded::get_main_loop() const {
	return main_loop;
}

void OS_AppleEmbedded::delete_main_loop() {
	if (main_loop) {
		main_loop->finalize();
		memdelete(main_loop);
	}

	main_loop = nullptr;
}

bool OS_AppleEmbedded::iterate() {
	if (!main_loop) {
		return true;
	}

	GodotProfileFrameMark;
	GodotProfileZone("OS_AppleEmbedded::iterate");

	if (DisplayServer::get_singleton()) {
		DisplayServer::get_singleton()->process_events();
	}

	joypad_apple->process_joypads();

	return Main::iteration();
}

void OS_AppleEmbedded::start() {
	if (Main::start() == EXIT_SUCCESS) {
		main_loop->initialize();
	}
}

void OS_AppleEmbedded::finalize() {
	deinitialize_modules();

	// Already gets called
	//delete_main_loop();
}

// MARK: Dynamic Libraries

_FORCE_INLINE_ String OS_AppleEmbedded::get_framework_executable(const String &p_path) {
	Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);

	// Read framework bundle to get executable name.
	NSURL *url = [NSURL fileURLWithPath:@(p_path.utf8().get_data())];
	NSBundle *bundle = [NSBundle bundleWithURL:url];
	if (bundle) {
		String exe_path = String::utf8([[bundle executablePath] UTF8String]);
		if (da->file_exists(exe_path)) {
			return exe_path;
		}
	}

	// Try default executable name (invalid framework).
	if (da->dir_exists(p_path) && da->file_exists(p_path.path_join(p_path.get_file().get_basename()))) {
		return p_path.path_join(p_path.get_file().get_basename());
	}

	// Not a framework, try loading as .dylib.
	return p_path;
}

Error OS_AppleEmbedded::open_dynamic_library(const String &p_path, void *&p_library_handle, GDExtensionData *p_data) {
	if (p_path.length() == 0) {
		// Static xcframework.
		p_library_handle = RTLD_SELF;

		if (p_data != nullptr && p_data->r_resolved_path != nullptr) {
			*p_data->r_resolved_path = p_path;
		}

		return OK;
	}

	String path = get_framework_executable(p_path);

	if (!FileAccess::exists(path)) {
		// Load .dylib or framework from within the executable path.
		path = get_framework_executable(get_executable_path().get_base_dir().path_join(p_path.get_file()));
	}

	if (!FileAccess::exists(path)) {
		// Load .dylib converted to framework from within the executable path.
		path = get_framework_executable(get_executable_path().get_base_dir().path_join(p_path.get_file().get_basename() + ".framework"));
	}

	if (!FileAccess::exists(path)) {
		// Load .dylib from within the executable path.
		path = get_framework_executable(get_executable_path().get_base_dir().path_join(p_path.get_file().get_basename() + ".dylib"));
	}

	if (!FileAccess::exists(path)) {
		// Load .dylib or framework from a standard iOS location.
		path = get_framework_executable(get_executable_path().get_base_dir().path_join("Frameworks").path_join(p_path.get_file()));
	}

	if (!FileAccess::exists(path)) {
		// Load .dylib converted to framework from a standard iOS location.
		path = get_framework_executable(get_executable_path().get_base_dir().path_join("Frameworks").path_join(p_path.get_file().get_basename() + ".framework"));
	}

	if (!FileAccess::exists(path)) {
		// Load .dylib from a standard iOS location.
		path = get_framework_executable(get_executable_path().get_base_dir().path_join("Frameworks").path_join(p_path.get_file().get_basename() + ".dylib"));
	}

	if (!FileAccess::exists(path) && (p_path.ends_with(".a") || p_path.ends_with(".xcframework"))) {
		path = String(); // Try loading static library.
	} else {
		ERR_FAIL_COND_V(!FileAccess::exists(path), ERR_FILE_NOT_FOUND);
	}
	p_library_handle = dlopen(path.utf8().get_data(), RTLD_NOW);
	ERR_FAIL_NULL_V_MSG(p_library_handle, ERR_CANT_OPEN, vformat("Can't open dynamic library: %s. Error: %s.", p_path, dlerror()));

	if (p_data != nullptr && p_data->r_resolved_path != nullptr) {
		*p_data->r_resolved_path = path;
	}

	return OK;
}

Error OS_AppleEmbedded::close_dynamic_library(void *p_library_handle) {
	if (p_library_handle == RTLD_SELF) {
		return OK;
	}
	return OS_Unix::close_dynamic_library(p_library_handle);
}

Error OS_AppleEmbedded::get_dynamic_library_symbol_handle(void *p_library_handle, const String &p_name, void *&p_symbol_handle, bool p_optional) {
	if (p_library_handle == RTLD_SELF) {
		void **ptr = OS_AppleEmbedded::dynamic_symbol_lookup_table.getptr(p_name);
		if (ptr) {
			p_symbol_handle = *ptr;
			return OK;
		}
	}
	return OS_Unix::get_dynamic_library_symbol_handle(p_library_handle, p_name, p_symbol_handle, p_optional);
}

String OS_AppleEmbedded::get_distribution_name() const {
	return get_name();
}

String OS_AppleEmbedded::get_version() const {
	NSOperatingSystemVersion ver = [NSProcessInfo processInfo].operatingSystemVersion;
	return vformat("%d.%d.%d", (int64_t)ver.majorVersion, (int64_t)ver.minorVersion, (int64_t)ver.patchVersion);
}

String OS_AppleEmbedded::get_model_name() const {
	String model = apple_embedded->get_model();
	if (model != "") {
		return model;
	}

	return OS_Unix::get_model_name();
}

Error OS_AppleEmbedded::shell_open(const String &p_uri) {
	NSString *urlPath = [[NSString alloc] initWithUTF8String:p_uri.utf8().get_data()];
	NSURL *url = [NSURL URLWithString:urlPath];

	if (![[UIApplication sharedApplication] canOpenURL:url]) {
		return ERR_CANT_OPEN;
	}

	print_verbose(vformat("Opening URL %s", p_uri));

	[[UIApplication sharedApplication] openURL:url options:@{} completionHandler:nil];

	return OK;
}

String OS_AppleEmbedded::get_user_data_dir(const String &p_user_dir) const {
	static String ret;
	if (ret.is_empty()) {
		NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
		if (paths && [paths count] >= 1) {
			ret.append_utf8([[paths firstObject] UTF8String]);
		}
	}
	return ret;
}

String OS_AppleEmbedded::get_cache_path() const {
	static String ret;
	if (ret.is_empty()) {
		NSArray *paths = NSSearchPathForDirectoriesInDomains(NSCachesDirectory, NSUserDomainMask, YES);
		if (paths && [paths count] >= 1) {
			ret.append_utf8([[paths firstObject] UTF8String]);
		}
	}
	return ret;
}

String OS_AppleEmbedded::get_temp_path() const {
	static String ret;
	if (ret.is_empty()) {
		NSURL *url = [NSURL fileURLWithPath:NSTemporaryDirectory()
								isDirectory:YES];
		if (url) {
			ret = String::utf8([url.path UTF8String]);
			ret = ret.trim_prefix("file://");
		}
	}
	return ret;
}

String OS_AppleEmbedded::get_resource_dir() const {
#ifdef TOOLS_ENABLED
	return OS_Unix::get_resource_dir();
#else
	if (remote_fs_dir.is_empty()) {
		return OS_Unix::get_resource_dir();
	} else {
		return remote_fs_dir;
	}
#endif
}

String OS_AppleEmbedded::get_bundle_resource_dir() const {
	NSString *str = [[[NSBundle mainBundle] infoDictionary] objectForKey:@"godot_path"];
	if (!str) {
		return OS_Unix::get_bundle_resource_dir();
	} else {
		return String::utf8([str cStringUsingEncoding:NSUTF8StringEncoding]);
	}
}

String OS_AppleEmbedded::get_locale() const {
	NSString *preferredLanguage = [NSLocale preferredLanguages].firstObject;

	if (preferredLanguage) {
		return String::utf8([preferredLanguage UTF8String]).replace_char('-', '_');
	}

	NSString *localeIdentifier = [[NSLocale currentLocale] localeIdentifier];
	return String::utf8([localeIdentifier UTF8String]).replace_char('-', '_');
}

String OS_AppleEmbedded::get_unique_id() const {
	NSString *uuid = [UIDevice currentDevice].identifierForVendor.UUIDString;
	return String::utf8([uuid UTF8String]);
}

struct _ModelInfo {
	Vector<String> model;
	String soc;
};

static const _ModelInfo _models[] = {
	{ { "iPhone1,1", "iPhone1,2", "iPod1,1" }, "Samsung S5L8900" },
	{ { "iPod2,1" }, "Samsung S5L8720" },
	{ { "iPhone2,1" }, "Samsung S5L8920" },
	{ { "iPod3,1" }, "Samsung S5L8922" },
	{ { "iPhone3,1", "iPhone3,2", "iPhone3,3", "iPad1,1", "iPad1,2", "iPod4,1", "AppleTV2,1" }, "Apple A4" },
	{ { "iPhone4,1", "iPad2,1", "iPad2,2", "iPad2,3", "iPad2,4", "iPad2,5", "iPad2,6", "iPad2,7", "iPod5,1", "AppleTV3,1", "AppleTV3,2" }, "Apple A5" },
	{ { "iPad3,1", "iPad3,2", "iPad3,3" }, "Apple A5X" },
	{ { "iPhone5,1", "iPhone5,2", "iPhone5,3", "iPhone5,4" }, "Apple A6" },
	{ { "iPad3,4", "iPad3,5", "iPad3,6" }, "Apple A6X" },
	{ { "iPhone6,1", "iPhone6,2", "iPad4,1", "iPad4,2", "iPad4,3", "iPad4,4", "iPad4,5", "iPad4,6", "iPad4,7", "iPad4,8", "iPad4,9" }, "Apple A7" },
	{ { "iPhone7,1", "iPhone7,2", "iPad5,1", "iPad5,2", "iPod7,1", "AppleTV5,3" }, "Apple A8" },
	{ { "iPad5,3", "iPad5,4" }, "Apple A8X" },
	{ { "iPhone8,1", "iPhone8,2", "iPhone8,4", "iPad6,11", "iPad6,12" }, "Apple A9" },
	{ { "iPad6,3", "iPad6,4", "iPad6,7", "iPad6,8" }, "Apple A9X" },
	{ { "iPhone9,1", "iPhone9,2", "iPhone9,3", "iPhone9,4", "iPad7,5", "iPad7,6", "iPad7,11", "iPad7,12", "iPod9,1" }, "Apple A10 Fusion" },
	{ { "iPad7,1", "iPad7,2", "iPad7,3", "iPad7,4", "AppleTV6,2" }, "Apple A10X Fusion" },
	{ { "iPhone10,1", "iPhone10,2", "iPhone10,3", "iPhone10,4", "iPhone10,5", "iPhone10,6" }, "Apple A11 Bionic" },
	{ { "iPhone11,2", "iPhone11,4", "iPhone11,6", "iPhone11,8", "iPad11,1", "iPad11,2", "iPad11,3", "iPad11,4", "iPad11,6", "iPad11,7", "AppleTV11,1" }, "Apple A12 Bionic" },
	{ { "iPad8,1", "iPad8,2", "iPad8,3", "iPad8,4", "iPad8,5", "iPad8,6", "iPad8,7", "iPad8,8" }, "Apple A12X Bionic" },
	{ { "iPad8,9", "iPad8,10", "iPad8,11", "iPad8,12" }, "Apple A12Z Bionic" },
	{ { "iPhone12,1", "iPhone12,3", "iPhone12,5", "iPhone12,8", "iPad12,1", "iPad12,2" }, "Apple A13 Bionic" },
	{ { "iPhone13,1", "iPhone13,2", "iPhone13,3", "iPhone13,4", "iPad13,1", "iPad13,2", "iPad13,18", "iPad13,19" }, "Apple A14 Bionic" },
	{ { "iPad13,4", "iPad13,5", "iPad13,6", "iPad13,7", "iPad13,8", "iPad13,9", "iPad13,10", "iPad13,11", "iPad13,16", "iPad13,17" }, "Apple M1" },
	{ { "iPhone14,2", "iPhone14,3", "iPhone14,4", "iPhone14,5", "iPhone14,6", "iPhone14,7", "iPhone14,8", "iPad14,1", "iPad14,2", "AppleTV14,1" }, "Apple A15 Bionic" },
	{ { "iPhone15,2", "iPhone15,3", "iPhone15,4", "iPhone15,5", "iPad15,7", "iPad15,8" }, "Apple A16 Bionic" },
	{ { "iPad14,3", "iPad14,4", "iPad14,5", "iPad14,6", "iPad14,8", "iPad14,9", "iPad14,10", "iPad14,11", "RealityDevice14,1" }, "Apple M2" },
	{ { "iPhone16,1", "iPhone16,2", "iPad16,1", "iPad16,2" }, "Apple A17 Pro" },
	{ { "iPad15,3", "iPad15,4", "iPad15,5", "iPad15,6" }, "Apple M3" },
	{ { "iPad16,3", "iPad16,4", "iPad16,5", "iPad16,6" }, "Apple M4" },
	{ { "iPad17,1", "iPad17,2", "iPad17,3", "iPad17,4", "RealityDevice17,1" }, "Apple M5" },
	{ { "iPhone17,3", "iPhone17,4", "iPhone17,5" }, "Apple A18" },
	{ { "iPhone17,1", "iPhone17,2" }, "Apple A18 Pro" },
	{ { "iPhone18,3" }, "Apple A19" },
	{ { "iPhone18,1", "iPhone18,2", "iPhone18,4" }, "Apple A19 Pro" },
};

String OS_AppleEmbedded::get_processor_name() const {
#if defined(IOS_SIMULATOR) || defined(VISIONOS_SIMULATOR)
	return "Simulator";
#else
	if (apple_embedded) {
		String model = apple_embedded->get_model();
		for (unsigned int i = 0; i < std::size(_models); i++) {
			for (const String &m : _models[i].model) {
				if (model.contains(m)) {
					return _models[i].soc;
				}
			}
		}
	}
#endif
	return OS::get_processor_name();
}

Vector<String> OS_AppleEmbedded::get_system_fonts() const {
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

String OS_AppleEmbedded::_get_default_fontname(const String &p_font_name) const {
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
	return font_name;
}

CGFloat OS_AppleEmbedded::_weight_to_ct(int p_weight) const {
	if (p_weight < 150) {
		return -0.80;
	} else if (p_weight < 250) {
		return -0.60;
	} else if (p_weight < 350) {
		return -0.40;
	} else if (p_weight < 450) {
		return 0.0;
	} else if (p_weight < 550) {
		return 0.23;
	} else if (p_weight < 650) {
		return 0.30;
	} else if (p_weight < 750) {
		return 0.40;
	} else if (p_weight < 850) {
		return 0.56;
	} else if (p_weight < 925) {
		return 0.62;
	} else {
		return 1.00;
	}
}

CGFloat OS_AppleEmbedded::_stretch_to_ct(int p_stretch) const {
	if (p_stretch < 56) {
		return -0.5;
	} else if (p_stretch < 69) {
		return -0.37;
	} else if (p_stretch < 81) {
		return -0.25;
	} else if (p_stretch < 93) {
		return -0.13;
	} else if (p_stretch < 106) {
		return 0.0;
	} else if (p_stretch < 137) {
		return 0.13;
	} else if (p_stretch < 144) {
		return 0.25;
	} else if (p_stretch < 162) {
		return 0.37;
	} else {
		return 0.5;
	}
}

Vector<String> OS_AppleEmbedded::get_system_font_path_for_text(const String &p_font_name, const String &p_text, const String &p_locale, const String &p_script, int p_weight, int p_stretch, bool p_italic) const {
	Vector<String> ret;
	String font_name = _get_default_fontname(p_font_name);

	CFStringRef name = CFStringCreateWithCString(kCFAllocatorDefault, font_name.utf8().get_data(), kCFStringEncodingUTF8);
	CTFontSymbolicTraits traits = 0;
	if (p_weight >= 700) {
		traits |= kCTFontBoldTrait;
	}
	if (p_italic) {
		traits |= kCTFontItalicTrait;
	}
	if (p_stretch < 100) {
		traits |= kCTFontCondensedTrait;
	} else if (p_stretch > 100) {
		traits |= kCTFontExpandedTrait;
	}

	CFNumberRef sym_traits = CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt32Type, &traits);
	CFMutableDictionaryRef traits_dict = CFDictionaryCreateMutable(kCFAllocatorDefault, 0, nullptr, nullptr);
	CFDictionaryAddValue(traits_dict, kCTFontSymbolicTrait, sym_traits);

	CGFloat weight = _weight_to_ct(p_weight);
	CFNumberRef font_weight = CFNumberCreate(kCFAllocatorDefault, kCFNumberCGFloatType, &weight);
	CFDictionaryAddValue(traits_dict, kCTFontWeightTrait, font_weight);

	CGFloat stretch = _stretch_to_ct(p_stretch);
	CFNumberRef font_stretch = CFNumberCreate(kCFAllocatorDefault, kCFNumberCGFloatType, &stretch);
	CFDictionaryAddValue(traits_dict, kCTFontWidthTrait, font_stretch);

	CFMutableDictionaryRef attributes = CFDictionaryCreateMutable(kCFAllocatorDefault, 0, nullptr, nullptr);
	CFDictionaryAddValue(attributes, kCTFontFamilyNameAttribute, name);
	CFDictionaryAddValue(attributes, kCTFontTraitsAttribute, traits_dict);

	CTFontDescriptorRef font = CTFontDescriptorCreateWithAttributes(attributes);
	if (font) {
		CTFontRef family = CTFontCreateWithFontDescriptor(font, 0, nullptr);
		if (family) {
			CFStringRef string = CFStringCreateWithCString(kCFAllocatorDefault, p_text.utf8().get_data(), kCFStringEncodingUTF8);
			CFRange range = CFRangeMake(0, CFStringGetLength(string));
			CTFontRef fallback_family = CTFontCreateForString(family, string, range);
			if (fallback_family) {
				CTFontDescriptorRef fallback_font = CTFontCopyFontDescriptor(fallback_family);
				if (fallback_font) {
					CFURLRef url = (CFURLRef)CTFontDescriptorCopyAttribute(fallback_font, kCTFontURLAttribute);
					if (url) {
						NSString *font_path = [NSString stringWithString:[(__bridge NSURL *)url path]];
						ret.push_back(String::utf8([font_path UTF8String]));
						CFRelease(url);
					}
					CFRelease(fallback_font);
				}
				CFRelease(fallback_family);
			}
			CFRelease(string);
			CFRelease(family);
		}
		CFRelease(font);
	}

	CFRelease(attributes);
	CFRelease(traits_dict);
	CFRelease(sym_traits);
	CFRelease(font_stretch);
	CFRelease(font_weight);
	CFRelease(name);

	return ret;
}

String OS_AppleEmbedded::get_system_font_path(const String &p_font_name, int p_weight, int p_stretch, bool p_italic) const {
	String ret;
	String font_name = _get_default_fontname(p_font_name);

	CFStringRef name = CFStringCreateWithCString(kCFAllocatorDefault, font_name.utf8().get_data(), kCFStringEncodingUTF8);

	CTFontSymbolicTraits traits = 0;
	if (p_weight >= 700) {
		traits |= kCTFontBoldTrait;
	}
	if (p_italic) {
		traits |= kCTFontItalicTrait;
	}
	if (p_stretch < 100) {
		traits |= kCTFontCondensedTrait;
	} else if (p_stretch > 100) {
		traits |= kCTFontExpandedTrait;
	}

	CFNumberRef sym_traits = CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt32Type, &traits);
	CFMutableDictionaryRef traits_dict = CFDictionaryCreateMutable(kCFAllocatorDefault, 0, nullptr, nullptr);
	CFDictionaryAddValue(traits_dict, kCTFontSymbolicTrait, sym_traits);

	CGFloat weight = _weight_to_ct(p_weight);
	CFNumberRef font_weight = CFNumberCreate(kCFAllocatorDefault, kCFNumberCGFloatType, &weight);
	CFDictionaryAddValue(traits_dict, kCTFontWeightTrait, font_weight);

	CGFloat stretch = _stretch_to_ct(p_stretch);
	CFNumberRef font_stretch = CFNumberCreate(kCFAllocatorDefault, kCFNumberCGFloatType, &stretch);
	CFDictionaryAddValue(traits_dict, kCTFontWidthTrait, font_stretch);

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
	CFRelease(font_stretch);
	CFRelease(font_weight);
	CFRelease(name);

	return ret;
}

void OS_AppleEmbedded::vibrate_handheld(int p_duration_ms, float p_amplitude) {
	if (apple_embedded->supports_haptic_engine()) {
		if (p_amplitude > 0.0) {
			p_amplitude = CLAMP(p_amplitude, 0.0, 1.0);
		}

		apple_embedded->vibrate_haptic_engine((float)p_duration_ms / 1000.f, p_amplitude);
	} else {
		// iOS <13 does not support duration for vibration
		AudioServicesPlaySystemSound(kSystemSoundID_Vibrate);
	}
}

bool OS_AppleEmbedded::_check_internal_feature_support(const String &p_feature) {
	if (p_feature == "system_fonts") {
		return true;
	}
	if (p_feature == "mobile") {
		return true;
	}

	return false;
}

Error OS_AppleEmbedded::setup_remote_filesystem(const String &p_server_host, int p_port, const String &p_password, String &r_project_path) {
	r_project_path = OS::get_user_data_dir();
	Error err = OS_Unix::setup_remote_filesystem(p_server_host, p_port, p_password, r_project_path);
	if (err == OK) {
		remote_fs_dir = r_project_path;
	}
	return err;
}

void OS_AppleEmbedded::on_focus_out() {
	if (is_focused) {
		is_focused = false;

		if (DisplayServerAppleEmbedded::get_singleton()) {
			DisplayServerAppleEmbedded::get_singleton()->send_window_event(DisplayServer::WINDOW_EVENT_FOCUS_OUT);
		}

		if (OS::get_singleton()->get_main_loop()) {
			OS::get_singleton()->get_main_loop()->notification(MainLoop::NOTIFICATION_APPLICATION_FOCUS_OUT);
		}

		[GDTAppDelegateService.viewController.godotView stopRendering];

		audio_driver.stop();
	}
}

void OS_AppleEmbedded::on_focus_in() {
	if (!is_focused) {
		is_focused = true;

		if (DisplayServerAppleEmbedded::get_singleton()) {
			DisplayServerAppleEmbedded::get_singleton()->send_window_event(DisplayServer::WINDOW_EVENT_FOCUS_IN);
		}

		if (OS::get_singleton()->get_main_loop()) {
			OS::get_singleton()->get_main_loop()->notification(MainLoop::NOTIFICATION_APPLICATION_FOCUS_IN);
		}

		[GDTAppDelegateService.viewController.godotView startRendering];

		audio_driver.start();
	}
}

void OS_AppleEmbedded::on_enter_background() {
	// Do not check for is_focused, because on_focus_out will always be fired first by applicationWillResignActive.

	if (OS::get_singleton()->get_main_loop()) {
		OS::get_singleton()->get_main_loop()->notification(MainLoop::NOTIFICATION_APPLICATION_PAUSED);
	}

	on_focus_out();
}

void OS_AppleEmbedded::on_exit_background() {
	if (!is_focused) {
		on_focus_in();

		if (OS::get_singleton()->get_main_loop()) {
			OS::get_singleton()->get_main_loop()->notification(MainLoop::NOTIFICATION_APPLICATION_RESUMED);
		}
	}
}

Rect2 OS_AppleEmbedded::calculate_boot_screen_rect(const Size2 &p_window_size, const Size2 &p_imgrect_size) const {
	String scalemodestr = GLOBAL_GET("ios/launch_screen_image_mode");

	if (scalemodestr == "scaleAspectFit") {
		return fit_keep_aspect_centered(p_window_size, p_imgrect_size);
	} else if (scalemodestr == "scaleAspectFill") {
		return fit_keep_aspect_covered(p_window_size, p_imgrect_size);
	} else if (scalemodestr == "scaleToFill") {
		return Rect2(Point2(), p_window_size);
	} else if (scalemodestr == "center") {
		return OS_Unix::calculate_boot_screen_rect(p_window_size, p_imgrect_size);
	} else {
		WARN_PRINT(vformat("Boot screen scale mode mismatch between iOS and Godot: %s not supported", scalemodestr));
		return OS_Unix::calculate_boot_screen_rect(p_window_size, p_imgrect_size);
	}
}

bool OS_AppleEmbedded::request_permission(const String &p_name) {
	if (p_name == "appleembedded.permission.AUDIO_RECORD") {
		if (@available(iOS 17.0, *)) {
			AVAudioApplicationRecordPermission permission = [AVAudioApplication sharedInstance].recordPermission;
			if (permission == AVAudioApplicationRecordPermissionGranted) {
				// Permission already granted, you can start recording.
				return true;
			} else if (permission == AVAudioApplicationRecordPermissionDenied) {
				// Permission denied, or not yet granted.
				return false;
			} else {
				// Request the permission, but for now return false as documented.
				[AVAudioApplication requestRecordPermissionWithCompletionHandler:^(BOOL granted) {
					get_main_loop()->emit_signal(SNAME("on_request_permissions_result"), p_name, granted);
				}];
			}
		}
	}
	return false;
}

Vector<String> OS_AppleEmbedded::get_granted_permissions() const {
	Vector<String> ret;

	if (@available(iOS 17.0, *)) {
		if ([AVAudioApplication sharedInstance].recordPermission == AVAudioApplicationRecordPermissionGranted) {
			ret.push_back("appleembedded.permission.AUDIO_RECORD");
		}
	}
	return ret;
}
#endif // APPLE_EMBEDDED_ENABLED
