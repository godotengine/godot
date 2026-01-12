/**************************************************************************/
/*  os_macos.mm                                                           */
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

#import "os_macos.h"

#import "dir_access_macos.h"
#ifdef TOOLS_ENABLED
#import "display_server_embedded.h"
#endif
#import "display_server_macos.h"
#import "godot_application.h"
#import "godot_application_delegate.h"

#include "core/crypto/crypto_core.h"
#include "core/io/file_access.h"
#include "core/os/main_loop.h"
#include "core/profiling/profiling.h"
#include "core/version_generated.gen.h"
#include "drivers/apple/os_log_logger.h"
#include "main/main.h"

#ifdef SDL_ENABLED
#include "drivers/sdl/joypad_sdl.h"
#endif

#include <dlfcn.h>
#include <libproc.h>
#import <mach-o/dyld.h>
#include <os/log.h>
#include <sys/sysctl.h>

void OS_MacOS::add_frame_delay(bool p_can_draw, bool p_wake_for_events) {
	if (p_wake_for_events) {
		uint64_t delay = get_frame_delay(p_can_draw);
		if (delay == 0) {
			return;
		}
		if (wait_timer) {
			CFRunLoopTimerInvalidate(wait_timer);
			CFRelease(wait_timer);
		}
		wait_timer = CFRunLoopTimerCreateWithHandler(kCFAllocatorDefault, CFAbsoluteTimeGetCurrent() + (double(delay) / 1000000.0), 0, 0, 0,
				^(CFRunLoopTimerRef timer) {
					CFRunLoopTimerInvalidate(wait_timer);
					CFRelease(wait_timer);
					wait_timer = nil;
				});
		CFRunLoopAddTimer(CFRunLoopGetCurrent(), wait_timer, kCFRunLoopCommonModes);
		return;
	}
	OS_Unix::add_frame_delay(p_can_draw, p_wake_for_events);
}

void OS_MacOS::initialize() {
	crash_handler.initialize();

	initialize_core();
}

String OS_MacOS::get_model_name() const {
	char buffer[256];
	size_t buffer_len = 256;
	if (sysctlbyname("hw.model", &buffer, &buffer_len, nullptr, 0) == 0 && buffer_len != 0) {
		return String::utf8(buffer, buffer_len);
	}
	return OS_Unix::get_model_name();
}

String OS_MacOS::get_processor_name() const {
	char buffer[256];
	size_t buffer_len = 256;
	if (sysctlbyname("machdep.cpu.brand_string", &buffer, &buffer_len, nullptr, 0) == 0) {
		return String::utf8(buffer, buffer_len);
	}
	ERR_FAIL_V_MSG("", String("Couldn't get the CPU model name. Returning an empty string."));
}

bool OS_MacOS::is_sandboxed() const {
	return has_environment("APP_SANDBOX_CONTAINER_ID");
}

bool OS_MacOS::request_permission(const String &p_name) {
	if (@available(macOS 11.0, *)) {
		if (p_name == "macos.permission.RECORD_SCREEN") {
			if (CGPreflightScreenCaptureAccess()) {
				return true;
			} else {
				CGRequestScreenCaptureAccess();
				return false;
			}
		}
	} else {
		if (p_name == "macos.permission.RECORD_SCREEN") {
			return true;
		}
	}
	return false;
}

Vector<String> OS_MacOS::get_granted_permissions() const {
	Vector<String> ret;

	if (@available(macOS 11.0, *)) {
		if (CGPreflightScreenCaptureAccess()) {
			ret.push_back("macos.permission.RECORD_SCREEN");
		}
	} else {
		ret.push_back("macos.permission.RECORD_SCREEN");
	}

	if (is_sandboxed()) {
		NSArray *bookmarks = [[NSUserDefaults standardUserDefaults] arrayForKey:@"sec_bookmarks"];
		for (id bookmark in bookmarks) {
			NSError *error = nil;
			BOOL isStale = NO;
			NSURL *url = [NSURL URLByResolvingBookmarkData:bookmark options:NSURLBookmarkResolutionWithSecurityScope relativeToURL:nil bookmarkDataIsStale:&isStale error:&error];
			if (!error && !isStale) {
				String url_string = String::utf8([[url path] UTF8String]);
				ret.push_back(url_string);
			}
		}
	}

	return ret;
}

void OS_MacOS::revoke_granted_permissions() {
	if (is_sandboxed()) {
		[[NSUserDefaults standardUserDefaults] setObject:nil forKey:@"sec_bookmarks"];
	}
}

#if TOOLS_ENABLED

// Function to check if a debugger is attached to the current process
bool OS_MacOS::is_debugger_attached() {
	int mib[4];
	struct kinfo_proc info{};
	size_t size = sizeof(info);

	// Initialize the flags so that, if sysctl fails, info.kp_proc.p_flag will be 0.
	info.kp_proc.p_flag = 0;

	// Initialize mib, which tells sysctl the info we want, in this case we're looking for information
	// about a specific process ID.
	mib[0] = CTL_KERN;
	mib[1] = KERN_PROC;
	mib[2] = KERN_PROC_PID;
	mib[3] = getpid();

	if (sysctl(mib, sizeof(mib) / sizeof(*mib), &info, &size, nullptr, 0) != 0) {
		perror("sysctl");
		return false;
	}

	return (info.kp_proc.p_flag & P_TRACED) != 0;
}

void OS_MacOS::wait_for_debugger(uint32_t p_msec) {
	if (p_msec == 0) {
		return;
	}

	CFAbsoluteTime start = CFAbsoluteTimeGetCurrent();
	CFTimeInterval wait_time = p_msec / 1000.0;

	NSTimer *timer = [NSTimer timerWithTimeInterval:0.100
											repeats:YES
											  block:^(NSTimer *t) {
												  if (is_debugger_attached() || CFAbsoluteTimeGetCurrent() > start + wait_time) {
													  [NSApp stopModalWithCode:NSModalResponseContinue];
													  [t invalidate];
												  }
											  }];

	[[NSRunLoop mainRunLoop] addTimer:timer forMode:NSModalPanelRunLoopMode];

	pid_t pid = getpid();
	alert(vformat("Attach debugger to pid: %d", pid));

	print("continue...");
}

#endif

void OS_MacOS::initialize_core() {
	OS_Unix::initialize_core();

	DirAccess::make_default<DirAccessMacOS>(DirAccess::ACCESS_RESOURCES);
	DirAccess::make_default<DirAccessMacOS>(DirAccess::ACCESS_USERDATA);
	DirAccess::make_default<DirAccessMacOS>(DirAccess::ACCESS_FILESYSTEM);
}

void OS_MacOS::finalize() {
	if (is_sandboxed()) {
		NSArray *bookmarks = [[NSUserDefaults standardUserDefaults] arrayForKey:@"sec_bookmarks"];
		for (id bookmark in bookmarks) {
			NSError *error = nil;
			BOOL isStale = NO;
			NSURL *url = [NSURL URLByResolvingBookmarkData:bookmark options:NSURLBookmarkResolutionWithSecurityScope relativeToURL:nil bookmarkDataIsStale:&isStale error:&error];
			if (!error && !isStale) {
				[url stopAccessingSecurityScopedResource];
			}
		}
	}

#ifdef COREMIDI_ENABLED
	midi_driver.close();
#endif

	delete_main_loop();

#ifdef SDL_ENABLED
	if (joypad_sdl) {
		memdelete(joypad_sdl);
	}
#endif
}

void OS_MacOS::initialize_joypads() {
#ifdef SDL_ENABLED
	joypad_sdl = memnew(JoypadSDL());
	if (joypad_sdl->initialize() != OK) {
		ERR_PRINT("Couldn't initialize SDL joypad input driver.");
		memdelete(joypad_sdl);
		joypad_sdl = nullptr;
	}
#endif
}

void OS_MacOS::set_main_loop(MainLoop *p_main_loop) {
	main_loop = p_main_loop;
}

void OS_MacOS::delete_main_loop() {
	if (!main_loop) {
		return;
	}

	memdelete(main_loop);
	main_loop = nullptr;
}

void OS_MacOS::set_cmdline_platform_args(const List<String> &p_args) {
	launch_service_args = p_args;
}

List<String> OS_MacOS::get_cmdline_platform_args() const {
	return launch_service_args;
}

void OS_MacOS::load_shell_environment() const {
	static bool shell_env_loaded = false;
	if (unlikely(!shell_env_loaded)) {
		shell_env_loaded = true;
		if (OS::get_singleton()->has_environment("TERM") || OS::get_singleton()->has_environment("__GODOT_SHELL_ENV_SET")) {
			return; // Already started from terminal, or other the instance with the shell environment, do nothing.
		}
		String pipe;
		List<String> args;
		args.push_back("-c");
		args.push_back(". /etc/zshrc;. /etc/zprofile;. ~/.zshenv;. ~/.zshrc;. ~/.zprofile;env");
		Error err = OS::get_singleton()->execute("zsh", args, &pipe);
		if (err == OK) {
			Vector<String> env_vars = pipe.split("\n");
			for (const String &E : env_vars) {
				Vector<String> tags = E.split("=", 2);
				if (tags.size() != 2 || tags[0] == "SHELL" || tags[0] == "USER" || tags[0] == "COMMAND_MODE" || tags[0] == "TMPDIR" || tags[0] == "TERM_SESSION_ID" || tags[0] == "PWD" || tags[0] == "OLDPWD" || tags[0] == "SHLVL" || tags[0] == "HOME" || tags[0] == "DISPLAY" || tags[0] == "LOGNAME" || tags[0] == "TERM" || tags[0] == "COLORTERM" || tags[0] == "_" || tags[0].begins_with("__CF") || tags[0].begins_with("XPC_") || tags[0].begins_with("__GODOT")) {
					continue;
				}
				OS::get_singleton()->set_environment(tags[0], tags[1]);
			}
		}
		OS::get_singleton()->set_environment("__GODOT_SHELL_ENV_SET", "1");
	}
}

String OS_MacOS::get_name() const {
	return "macOS";
}

String OS_MacOS::get_distribution_name() const {
	return get_name();
}

String OS_MacOS::get_version() const {
	NSOperatingSystemVersion ver = [NSProcessInfo processInfo].operatingSystemVersion;
	return vformat("%d.%d.%d", (int64_t)ver.majorVersion, (int64_t)ver.minorVersion, (int64_t)ver.patchVersion);
}

String OS_MacOS::get_version_alias() const {
	NSOperatingSystemVersion ver = [NSProcessInfo processInfo].operatingSystemVersion;
	String macos_string;
	if (ver.majorVersion == 26) {
		macos_string += "Tahoe";
	} else if (ver.majorVersion == 15) {
		macos_string += "Sequoia";
	} else if (ver.majorVersion == 14) {
		macos_string += "Sonoma";
	} else if (ver.majorVersion == 13) {
		macos_string += "Ventura";
	} else if (ver.majorVersion == 12) {
		macos_string += "Monterey";
	} else if (ver.majorVersion == 11 || (ver.majorVersion == 10 && ver.minorVersion == 16)) {
		// Big Sur was 10.16 during beta, but it became 11 for the stable version.
		macos_string += "Big Sur";
	} else if (ver.majorVersion == 10 && ver.minorVersion == 15) {
		macos_string += "Catalina";
	} else if (ver.majorVersion == 10 && ver.minorVersion == 14) {
		macos_string += "Mojave";
	} else if (ver.majorVersion == 10 && ver.minorVersion == 13) {
		macos_string += "High Sierra";
	} else {
		macos_string += "Unknown";
	}
	// macOS versions older than 10.13 cannot run Godot.
	return vformat("%s (%s)", macos_string, get_version());
}

void OS_MacOS::alert(const String &p_alert, const String &p_title) {
	NSAlert *window = [[NSAlert alloc] init];
	NSString *ns_title = [NSString stringWithUTF8String:p_title.utf8().get_data()];
	NSString *ns_alert = [NSString stringWithUTF8String:p_alert.utf8().get_data()];

	NSTextField *text_field = [NSTextField labelWithString:ns_alert];
	[text_field setAlignment:NSTextAlignmentCenter];
	[window addButtonWithTitle:@"OK"];
	[window setMessageText:ns_title];
	[window setAccessoryView:text_field];
	[window setAlertStyle:NSAlertStyleWarning];

	id key_window = [[NSApplication sharedApplication] keyWindow];
	[window runModal];
	if (key_window) {
		[key_window makeKeyAndOrderFront:nil];
	}
}

_FORCE_INLINE_ String OS_MacOS::get_framework_executable(const String &p_path) {
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

Error OS_MacOS::open_dynamic_library(const String &p_path, void *&p_library_handle, GDExtensionData *p_data) {
	String path = get_framework_executable(p_path);

	if (!FileAccess::exists(path)) {
		// Load .dylib or framework from within the executable path.
		path = get_framework_executable(get_executable_path().get_base_dir().path_join(p_path.get_file()));
	}

	if (!FileAccess::exists(path)) {
		// Load .dylib or framework from a standard macOS location.
		path = get_framework_executable(get_executable_path().get_base_dir().path_join("../Frameworks").path_join(p_path.get_file()));
	}

	if (!FileAccess::exists(path)) {
		// Try using path as is. macOS system libraries with `/usr/lib/*` path do not exist as physical files and are loaded from shared cache.
		path = p_path;
	}

	p_library_handle = dlopen(path.utf8().get_data(), RTLD_NOW);
	ERR_FAIL_NULL_V_MSG(p_library_handle, ERR_CANT_OPEN, vformat("Can't open dynamic library: %s. Error: %s.", p_path, dlerror()));

	if (p_data != nullptr && p_data->r_resolved_path != nullptr) {
		*p_data->r_resolved_path = path;
	}

	return OK;
}

MainLoop *OS_MacOS::get_main_loop() const {
	return main_loop;
}

String OS_MacOS::get_config_path() const {
	if (has_environment("HOME")) {
		return get_environment("HOME").path_join("Library/Application Support");
	}
	return ".";
}

String OS_MacOS::get_data_path() const {
	return get_config_path();
}

String OS_MacOS::get_cache_path() const {
	if (has_environment("HOME")) {
		return get_environment("HOME").path_join("Library/Caches");
	}
	return get_config_path();
}

String OS_MacOS::get_temp_path() const {
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

String OS_MacOS::get_bundle_resource_dir() const {
	String ret;

	NSBundle *main = [NSBundle mainBundle];
	if (main) {
		NSString *resource_path = [main resourcePath];
		ret.append_utf8([resource_path UTF8String]);
	}
	return ret;
}

String OS_MacOS::get_bundle_icon_path() const {
	String ret;

	NSBundle *main = [NSBundle mainBundle];
	if (main) {
		NSString *icon_path = [[main infoDictionary] objectForKey:@"CFBundleIconFile"];
		if (icon_path) {
			ret.append_utf8([icon_path UTF8String]);
		}
	}
	return ret;
}

String OS_MacOS::get_bundle_icon_name() const {
	String ret;

	NSBundle *main = [NSBundle mainBundle];
	if (main) {
		NSString *icon_name = [[main infoDictionary] objectForKey:@"CFBundleIconName"];
		if (icon_name) {
			ret.append_utf8([icon_name UTF8String]);
		}
	}
	return ret;
}

// Get properly capitalized engine name for system paths
String OS_MacOS::get_godot_dir_name() const {
	return String(GODOT_VERSION_SHORT_NAME).capitalize();
}

String OS_MacOS::get_system_dir(SystemDir p_dir, bool p_shared_storage) const {
	NSSearchPathDirectory id;
	bool found = true;

	switch (p_dir) {
		case SYSTEM_DIR_DESKTOP: {
			id = NSDesktopDirectory;
		} break;
		case SYSTEM_DIR_DOCUMENTS: {
			id = NSDocumentDirectory;
		} break;
		case SYSTEM_DIR_DOWNLOADS: {
			id = NSDownloadsDirectory;
		} break;
		case SYSTEM_DIR_MOVIES: {
			id = NSMoviesDirectory;
		} break;
		case SYSTEM_DIR_MUSIC: {
			id = NSMusicDirectory;
		} break;
		case SYSTEM_DIR_PICTURES: {
			id = NSPicturesDirectory;
		} break;
		default: {
			found = false;
		}
	}

	String ret;
	if (found) {
		NSArray *paths = NSSearchPathForDirectoriesInDomains(id, NSUserDomainMask, YES);
		if (paths && [paths count] >= 1) {
			ret.append_utf8([[paths firstObject] UTF8String]);
		}
	}

	return ret;
}

Error OS_MacOS::shell_show_in_file_manager(String p_path, bool p_open_folder) {
	bool open_folder = false;
	if (DirAccess::dir_exists_absolute(p_path) && p_open_folder) {
		open_folder = true;
	}

	if (!p_path.begins_with("file://")) {
		p_path = String("file://") + p_path;
	}

	NSString *string = [NSString stringWithUTF8String:p_path.utf8().get_data()];
	NSURL *uri = [[NSURL alloc] initWithString:[string stringByAddingPercentEncodingWithAllowedCharacters:[NSCharacterSet URLFragmentAllowedCharacterSet]]];

	if (open_folder) {
		[[NSWorkspace sharedWorkspace] openURL:uri];
	} else {
		[[NSWorkspace sharedWorkspace] activateFileViewerSelectingURLs:@[ uri ]];
	}
	return OK;
}

Error OS_MacOS::shell_open(const String &p_uri) {
	NSString *string = [NSString stringWithUTF8String:p_uri.utf8().get_data()];
	NSURL *uri = [[NSURL alloc] initWithString:string];
	if (!uri || !uri.scheme || [uri.scheme isEqual:@"file"]) {
		// No scheme set, assume "file://" and escape special characters.
		if (!p_uri.begins_with("file://")) {
			string = [NSString stringWithUTF8String:("file://" + p_uri).utf8().get_data()];
		}
		uri = [[NSURL alloc] initWithString:[string stringByAddingPercentEncodingWithAllowedCharacters:[NSCharacterSet URLFragmentAllowedCharacterSet]]];
	}
	[[NSWorkspace sharedWorkspace] openURL:uri];
	return OK;
}

String OS_MacOS::get_locale() const {
	NSString *locale_code = [[NSLocale preferredLanguages] objectAtIndex:0];
	return String([locale_code UTF8String]).replace_char('-', '_');
}

Vector<String> OS_MacOS::get_system_fonts() const {
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

String OS_MacOS::_get_default_fontname(const String &p_font_name) const {
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

CGFloat OS_MacOS::_weight_to_ct(int p_weight) const {
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

CGFloat OS_MacOS::_stretch_to_ct(int p_stretch) const {
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

Vector<String> OS_MacOS::get_system_font_path_for_text(const String &p_font_name, const String &p_text, const String &p_locale, const String &p_script, int p_weight, int p_stretch, bool p_italic) const {
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

String OS_MacOS::get_system_font_path(const String &p_font_name, int p_weight, int p_stretch, bool p_italic) const {
	String ret;
	String font_name = _get_default_fontname(p_font_name);

	CFStringRef name = CFStringCreateWithCString(kCFAllocatorDefault, font_name.utf8().get_data(), kCFStringEncodingUTF8);

	CTFontSymbolicTraits traits = 0;
	if (p_weight > 700) {
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

String OS_MacOS::get_executable_path() const {
	char pathbuf[PROC_PIDPATHINFO_MAXSIZE];
	int pid = getpid();
	pid_t ret = proc_pidpath(pid, pathbuf, sizeof(pathbuf));
	if (ret <= 0) {
		return OS::get_executable_path();
	} else {
		return String::utf8(pathbuf);
	}
}

Error OS_MacOS::create_process(const String &p_path, const List<String> &p_arguments, ProcessID *r_child_id, bool p_open_console) {
	// Use NSWorkspace if path is an .app bundle.
	NSURL *url = [NSURL fileURLWithPath:@(p_path.utf8().get_data())];
	NSBundle *bundle = [NSBundle bundleWithURL:url];
	if (bundle) {
		NSMutableArray *arguments = [[NSMutableArray alloc] init];
		for (const String &arg : p_arguments) {
			[arguments addObject:[NSString stringWithUTF8String:arg.utf8().get_data()]];
		}
#if defined(__x86_64__)
		if (@available(macOS 10.15, *)) {
#endif
			NSWorkspaceOpenConfiguration *configuration = [[NSWorkspaceOpenConfiguration alloc] init];
			[configuration setArguments:arguments];
			[configuration setCreatesNewApplicationInstance:YES];
			__block dispatch_semaphore_t lock = dispatch_semaphore_create(0);
			__block Error err = ERR_TIMEOUT;
			__block pid_t pid = 0;

			[[NSWorkspace sharedWorkspace] openApplicationAtURL:url
												  configuration:configuration
											  completionHandler:^(NSRunningApplication *app, NSError *error) {
												  if (error) {
													  err = ERR_CANT_FORK;
													  NSLog(@"Failed to execute: %@", error.localizedDescription);
												  } else {
													  pid = [app processIdentifier];
													  err = OK;
												  }
												  dispatch_semaphore_signal(lock);
											  }];
			dispatch_semaphore_wait(lock, dispatch_time(DISPATCH_TIME_NOW, 20000000000)); // 20 sec timeout, wait for app to launch.

			if (err == OK) {
				if (r_child_id) {
					*r_child_id = (ProcessID)pid;
				}
			}

			return err;
#if defined(__x86_64__)
		} else {
			Error err = ERR_TIMEOUT;
			NSError *error = nullptr;
			NSRunningApplication *app = [[NSWorkspace sharedWorkspace] launchApplicationAtURL:url options:NSWorkspaceLaunchNewInstance configuration:[NSDictionary dictionaryWithObject:arguments forKey:NSWorkspaceLaunchConfigurationArguments] error:&error];
			if (error) {
				err = ERR_CANT_FORK;
				NSLog(@"Failed to execute: %@", error.localizedDescription);
			} else {
				if (r_child_id) {
					*r_child_id = (ProcessID)[app processIdentifier];
				}
				err = OK;
			}
			return err;
		}
#endif
	} else {
		return OS_Unix::create_process(p_path, p_arguments, r_child_id, p_open_console);
	}
}

Error OS_MacOS::create_instance(const List<String> &p_arguments, ProcessID *r_child_id) {
	// Do not run headless instance as app bundle, since it will never send `applicationDidFinishLaunching` and register as failed start after timeout.
	for (size_t i = 0; i < std::size(OS_MacOS::headless_args); i++) {
		if (p_arguments.find(String(OS_MacOS::headless_args[i]))) {
			return OS_Unix::create_process(get_executable_path(), p_arguments, r_child_id, false);
		}
	}

	// If executable is bundled, always execute editor instances as an app bundle to ensure app window is registered and activated correctly.
	NSString *nsappname = [[[NSBundle mainBundle] infoDictionary] objectForKey:@"CFBundleName"];
	if (nsappname != nil) {
		String path = String::utf8([[[NSBundle mainBundle] bundlePath] UTF8String]);
#ifdef TOOLS_ENABLED
		if (Engine::get_singleton() && !Engine::get_singleton()->is_project_manager_hint() && !Engine::get_singleton()->is_editor_hint()) {
			// Project started from the editor, inject "path" argument to set instance working directory.
			char cwd[PATH_MAX];
			if (::getcwd(cwd, sizeof(cwd)) != nullptr) {
				List<String> arguments = p_arguments;
				arguments.push_back("--path");
				arguments.push_back(String::utf8(cwd));
				return create_process(path, arguments, r_child_id, false);
			}
		}
#endif
		return create_process(path, p_arguments, r_child_id, false);
	} else {
		return create_process(get_executable_path(), p_arguments, r_child_id, false);
	}
}

Error OS_MacOS::open_with_program(const String &p_program_path, const List<String> &p_paths) {
	NSURL *app_url = [NSURL fileURLWithPath:@(p_program_path.utf8().get_data())];
	if (!app_url) {
		return ERR_INVALID_PARAMETER;
	}

	NSBundle *bundle = [NSBundle bundleWithURL:app_url];
	if (!bundle) {
		return OS_Unix::create_process(p_program_path, p_paths);
	}

	NSMutableArray *urls_to_open = [[NSMutableArray alloc] init];
	for (const String &path : p_paths) {
		NSURL *file_url = [NSURL fileURLWithPath:@(path.utf8().get_data())];
		if (file_url) {
			[urls_to_open addObject:file_url];
		}
	}

	if ([urls_to_open count] == 0) {
		return ERR_INVALID_PARAMETER;
	}

#if defined(__x86_64__)
	if (@available(macOS 10.15, *)) {
#endif
		NSWorkspaceOpenConfiguration *configuration = [[NSWorkspaceOpenConfiguration alloc] init];
		[configuration setCreatesNewApplicationInstance:NO];
		__block dispatch_semaphore_t lock = dispatch_semaphore_create(0);
		__block Error err = ERR_TIMEOUT;

		[[NSWorkspace sharedWorkspace] openURLs:urls_to_open
						   withApplicationAtURL:app_url
								  configuration:configuration
							  completionHandler:^(NSRunningApplication *app, NSError *error) {
								  if (error) {
									  err = ERR_CANT_FORK;
									  NSLog(@"Failed to open paths: %@", error.localizedDescription);
								  } else {
									  err = OK;
								  }
								  dispatch_semaphore_signal(lock);
							  }];
		dispatch_semaphore_wait(lock, dispatch_time(DISPATCH_TIME_NOW, 20000000000)); // 20 sec timeout, wait for app to launch.

		return err;
#if defined(__x86_64__)
	} else {
		NSError *error = nullptr;
		[[NSWorkspace sharedWorkspace] openURLs:urls_to_open withApplicationAtURL:app_url options:NSWorkspaceLaunchDefault configuration:@{} error:&error];
		if (error) {
			return ERR_CANT_FORK;
		}
		return OK;
	}
#endif
}

bool OS_MacOS::is_process_running(const ProcessID &p_pid) const {
	NSRunningApplication *app = [NSRunningApplication runningApplicationWithProcessIdentifier:(pid_t)p_pid];
	if (!app) {
		return OS_Unix::is_process_running(p_pid);
	}

	return ![app isTerminated];
}

String OS_MacOS::get_unique_id() const {
	static String serial_number;

	if (serial_number.is_empty()) {
		io_service_t platform_expert = IOServiceGetMatchingService(kIOMasterPortDefault, IOServiceMatching("IOPlatformExpertDevice"));
		CFStringRef serial_number_cf_string = nullptr;
		if (platform_expert) {
			serial_number_cf_string = (CFStringRef)IORegistryEntryCreateCFProperty(platform_expert, CFSTR(kIOPlatformSerialNumberKey), kCFAllocatorDefault, 0);
			IOObjectRelease(platform_expert);
		}

		NSString *serial_number_ns_string = nil;
		if (serial_number_cf_string) {
			serial_number_ns_string = [NSString stringWithString:(__bridge NSString *)serial_number_cf_string];
			CFRelease(serial_number_cf_string);
		}

		if (serial_number_ns_string) {
			serial_number.append_utf8([serial_number_ns_string UTF8String]);
		}
	}

	return serial_number;
}

bool OS_MacOS::_check_internal_feature_support(const String &p_feature) {
	if (p_feature == "system_fonts") {
		return true;
	}
	if (p_feature == "pc") {
		return true;
	}

	return false;
}

void OS_MacOS::disable_crash_handler() {
	crash_handler.disable();
}

bool OS_MacOS::is_disable_crash_handler() const {
	return crash_handler.is_disabled();
}

Error OS_MacOS::move_to_trash(const String &p_path) {
	NSFileManager *fm = [NSFileManager defaultManager];
	NSURL *url = [NSURL fileURLWithPath:@(p_path.utf8().get_data())];
	NSError *err;

	if (![fm trashItemAtURL:url resultingItemURL:nil error:&err]) {
		ERR_PRINT("trashItemAtURL error: " + String::utf8(err.localizedDescription.UTF8String));
		return FAILED;
	}

	return OK;
}

String OS_MacOS::get_system_ca_certificates() {
	CFArrayRef result;
	SecCertificateRef item;
	CFDataRef der;

	OSStatus ret = SecTrustCopyAnchorCertificates(&result);
	ERR_FAIL_COND_V(ret != noErr, "");

	CFIndex l = CFArrayGetCount(result);
	String certs;
	PackedByteArray pba;
	for (CFIndex i = 0; i < l; i++) {
		item = (SecCertificateRef)CFArrayGetValueAtIndex(result, i);
		der = SecCertificateCopyData(item);
		int derlen = CFDataGetLength(der);
		if (pba.size() < derlen * 3) {
			pba.resize(derlen * 3);
		}
		size_t b64len = 0;
		Error err = CryptoCore::b64_encode(pba.ptrw(), pba.size(), &b64len, (unsigned char *)CFDataGetBytePtr(der), derlen);
		CFRelease(der);
		ERR_CONTINUE(err != OK);
		// Certificate is bas64 encoded, aka ascii.
		certs += "-----BEGIN CERTIFICATE-----\n" + String::ascii(Span((char *)pba.ptr(), b64len)) + "\n-----END CERTIFICATE-----\n";
	}
	CFRelease(result);
	return certs;
}

OS::PreferredTextureFormat OS_MacOS::get_preferred_texture_format() const {
	// macOS supports both formats on ARM. Prefer S3TC/BPTC
	// for better compatibility with x86 platforms.
	return PREFERRED_TEXTURE_FORMAT_S3TC_BPTC;
}

OS_MacOS::OS_MacOS(const char *p_execpath, int p_argc, char **p_argv) {
	execpath = p_execpath;
	argc = p_argc;
	argv = p_argv;

	if (is_sandboxed()) {
		// Load security-scoped bookmarks, request access, remove stale or invalid bookmarks.
		NSArray *bookmarks = [[NSUserDefaults standardUserDefaults] arrayForKey:@"sec_bookmarks"];
		NSMutableArray *new_bookmarks = [[NSMutableArray alloc] init];
		for (id bookmark in bookmarks) {
			NSError *error = nil;
			BOOL isStale = NO;
			NSURL *url = [NSURL URLByResolvingBookmarkData:bookmark options:NSURLBookmarkResolutionWithSecurityScope relativeToURL:nil bookmarkDataIsStale:&isStale error:&error];
			if (!error && !isStale) {
				if ([url startAccessingSecurityScopedResource]) {
					[new_bookmarks addObject:bookmark];
				}
			}
		}
		[[NSUserDefaults standardUserDefaults] setObject:new_bookmarks forKey:@"sec_bookmarks"];
	}
	Vector<Logger *> loggers;
	loggers.push_back(memnew(OsLogLogger(NSBundle.mainBundle.bundleIdentifier.UTF8String)));
	loggers.push_back(memnew(UnixTerminalLogger));
	_set_logger(memnew(CompositeLogger(loggers)));

#ifdef COREAUDIO_ENABLED
	AudioDriverManager::add_driver(&audio_driver);
#endif

	DisplayServerMacOS::register_macos_driver();
}

// MARK: - OS_MacOS_NSApp

void OS_MacOS_NSApp::run() {
	[NSApp run]; // Note: this call will never return. Use `OS_MacOS_NSApp::cleanup()` for cleanup.
}

static bool sig_received = false;

static void handle_interrupt(int sig) {
	if (sig == SIGINT) {
		sig_received = true;
	}
}

void OS_MacOS_NSApp::start_main() {
	Error err;
	@autoreleasepool {
		err = Main::setup(execpath, argc, argv);
	}

	if (err == OK) {
		main_started = true;

		int ret;
		@autoreleasepool {
			ret = Main::start();
		}
		if (ret == EXIT_SUCCESS) {
			if (main_loop) {
				@autoreleasepool {
					main_loop->initialize();
				}
				DisplayServer *ds = DisplayServer::get_singleton();
				DisplayServerMacOS *ds_mac = Object::cast_to<DisplayServerMacOS>(ds);

				pre_wait_observer = CFRunLoopObserverCreateWithHandler(kCFAllocatorDefault, kCFRunLoopBeforeWaiting, true, 0, ^(CFRunLoopObserverRef observer, CFRunLoopActivity activity) {
					@autoreleasepool {
						@try {
							GodotProfileFrameMark;
							GodotProfileZone("macOS main loop");

							if (ds_mac) {
								ds_mac->_process_events(false);
							} else if (ds) {
								ds->process_events();
							}
#ifdef SDL_ENABLED
							if (joypad_sdl) {
								joypad_sdl->process_events();
							}
#endif

							if (Main::iteration() || sig_received) {
								terminate();
							}
						} @catch (NSException *exception) {
							ERR_PRINT("NSException: " + String::utf8([exception reason].UTF8String));
						}
					}
					if (wait_timer == nil) {
						CFRunLoopWakeUp(CFRunLoopGetCurrent()); // Prevent main loop from sleeping.
					}
				});
				CFRunLoopAddObserver(CFRunLoopGetCurrent(), pre_wait_observer, kCFRunLoopCommonModes);
				return;
			}
		} else {
			set_exit_code(EXIT_FAILURE);
		}
	} else if (err == ERR_HELP) { // Returned by --help and --version, so success.
		set_exit_code(EXIT_SUCCESS);
	} else {
		set_exit_code(EXIT_FAILURE);
	}

	terminate();
}

void OS_MacOS_NSApp::terminate() {
	// Note: This method only sends app termination request. Use `OS_MacOS_NSApp::cleanup()` for cleanup.
	if (pre_wait_observer) {
		CFRunLoopRemoveObserver(CFRunLoopGetCurrent(), pre_wait_observer, kCFRunLoopCommonModes);
		CFRelease(pre_wait_observer);
		pre_wait_observer = nil;
	}

	should_terminate = true;
	[NSApp terminate:nil];
}

void OS_MacOS_NSApp::cleanup() {
	if (main_loop) {
		main_loop->finalize();
	}
	if (main_started) {
		@autoreleasepool {
			Main::cleanup();
		}
	}
	godot_cleanup_profiler();
}

OS_MacOS_NSApp::OS_MacOS_NSApp(const char *p_execpath, int p_argc, char **p_argv) :
		OS_MacOS(p_execpath, p_argc, p_argv) {
	// Implicitly create shared NSApplication instance.
	[GodotApplication sharedApplication];

	// In case we are unbundled, make us a proper UI application.
	[NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];

	// Menu bar setup must go between sharedApplication above and
	// finishLaunching below, in order to properly emulate the behavior
	// of NSApplicationMain.

	NSMenu *main_menu = [[NSMenu alloc] initWithTitle:@""];
	[NSApp setMainMenu:main_menu];

	delegate = [[GodotApplicationDelegate alloc] initWithOS:this];
	ERR_FAIL_NULL(delegate);
	[NSApp setDelegate:delegate];
	[NSApp registerUserInterfaceItemSearchHandler:delegate];

	struct sigaction action;
	memset(&action, 0, sizeof(action));
	action.sa_handler = handle_interrupt;
	sigaction(SIGINT, &action, nullptr);
}

// MARK: - OS_MacOS_Headless

void OS_MacOS_Headless::run() {
	CFRunLoopGetCurrent();

	@autoreleasepool {
		Error err = Main::setup(execpath, argc, argv);
		if (err != OK) {
			if (err == ERR_HELP) {
				return set_exit_code(EXIT_SUCCESS);
			}
			return set_exit_code(EXIT_FAILURE);
		}
	}

	int ret;
	@autoreleasepool {
		ret = Main::start();
	}

	if (ret == EXIT_SUCCESS && main_loop) {
		@autoreleasepool {
			main_loop->initialize();
		}

		while (true) {
			@autoreleasepool {
				@try {
					if (Input::get_singleton()) {
						Input::get_singleton()->flush_buffered_events();
					}

					if (Main::iteration()) {
						break;
					}

					CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0, 0);
				} @catch (NSException *exception) {
					ERR_PRINT("NSException: " + String::utf8([exception reason].UTF8String));
				}
			}
		}

		main_loop->finalize();
	}

	Main::cleanup();
}

OS_MacOS_Headless::OS_MacOS_Headless(const char *p_execpath, int p_argc, char **p_argv) :
		OS_MacOS(p_execpath, p_argc, p_argv) {
}

// MARK: - OS_MacOS_Embedded

#ifdef TOOLS_ENABLED

void OS_MacOS_Embedded::run() {
	CFRunLoopGetCurrent();

	@autoreleasepool {
		Error err = Main::setup(execpath, argc, argv);
		if (err != OK) {
			if (err == ERR_HELP) {
				return set_exit_code(EXIT_SUCCESS);
			}
			return set_exit_code(EXIT_FAILURE);
		}
	}

	int ret;
	@autoreleasepool {
		ret = Main::start();
	}

	DisplayServerEmbedded *ds = Object::cast_to<DisplayServerEmbedded>(DisplayServer::get_singleton());
	if (!ds) {
		ERR_FAIL_MSG("DisplayServerEmbedded is not initialized.");
	}

	if (ds && ret == EXIT_SUCCESS && main_loop) {
		@autoreleasepool {
			main_loop->initialize();
		}

		while (true) {
			@autoreleasepool {
				@try {
					GodotProfileFrameMark;
					GodotProfileZone("macOS embedded main loop");

					ds->process_events();

#ifdef SDL_ENABLED
					if (joypad_sdl) {
						joypad_sdl->process_events();
					}
#endif
					if (Main::iteration()) {
						break;
					}

					CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0, 0);
				} @catch (NSException *exception) {
					ERR_PRINT("NSException: " + String::utf8([exception reason].UTF8String));
				}
			}
		}

		main_loop->finalize();
	}

	Main::cleanup();
}

OS_MacOS_Embedded::OS_MacOS_Embedded(const char *p_execpath, int p_argc, char **p_argv) :
		OS_MacOS(p_execpath, p_argc, p_argv) {
	DisplayServerEmbedded::register_embedded_driver();
}

#endif
