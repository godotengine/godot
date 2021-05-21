/*************************************************************************/
/*  os_osx.mm                                                            */
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

#include "os_osx.h"

#include "core/version_generated.gen.h"

#include "dir_access_osx.h"
#include "display_server_osx.h"
#include "main/main.h"

#include <dlfcn.h>
#include <libproc.h>
#include <mach-o/dyld.h>
#include <os/log.h>

/*************************************************************************/
/* OSXTerminalLogger                                                     */
/*************************************************************************/

class OSXTerminalLogger : public StdLogger {
public:
	virtual void log_error(const char *p_function, const char *p_file, int p_line, const char *p_code, const char *p_rationale, ErrorType p_type = ERR_ERROR) {
		if (!should_log(true)) {
			return;
		}

		const char *err_details;
		if (p_rationale && p_rationale[0])
			err_details = p_rationale;
		else
			err_details = p_code;

		switch (p_type) {
			case ERR_WARNING:
				os_log_info(OS_LOG_DEFAULT,
						"WARNING: %{public}s\nat: %{public}s (%{public}s:%i)",
						err_details, p_function, p_file, p_line);
				logf_error("\E[1;33mWARNING:\E[0;93m %s\n", err_details);
				logf_error("\E[0;90m     at: %s (%s:%i)\E[0m\n", p_function, p_file, p_line);
				break;
			case ERR_SCRIPT:
				os_log_error(OS_LOG_DEFAULT,
						"SCRIPT ERROR: %{public}s\nat: %{public}s (%{public}s:%i)",
						err_details, p_function, p_file, p_line);
				logf_error("\E[1;35mSCRIPT ERROR:\E[0;95m %s\n", err_details);
				logf_error("\E[0;90m          at: %s (%s:%i)\E[0m\n", p_function, p_file, p_line);
				break;
			case ERR_SHADER:
				os_log_error(OS_LOG_DEFAULT,
						"SHADER ERROR: %{public}s\nat: %{public}s (%{public}s:%i)",
						err_details, p_function, p_file, p_line);
				logf_error("\E[1;36mSHADER ERROR:\E[0;96m %s\n", err_details);
				logf_error("\E[0;90m          at: %s (%s:%i)\E[0m\n", p_function, p_file, p_line);
				break;
			case ERR_ERROR:
			default:
				os_log_error(OS_LOG_DEFAULT,
						"ERROR: %{public}s\nat: %{public}s (%{public}s:%i)",
						err_details, p_function, p_file, p_line);
				logf_error("\E[1;31mERROR:\E[0;91m %s\n", err_details);
				logf_error("\E[0;90m   at: %s (%s:%i)\E[0m\n", p_function, p_file, p_line);
				break;
		}
	}
};

/*************************************************************************/
/* OS_OSX                                                                */
/*************************************************************************/

String OS_OSX::get_unique_id() const {
	static String serial_number;

	if (serial_number.is_empty()) {
		io_service_t platformExpert = IOServiceGetMatchingService(kIOMasterPortDefault, IOServiceMatching("IOPlatformExpertDevice"));
		CFStringRef serialNumberAsCFString = nullptr;
		if (platformExpert) {
			serialNumberAsCFString = (CFStringRef)IORegistryEntryCreateCFProperty(platformExpert, CFSTR(kIOPlatformSerialNumberKey), kCFAllocatorDefault, 0);
			IOObjectRelease(platformExpert);
		}

		NSString *serialNumberAsNSString = nil;
		if (serialNumberAsCFString) {
			serialNumberAsNSString = [NSString stringWithString:(NSString *)serialNumberAsCFString];
			CFRelease(serialNumberAsCFString);
		}

		serial_number = [serialNumberAsNSString UTF8String];
	}

	return serial_number;
}

void OS_OSX::initialize_core() {
	OS_Unix::initialize_core();

	DirAccess::make_default<DirAccessOSX>(DirAccess::ACCESS_RESOURCES);
	DirAccess::make_default<DirAccessOSX>(DirAccess::ACCESS_USERDATA);
	DirAccess::make_default<DirAccessOSX>(DirAccess::ACCESS_FILESYSTEM);
}

void OS_OSX::initialize_joypads() {
	joypad_osx = memnew(JoypadOSX(Input::get_singleton()));
}

void OS_OSX::initialize() {
	crash_handler.initialize();

	initialize_core();
	//ensure_user_data_dir();
}

void OS_OSX::finalize() {
#ifdef COREMIDI_ENABLED
	midi_driver.close();
#endif

	delete_main_loop();

	if (joypad_osx) {
		memdelete(joypad_osx);
	}
}

void OS_OSX::set_main_loop(MainLoop *p_main_loop) {
	main_loop = p_main_loop;
}

void OS_OSX::delete_main_loop() {
	if (!main_loop)
		return;
	memdelete(main_loop);
	main_loop = nullptr;
}

String OS_OSX::get_name() const {
	return "macOS";
}

Error OS_OSX::open_dynamic_library(const String p_path, void *&p_library_handle, bool p_also_set_library_path) {
	String path = p_path;

	if (!FileAccess::exists(path)) {
		//this code exists so gdnative can load .dylib files from within the executable path
		path = get_executable_path().get_base_dir().plus_file(p_path.get_file());
	}

	if (!FileAccess::exists(path)) {
		//this code exists so gdnative can load .dylib files from a standard macOS location
		path = get_executable_path().get_base_dir().plus_file("../Frameworks").plus_file(p_path.get_file());
	}

	p_library_handle = dlopen(path.utf8().get_data(), RTLD_NOW);
	ERR_FAIL_COND_V_MSG(!p_library_handle, ERR_CANT_OPEN, "Can't open dynamic library: " + p_path + ", error: " + dlerror() + ".");
	return OK;
}

MainLoop *OS_OSX::get_main_loop() const {
	return main_loop;
}

String OS_OSX::get_config_path() const {
	// The XDG Base Directory specification technically only applies on Linux/*BSD, but it doesn't hurt to support it on macOS as well.
	if (has_environment("XDG_CONFIG_HOME")) {
		if (get_environment("XDG_CONFIG_HOME").is_abs_path()) {
			return get_environment("XDG_CONFIG_HOME");
		} else {
			WARN_PRINT_ONCE("`XDG_CONFIG_HOME` is a relative path. Ignoring its value and falling back to `$HOME/Library/Application Support` or `.` per the XDG Base Directory specification.");
		}
	}
	if (has_environment("HOME")) {
		return get_environment("HOME").plus_file("Library/Application Support");
	}
	return ".";
}

String OS_OSX::get_data_path() const {
	// The XDG Base Directory specification technically only applies on Linux/*BSD, but it doesn't hurt to support it on macOS as well.
	if (has_environment("XDG_DATA_HOME")) {
		if (get_environment("XDG_DATA_HOME").is_abs_path()) {
			return get_environment("XDG_DATA_HOME");
		} else {
			WARN_PRINT_ONCE("`XDG_DATA_HOME` is a relative path. Ignoring its value and falling back to `get_config_path()` per the XDG Base Directory specification.");
		}
	}
	return get_config_path();
}

String OS_OSX::get_cache_path() const {
	// The XDG Base Directory specification technically only applies on Linux/*BSD, but it doesn't hurt to support it on macOS as well.
	if (has_environment("XDG_CACHE_HOME")) {
		if (get_environment("XDG_CACHE_HOME").is_abs_path()) {
			return get_environment("XDG_CACHE_HOME");
		} else {
			WARN_PRINT_ONCE("`XDG_CACHE_HOME` is a relative path. Ignoring its value and falling back to `$HOME/Libary/Caches` or `get_config_path()` per the XDG Base Directory specification.");
		}
	}
	if (has_environment("HOME")) {
		return get_environment("HOME").plus_file("Library/Caches");
	}
	return get_config_path();
}

String OS_OSX::get_bundle_resource_dir() const {
	NSBundle *main = [NSBundle mainBundle];
	NSString *resourcePath = [main resourcePath];

	char *utfs = strdup([resourcePath UTF8String]);
	String ret;
	ret.parse_utf8(utfs);
	free(utfs);

	return ret;
}

// Get properly capitalized engine name for system paths
String OS_OSX::get_godot_dir_name() const {
	return String(VERSION_SHORT_NAME).capitalize();
}

String OS_OSX::get_system_dir(SystemDir p_dir) const {
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
			char *utfs = strdup([[paths firstObject] UTF8String]);
			ret.parse_utf8(utfs);
			free(utfs);
		}
	}

	return ret;
}

Error OS_OSX::shell_open(String p_uri) {
	NSString *string = [NSString stringWithUTF8String:p_uri.utf8().get_data()];
	NSURL *uri = [[NSURL alloc] initWithString:string];
	// Escape special characters in filenames
	if (!uri || !uri.scheme || [uri.scheme isEqual:@"file"]) {
		uri = [[NSURL alloc] initWithString:[string stringByAddingPercentEncodingWithAllowedCharacters:[NSCharacterSet URLFragmentAllowedCharacterSet]]];
	}
	[[NSWorkspace sharedWorkspace] openURL:uri];
	return OK;
}

String OS_OSX::get_locale() const {
	NSString *locale_code = [[NSLocale preferredLanguages] objectAtIndex:0];
	return String([locale_code UTF8String]).replace("-", "_");
}

String OS_OSX::get_executable_path() const {
	int ret;
	pid_t pid;
	char pathbuf[PROC_PIDPATHINFO_MAXSIZE];

	pid = getpid();
	ret = proc_pidpath(pid, pathbuf, sizeof(pathbuf));
	if (ret <= 0) {
		return OS::get_executable_path();
	} else {
		String path;
		path.parse_utf8(pathbuf);

		return path;
	}
}

void OS_OSX::run() {
	force_quit = false;

	if (!main_loop)
		return;

	main_loop->initialize();

	bool quit = false;
	while (!force_quit && !quit) {
		@try {
			if (DisplayServer::get_singleton()) {
				DisplayServer::get_singleton()->process_events(); // get rid of pending events
			}
			joypad_osx->process_joypads();

			if (Main::iteration()) {
				quit = true;
			}
		} @catch (NSException *exception) {
			ERR_PRINT("NSException: " + String([exception reason].UTF8String));
		}
	};
	main_loop->finalize();
}

Error OS_OSX::move_to_trash(const String &p_path) {
	NSFileManager *fm = [NSFileManager defaultManager];
	NSURL *url = [NSURL fileURLWithPath:@(p_path.utf8().get_data())];
	NSError *err;

	if (![fm trashItemAtURL:url resultingItemURL:nil error:&err]) {
		ERR_PRINT("trashItemAtURL error: " + String(err.localizedDescription.UTF8String));
		return FAILED;
	}

	return OK;
}

OS_OSX::OS_OSX() {
	main_loop = nullptr;
	force_quit = false;

	Vector<Logger *> loggers;
	loggers.push_back(memnew(OSXTerminalLogger));
	_set_logger(memnew(CompositeLogger(loggers)));

#ifdef COREAUDIO_ENABLED
	AudioDriverManager::add_driver(&audio_driver);
#endif

	DisplayServerOSX::register_osx_driver();
}

bool OS_OSX::_check_internal_feature_support(const String &p_feature) {
	return p_feature == "pc";
}

void OS_OSX::disable_crash_handler() {
	crash_handler.disable();
}

bool OS_OSX::is_disable_crash_handler() const {
	return crash_handler.is_disabled();
}
