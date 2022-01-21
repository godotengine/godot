/*************************************************************************/
/*  os_osx.mm                                                            */
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

#include "os_osx.h"

#include "core/version_generated.gen.h"

#include "dir_access_osx.h"
#include "display_server_osx.h"
#include "main/main.h"

#include <dlfcn.h>
#include <libproc.h>
#include <mach-o/dyld.h>
#include <os/log.h>

#define DS_OSX ((DisplayServerOSX *)(DisplayServerOSX::get_singleton()))

/*************************************************************************/
/* GodotApplication                                                      */
/*************************************************************************/

@interface GodotApplication : NSApplication
@end

@implementation GodotApplication

- (void)sendEvent:(NSEvent *)event {
	if (DS_OSX) {
		DS_OSX->_send_event(event);
	}

	// From http://cocoadev.com/index.pl?GameKeyboardHandlingAlmost
	// This works around an AppKit bug, where key up events while holding
	// down the command key don't get sent to the key window.
	if ([event type] == NSEventTypeKeyUp && ([event modifierFlags] & NSEventModifierFlagCommand)) {
		[[self keyWindow] sendEvent:event];
	} else {
		[super sendEvent:event];
	}
}

@end

/*************************************************************************/
/* GodotApplicationDelegate                                              */
/*************************************************************************/

@interface GodotApplicationDelegate : NSObject
- (void)forceUnbundledWindowActivationHackStep1;
- (void)forceUnbundledWindowActivationHackStep2;
- (void)forceUnbundledWindowActivationHackStep3;
@end

@implementation GodotApplicationDelegate

- (void)forceUnbundledWindowActivationHackStep1 {
	// Step 1: Switch focus to macOS SystemUIServer process.
	// Required to perform step 2, TransformProcessType will fail if app is already the in focus.
	for (NSRunningApplication *app in [NSRunningApplication runningApplicationsWithBundleIdentifier:@"com.apple.systemuiserver"]) {
		[app activateWithOptions:NSApplicationActivateIgnoringOtherApps];
		break;
	}
	[self performSelector:@selector(forceUnbundledWindowActivationHackStep2)
			   withObject:nil
			   afterDelay:0.02];
}

- (void)forceUnbundledWindowActivationHackStep2 {
	// Step 2: Register app as foreground process.
	ProcessSerialNumber psn = { 0, kCurrentProcess };
	(void)TransformProcessType(&psn, kProcessTransformToForegroundApplication);
	[self performSelector:@selector(forceUnbundledWindowActivationHackStep3) withObject:nil afterDelay:0.02];
}

- (void)forceUnbundledWindowActivationHackStep3 {
	// Step 3: Switch focus back to app window.
	[[NSRunningApplication currentApplication] activateWithOptions:NSApplicationActivateIgnoringOtherApps];
}

- (void)applicationDidFinishLaunching:(NSNotification *)notice {
	NSString *nsappname = [[[NSBundle mainBundle] infoDictionary] objectForKey:@"CFBundleName"];
	if (nsappname == nil || isatty(STDOUT_FILENO) || isatty(STDIN_FILENO) || isatty(STDERR_FILENO)) {
		// If the executable is started from terminal or is not bundled, macOS WindowServer won't register and activate app window correctly (menu and title bar are grayed out and input ignored).
		[self performSelector:@selector(forceUnbundledWindowActivationHackStep1) withObject:nil afterDelay:0.02];
	}
}

- (void)applicationDidResignActive:(NSNotification *)notification {
	if (OS::get_singleton()->get_main_loop()) {
		OS::get_singleton()->get_main_loop()->notification(MainLoop::NOTIFICATION_APPLICATION_FOCUS_OUT);
	}
}

- (void)applicationDidBecomeActive:(NSNotification *)notification {
	if (OS::get_singleton()->get_main_loop()) {
		OS::get_singleton()->get_main_loop()->notification(MainLoop::NOTIFICATION_APPLICATION_FOCUS_IN);
	}
}

- (void)globalMenuCallback:(id)sender {
	if (DS_OSX) {
		return DS_OSX->_menu_callback(sender);
	}
}

- (NSMenu *)applicationDockMenu:(NSApplication *)sender {
	if (DS_OSX) {
		return DS_OSX->_get_dock_menu();
	} else {
		return nullptr;
	}
}

- (BOOL)application:(NSApplication *)sender openFile:(NSString *)filename {
	// Note: may be called called before main loop init!
	char *utfs = strdup([filename UTF8String]);
	((OS_OSX *)OS_OSX::get_singleton())->open_with_filename.parse_utf8(utfs);
	free(utfs);

#ifdef TOOLS_ENABLED
	// Open new instance
	if (OS_OSX::get_singleton()->get_main_loop()) {
		List<String> args;
		args.push_back(((OS_OSX *)OS_OSX::get_singleton())->open_with_filename);
		String exec = OS_OSX::get_singleton()->get_executable_path();
		OS_OSX::get_singleton()->create_process(exec, args);
	}
#endif
	return YES;
}

- (NSApplicationTerminateReply)applicationShouldTerminate:(NSApplication *)sender {
	if (DS_OSX) {
		DS_OSX->_send_window_event(DS_OSX->windows[DisplayServerOSX::MAIN_WINDOW_ID], DisplayServerOSX::WINDOW_EVENT_CLOSE_REQUEST);
	}
	return NSTerminateCancel;
}

- (void)showAbout:(id)sender {
	if (OS_OSX::get_singleton()->get_main_loop()) {
		OS_OSX::get_singleton()->get_main_loop()->notification(MainLoop::NOTIFICATION_WM_ABOUT);
	}
}

@end

/*************************************************************************/
/* OSXTerminalLogger                                                     */
/*************************************************************************/

class OSXTerminalLogger : public StdLogger {
public:
	virtual void log_error(const char *p_function, const char *p_file, int p_line, const char *p_code, const char *p_rationale, bool p_editor_notify = false, ErrorType p_type = ERR_ERROR) {
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

void OS_OSX::alert(const String &p_alert, const String &p_title) {
	NSAlert *window = [[NSAlert alloc] init];
	NSString *ns_title = [NSString stringWithUTF8String:p_title.utf8().get_data()];
	NSString *ns_alert = [NSString stringWithUTF8String:p_alert.utf8().get_data()];

	[window addButtonWithTitle:@"OK"];
	[window setMessageText:ns_title];
	[window setInformativeText:ns_alert];
	[window setAlertStyle:NSAlertStyleWarning];

	id key_window = [[NSApplication sharedApplication] keyWindow];
	[window runModal];
	[window release];
	if (key_window) {
		[key_window makeKeyAndOrderFront:nil];
	}
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

_FORCE_INLINE_ String _get_framework_executable(const String p_path) {
	// Append framework executable name, or return as is if p_path is not a framework.
	DirAccessRef da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	if (da->dir_exists(p_path) && da->file_exists(p_path.plus_file(p_path.get_file().get_basename()))) {
		return p_path.plus_file(p_path.get_file().get_basename());
	} else {
		return p_path;
	}
}

Error OS_OSX::open_dynamic_library(const String p_path, void *&p_library_handle, bool p_also_set_library_path) {
	String path = _get_framework_executable(p_path);

	if (!FileAccess::exists(path)) {
		// This code exists so gdnative can load .dylib files from within the executable path.
		path = _get_framework_executable(get_executable_path().get_base_dir().plus_file(p_path.get_file()));
	}

	if (!FileAccess::exists(path)) {
		// This code exists so gdnative can load .dylib files from a standard macOS location.
		path = _get_framework_executable(get_executable_path().get_base_dir().plus_file("../Frameworks").plus_file(p_path.get_file()));
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
		if (get_environment("XDG_CONFIG_HOME").is_absolute_path()) {
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
		if (get_environment("XDG_DATA_HOME").is_absolute_path()) {
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
		if (get_environment("XDG_CACHE_HOME").is_absolute_path()) {
			return get_environment("XDG_CACHE_HOME");
		} else {
			WARN_PRINT_ONCE("`XDG_CACHE_HOME` is a relative path. Ignoring its value and falling back to `$HOME/Library/Caches` or `get_config_path()` per the XDG Base Directory specification.");
		}
	}
	if (has_environment("HOME")) {
		return get_environment("HOME").plus_file("Library/Caches");
	}
	return get_config_path();
}

String OS_OSX::get_bundle_resource_dir() const {
	String ret;

	NSBundle *main = [NSBundle mainBundle];
	if (main) {
		NSString *resourcePath = [main resourcePath];
		ret.parse_utf8([resourcePath UTF8String]);
	}
	return ret;
}

String OS_OSX::get_bundle_icon_path() const {
	String ret;

	NSBundle *main = [NSBundle mainBundle];
	if (main) {
		NSString *iconPath = [[main infoDictionary] objectForKey:@"CFBundleIconFile"];
		if (iconPath) {
			ret.parse_utf8([iconPath UTF8String]);
		}
	}
	return ret;
}

// Get properly capitalized engine name for system paths
String OS_OSX::get_godot_dir_name() const {
	return String(VERSION_SHORT_NAME).capitalize();
}

String OS_OSX::get_system_dir(SystemDir p_dir, bool p_shared_storage) const {
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

Error OS_OSX::create_instance(const List<String> &p_arguments, ProcessID *r_child_id) {
	// If executable is bundled, always execute editor instances as an app bundle to ensure app window is registered and activated correctly.
	NSString *nsappname = [[[NSBundle mainBundle] infoDictionary] objectForKey:@"CFBundleName"];
	if (nsappname != nil) {
		String path;
		path.parse_utf8([[[NSBundle mainBundle] bundlePath] UTF8String]);
		return create_process(path, p_arguments, r_child_id, false);
	} else {
		return create_process(get_executable_path(), p_arguments, r_child_id, false);
	}
}

Error OS_OSX::create_process(const String &p_path, const List<String> &p_arguments, ProcessID *r_child_id, bool p_open_console) {
	if (@available(macOS 10.15, *)) {
		// Use NSWorkspace if path is an .app bundle.
		NSURL *url = [NSURL fileURLWithPath:@(p_path.utf8().get_data())];
		NSBundle *bundle = [NSBundle bundleWithURL:url];
		if (bundle) {
			NSMutableArray *arguments = [[NSMutableArray alloc] init];
			for (const List<String>::Element *E = p_arguments.front(); E; E = E->next()) {
				[arguments addObject:[NSString stringWithUTF8String:E->get().utf8().get_data()]];
			}
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
			dispatch_release(lock);

			if (err == OK) {
				if (r_child_id) {
					*r_child_id = (ProcessID)pid;
				}
			}

			return err;
		} else {
			return OS_Unix::create_process(p_path, p_arguments, r_child_id, p_open_console);
		}
	} else {
		return OS_Unix::create_process(p_path, p_arguments, r_child_id, p_open_console);
	}
}

void OS_OSX::pre_wait_observer_cb(CFRunLoopObserverRef p_observer, CFRunLoopActivity p_activiy, void *p_context) {
	// Prevent main loop from sleeping and redraw window during resize / modal popups.

	if (get_singleton()->get_main_loop()) {
		Main::force_redraw();
		if (!Main::is_iterating()) { // Avoid cyclic loop.
			Main::iteration();
		}
	}

	CFRunLoopWakeUp(CFRunLoopGetCurrent()); // Prevent main loop from sleeping.
}

void OS_OSX::run() {
	force_quit = false;

	if (!main_loop) {
		return;
	}

	main_loop->initialize();

	CFRunLoopObserverRef pre_wait_observer = CFRunLoopObserverCreate(kCFAllocatorDefault, kCFRunLoopBeforeWaiting, true, 0, &pre_wait_observer_cb, nullptr);
	CFRunLoopAddObserver(CFRunLoopGetCurrent(), pre_wait_observer, kCFRunLoopCommonModes);

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
			ERR_PRINT("NSException: " + String::utf8([exception reason].UTF8String));
		}
	};

	CFRunLoopRemoveObserver(CFRunLoopGetCurrent(), pre_wait_observer, kCFRunLoopCommonModes);
	CFRelease(pre_wait_observer);

	main_loop->finalize();
}

Error OS_OSX::move_to_trash(const String &p_path) {
	NSFileManager *fm = [NSFileManager defaultManager];
	NSURL *url = [NSURL fileURLWithPath:@(p_path.utf8().get_data())];
	NSError *err;

	if (![fm trashItemAtURL:url resultingItemURL:nil error:&err]) {
		ERR_PRINT("trashItemAtURL error: " + String::utf8(err.localizedDescription.UTF8String));
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

	// Implicitly create shared NSApplication instance
	[GodotApplication sharedApplication];

	// In case we are unbundled, make us a proper UI application
	[NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];

	// Menu bar setup must go between sharedApplication above and
	// finishLaunching below, in order to properly emulate the behavior
	// of NSApplicationMain

	NSMenu *main_menu = [[[NSMenu alloc] initWithTitle:@""] autorelease];
	[NSApp setMainMenu:main_menu];
	[NSApp finishLaunching];

	id delegate = [[GodotApplicationDelegate alloc] init];
	ERR_FAIL_COND(!delegate);
	[NSApp setDelegate:delegate];

	//process application:openFile: event
	while (true) {
		NSEvent *event = [NSApp
				nextEventMatchingMask:NSEventMaskAny
							untilDate:[NSDate distantPast]
							   inMode:NSDefaultRunLoopMode
							  dequeue:YES];

		if (event == nil) {
			break;
		}

		[NSApp sendEvent:event];
	}

	[NSApp activateIgnoringOtherApps:YES];
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
