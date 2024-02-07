/**************************************************************************/
/*  godot_application_delegate.mm                                         */
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

#include "godot_application_delegate.h"

#include "display_server_macos.h"
#include "os_macos.h"

@implementation GodotApplicationDelegate

- (BOOL)applicationSupportsSecureRestorableState:(NSApplication *)app {
	return YES;
}

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
	NSString *nsbundleid_env = [NSString stringWithUTF8String:getenv("__CFBundleIdentifier")];
	NSString *nsbundleid = [[NSBundle mainBundle] bundleIdentifier];
	if (nsappname == nil || isatty(STDOUT_FILENO) || isatty(STDIN_FILENO) || isatty(STDERR_FILENO) || ![nsbundleid isEqualToString:nsbundleid_env]) {
		// If the executable is started from terminal or is not bundled, macOS WindowServer won't register and activate app window correctly (menu and title bar are grayed out and input ignored).
		[self performSelector:@selector(forceUnbundledWindowActivationHackStep1) withObject:nil afterDelay:0.02];
	}
}

- (id)init {
	self = [super init];

	NSAppleEventManager *aem = [NSAppleEventManager sharedAppleEventManager];
	[aem setEventHandler:self andSelector:@selector(handleAppleEvent:withReplyEvent:) forEventClass:kInternetEventClass andEventID:kAEGetURL];
	[aem setEventHandler:self andSelector:@selector(handleAppleEvent:withReplyEvent:) forEventClass:kCoreEventClass andEventID:kAEOpenDocuments];

	return self;
}

- (void)handleAppleEvent:(NSAppleEventDescriptor *)event withReplyEvent:(NSAppleEventDescriptor *)replyEvent {
	OS_MacOS *os = (OS_MacOS *)OS::get_singleton();
	if (!event || !os) {
		return;
	}

	List<String> args;
	if (([event eventClass] == kInternetEventClass) && ([event eventID] == kAEGetURL)) {
		// Opening URL scheme.
		NSString *url = [[event paramDescriptorForKeyword:keyDirectObject] stringValue];
		args.push_back(vformat("--uri=\"%s\"", String::utf8([url UTF8String])));
	}

	if (([event eventClass] == kCoreEventClass) && ([event eventID] == kAEOpenDocuments)) {
		// Opening file association.
		NSAppleEventDescriptor *files = [event paramDescriptorForKeyword:keyDirectObject];
		if (files) {
			NSInteger count = [files numberOfItems];
			for (NSInteger i = 1; i <= count; i++) {
				NSURL *url = [NSURL URLWithString:[[files descriptorAtIndex:i] stringValue]];
				args.push_back(String::utf8([url.path UTF8String]));
			}
		}
	}

	if (!args.is_empty()) {
		if (os->get_main_loop()) {
			// Application is already running, open a new instance with the URL/files as command line arguments.
			os->create_instance(args);
		} else {
			// Application is just started, add to the list of command line arguments and continue.
			os->set_cmdline_platform_args(args);
		}
	}
}

- (void)applicationDidResignActive:(NSNotification *)notification {
	DisplayServerMacOS *ds = (DisplayServerMacOS *)DisplayServer::get_singleton();
	if (ds) {
		ds->mouse_process_popups(true);
	}
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
	DisplayServerMacOS *ds = (DisplayServerMacOS *)DisplayServer::get_singleton();
	if (ds) {
		return ds->menu_callback(sender);
	}
}

- (NSMenu *)applicationDockMenu:(NSApplication *)sender {
	DisplayServerMacOS *ds = (DisplayServerMacOS *)DisplayServer::get_singleton();
	if (ds) {
		return ds->get_dock_menu();
	} else {
		return nullptr;
	}
}

- (NSApplicationTerminateReply)applicationShouldTerminate:(NSApplication *)sender {
	DisplayServerMacOS *ds = (DisplayServerMacOS *)DisplayServer::get_singleton();
	if (ds) {
		ds->send_window_event(ds->get_window(DisplayServerMacOS::MAIN_WINDOW_ID), DisplayServerMacOS::WINDOW_EVENT_CLOSE_REQUEST);
	}
	return NSTerminateCancel;
}

- (void)showAbout:(id)sender {
	OS_MacOS *os = (OS_MacOS *)OS::get_singleton();
	if (os && os->get_main_loop()) {
		os->get_main_loop()->notification(MainLoop::NOTIFICATION_WM_ABOUT);
	}
}

@end
