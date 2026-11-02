/**************************************************************************/
/*  libgodot_ios.mm                                                       */
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

#import "libgodot_ios.h"

#include "core/config/project_settings.h"
#include "core/core_globals.h"
#include "core/os/main_loop.h"
#include "core/profiling/profiling.h"
#import "drivers/apple_embedded/app_delegate_service.h"
#import "drivers/apple_embedded/display_server_apple_embedded.h"
#import "drivers/apple_embedded/godot_view_controller.h"
#import "drivers/apple_embedded/main_utilities.h"
#include "main/main.h"
#import "os_ios.h"

#import <AVFoundation/AVFoundation.h>
#import <UIKit/UIKit.h>

// 2dog: mirrors app_delegate_service.mm and main_ios.mm for a process whose UIApplication delegate is not Godot's.

static OS_IOS *os = nullptr;
static GDTViewController *view_controller = nil;
static libgodot_ios_callback started_callback = nullptr;
static libgodot_ios_callback quit_callback = nullptr;
static void *callback_userdata = nullptr;
static id lifecycle_observer = nil;

// Godot's Xcode export generates these for its .gdip plugins; a libgodot app has none (GDExtensions still load).
void godot_apple_embedded_plugins_initialize() {}
void godot_apple_embedded_plugins_deinitialize() {}

@interface TwoDogLifecycleObserver : NSObject
@end

@implementation TwoDogLifecycleObserver

- (void)didBecomeActive:(NSNotification *)notification {
	libgodot_ios_on_focus_in();
}

- (void)willResignActive:(NSNotification *)notification {
	libgodot_ios_on_focus_out();
}

- (void)didEnterBackground:(NSNotification *)notification {
	libgodot_ios_on_enter_background();
}

- (void)willEnterForeground:(NSNotification *)notification {
	libgodot_ios_on_exit_background();
}

- (void)didReceiveMemoryWarning:(NSNotification *)notification {
	libgodot_ios_on_memory_warning();
}

- (void)willTerminate:(NSNotification *)notification {
	libgodot_ios_terminate();
}

- (void)onAudioInterruption:(NSNotification *)notification {
	NSNumber *type = [notification.userInfo valueForKey:AVAudioSessionInterruptionTypeKey];
	if ([type isEqualToNumber:[NSNumber numberWithInt:AVAudioSessionInterruptionTypeBegan]]) {
		libgodot_ios_on_focus_out();
	} else if ([type isEqualToNumber:[NSNumber numberWithInt:AVAudioSessionInterruptionTypeEnded]]) {
		libgodot_ios_on_focus_in();
	}
}

@end

// OS_AppleEmbedded overrides get_main_loop privately; the OS interface keeps it public.
static MainLoop *main_loop() {
	return os ? static_cast<OS *>(os)->get_main_loop() : nullptr;
}

static void on_main_loop_started(void *p_userdata) {
	if (started_callback) {
		started_callback(callback_userdata);
	}
}

static void on_quit_requested(void *p_userdata) {
	if (quit_callback) {
		quit_callback(callback_userdata);
	}
}

static void configure_audio_session() {
	enum {
		SESSION_CATEGORY_AMBIENT,
		SESSION_CATEGORY_MULTI_ROUTE,
		SESSION_CATEGORY_PLAY_AND_RECORD,
		SESSION_CATEGORY_PLAYBACK,
		SESSION_CATEGORY_RECORD,
		SESSION_CATEGORY_SOLO_AMBIENT
	};

	int setting = GLOBAL_GET("audio/general/ios/session_category");
	AVAudioSessionCategory category = AVAudioSessionCategoryAmbient;
	AVAudioSessionCategoryOptions options = 0;
	if (GLOBAL_GET("audio/general/ios/mix_with_others")) {
		options |= AVAudioSessionCategoryOptionMixWithOthers;
	}
	if (setting == SESSION_CATEGORY_MULTI_ROUTE) {
		category = AVAudioSessionCategoryMultiRoute;
	} else if (setting == SESSION_CATEGORY_PLAY_AND_RECORD) {
		category = AVAudioSessionCategoryPlayAndRecord;
		options |= AVAudioSessionCategoryOptionDefaultToSpeaker;
		options |= AVAudioSessionCategoryOptionAllowBluetoothA2DP;
		options |= AVAudioSessionCategoryOptionAllowAirPlay;
	} else if (setting == SESSION_CATEGORY_PLAYBACK) {
		category = AVAudioSessionCategoryPlayback;
	} else if (setting == SESSION_CATEGORY_RECORD) {
		category = AVAudioSessionCategoryRecord;
	} else if (setting == SESSION_CATEGORY_SOLO_AMBIENT) {
		category = AVAudioSessionCategorySoloAmbient;
	}
	[[AVAudioSession sharedInstance] setCategory:category withOptions:options error:nil];
}

static void install_observers() {
	if (lifecycle_observer) {
		return;
	}
	TwoDogLifecycleObserver *observer = [TwoDogLifecycleObserver new];
	NSNotificationCenter *center = [NSNotificationCenter defaultCenter];
	[center addObserver:observer selector:@selector(didBecomeActive:) name:UIApplicationDidBecomeActiveNotification object:nil];
	[center addObserver:observer selector:@selector(willResignActive:) name:UIApplicationWillResignActiveNotification object:nil];
	[center addObserver:observer selector:@selector(didEnterBackground:) name:UIApplicationDidEnterBackgroundNotification object:nil];
	[center addObserver:observer selector:@selector(willEnterForeground:) name:UIApplicationWillEnterForegroundNotification object:nil];
	[center addObserver:observer selector:@selector(didReceiveMemoryWarning:) name:UIApplicationDidReceiveMemoryWarningNotification object:nil];
	[center addObserver:observer selector:@selector(willTerminate:) name:UIApplicationWillTerminateNotification object:nil];
	[center addObserver:observer selector:@selector(onAudioInterruption:) name:AVAudioSessionInterruptionNotification object:[AVAudioSession sharedInstance]];
	lifecycle_observer = observer;
}

static void remove_observers() {
	if (!lifecycle_observer) {
		return;
	}
	[[NSNotificationCenter defaultCenter] removeObserver:lifecycle_observer];
	lifecycle_observer = nil;
}

int libgodot_ios_create(int p_argc, char *p_argv[], GDExtensionInitializationFunction p_init_func) {
	ERR_FAIL_COND_V_MSG(os != nullptr, (int)ERR_ALREADY_IN_USE, "The Godot engine was already created in this process.");
	ERR_FAIL_COND_V_MSG(p_argc < 1 || p_argc > 60, (int)ERR_INVALID_PARAMETER, "libgodot_ios_create needs 1 to 60 arguments.");

	CoreGlobals::global_init_func_libgodot = p_init_func;
	CoreGlobals::libgodot_host_instance = true;

	os = new OS_IOS();
	os->set_libgodot_callbacks(on_main_loop_started, on_quit_requested, nullptr);

	// Info.plist's godot_cmdline entries are appended, as the app template does.
	char *fargv[64];
	int argc = process_args(p_argc, p_argv, fargv);

	godot_init_profiler();

	Error err = Main::setup(fargv[0], argc - 1, &fargv[1], false);
	if (err != OK) {
		godot_cleanup_profiler();
		delete os;
		os = nullptr;
		CoreGlobals::global_init_func_libgodot = nullptr;
		CoreGlobals::libgodot_host_instance = false;
		return (int)err;
	}

	os->initialize_modules();
	configure_audio_session();
	install_observers();
	return 0;
}

void *libgodot_ios_view_controller(void) {
	ERR_FAIL_NULL_V_MSG(os, nullptr, "Create the Godot engine before asking for its view controller.");
	if (!view_controller) {
		view_controller = [[GDTViewController alloc] initWithNibName:nil bundle:nil];
		// The display server and the renderer find Godot's view through this class property.
		GDTAppDelegateService.viewController = view_controller;
	}
	return (__bridge void *)view_controller;
}

void libgodot_ios_set_callbacks(libgodot_ios_callback p_started, libgodot_ios_callback p_quit_requested, void *p_userdata) {
	started_callback = p_started;
	quit_callback = p_quit_requested;
	callback_userdata = p_userdata;
}

bool libgodot_ios_is_created(void) {
	return os != nullptr;
}

bool libgodot_ios_is_running(void) {
	return main_loop() != nullptr;
}

void libgodot_ios_request_quit(void) {
	DisplayServerAppleEmbedded *display_server = DisplayServerAppleEmbedded::get_singleton();
	if (display_server && main_loop()) {
		display_server->send_window_event(DisplayServerEnums::WINDOW_EVENT_CLOSE_REQUEST);
	}
}

void libgodot_ios_on_focus_in(void) {
	if (os) {
		os->on_focus_in();
	}
}

void libgodot_ios_on_focus_out(void) {
	if (os) {
		os->on_focus_out();
	}
}

void libgodot_ios_on_enter_background(void) {
	if (os) {
		os->on_enter_background();
	}
}

void libgodot_ios_on_exit_background(void) {
	if (os) {
		os->on_exit_background();
	}
}

void libgodot_ios_on_memory_warning(void) {
	if (MainLoop *loop = main_loop()) {
		loop->notification(MainLoop::NOTIFICATION_OS_MEMORY_WARNING);
	}
}

void libgodot_ios_terminate(void) {
	if (!os) {
		return;
	}
	remove_observers();
	Main::cleanup();
	godot_cleanup_profiler();
	delete os;
	os = nullptr;
	view_controller = nil;
	GDTAppDelegateService.viewController = nil;
	CoreGlobals::global_init_func_libgodot = nullptr;
	CoreGlobals::libgodot_host_instance = false;
}
