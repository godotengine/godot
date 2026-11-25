/**************************************************************************/
/*  godot_renderer.mm                                                     */
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

#import "godot_renderer.h"

#include "core/config/project_settings.h"
#import "drivers/apple_embedded/display_server_apple_embedded.h"
#import "drivers/apple_embedded/os_apple_embedded.h"
#include "main/main.h"

// The engine is set up once per process, not once per renderer. The system can give a running app a new window
// (iPadOS disconnects a background scene to reclaim memory and connects a new one when the user returns), and each
// window gets a new renderer, which would otherwise run Main::setup2() again over the live servers.
static bool has_called_project_data_set_up = false;
static bool has_started_main = false;
static bool has_finished_set_up = false;

@implementation GDTRenderer

- (BOOL)hasFinishedSetup {
	return has_finished_set_up;
}

- (BOOL)setUp {
	if (has_finished_set_up) {
		return NO;
	}

	if (!OS::get_singleton()) {
		exit(0);
	}

	if (!has_called_project_data_set_up) {
		[self setUpProjectDataShowingBootLogo:YES];
	}

	if (!has_started_main) {
		[self startMain];
	}

	has_finished_set_up = true;

	return NO;
}

- (void)setUpProjectDataShowingBootLogo:(BOOL)p_show_boot_logo {
	has_called_project_data_set_up = true;
	safeDispatchSyncToMain(^{
		Main::setup2(p_show_boot_logo);

		// this might be necessary before here
		NSDictionary *dict = [[NSBundle mainBundle] infoDictionary];
		for (NSString *key in dict) {
			NSObject *value = [dict objectForKey:key];
			String ukey = String::utf8([key UTF8String]);

			// we need a NSObject to Variant conversor

			if ([value isKindOfClass:[NSString class]]) {
				NSString *str = (NSString *)value;
				String uval = String::utf8([str UTF8String]);

				ProjectSettings::get_singleton()->set("Info.plist/" + ukey, uval);

			} else if ([value isKindOfClass:[NSNumber class]]) {
				NSNumber *n = (NSNumber *)value;
				double dval = [n doubleValue];

				ProjectSettings::get_singleton()->set("Info.plist/" + ukey, dval);
			}
			// do stuff
		}
	});
}

- (void)startMain {
	has_started_main = true;
	safeDispatchSyncToMain(^{
		OS_AppleEmbedded::get_singleton()->start();
	});
}

@end
