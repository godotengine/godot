/**************************************************************************/
/*  godot_notifications.mm                                                */
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

#include "godot_notifications.h"

#include <TargetConditionals.h>

#if TARGET_OS_IPHONE
#import <UIKit/UIKit.h>
#endif
#import <UserNotifications/UserNotifications.h>

GodotNotifications *GodotNotifications::singleton = nullptr;

GodotNotifications::GodotNotifications() {
	ERR_FAIL_COND(singleton != nullptr);
	singleton = this;
}

GodotNotifications::~GodotNotifications() {
	singleton = nullptr;
}

void GodotNotifications::request_notifications() {
#if TARGET_OS_IPHONE
	UNUserNotificationCenter *center = [UNUserNotificationCenter currentNotificationCenter];
	[center requestAuthorizationWithOptions:(UNAuthorizationOptionAlert + UNAuthorizationOptionSound)
						  completionHandler:^(BOOL granted, NSError *_Nullable error) {
							  if (granted) {
								  [center getNotificationSettingsWithCompletionHandler:^(UNNotificationSettings *_Nonnull settings) {
									  if (settings.authorizationStatus == UNAuthorizationStatusAuthorized) {
										  [[UIApplication sharedApplication] registerForRemoteNotifications]; // you can also set here for local notification.
									  }
								  }];
							  }
						  }];
#elif TARGET_OS_OSX
#endif
}

void GodotNotifications::_bind_methods() {
	ClassDB::bind_method(D_METHOD("request_notifications"), &GodotNotifications::request_notifications);
	ADD_SIGNAL(MethodInfo("rate_success"));
	ADD_SIGNAL(MethodInfo("rate_failed", PropertyInfo(Variant::STRING, "message")));
}
