/**************************************************************************/
/*  godot_scene_delegate.mm                                               */
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

#import "godot_scene_delegate.h"

#import "app_delegate_service.h"

@implementation GDTSceneDelegate

API_AVAILABLE(ios(13.0), tvos(13.0), visionos(1.0))
static NSMutableArray<GDTSceneDelegateServiceProtocol *> *services = nil;

+ (NSArray<GDTSceneDelegateServiceProtocol *> *)services API_AVAILABLE(ios(13.0), tvos(13.0), visionos(1.0)) {
	return services;
}

+ (void)load {
	if (@available(iOS 13, tvOS 13, visionOS 1, *)) {
		services = [NSMutableArray new];
		[services addObject:[GDTAppDelegateService new]];
	}
}

+ (void)addService:(GDTSceneDelegateServiceProtocol *)service API_AVAILABLE(ios(13.0), tvos(13.0), visionos(1.0)) {
	if (!services || !service) {
		return;
	}
	[services addObject:service];
}

// MARK: Scene

- (void)scene:(UIScene *)scene willConnectToSession:(UISceneSession *)session options:(UISceneConnectionOptions *)connectionOptions API_AVAILABLE(ios(13.0), tvos(13.0), visionos(1.0)) {
	for (GDTSceneDelegateServiceProtocol *service in services) {
		if (![service respondsToSelector:_cmd]) {
			continue;
		}

		[service scene:scene willConnectToSession:session options:connectionOptions];
	}
}

// MARK: Life-Cycle

- (void)sceneDidDisconnect:(UIScene *)scene API_AVAILABLE(ios(13.0), tvos(13.0), visionos(1.0)) {
	for (GDTSceneDelegateServiceProtocol *service in services) {
		if (![service respondsToSelector:_cmd]) {
			continue;
		}

		[service sceneDidDisconnect:scene];
	}
}

- (void)sceneDidBecomeActive:(UIScene *)scene API_AVAILABLE(ios(13.0), tvos(13.0), visionos(1.0)) {
	for (GDTSceneDelegateServiceProtocol *service in services) {
		if (![service respondsToSelector:_cmd]) {
			continue;
		}

		[service sceneDidBecomeActive:scene];
	}
}

- (void)sceneWillResignActive:(UIScene *)scene API_AVAILABLE(ios(13.0), tvos(13.0), visionos(1.0)) {
	for (GDTSceneDelegateServiceProtocol *service in services) {
		if (![service respondsToSelector:_cmd]) {
			continue;
		}

		[service sceneWillResignActive:scene];
	}
}

- (void)sceneDidEnterBackground:(UIScene *)scene API_AVAILABLE(ios(13.0), tvos(13.0), visionos(1.0)) {
	for (GDTSceneDelegateServiceProtocol *service in services) {
		if (![service respondsToSelector:_cmd]) {
			continue;
		}

		[service sceneDidEnterBackground:scene];
	}
}

- (void)sceneWillEnterForeground:(UIScene *)scene API_AVAILABLE(ios(13.0), tvos(13.0), visionos(1.0)) {
	for (GDTSceneDelegateServiceProtocol *service in services) {
		if (![service respondsToSelector:_cmd]) {
			continue;
		}

		[service sceneWillEnterForeground:scene];
	}
}

@end
