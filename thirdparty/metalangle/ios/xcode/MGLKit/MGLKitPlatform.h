//
// Copyright 2020 Le Hoang Quyen. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#ifndef MGLKitPlatform_h
#define MGLKitPlatform_h

#include <TargetConditionals.h>

#if TARGET_OS_IOS || TARGET_OS_MACCATALYST || TARGET_OS_TV
#    include <UIKit/UIKit.h>

@compatibility_alias MGLKNativeView UIView;
@compatibility_alias MGLKNativeViewController UIViewController;

#    define MGLKApplicationWillResignActiveNotification UIApplicationWillResignActiveNotification
#    define MGLKApplicationDidBecomeActiveNotification UIApplicationDidBecomeActiveNotification

#elif TARGET_OS_OSX
#    include <Cocoa/Cocoa.h>

@compatibility_alias MGLKNativeView NSView;
@compatibility_alias MGLKNativeViewController NSViewController;

#    define MGLKApplicationWillResignActiveNotification NSApplicationWillResignActiveNotification
#    define MGLKApplicationDidBecomeActiveNotification NSApplicationDidBecomeActiveNotification
#else
#    error "Unsupported platform"
#endif

#endif /* MGLKitPlatform_h */
