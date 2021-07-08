//
// Copyright 2019 Le Hoang Quyen. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#ifndef MGLKitCommon_h
#define MGLKitCommon_h

#import <Foundation/Foundation.h>

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <EGL/eglext_angle.h>
#include <EGL/eglplatform.h>

@interface MGLDisplay : NSObject

@property(nonatomic, readonly) EGLDisplay eglDisplay;

+ (MGLDisplay *)defaultDisplay;

@end

#endif /* MGLKitCommon_h */
