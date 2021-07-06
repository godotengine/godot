//
// Copyright 2019 Le Hoang Quyen. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#ifndef MGLContext_Private_h
#define MGLContext_Private_h

#import "MGLContext.h"
#import "MGLDisplay.h"

#include <EGL/egl.h>

@interface MGLContext () {
    // ANGLE won't allow context to be current without surface.
    // Create a dummy surface for it using this dummy layer.
    CALayer *_dummyLayer;
    EGLSurface _dummySurface;
    MGLRenderingAPI _renderingApi;
    MGLDisplay *_display;
}

@property(nonatomic, readonly) EGLContext eglContext;

@end

#endif /* MGLContext_Private_h */
