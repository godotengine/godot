//
// Copyright 2019 Le Hoang Quyen. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#import "MGLDisplay.h"

namespace
{
void Throw(NSString *msg)
{
    [NSException raise:@"MGLSurfaceException" format:@"%@", msg];
}
}

// EGLDisplayHolder
@interface EGLDisplayHolder : NSObject
@property(nonatomic) EGLDisplay eglDisplay;
@end

@implementation EGLDisplayHolder

- (id)init
{
    if (self = [super init])
    {
        // Init display
        EGLAttrib displayAttribs[] = {EGL_NONE};
        _eglDisplay = eglGetPlatformDisplay(EGL_PLATFORM_ANGLE_ANGLE, nullptr, displayAttribs);
        if (_eglDisplay == EGL_NO_DISPLAY)
        {
            Throw(@"Failed To call eglGetPlatformDisplay()");
        }
        if (!eglInitialize(_eglDisplay, NULL, NULL))
        {
            Throw(@"Failed To call eglInitialize()");
        }
    }

    return self;
}

- (void)dealloc
{
    if (_eglDisplay != EGL_NO_DISPLAY)
    {
        eglTerminate(_eglDisplay);
        _eglDisplay = EGL_NO_DISPLAY;
    }
}

@end

static EGLDisplayHolder *gGlobalDisplayHolder;
static MGLDisplay *gDefaultDisplay;

// MGLDisplay implementation
@interface MGLDisplay () {
    EGLDisplayHolder *_eglDisplayHolder;
}

@end

@implementation MGLDisplay

+ (MGLDisplay *)defaultDisplay
{
    if (!gDefaultDisplay)
    {
        gDefaultDisplay = [[MGLDisplay alloc] init];
    }
    return gDefaultDisplay;
}

- (id)init
{
    if (self = [super init])
    {
        if (!gGlobalDisplayHolder)
        {
            gGlobalDisplayHolder = [[EGLDisplayHolder alloc] init];
        }
        _eglDisplayHolder = gGlobalDisplayHolder;
        _eglDisplay       = _eglDisplayHolder.eglDisplay;
    }

    return self;
}

@end
