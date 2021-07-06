//
// Copyright 2019 Le Hoang Quyen. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#ifndef MGLKViewController_Private_h
#define MGLKViewController_Private_h

#import "MGLKViewController.h"

@interface MGLKViewController () {
    __weak MGLKView *_glView;
#if TARGET_OS_OSX
    NSTimer *_displayTimer;            // Used to render with irregular framerate
    CVDisplayLinkRef _displayLink;     // Used to render in sync with display refresh rate
    dispatch_source_t _displaySource;  // Used together with displayLink
    double _currentScreenRefreshRate;
    NSWindow *_observedWindow;
#else
    CADisplayLink *_displayLink;
#endif
    CFTimeInterval _lastUpdateTime;
    BOOL _needDisableVsync;
    BOOL _needEnableVsync;

    BOOL _appWasInBackground;
}

- (void)viewDidMoveToWindow;

// Platform specific
- (void)initImpl;
- (void)deallocImpl;
- (void)pause;
- (void)resume;

// Common
- (void)frameStep;

@end

#endif /* MGLKViewController_Private_h */
