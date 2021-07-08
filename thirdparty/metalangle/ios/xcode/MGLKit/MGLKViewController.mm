//
// Copyright 2019 Le Hoang Quyen. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#import "MGLKViewController+Private.h"

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <EGL/eglext_angle.h>
#include <EGL/eglplatform.h>
#include <common/apple_platform_utils.h>
#include <common/debug.h>

#import "MGLDisplay.h"
#import "MGLKView+Private.h"

@implementation MGLKViewController

#if TARGET_OS_OSX
#    include "MGLKViewController+Mac.mm"
#else
#    include "MGLKViewController+iOS.mm"
#endif

- (instancetype)init
{
    if (self = [super init])
    {
        [self constructor];
    }
    return self;
}

- (instancetype)initWithNibName:(NSString *)nibNameOrNil bundle:(NSBundle *)nibBundleOrNil
{
    if (self = [super initWithNibName:nibNameOrNil bundle:nibBundleOrNil])
    {
        [self constructor];
    }
    return self;
}

- (instancetype)initWithCoder:(NSCoder *)coder
{
    if (self = [super initWithCoder:coder])
    {
        [self constructor];
    }
    return self;
}

- (void)constructor
{
    _appWasInBackground       = YES;
    _preferredFramesPerSecond = 30;
    _pauseOnWillResignActive  = YES;
    _resumeOnDidBecomeActive  = YES;
    // not-paused corresponds to having a DisplayLink or timer active and driving the frame loop
    _isPaused                 = YES;
}

- (void)dealloc
{
    [self deallocImpl];
}

- (void)setView:(MGLKNativeView *)view
{
    [super setView:view];
    if ([view isKindOfClass:MGLKView.class])
    {
        _glView = (MGLKView *)view;
#if TARGET_OS_IOS || TARGET_OS_TV
        _glView.enableSetNeedsDisplay = NO;
#endif
        if (!_glView.delegate)
        {
            // If view has no delegate, set this controller as its delegate
            _glView.delegate = self;
        }
        // Store this object inside the view itself so that the view can notify
        // this controller about certain events such as moving to new window.
        _glView.controller = self;
    }
    else
    {
        if (_glView.delegate == self)
        {
            // Detach from old view
            _glView.delegate = nil;
        }
        if (_glView.controller == self)
        {
            _glView.controller = nil;
        }
        _glView = nil;
    }
}

- (void)setIsPaused:(BOOL)isPaused
{
    if (isPaused != _isPaused) {
        if (isPaused) {
            [self pause];
        } else {
            [self resume];
        }
    }
}

- (BOOL)paused
{
    return _isPaused;
}

- (void)mglkView:(MGLKView *)view drawInRect:(CGRect)rect
{
    // Default implementation do nothing.
}

// viewDidAppear callback
#if TARGET_OS_OSX
- (void)viewDidAppear
{
    [super viewDidAppear];
#else   // TARGET_OS_OSX
- (void)viewDidAppear:(BOOL)animated
{
    [super viewDidAppear:animated];
#endif  // TARGET_OS_OSX
    NSLog(@"MGLKViewController viewDidAppear");

    // Implementation dependent
    [self resume];

    // Register callbacks to be called when app enters/exits background
    [[NSNotificationCenter defaultCenter] addObserver:self
                                             selector:@selector(appWillPause:)
                                                 name:MGLKApplicationWillResignActiveNotification
                                               object:nil];

    [[NSNotificationCenter defaultCenter] addObserver:self
                                             selector:@selector(appDidBecomeActive:)
                                                 name:MGLKApplicationDidBecomeActiveNotification
                                               object:nil];
}

// viewDidDisappear callback
#if TARGET_OS_OSX
- (void)viewDidDisappear
{
    [super viewDidDisappear];
#else   // TARGET_OS_OSX
- (void)viewDidDisappear:(BOOL)animated
{
    [super viewDidDisappear:animated];
#endif  // TARGET_OS_OSX
    NSLog(@"MGLKViewController viewDidDisappear");
    _appWasInBackground = YES;

    // Implementation dependent
    [self pause];

    // Unregister callbacks that are called when app enters/exits background
    [[NSNotificationCenter defaultCenter] removeObserver:self
                                                    name:MGLKApplicationWillResignActiveNotification
                                                  object:nil];

    [[NSNotificationCenter defaultCenter] removeObserver:self
                                                    name:MGLKApplicationDidBecomeActiveNotification
                                                  object:nil];
}

- (void)appWillPause:(NSNotification *)note
{
    NSLog(@"MGLKViewController appWillPause:");
    if (_pauseOnWillResignActive) {
        _appWasInBackground = YES;
        [self pause];
    }
}

- (void)appDidBecomeActive:(NSNotification *)note
{
    NSLog(@"MGLKViewController appDidBecomeActive:");
    if (_resumeOnDidBecomeActive) {
        [self resume];
    }
}

- (void)handleAppWasInBackground
{
    if (!_appWasInBackground)
    {
        return;
    }
    // To avoid time jump when the app goes to background
    // for a long period of time.
    _lastUpdateTime = CACurrentMediaTime();

    _appWasInBackground = NO;
}

- (void)frameStep
{
    [self handleAppWasInBackground];

    CFTimeInterval now   = CACurrentMediaTime();
    _timeSinceLastUpdate = now - _lastUpdateTime;

    [self update];
    [_glView display];

    if (_needDisableVsync)
    {
        eglSwapInterval([MGLDisplay defaultDisplay].eglDisplay, 0);
        _needDisableVsync = NO;
    }
    else if (_needEnableVsync)
    {
        eglSwapInterval([MGLDisplay defaultDisplay].eglDisplay, 1);
        _needEnableVsync = NO;
    }

    _framesDisplayed++;
    _lastUpdateTime = now;

#if 0
    if (_timeSinceLastUpdate > 2 * _displayLink.duration)
    {
        NSLog(@"frame was jump by %fs", _timeSinceLastUpdate);
    }
#endif
}

- (void)update
{
    if (_delegate)
    {
        [_delegate mglkViewControllerUpdate:self];
    }
}

@end
