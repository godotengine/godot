//
// Copyright 2019 Le Hoang Quyen. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

- (void)initImpl {}

- (void)releaseTimer
{
    if (_displayLink)
    {
        CVDisplayLinkRelease(_displayLink);
        _displayLink = nullptr;
    }
    if (_displaySource)
    {
        dispatch_source_cancel(_displaySource);
        _displaySource = nullptr;
    }
    if (_displayTimer)
    {
        [_displayTimer invalidate];
        _displayTimer = nil;
    }
}

- (void)deallocImpl
{
    NSLog(@"MGLKViewController deallocImpl");
    [self releaseTimer];
}

- (void)viewDidMoveToWindow
{
    NSLog(@"MGLKViewController viewDidMoveToWindow");
    if (self.view.window)
    {
        // Obtain current window's screen refresh rate.
        CGDirectDisplayID displayID =
            (CGDirectDisplayID)[self.view.window.screen
                                    .deviceDescription[@"NSScreenNumber"] unsignedIntegerValue];

        CGDisplayModeRef displayModeRef = CGDisplayCopyDisplayMode(displayID);
        if (displayModeRef)
        {
            _currentScreenRefreshRate = CGDisplayModeGetRefreshRate(displayModeRef);
        }
        CGDisplayModeRelease(displayModeRef);

        // Call resume to reset display link's window
        [self pause];
        [self resume];

        // Register callback to be called when this window is closed.
        _observedWindow = self.view.window;
        [[NSNotificationCenter defaultCenter] addObserver:self
                                                 selector:@selector(windowWillClose:)
                                                     name:NSWindowWillCloseNotification
                                                   object:self.view.window];
    }
    else
    {
        // View is removed from window
        [self releaseTimer];

        // Unregister window closed callback.
        if (_observedWindow)
        {
            [[NSNotificationCenter defaultCenter] removeObserver:self
                                                            name:NSWindowWillCloseNotification
                                                          object:_observedWindow];
            _observedWindow = nil;
        }
    }
}

- (void)windowWillClose:(NSNotification *)notification
{
    NSLog(@"MGLKViewController windowWillClose:");
    [self releaseTimer];
}

static CVReturn CVFrameDisplayCallback(CVDisplayLinkRef displayLink,
                                       const CVTimeStamp *now,
                                       const CVTimeStamp *outputTime,
                                       CVOptionFlags flagsIn,
                                       CVOptionFlags *flagsOut,
                                       void *displayLinkContext)
{
    // 'CVFrameDisplayCallback' is always called on a secondary thread.  Merge the dispatch source
    // setup for the main queue so that rendering occurs on the main thread
    __weak dispatch_source_t source = (__bridge dispatch_source_t)displayLinkContext;
    dispatch_source_merge_data(source, 1);

    return kCVReturnSuccess;
}

- (void)setPreferredFramesPerSecond:(NSInteger)preferredFramesPerSecond
{
    _preferredFramesPerSecond = preferredFramesPerSecond;

    [self pause];
    [self resume];
}

- (void)pause
{
    if (_isPaused)
    {
        return;
    }
    NSLog(@"MGLKViewController pause");

    if (_displayLink)
    {
        CVDisplayLinkStop(_displayLink);
    }
    if (_displayTimer)
    {
        [_displayTimer invalidate];
        _displayTimer = nil;
    }

    _isPaused = YES;
}

- (void)resume
{
    if (!_isPaused)
    {
        return;
    }

    [self pause];
    NSLog(@"MGLKViewController resume");

    if (!_glView)
    {
        return;
    }

    if (_preferredFramesPerSecond == 1 ||
        (_currentScreenRefreshRate &&
         fabs(_preferredFramesPerSecond - _currentScreenRefreshRate) < 0.00001))
    {
        NSWindow *window = _glView.window;
        if (!window)
        {
            return;
        }
        // The CVDisplayLink callback, CVFrameDisplayCallback, never executes
        // on the main thread. To execute rendering on the main thread, create
        // a dispatch source using the main queue (the main thread).
        // CVFrameDisplayCallback merges this dispatch source in each call
        // to execute rendering on the main thread.
        if (!_displaySource)
        {
            _displaySource = dispatch_source_create(DISPATCH_SOURCE_TYPE_DATA_ADD, 0, 0,
                                                    dispatch_get_main_queue());
            __weak MGLKViewController *weakSelf = self;
            dispatch_source_set_event_handler(_displaySource, ^() {
              [weakSelf frameStep];
            });
            dispatch_resume(_displaySource);
        }

        // Sync to display refresh rate using CVDisplayLink
        _needEnableVsync = YES;
        if (!_displayLink)
        {
            CVDisplayLinkCreateWithActiveCGDisplays(&_displayLink);
            CVDisplayLinkSetOutputCallback(_displayLink, CVFrameDisplayCallback,
                                           (__bridge void *)_displaySource);
        }
        CGDirectDisplayID displayID =
            (CGDirectDisplayID)[window.screen
                                    .deviceDescription[@"NSScreenNumber"] unsignedIntegerValue];
        CVDisplayLinkSetCurrentCGDisplay(_displayLink, displayID);

        CVDisplayLinkStart(_displayLink);
    }
    else
    {
        // Render the frames without in sync with refresh rate.
        _needDisableVsync = YES;

        ASSERT(!_displayTimer);
        NSTimeInterval frameInterval =
            (_preferredFramesPerSecond <= 0) ? 0 : (1.0 / _preferredFramesPerSecond);
        _displayTimer = [NSTimer timerWithTimeInterval:frameInterval
                                                target:self
                                              selector:@selector(frameStep)
                                              userInfo:nil
                                               repeats:YES];
        [[NSRunLoop currentRunLoop] addTimer:_displayTimer forMode:NSDefaultRunLoopMode];
        [[NSRunLoop currentRunLoop] addTimer:_displayTimer forMode:NSModalPanelRunLoopMode];
    }

    _isPaused = NO;
}
