//
// Copyright 2019 Le Hoang Quyen. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

- (void)initImpl {}
- (void)deallocImpl {}
- (void)viewDidMoveToWindow {}

- (void)viewDidLoad
{
    NSLog(@"MGLKViewController viewDidLoad");
    [super viewDidLoad];
}

- (void)setPreferredFramesPerSecond:(NSInteger)preferredFramesPerSecond
{
    _preferredFramesPerSecond = preferredFramesPerSecond;
    if (_displayLink)
    {
        if (ANGLE_APPLE_AVAILABLE_CI(13.0, 10.0))
        {
            _displayLink.preferredFramesPerSecond = _preferredFramesPerSecond;
        }
        else
        {
            _displayLink.frameInterval = 60 / std::max<NSInteger>(_preferredFramesPerSecond, 1);
        }
    }
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
        [_displayLink removeFromRunLoop:[NSRunLoop mainRunLoop] forMode:NSDefaultRunLoopMode];
        _displayLink = nil;
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

    if (!_displayLink)
    {
        _displayLink = [CADisplayLink displayLinkWithTarget:self selector:@selector(frameStep)];
        if (ANGLE_APPLE_AVAILABLE_CI(13.0, 10.0))
        {
            _displayLink.preferredFramesPerSecond = _preferredFramesPerSecond == 1 ? 0 : _preferredFramesPerSecond;
        }
        else
        {
            if (_preferredFramesPerSecond <= 1)
            {
                _displayLink.frameInterval = 1;
            }
            else {
                _displayLink.frameInterval = 60 / _preferredFramesPerSecond;
            }
        }
    }

    [_displayLink addToRunLoop:[NSRunLoop mainRunLoop] forMode:NSDefaultRunLoopMode];

    _isPaused = NO;
}
