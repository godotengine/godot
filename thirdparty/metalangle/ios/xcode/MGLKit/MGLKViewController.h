//
// Copyright 2019 Le Hoang Quyen. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#import "MGLKView.h"

NS_ASSUME_NONNULL_BEGIN

@class MGLKViewController;

@protocol MGLKViewControllerDelegate <NSObject>

- (void)mglkViewControllerUpdate:(MGLKViewController *)controller;

@end

@interface MGLKViewController : MGLKNativeViewController <MGLKViewDelegate>

@property(nonatomic, assign) IBOutlet id<MGLKViewControllerDelegate> delegate;

// The default value is 30.
// On iOS:
//  - Setting to 0 or 1 will sync the framerate with display's refresh rate
// On macOS:
//  - Setting to 1 will sync the framerate with display's refresh rate
//  - Setting to 0 will display the frames as fast as possible.
@property(nonatomic) NSInteger preferredFramesPerSecond;

@property(nonatomic, readonly) NSInteger framesDisplayed;
@property(nonatomic, readonly) NSTimeInterval timeSinceLastUpdate;

@property(nonatomic) BOOL isPaused;
@property(nonatomic, setter=setIsPaused:) BOOL paused;
@property(nonatomic) BOOL pauseOnWillResignActive;
@property(nonatomic) BOOL resumeOnDidBecomeActive;

@property(weak, nonatomic, readonly) MGLKView *glView;

@end

NS_ASSUME_NONNULL_END
