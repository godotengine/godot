//
// Copyright (c) 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#include "util/ios/IOSWindow.h"

#import <QuartzCore/CALayer.h>
#import <UIKit/UIApplication.h>
#import <UIKit/UIColor.h>
#import <UIKit/UIScreen.h>
#import <UIKit/UIView.h>
#import <UIKit/UIViewController.h>
#import <UIKit/UIWindow.h>

#include <set>

#include "common/debug.h"

static IOSWindow *gMainIOSWindow;

#pragma region AppDelegate

// AppDelegate implementation.
@interface AppDelegate : UIResponder <UIApplicationDelegate>

@end

@implementation AppDelegate

- (BOOL)application:(UIApplication *)application
    didFinishLaunchingWithOptions:(NSDictionary *)launchOptions
{
#if !TARGET_OS_TV
    [[UIApplication sharedApplication] setStatusBarHidden:YES];  // hide status bar

    // Listen to orientation change.
    [[UIDevice currentDevice] beginGeneratingDeviceOrientationNotifications];
#endif

    if (gMainIOSWindow)
    {
        gMainIOSWindow->appDidFinishLaunching();
    }

    return YES;
}

- (void)applicationWillResignActive:(UIApplication *)application
{}

- (void)applicationDidBecomeActive:(UIApplication *)application
{}

- (void)applicationWillTerminate:(UIApplication *)application
{
    if (gMainIOSWindow)
    {
        gMainIOSWindow->appWillTerminate();
    }

#if !TARGET_OS_TV
    [[UIDevice currentDevice] endGeneratingDeviceOrientationNotifications];
#endif
}

- (void)applicationDidEnterBackground:(UIApplication *)application
{
    // Handle any background procedures not related to animation here.
}

- (void)applicationWillEnterForeground:(UIApplication *)application
{
    // Handle any foreground procedures not related to animation here.
}

@end  // AppDelegate

#pragma region IOSWindowViewController

// IOSWindowViewController implementation
@interface IOSWindowViewController : UIViewController

- (void)loopIteration;

@end

@implementation IOSWindowViewController

#if !TARGET_OS_TV
- (BOOL)shouldAutorotate
{
    return YES;
}

- (UIInterfaceOrientation)preferredInterfaceOrientationForPresentation
{
    return UIInterfaceOrientationLandscapeRight;
}

- (NSUInteger)supportedInterfaceOrientations
{
    return UIInterfaceOrientationMaskLandscapeRight;
}
#endif // !TARGET_OS_TV

- (void)deviceOrientationDidChange:(NSNotification *)notification
{
    if (gMainIOSWindow)
    {
        gMainIOSWindow->deviceOrientationDidChange();
    }
}

- (void)viewDidAppear:(BOOL)animated
{
    if (gMainIOSWindow)
    {
        gMainIOSWindow->viewDidAppear();
    }
}

- (void)viewDidLayoutSubviews
{
    if (gMainIOSWindow)
    {
        gMainIOSWindow->viewDidLayoutSubviews();
    }
}

- (void)loopIteration
{
    if (gMainIOSWindow)
    {
        gMainIOSWindow->loopIteration();
    }
}

@end  // IOSWindowViewController

#pragma region IOSWindowView

// IOSWindowView implementation.
@interface IOSWindowView : UIView

@end

@implementation IOSWindowView

- (id)initWithFrame:(CGRect)frameRect
{
    if ((self = [super initWithFrame:frameRect]) != nil)
    {
#if !TARGET_OS_TV
        self.multipleTouchEnabled = YES;
#endif
    }
    return self;
}

- (BOOL)canBecomeFirstResponder
{
    return YES;
}

@end  // IOSWindowView

// IOSWindow implementation.
struct IOSWindow::Impl
{
    ~Impl()
    {
        [view release];
        view = nil;

        [viewController release];
        viewController = nil;

        [window release];
        window = nil;

        [displayLink release];
        displayLink = nil;
    }

    UIWindow *window                        = nil;
    IOSWindowView *view                     = nil;
    IOSWindowViewController *viewController = nil;
    CADisplayLink *displayLink              = nil;
};

namespace
{

std::vector<IOSWindow::Delegate> &GetRegisteredAppStartHandlers()
{
    static std::vector<IOSWindow::Delegate> sAppStartedHandlers;
    return sAppStartedHandlers;
}
}

IOSWindow::IOSWindow() : mImpl(new Impl()), mRunning(false), mLoopInitializedDelegate(nullptr) {}

IOSWindow::~IOSWindow()
{
    destroy();
}

bool IOSWindow::initialize(const std::string &name, int width, int height)
{
    ASSERT(!gMainIOSWindow);

    gMainIOSWindow = this;

    return true;
}

void IOSWindow::destroy()
{
    mRunning = false;

    mLoopInitializedDelegate = nullptr;
    mLoopIterationDelegate   = nullptr;

    ASSERT(this == gMainIOSWindow);
    gMainIOSWindow = nullptr;
}

void IOSWindow::resetNativeWindow() {}

EGLNativeWindowType IOSWindow::getNativeWindow() const
{
    return [mImpl->view layer];
}

EGLNativeDisplayType IOSWindow::getNativeDisplay() const
{
    // TODO(cwallez): implement it once we have defined what EGLNativeDisplayType is
    return static_cast<EGLNativeDisplayType>(0);
}

void IOSWindow::messageLoop()
{
    // Do nothing.
}

void IOSWindow::setMousePosition(int x, int y)
{
    // No supported.
}

bool IOSWindow::setPosition(int x, int y)
{
    // No supported.
    return true;
}

bool IOSWindow::resize(int width, int height)
{
    // No supported.
    return true;
}

void IOSWindow::setVisible(bool isVisible)
{
    if (isVisible)
    {
        mImpl->window.hidden = NO;
    }
    else
    {
        mImpl->window.hidden = YES;
    }
}

void IOSWindow::signalTestEvent()
{
    // TODO(hqle)
}

void IOSWindow::appDidFinishLaunching()
{
    NSLog(@"IOSWindow::appDidFinishLaunching()");

    // Invoke registered callbacks first.
    for (auto callback : GetRegisteredAppStartHandlers())
    {
        callback();
    }

    // Create window and make key.
    mImpl->window = [[UIWindow alloc] initWithFrame:[[UIScreen mainScreen] bounds]];

    [mImpl->window makeKeyWindow];

    // Create view.
    UIScreen *mainScreen    = [UIScreen mainScreen];
    CGRect screenRect       = [mainScreen bounds];
    CGRect screenNativeRect = [mainScreen nativeBounds];

    mScreenWidth = mWidth = screenNativeRect.size.width;
    mScreenHeight = mHeight = screenNativeRect.size.height;

    CGRect rect = CGRectMake(0, 0, screenRect.size.width, screenRect.size.height);
    mImpl->view = [[IOSWindowView alloc] initWithFrame:rect];

    mImpl->view.contentScaleFactor = mainScreen.nativeScale;

    // Create view controller.
    mImpl->viewController      = [[IOSWindowViewController alloc] init];
    mImpl->viewController.view = mImpl->view;

    [mImpl->window setRootViewController:mImpl->viewController];

    [mImpl->view becomeFirstResponder];

    // Show window.
    [mImpl->window makeKeyAndVisible];
}

void IOSWindow::viewDidLayoutSubviews() {}

void IOSWindow::viewDidAppear()
{
    NSLog(@"IOSWindow::viewDidLayoutSubviews()");

    if (mLoopInitializedDelegate)
    {
        if (mLoopInitializedDelegate())
        {
            // Stop creating rendering loop.
            stopRunning();
            return;
        }
    }

    // Start rendering loop
    mImpl->displayLink = [CADisplayLink displayLinkWithTarget:mImpl->viewController
                                                     selector:@selector(loopIteration)];
    mImpl->displayLink.preferredFramesPerSecond = 60;
    [mImpl->displayLink addToRunLoop:[NSRunLoop mainRunLoop] forMode:NSDefaultRunLoopMode];
}

void IOSWindow::appWillTerminate()
{
    stopRunning();
}

void IOSWindow::deviceOrientationDidChange() {}

void IOSWindow::stopRunning()
{
    mRunning = false;

    Event event;
    event.Type = Event::EVENT_CLOSED;
    pushEvent(event);

    [mImpl->displayLink removeFromRunLoop:[NSRunLoop mainRunLoop] forMode:NSDefaultRunLoopMode];
}

void IOSWindow::loopIteration()
{
    @autoreleasepool
    {
        if (!mRunning)
        {
            Event event;
            event.Type = Event::EVENT_CLOSED;
            pushEvent(event);

            return;
        }

        mWidth  = mImpl->view.layer.frame.size.width * mImpl->view.layer.contentsScale;
        mHeight = mImpl->view.layer.frame.size.height * mImpl->view.layer.contentsScale;

        if (mLoopIterationDelegate)
        {
            if (mLoopIterationDelegate())
            {
                stopRunning();
            }
        }
    }
}

int IOSWindow::runOwnLoop(LoopStartDelegate initDelegate, LoopDelegate loopDelegate)
{
    if (mRunning)
    {
        // Already running.
        return -1;
    }

    mRunning = true;

    mLoopInitializedDelegate = initDelegate;
    mLoopIterationDelegate   = loopDelegate;

    char *argv[1] = {nullptr};

    @autoreleasepool
    {
        return UIApplicationMain(0, argv, nil, NSStringFromClass(AppDelegate.class));
    }
}

// static
void IOSWindow::RegisterAppStartDelegate(Delegate delegate)
{
    GetRegisteredAppStartHandlers().push_back(delegate);
}

// static
OSWindow *OSWindow::New()
{
    return new IOSWindow;
}
