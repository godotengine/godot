/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2025 Sam Lantinga <slouken@libsdl.org>

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/
#include "SDL_internal.h"

#ifdef SDL_VIDEO_DRIVER_UIKIT

#include "../SDL_sysvideo.h"
#include "../SDL_pixels_c.h"
#include "../../events/SDL_events_c.h"

#include "SDL_uikitvideo.h"
#include "SDL_uikitevents.h"
#include "SDL_uikitmodes.h"
#include "SDL_uikitwindow.h"
#include "SDL_uikitappdelegate.h"
#include "SDL_uikitview.h"
#include "SDL_uikitopenglview.h"

#include <Foundation/Foundation.h>

@implementation SDL_UIKitWindowData

@synthesize uiwindow;
@synthesize viewcontroller;
@synthesize views;

- (instancetype)init
{
    if ((self = [super init])) {
        views = [NSMutableArray new];
    }

    return self;
}

@end

@interface SDL_uikitwindow : UIWindow

- (void)layoutSubviews;

@end

@implementation SDL_uikitwindow

- (void)layoutSubviews
{
#ifndef SDL_PLATFORM_VISIONOS
    // Workaround to fix window orientation issues in iOS 8.
    /* As of July 1 2019, I haven't been able to reproduce any orientation
     * issues with this disabled on iOS 12. The issue this is meant to fix might
     * only happen on iOS 8, or it might have been fixed another way with other
     * code... This code prevents split view (iOS 9+) from working on iPads, so
     * we want to avoid using it if possible. */
    if (!UIKit_IsSystemVersionAtLeast(9.0)) {
        self.frame = self.screen.bounds;
    }
#endif
    [super layoutSubviews];
}

@end

static bool SetupWindowData(SDL_VideoDevice *_this, SDL_Window *window, UIWindow *uiwindow, bool created)
{
    SDL_VideoDisplay *display = SDL_GetVideoDisplayForWindow(window);
    SDL_UIKitDisplayData *displaydata = (__bridge SDL_UIKitDisplayData *)display->internal;
    SDL_uikitview *view;

#ifdef SDL_PLATFORM_VISIONOS
    CGRect frame = UIKit_ComputeViewFrame(window);
#else
    CGRect frame = UIKit_ComputeViewFrame(window, displaydata.uiscreen);
#endif

    int width = (int)frame.size.width;
    int height = (int)frame.size.height;

    SDL_UIKitWindowData *data = [[SDL_UIKitWindowData alloc] init];
    if (!data) {
        return SDL_OutOfMemory();
    }

    window->internal = (SDL_WindowData *)CFBridgingRetain(data);

    data.uiwindow = uiwindow;

#ifndef SDL_PLATFORM_VISIONOS
    if (displaydata.uiscreen != [UIScreen mainScreen]) {
        window->flags &= ~SDL_WINDOW_RESIZABLE;   // window is NEVER resizable
        window->flags &= ~SDL_WINDOW_INPUT_FOCUS; // never has input focus
        window->flags |= SDL_WINDOW_BORDERLESS;   // never has a status bar.
    }
#endif

#if !defined(SDL_PLATFORM_TVOS) && !defined(SDL_PLATFORM_VISIONOS)
    if (displaydata.uiscreen == [UIScreen mainScreen]) {
        NSUInteger orients = UIKit_GetSupportedOrientations(window);
        BOOL supportsLandscape = (orients & UIInterfaceOrientationMaskLandscape) != 0;
        BOOL supportsPortrait = (orients & (UIInterfaceOrientationMaskPortrait | UIInterfaceOrientationMaskPortraitUpsideDown)) != 0;

        // Make sure the width/height are oriented correctly
        if ((width > height && !supportsLandscape) || (height > width && !supportsPortrait)) {
            int temp = width;
            width = height;
            height = temp;
        }
    }
#endif // !SDL_PLATFORM_TVOS

#if 0 // Don't set the x/y position, it's already placed on a display
    window->x = 0;
    window->y = 0;
#endif
    window->w = width;
    window->h = height;

    /* The View Controller will handle rotating the view when the device
     * orientation changes. This will trigger resize events, if appropriate. */
    data.viewcontroller = [[SDL_uikitviewcontroller alloc] initWithSDLWindow:window];

    /* The window will initially contain a generic view so resizes, touch events,
     * etc. can be handled without an active OpenGL view/context. */
    view = [[SDL_uikitview alloc] initWithFrame:frame];

    /* Sets this view as the controller's view, and adds the view to the window
     * hierarchy. */
    [view setSDLWindow:window];

    SDL_PropertiesID props = SDL_GetWindowProperties(window);
    SDL_SetPointerProperty(props, SDL_PROP_WINDOW_UIKIT_WINDOW_POINTER, (__bridge void *)data.uiwindow);
    SDL_SetNumberProperty(props, SDL_PROP_WINDOW_UIKIT_METAL_VIEW_TAG_NUMBER, SDL_METALVIEW_TAG);

    return true;
}

bool UIKit_CreateWindow(SDL_VideoDevice *_this, SDL_Window *window, SDL_PropertiesID create_props)
{
    @autoreleasepool {
        SDL_VideoDisplay *display = SDL_GetVideoDisplayForWindow(window);
        SDL_UIKitDisplayData *data = (__bridge SDL_UIKitDisplayData *)display->internal;
        SDL_Window *other;

        // We currently only handle a single window per display on iOS
        for (other = _this->windows; other; other = other->next) {
            if (other != window && SDL_GetVideoDisplayForWindow(other) == display) {
                return SDL_SetError("Only one window allowed per display.");
            }
        }

        /* If monitor has a resolution of 0x0 (hasn't been explicitly set by the
         * user, so it's in standby), try to force the display to a resolution
         * that most closely matches the desired window size. */
#if !defined(SDL_PLATFORM_TVOS) && !defined(SDL_PLATFORM_VISIONOS)
        const CGSize origsize = data.uiscreen.currentMode.size;
        if ((origsize.width == 0.0f) && (origsize.height == 0.0f)) {
            SDL_DisplayMode bestmode;
            bool include_high_density_modes = false;
            if (window->flags & SDL_WINDOW_HIGH_PIXEL_DENSITY) {
                include_high_density_modes = true;
            }
            if (SDL_GetClosestFullscreenDisplayMode(display->id, window->w, window->h, 0.0f, include_high_density_modes, &bestmode)) {
                SDL_UIKitDisplayModeData *modedata = (__bridge SDL_UIKitDisplayModeData *)bestmode.internal;
                [data.uiscreen setCurrentMode:modedata.uiscreenmode];

                /* desktop_mode doesn't change here (the higher level will
                 * use it to set all the screens back to their defaults
                 * upon window destruction, SDL_Quit(), etc. */
                SDL_SetCurrentDisplayMode(display, &bestmode);
            }
        }

        if (data.uiscreen == [UIScreen mainScreen]) {
            if (window->flags & (SDL_WINDOW_FULLSCREEN | SDL_WINDOW_BORDERLESS)) {
                [UIApplication sharedApplication].statusBarHidden = YES;
            } else {
                [UIApplication sharedApplication].statusBarHidden = NO;
            }
        }
#endif // !SDL_PLATFORM_TVOS

        // ignore the size user requested, and make a fullscreen window
        // !!! FIXME: can we have a smaller view?
#ifdef SDL_PLATFORM_VISIONOS
        UIWindow *uiwindow = [[SDL_uikitwindow alloc] initWithFrame:CGRectMake(window->x, window->y, window->w, window->h)];
#else
        UIWindow *uiwindow = [[SDL_uikitwindow alloc] initWithFrame:data.uiscreen.bounds];
#endif

        // put the window on an external display if appropriate.
#ifndef SDL_PLATFORM_VISIONOS
        if (data.uiscreen != [UIScreen mainScreen]) {
            [uiwindow setScreen:data.uiscreen];
        }
#endif

        if (!SetupWindowData(_this, window, uiwindow, true)) {
            return false;
        }
    }

    return true;
}

void UIKit_SetWindowTitle(SDL_VideoDevice *_this, SDL_Window *window)
{
    @autoreleasepool {
        SDL_UIKitWindowData *data = (__bridge SDL_UIKitWindowData *)window->internal;
        data.viewcontroller.title = @(window->title);
    }
}

void UIKit_ShowWindow(SDL_VideoDevice *_this, SDL_Window *window)
{
    @autoreleasepool {
        SDL_UIKitWindowData *data = (__bridge SDL_UIKitWindowData *)window->internal;
        [data.uiwindow makeKeyAndVisible];

        // Make this window the current mouse focus for touch input
        SDL_VideoDisplay *display = SDL_GetVideoDisplayForWindow(window);
        SDL_UIKitDisplayData *displaydata = (__bridge SDL_UIKitDisplayData *)display->internal;
#ifndef SDL_PLATFORM_VISIONOS
        if (displaydata.uiscreen == [UIScreen mainScreen])
#endif
        {
            SDL_SetMouseFocus(window);
            SDL_SetKeyboardFocus(window);
        }
    }
}

void UIKit_HideWindow(SDL_VideoDevice *_this, SDL_Window *window)
{
    @autoreleasepool {
        SDL_UIKitWindowData *data = (__bridge SDL_UIKitWindowData *)window->internal;
        data.uiwindow.hidden = YES;
    }
}

void UIKit_RaiseWindow(SDL_VideoDevice *_this, SDL_Window *window)
{
#if defined(SDL_VIDEO_OPENGL_ES) || defined(SDL_VIDEO_OPENGL_ES2)
    /* We don't currently offer a concept of "raising" the SDL window, since
     * we only allow one per display, in the iOS fashion.
     * However, we use this entry point to rebind the context to the view
     * during OnWindowRestored processing. */
    _this->GL_MakeCurrent(_this, _this->current_glwin, _this->current_glctx);
#endif
}

static void UIKit_UpdateWindowBorder(SDL_VideoDevice *_this, SDL_Window *window)
{
    SDL_UIKitWindowData *data = (__bridge SDL_UIKitWindowData *)window->internal;
    SDL_uikitviewcontroller *viewcontroller = data.viewcontroller;

#if !defined(SDL_PLATFORM_TVOS) && !defined(SDL_PLATFORM_VISIONOS)
    if (data.uiwindow.screen == [UIScreen mainScreen]) {
        if (window->flags & (SDL_WINDOW_FULLSCREEN | SDL_WINDOW_BORDERLESS)) {
            [UIApplication sharedApplication].statusBarHidden = YES;
        } else {
            [UIApplication sharedApplication].statusBarHidden = NO;
        }

        [viewcontroller setNeedsStatusBarAppearanceUpdate];
    }

    // Update the view's frame to account for the status bar change.
    viewcontroller.view.frame = UIKit_ComputeViewFrame(window, data.uiwindow.screen);
#endif // !SDL_PLATFORM_TVOS

#ifdef SDL_IPHONE_KEYBOARD
    // Make sure the view is offset correctly when the keyboard is visible.
    [viewcontroller updateKeyboard];
#endif

    [viewcontroller.view setNeedsLayout];
    [viewcontroller.view layoutIfNeeded];
}

void UIKit_SetWindowBordered(SDL_VideoDevice *_this, SDL_Window *window, bool bordered)
{
    @autoreleasepool {
        if (bordered) {
            window->flags &= ~SDL_WINDOW_BORDERLESS;
        } else {
            window->flags |= SDL_WINDOW_BORDERLESS;
        }
        UIKit_UpdateWindowBorder(_this, window);
    }
}

SDL_FullscreenResult UIKit_SetWindowFullscreen(SDL_VideoDevice *_this, SDL_Window *window, SDL_VideoDisplay *display, SDL_FullscreenOp fullscreen)
{
    @autoreleasepool {
        SDL_SendWindowEvent(window, fullscreen ? SDL_EVENT_WINDOW_ENTER_FULLSCREEN : SDL_EVENT_WINDOW_LEAVE_FULLSCREEN, 0, 0);
        UIKit_UpdateWindowBorder(_this, window);
    }
    return SDL_FULLSCREEN_SUCCEEDED;
}

void UIKit_UpdatePointerLock(SDL_VideoDevice *_this, SDL_Window *window)
{
#ifndef SDL_PLATFORM_TVOS
    @autoreleasepool {
        SDL_UIKitWindowData *data = (__bridge SDL_UIKitWindowData *)window->internal;
        SDL_uikitviewcontroller *viewcontroller = data.viewcontroller;
        if (@available(iOS 14.0, *)) {
            [viewcontroller setNeedsUpdateOfPrefersPointerLocked];
        }
    }
#endif // !SDL_PLATFORM_TVOS
}

void UIKit_DestroyWindow(SDL_VideoDevice *_this, SDL_Window *window)
{
    @autoreleasepool {
        if (window->internal != NULL) {
            SDL_UIKitWindowData *data = (__bridge SDL_UIKitWindowData *)window->internal;
            NSArray *views = nil;

            [data.viewcontroller stopAnimation];

            /* Detach all views from this window. We use a copy of the array
             * because setSDLWindow will remove the object from the original
             * array, which would be undesirable if we were iterating over it. */
            views = [data.views copy];
            for (SDL_uikitview *view in views) {
                [view setSDLWindow:NULL];
            }

            /* iOS may still hold a reference to the window after we release it.
             * We want to make sure the SDL view controller isn't accessed in
             * that case, because it would contain an invalid pointer to the old
             * SDL window. */
            data.uiwindow.rootViewController = nil;
            data.uiwindow.hidden = YES;

            CFRelease(window->internal);
            window->internal = NULL;
        }
    }
}

void UIKit_GetWindowSizeInPixels(SDL_VideoDevice *_this, SDL_Window *window, int *w, int *h)
{
    @autoreleasepool {
        SDL_UIKitWindowData *windata = (__bridge SDL_UIKitWindowData *)window->internal;
        UIView *view = windata.viewcontroller.view;
        CGSize size = view.bounds.size;
        CGFloat scale = 1.0;


        if (window->flags & SDL_WINDOW_HIGH_PIXEL_DENSITY) {
#ifndef SDL_PLATFORM_VISIONOS
            scale = windata.uiwindow.screen.nativeScale;
#else
            scale = 2.0;
#endif
        }


        /* Integer truncation of fractional values matches SDL_uikitmetalview and
         * SDL_uikitopenglview. */
        *w = (int)(size.width * scale);
        *h = (int)(size.height * scale);
    }
}

#ifndef SDL_PLATFORM_TVOS
NSUInteger
UIKit_GetSupportedOrientations(SDL_Window *window)
{
    const char *hint = SDL_GetHint(SDL_HINT_ORIENTATIONS);
    NSUInteger validOrientations = UIInterfaceOrientationMaskAll;
    NSUInteger orientationMask = 0;

    @autoreleasepool {
        SDL_UIKitWindowData *data = (__bridge SDL_UIKitWindowData *)window->internal;
        UIApplication *app = [UIApplication sharedApplication];

        /* Get all possible valid orientations. If the app delegate doesn't tell
         * us, we get the orientations from Info.plist via UIApplication. */
        if ([app.delegate respondsToSelector:@selector(application:supportedInterfaceOrientationsForWindow:)]) {
            validOrientations = [app.delegate application:app supportedInterfaceOrientationsForWindow:data.uiwindow];
        } else {
            validOrientations = [app supportedInterfaceOrientationsForWindow:data.uiwindow];
        }

        if (hint != NULL) {
            NSArray *orientations = [@(hint) componentsSeparatedByString:@" "];

            if ([orientations containsObject:@"LandscapeLeft"]) {
                orientationMask |= UIInterfaceOrientationMaskLandscapeLeft;
            }
            if ([orientations containsObject:@"LandscapeRight"]) {
                orientationMask |= UIInterfaceOrientationMaskLandscapeRight;
            }
            if ([orientations containsObject:@"Portrait"]) {
                orientationMask |= UIInterfaceOrientationMaskPortrait;
            }
            if ([orientations containsObject:@"PortraitUpsideDown"]) {
                orientationMask |= UIInterfaceOrientationMaskPortraitUpsideDown;
            }
        }

        if (orientationMask == 0 && (window->flags & SDL_WINDOW_RESIZABLE)) {
            // any orientation is okay.
            orientationMask = UIInterfaceOrientationMaskAll;
        }

        if (orientationMask == 0) {
            if (window->floating.w >= window->floating.h) {
                orientationMask |= UIInterfaceOrientationMaskLandscape;
            }
            if (window->floating.h >= window->floating.w) {
                orientationMask |= (UIInterfaceOrientationMaskPortrait | UIInterfaceOrientationMaskPortraitUpsideDown);
            }
        }

        // Don't allow upside-down orientation on phones, so answering calls is in the natural orientation
        if ([UIDevice currentDevice].userInterfaceIdiom == UIUserInterfaceIdiomPhone) {
            orientationMask &= ~UIInterfaceOrientationMaskPortraitUpsideDown;
        }

        /* If none of the specified orientations are actually supported by the
         * app, we'll revert to what the app supports. An exception would be
         * thrown by the system otherwise. */
        if ((validOrientations & orientationMask) == 0) {
            orientationMask = validOrientations;
        }
    }

    return orientationMask;
}
#endif // !SDL_PLATFORM_TVOS

bool SDL_SetiOSAnimationCallback(SDL_Window *window, int interval, SDL_iOSAnimationCallback callback, void *callbackParam)
{
    if (!window || !window->internal) {
        return SDL_SetError("Invalid window");
    }

    @autoreleasepool {
        SDL_UIKitWindowData *data = (__bridge SDL_UIKitWindowData *)window->internal;
        [data.viewcontroller setAnimationCallback:interval
                                         callback:callback
                                    callbackParam:callbackParam];
    }

    return true;
}

#endif // SDL_VIDEO_DRIVER_UIKIT
