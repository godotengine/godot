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

#import <UIKit/UIKit.h>

#include "../SDL_sysvideo.h"
#include "../SDL_pixels_c.h"
#include "../../events/SDL_events_c.h"

#include "SDL_uikitvideo.h"
#include "SDL_uikitevents.h"
#include "SDL_uikitmodes.h"
#include "SDL_uikitwindow.h"
#include "SDL_uikitopengles.h"
#include "SDL_uikitclipboard.h"
#include "SDL_uikitvulkan.h"
#include "SDL_uikitmetalview.h"
#include "SDL_uikitmessagebox.h"
#include "SDL_uikitpen.h"

#define UIKITVID_DRIVER_NAME "uikit"

@implementation SDL_UIKitVideoData

@end

// Initialization/Query functions
static bool UIKit_VideoInit(SDL_VideoDevice *_this);
static void UIKit_VideoQuit(SDL_VideoDevice *_this);

// DUMMY driver bootstrap functions

static void UIKit_DeleteDevice(SDL_VideoDevice *device)
{
    @autoreleasepool {
        if (device->internal){
            CFRelease(device->internal);
        }
        SDL_free(device);
    }
}

static SDL_VideoDevice *UIKit_CreateDevice(void)
{
    @autoreleasepool {
        SDL_VideoDevice *device;
        SDL_UIKitVideoData *data;

        // Initialize all variables that we clean on shutdown
        device = (SDL_VideoDevice *)SDL_calloc(1, sizeof(SDL_VideoDevice));
        if (!device) {
            return NULL;
        }

        data = [SDL_UIKitVideoData new];

        device->internal = (SDL_VideoData *)CFBridgingRetain(data);
        device->system_theme = UIKit_GetSystemTheme();

        // Set the function pointers
        device->VideoInit = UIKit_VideoInit;
        device->VideoQuit = UIKit_VideoQuit;
        device->GetDisplayModes = UIKit_GetDisplayModes;
        device->SetDisplayMode = UIKit_SetDisplayMode;
        device->PumpEvents = UIKit_PumpEvents;
        device->SuspendScreenSaver = UIKit_SuspendScreenSaver;
        device->CreateSDLWindow = UIKit_CreateWindow;
        device->SetWindowTitle = UIKit_SetWindowTitle;
        device->ShowWindow = UIKit_ShowWindow;
        device->HideWindow = UIKit_HideWindow;
        device->RaiseWindow = UIKit_RaiseWindow;
        device->SetWindowBordered = UIKit_SetWindowBordered;
        device->SetWindowFullscreen = UIKit_SetWindowFullscreen;
        device->DestroyWindow = UIKit_DestroyWindow;
        device->GetDisplayUsableBounds = UIKit_GetDisplayUsableBounds;
        device->GetWindowSizeInPixels = UIKit_GetWindowSizeInPixels;

#ifdef SDL_IPHONE_KEYBOARD
        device->HasScreenKeyboardSupport = UIKit_HasScreenKeyboardSupport;
        device->StartTextInput = UIKit_StartTextInput;
        device->StopTextInput = UIKit_StopTextInput;
        device->SetTextInputProperties = UIKit_SetTextInputProperties;
        device->IsScreenKeyboardShown = UIKit_IsScreenKeyboardShown;
        device->UpdateTextInputArea = UIKit_UpdateTextInputArea;
#endif

        device->SetClipboardText = UIKit_SetClipboardText;
        device->GetClipboardText = UIKit_GetClipboardText;
        device->HasClipboardText = UIKit_HasClipboardText;

        // OpenGL (ES) functions
#if defined(SDL_VIDEO_OPENGL_ES) || defined(SDL_VIDEO_OPENGL_ES2)
        device->GL_MakeCurrent = UIKit_GL_MakeCurrent;
        device->GL_SwapWindow = UIKit_GL_SwapWindow;
        device->GL_CreateContext = UIKit_GL_CreateContext;
        device->GL_DestroyContext = UIKit_GL_DestroyContext;
        device->GL_GetProcAddress = UIKit_GL_GetProcAddress;
        device->GL_LoadLibrary = UIKit_GL_LoadLibrary;
#endif
        device->free = UIKit_DeleteDevice;

#ifdef SDL_VIDEO_VULKAN
        device->Vulkan_LoadLibrary = UIKit_Vulkan_LoadLibrary;
        device->Vulkan_UnloadLibrary = UIKit_Vulkan_UnloadLibrary;
        device->Vulkan_GetInstanceExtensions = UIKit_Vulkan_GetInstanceExtensions;
        device->Vulkan_CreateSurface = UIKit_Vulkan_CreateSurface;
        device->Vulkan_DestroySurface = UIKit_Vulkan_DestroySurface;
#endif

#ifdef SDL_VIDEO_METAL
        device->Metal_CreateView = UIKit_Metal_CreateView;
        device->Metal_DestroyView = UIKit_Metal_DestroyView;
        device->Metal_GetLayer = UIKit_Metal_GetLayer;
#endif

        device->device_caps = VIDEO_DEVICE_CAPS_SENDS_FULLSCREEN_DIMENSIONS;

        device->gl_config.accelerated = 1;

        return device;
    }
}

VideoBootStrap UIKIT_bootstrap = {
    UIKITVID_DRIVER_NAME, "SDL UIKit video driver",
    UIKit_CreateDevice,
    UIKit_ShowMessageBox,
    false
};

static bool UIKit_VideoInit(SDL_VideoDevice *_this)
{
    _this->gl_config.driver_loaded = 1;

    if (!UIKit_InitModes(_this)) {
        return false;
    }

    SDL_InitGCKeyboard();
    SDL_InitGCMouse();

    UIKit_InitClipboard(_this);

    return true;
}

static void UIKit_VideoQuit(SDL_VideoDevice *_this)
{
    UIKit_QuitClipboard(_this);

    SDL_QuitGCKeyboard();
    SDL_QuitGCMouse();
    UIKit_QuitPen(_this);

    UIKit_QuitModes(_this);
}

bool UIKit_SuspendScreenSaver(SDL_VideoDevice *_this)
{
    @autoreleasepool {
        UIApplication *app = [UIApplication sharedApplication];

        // Prevent the display from dimming and going to sleep.
        app.idleTimerDisabled = (_this->suspend_screensaver != false);
    }
    return true;
}

bool UIKit_IsSystemVersionAtLeast(double version)
{
    return [[UIDevice currentDevice].systemVersion doubleValue] >= version;
}

SDL_SystemTheme UIKit_GetSystemTheme(void)
{
#ifndef SDL_PLATFORM_VISIONOS
    if (@available(iOS 12.0, tvOS 10.0, *)) {
        switch ([UIScreen mainScreen].traitCollection.userInterfaceStyle) {
        case UIUserInterfaceStyleDark:
            return SDL_SYSTEM_THEME_DARK;
        case UIUserInterfaceStyleLight:
            return SDL_SYSTEM_THEME_LIGHT;
        default:
            break;
        }
    }
#endif
    return SDL_SYSTEM_THEME_UNKNOWN;
}

#ifdef SDL_PLATFORM_VISIONOS
CGRect UIKit_ComputeViewFrame(SDL_Window *window){
    return CGRectMake(window->x, window->y, window->w, window->h);
}
#else
CGRect UIKit_ComputeViewFrame(SDL_Window *window, UIScreen *screen)
{
    SDL_UIKitWindowData *data = (__bridge SDL_UIKitWindowData *)window->internal;
    CGRect frame = screen.bounds;

    /* Use the UIWindow bounds instead of the UIScreen bounds, when possible.
     * The uiwindow bounds may be smaller than the screen bounds when Split View
     * is used on an iPad. */
    if (data != nil && data.uiwindow != nil) {
        frame = data.uiwindow.bounds;
    }

#ifndef SDL_PLATFORM_TVOS
    /* iOS 10 seems to have a bug where, in certain conditions, putting the
     * device to sleep with the a landscape-only app open, re-orienting the
     * device to portrait, and turning it back on will result in the screen
     * bounds returning portrait orientation despite the app being in landscape.
     * This is a workaround until a better solution can be found.
     * https://bugzilla.libsdl.org/show_bug.cgi?id=3505
     * https://bugzilla.libsdl.org/show_bug.cgi?id=3465
     * https://forums.developer.apple.com/thread/65337 */
    UIInterfaceOrientation orient = [UIApplication sharedApplication].statusBarOrientation;
    BOOL landscape = UIInterfaceOrientationIsLandscape(orient) ||
                    !(UIKit_GetSupportedOrientations(window) & (UIInterfaceOrientationMaskPortrait | UIInterfaceOrientationMaskPortraitUpsideDown));
    BOOL fullscreen = CGRectEqualToRect(screen.bounds, frame);

    /* The orientation flip doesn't make sense when the window is smaller
     * than the screen (iPad Split View, for example). */
    if (fullscreen && (landscape != (frame.size.width > frame.size.height))) {
        float height = frame.size.width;
        frame.size.width = frame.size.height;
        frame.size.height = height;
    }
#endif

    return frame;
}

#endif

void UIKit_ForceUpdateHomeIndicator(void)
{
#ifndef SDL_PLATFORM_TVOS
    // Force the main SDL window to re-evaluate home indicator state
    SDL_Window *focus = SDL_GetKeyboardFocus();
    if (focus) {
        SDL_UIKitWindowData *data = (__bridge SDL_UIKitWindowData *)focus->internal;
        if (data != nil) {
            [data.viewcontroller performSelectorOnMainThread:@selector(setNeedsUpdateOfHomeIndicatorAutoHidden) withObject:nil waitUntilDone:NO];
            [data.viewcontroller performSelectorOnMainThread:@selector(setNeedsUpdateOfScreenEdgesDeferringSystemGestures) withObject:nil waitUntilDone:NO];
        }
    }
#endif // !SDL_PLATFORM_TVOS
}

/*
 * iOS log support.
 *
 * This doesn't really have anything to do with the interfaces of the SDL video
 *  subsystem, but we need to stuff this into an Objective-C source code file.
 *
 * NOTE: This is copypasted from src/video/cocoa/SDL_cocoavideo.m! Thus, if
 *  Cocoa is supported, we use that one instead. Be sure both versions remain
 *  identical!
 */

#ifndef SDL_VIDEO_DRIVER_COCOA
void SDL_NSLog(const char *prefix, const char *text)
{
    @autoreleasepool {
        NSString *nsText = [NSString stringWithUTF8String:text];
        if (prefix && *prefix) {
            NSString *nsPrefix = [NSString stringWithUTF8String:prefix];
            NSLog(@"%@%@", nsPrefix, nsText);
        } else {
            NSLog(@"%@", nsText);
        }
    }
}
#endif // SDL_VIDEO_DRIVER_COCOA

/*
 * iOS Tablet, etc, detection
 *
 * This doesn't really have anything to do with the interfaces of the SDL video
 * subsystem, but we need to stuff this into an Objective-C source code file.
 */
bool SDL_IsIPad(void)
{
    return ([UIDevice currentDevice].userInterfaceIdiom == UIUserInterfaceIdiomPad);
}

bool SDL_IsAppleTV(void)
{
    return ([UIDevice currentDevice].userInterfaceIdiom == UIUserInterfaceIdiomTV);
}

#endif // SDL_VIDEO_DRIVER_UIKIT
