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

#ifdef SDL_VIDEO_DRIVER_COCOA

#if !__has_feature(objc_arc)
#error SDL must be built with Objective-C ARC (automatic reference counting) enabled
#endif

#include "SDL_cocoavideo.h"
#include "SDL_cocoavulkan.h"
#include "SDL_cocoametalview.h"
#include "SDL_cocoaopengles.h"
#include "SDL_cocoamessagebox.h"
#include "SDL_cocoashape.h"

#include "../../events/SDL_keyboard_c.h"
#include "../../events/SDL_mouse_c.h"

@implementation SDL_CocoaVideoData

@end

// Initialization/Query functions
static bool Cocoa_VideoInit(SDL_VideoDevice *_this);
static void Cocoa_VideoQuit(SDL_VideoDevice *_this);

// Cocoa driver bootstrap functions

static void Cocoa_DeleteDevice(SDL_VideoDevice *device)
{
    @autoreleasepool {
        if (device->wakeup_lock) {
            SDL_DestroyMutex(device->wakeup_lock);
        }
        CFBridgingRelease(device->internal);
        SDL_free(device);
    }
}

static SDL_VideoDevice *Cocoa_CreateDevice(void)
{
    @autoreleasepool {
        SDL_VideoDevice *device;
        SDL_CocoaVideoData *data;

        if (![NSThread isMainThread]) {
            return NULL;  // this doesn't SDL_SetError() because SDL_VideoInit is just going to overwrite it.
        }

        Cocoa_RegisterApp();

        // Initialize all variables that we clean on shutdown
        device = (SDL_VideoDevice *)SDL_calloc(1, sizeof(SDL_VideoDevice));
        if (device) {
            data = [[SDL_CocoaVideoData alloc] init];
        } else {
            data = nil;
        }
        if (!data) {
            SDL_free(device);
            return NULL;
        }
        device->internal = (SDL_VideoData *)CFBridgingRetain(data);
        device->wakeup_lock = SDL_CreateMutex();
        device->system_theme = Cocoa_GetSystemTheme();

        // Set the function pointers
        device->VideoInit = Cocoa_VideoInit;
        device->VideoQuit = Cocoa_VideoQuit;
        device->GetDisplayBounds = Cocoa_GetDisplayBounds;
        device->GetDisplayUsableBounds = Cocoa_GetDisplayUsableBounds;
        device->GetDisplayModes = Cocoa_GetDisplayModes;
        device->SetDisplayMode = Cocoa_SetDisplayMode;
        device->PumpEvents = Cocoa_PumpEvents;
        device->WaitEventTimeout = Cocoa_WaitEventTimeout;
        device->SendWakeupEvent = Cocoa_SendWakeupEvent;
        device->SuspendScreenSaver = Cocoa_SuspendScreenSaver;

        device->CreateSDLWindow = Cocoa_CreateWindow;
        device->SetWindowTitle = Cocoa_SetWindowTitle;
        device->SetWindowIcon = Cocoa_SetWindowIcon;
        device->SetWindowPosition = Cocoa_SetWindowPosition;
        device->SetWindowSize = Cocoa_SetWindowSize;
        device->SetWindowMinimumSize = Cocoa_SetWindowMinimumSize;
        device->SetWindowMaximumSize = Cocoa_SetWindowMaximumSize;
        device->SetWindowAspectRatio = Cocoa_SetWindowAspectRatio;
        device->SetWindowOpacity = Cocoa_SetWindowOpacity;
        device->GetWindowSizeInPixels = Cocoa_GetWindowSizeInPixels;
        device->ShowWindow = Cocoa_ShowWindow;
        device->HideWindow = Cocoa_HideWindow;
        device->RaiseWindow = Cocoa_RaiseWindow;
        device->MaximizeWindow = Cocoa_MaximizeWindow;
        device->MinimizeWindow = Cocoa_MinimizeWindow;
        device->RestoreWindow = Cocoa_RestoreWindow;
        device->SetWindowBordered = Cocoa_SetWindowBordered;
        device->SetWindowResizable = Cocoa_SetWindowResizable;
        device->SetWindowAlwaysOnTop = Cocoa_SetWindowAlwaysOnTop;
        device->SetWindowFullscreen = Cocoa_SetWindowFullscreen;
        device->GetWindowICCProfile = Cocoa_GetWindowICCProfile;
        device->GetDisplayForWindow = Cocoa_GetDisplayForWindow;
        device->SetWindowMouseRect = Cocoa_SetWindowMouseRect;
        device->SetWindowMouseGrab = Cocoa_SetWindowMouseGrab;
        device->SetWindowKeyboardGrab = Cocoa_SetWindowKeyboardGrab;
        device->DestroyWindow = Cocoa_DestroyWindow;
        device->SetWindowHitTest = Cocoa_SetWindowHitTest;
        device->AcceptDragAndDrop = Cocoa_AcceptDragAndDrop;
        device->UpdateWindowShape = Cocoa_UpdateWindowShape;
        device->FlashWindow = Cocoa_FlashWindow;
        device->SetWindowFocusable = Cocoa_SetWindowFocusable;
        device->SetWindowParent = Cocoa_SetWindowParent;
        device->SetWindowModal = Cocoa_SetWindowModal;
        device->SyncWindow = Cocoa_SyncWindow;

#ifdef SDL_VIDEO_OPENGL_CGL
        device->GL_LoadLibrary = Cocoa_GL_LoadLibrary;
        device->GL_GetProcAddress = Cocoa_GL_GetProcAddress;
        device->GL_UnloadLibrary = Cocoa_GL_UnloadLibrary;
        device->GL_CreateContext = Cocoa_GL_CreateContext;
        device->GL_MakeCurrent = Cocoa_GL_MakeCurrent;
        device->GL_SetSwapInterval = Cocoa_GL_SetSwapInterval;
        device->GL_GetSwapInterval = Cocoa_GL_GetSwapInterval;
        device->GL_SwapWindow = Cocoa_GL_SwapWindow;
        device->GL_DestroyContext = Cocoa_GL_DestroyContext;
        device->GL_GetEGLSurface = NULL;
#endif
#ifdef SDL_VIDEO_OPENGL_EGL
#ifdef SDL_VIDEO_OPENGL_CGL
        if (SDL_GetHintBoolean(SDL_HINT_VIDEO_FORCE_EGL, false)) {
#endif
            device->GL_LoadLibrary = Cocoa_GLES_LoadLibrary;
            device->GL_GetProcAddress = Cocoa_GLES_GetProcAddress;
            device->GL_UnloadLibrary = Cocoa_GLES_UnloadLibrary;
            device->GL_CreateContext = Cocoa_GLES_CreateContext;
            device->GL_MakeCurrent = Cocoa_GLES_MakeCurrent;
            device->GL_SetSwapInterval = Cocoa_GLES_SetSwapInterval;
            device->GL_GetSwapInterval = Cocoa_GLES_GetSwapInterval;
            device->GL_SwapWindow = Cocoa_GLES_SwapWindow;
            device->GL_DestroyContext = Cocoa_GLES_DestroyContext;
            device->GL_GetEGLSurface = Cocoa_GLES_GetEGLSurface;
#ifdef SDL_VIDEO_OPENGL_CGL
        }
#endif
#endif

#ifdef SDL_VIDEO_VULKAN
        device->Vulkan_LoadLibrary = Cocoa_Vulkan_LoadLibrary;
        device->Vulkan_UnloadLibrary = Cocoa_Vulkan_UnloadLibrary;
        device->Vulkan_GetInstanceExtensions = Cocoa_Vulkan_GetInstanceExtensions;
        device->Vulkan_CreateSurface = Cocoa_Vulkan_CreateSurface;
        device->Vulkan_DestroySurface = Cocoa_Vulkan_DestroySurface;
#endif

#ifdef SDL_VIDEO_METAL
        device->Metal_CreateView = Cocoa_Metal_CreateView;
        device->Metal_DestroyView = Cocoa_Metal_DestroyView;
        device->Metal_GetLayer = Cocoa_Metal_GetLayer;
#endif

        device->StartTextInput = Cocoa_StartTextInput;
        device->StopTextInput = Cocoa_StopTextInput;
        device->UpdateTextInputArea = Cocoa_UpdateTextInputArea;

        device->SetClipboardData = Cocoa_SetClipboardData;
        device->GetClipboardData = Cocoa_GetClipboardData;
        device->HasClipboardData = Cocoa_HasClipboardData;

        device->free = Cocoa_DeleteDevice;

        device->device_caps = VIDEO_DEVICE_CAPS_HAS_POPUP_WINDOW_SUPPORT |
                              VIDEO_DEVICE_CAPS_SENDS_FULLSCREEN_DIMENSIONS;
        return device;
    }
}

VideoBootStrap COCOA_bootstrap = {
    "cocoa", "SDL Cocoa video driver",
    Cocoa_CreateDevice,
    Cocoa_ShowMessageBox,
    false
};

static bool Cocoa_VideoInit(SDL_VideoDevice *_this)
{
    @autoreleasepool {
        SDL_CocoaVideoData *data = (__bridge SDL_CocoaVideoData *)_this->internal;

        Cocoa_InitModes(_this);
        Cocoa_InitKeyboard(_this);
        if (!Cocoa_InitMouse(_this)) {
            return false;
        }
        if (!Cocoa_InitPen(_this)) {
            return false;
        }

        // Assume we have a mouse and keyboard
        // We could use GCMouse and GCKeyboard if we needed to, as is done in SDL_uikitevents.m
        SDL_AddKeyboard(SDL_DEFAULT_KEYBOARD_ID, NULL, false);
        SDL_AddMouse(SDL_DEFAULT_MOUSE_ID, NULL, false);

        data.allow_spaces = SDL_GetHintBoolean(SDL_HINT_VIDEO_MAC_FULLSCREEN_SPACES, true);
        data.trackpad_is_touch_only = SDL_GetHintBoolean(SDL_HINT_TRACKPAD_IS_TOUCH_ONLY, false);
        SDL_AddHintCallback(SDL_HINT_VIDEO_MAC_FULLSCREEN_MENU_VISIBILITY, Cocoa_MenuVisibilityCallback, NULL);

        data.swaplock = SDL_CreateMutex();
        if (!data.swaplock) {
            return false;
        }

        return true;
    }
}

void Cocoa_VideoQuit(SDL_VideoDevice *_this)
{
    @autoreleasepool {
        SDL_CocoaVideoData *data = (__bridge SDL_CocoaVideoData *)_this->internal;
        Cocoa_QuitModes(_this);
        Cocoa_QuitKeyboard(_this);
        Cocoa_QuitMouse(_this);
        Cocoa_QuitPen(_this);
        SDL_DestroyMutex(data.swaplock);
        data.swaplock = NULL;
    }
}

// This function assumes that it's called from within an autorelease pool
SDL_SystemTheme Cocoa_GetSystemTheme(void)
{
    if (@available(macOS 10.14, *)) {
        NSAppearance* appearance = [[NSApplication sharedApplication] effectiveAppearance];

        if ([appearance.name containsString: @"Dark"]) {
            return SDL_SYSTEM_THEME_DARK;
        }
    }
    return SDL_SYSTEM_THEME_LIGHT;
}

// This function assumes that it's called from within an autorelease pool
NSImage *Cocoa_CreateImage(SDL_Surface *surface)
{
    NSImage *img;

    img = [[NSImage alloc] initWithSize:NSMakeSize(surface->w, surface->h)];
    if (img == nil) {
        return nil;
    }

    SDL_Surface **images = SDL_GetSurfaceImages(surface, NULL);
    if (!images) {
        return nil;
    }

    for (int i = 0; images[i]; ++i) {
        SDL_Surface *converted = SDL_ConvertSurface(images[i], SDL_PIXELFORMAT_RGBA32);
        if (!converted) {
            SDL_free(images);
            return nil;
        }

        // Premultiply the alpha channel
        SDL_PremultiplySurfaceAlpha(converted, false);

        NSBitmapImageRep *imgrep = [[NSBitmapImageRep alloc] initWithBitmapDataPlanes:NULL
                                                                           pixelsWide:converted->w
                                                                           pixelsHigh:converted->h
                                                                        bitsPerSample:8
                                                                      samplesPerPixel:4
                                                                             hasAlpha:YES
                                                                             isPlanar:NO
                                                                       colorSpaceName:NSDeviceRGBColorSpace
                                                                          bytesPerRow:converted->pitch
                                                                         bitsPerPixel:SDL_BITSPERPIXEL(converted->format)];
        if (imgrep == nil) {
            SDL_free(images);
            SDL_DestroySurface(converted);
            return nil;
        }

        // Copy the pixels
        Uint8 *pixels = [imgrep bitmapData];
        SDL_memcpy(pixels, converted->pixels, (size_t)converted->h * converted->pitch);
        SDL_DestroySurface(converted);

        // Add the image representation
        [img addRepresentation:imgrep];
    }
    SDL_free(images);

    return img;
}

/*
 * macOS log support.
 *
 * This doesn't really have anything to do with the interfaces of the SDL video
 *  subsystem, but we need to stuff this into an Objective-C source code file.
 *
 * NOTE: This is copypasted in src/video/uikit/SDL_uikitvideo.m! Be sure both
 *  versions remain identical!
 */

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
