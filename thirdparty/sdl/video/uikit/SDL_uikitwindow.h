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
#ifndef SDL_uikitwindow_h_
#define SDL_uikitwindow_h_

#include "../SDL_sysvideo.h"
#import "SDL_uikitvideo.h"
#import "SDL_uikitview.h"
#import "SDL_uikitviewcontroller.h"

extern bool UIKit_CreateWindow(SDL_VideoDevice *_this, SDL_Window *window, SDL_PropertiesID create_props);
extern void UIKit_SetWindowTitle(SDL_VideoDevice *_this, SDL_Window *window);
extern void UIKit_ShowWindow(SDL_VideoDevice *_this, SDL_Window *window);
extern void UIKit_HideWindow(SDL_VideoDevice *_this, SDL_Window *window);
extern void UIKit_RaiseWindow(SDL_VideoDevice *_this, SDL_Window *window);
extern void UIKit_SetWindowBordered(SDL_VideoDevice *_this, SDL_Window *window, bool bordered);
extern SDL_FullscreenResult UIKit_SetWindowFullscreen(SDL_VideoDevice *_this, SDL_Window *window, SDL_VideoDisplay *display, SDL_FullscreenOp fullscreen);
extern void UIKit_UpdatePointerLock(SDL_VideoDevice *_this, SDL_Window *window);
extern void UIKit_DestroyWindow(SDL_VideoDevice *_this, SDL_Window *window);
extern void UIKit_GetWindowSizeInPixels(SDL_VideoDevice *_this, SDL_Window *window, int *w, int *h);

extern NSUInteger UIKit_GetSupportedOrientations(SDL_Window *window);

#define SDL_METALVIEW_TAG 255

@class UIWindow;

@interface SDL_UIKitWindowData : NSObject

@property(nonatomic, strong) UIWindow *uiwindow;
@property(nonatomic, strong) SDL_uikitviewcontroller *viewcontroller;

// Array of SDL_uikitviews owned by this window.
@property(nonatomic, copy) NSMutableArray *views;

@end

#endif // SDL_uikitwindow_h_
