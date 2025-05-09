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

#ifndef SDL_uikitmodes_h_
#define SDL_uikitmodes_h_

#include "SDL_uikitvideo.h"

@interface SDL_UIKitDisplayData : NSObject

#ifndef SDL_PLATFORM_VISIONOS
- (instancetype)initWithScreen:(UIScreen *)screen;
@property(nonatomic, strong) UIScreen *uiscreen;
#endif

@end

@interface SDL_UIKitDisplayModeData : NSObject
#ifndef SDL_PLATFORM_VISIONOS
@property(nonatomic, strong) UIScreenMode *uiscreenmode;
#endif

@end

#ifndef SDL_PLATFORM_VISIONOS
extern bool UIKit_IsDisplayLandscape(UIScreen *uiscreen);
#endif

extern bool UIKit_InitModes(SDL_VideoDevice *_this);
#ifndef SDL_PLATFORM_VISIONOS
extern bool UIKit_AddDisplay(UIScreen *uiscreen, bool send_event);
extern void UIKit_DelDisplay(UIScreen *uiscreen, bool send_event);
#endif
extern bool UIKit_GetDisplayModes(SDL_VideoDevice *_this, SDL_VideoDisplay *display);
extern bool UIKit_SetDisplayMode(SDL_VideoDevice *_this, SDL_VideoDisplay *display, SDL_DisplayMode *mode);
extern void UIKit_QuitModes(SDL_VideoDevice *_this);
extern bool UIKit_GetDisplayUsableBounds(SDL_VideoDevice *_this, SDL_VideoDisplay *display, SDL_Rect *rect);

// because visionOS does not have a screen
// we create a fake display to maintain compatibility.
// By default, a window measures 1280x720 pt.
// https://developer.apple.com/design/human-interface-guidelines/windows#visionOS
#ifdef SDL_PLATFORM_VISIONOS
#define SDL_XR_SCREENWIDTH 1280
#define SDL_XR_SCREENHEIGHT 720
#endif

#endif // SDL_uikitmodes_h_
