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
#ifndef SDL_uikitvideo_h_
#define SDL_uikitvideo_h_

#include "../SDL_sysvideo.h"

#ifdef __OBJC__

#include <UIKit/UIKit.h>

@interface SDL_UIKitVideoData : NSObject

@property(nonatomic, assign) id pasteboardObserver;

@end

#ifdef SDL_PLATFORM_VISIONOS
CGRect UIKit_ComputeViewFrame(SDL_Window *window);
#else
CGRect UIKit_ComputeViewFrame(SDL_Window *window, UIScreen *screen);
#endif

#endif // __OBJC__

bool UIKit_SuspendScreenSaver(SDL_VideoDevice *_this);

void UIKit_ForceUpdateHomeIndicator(void);

bool UIKit_IsSystemVersionAtLeast(double version);

SDL_SystemTheme UIKit_GetSystemTheme(void);

#endif // SDL_uikitvideo_h_
