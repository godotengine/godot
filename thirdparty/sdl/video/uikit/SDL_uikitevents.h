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
#ifndef SDL_uikitevents_h_
#define SDL_uikitevents_h_

#import <UIKit/UIKit.h>

#include "../SDL_sysvideo.h"

extern void SDL_UpdateLifecycleObserver(void);

extern Uint64 UIKit_GetEventTimestamp(NSTimeInterval nsTimestamp);
extern void UIKit_PumpEvents(SDL_VideoDevice *_this);

extern void SDL_InitGCKeyboard(void);
extern void SDL_QuitGCKeyboard(void);

extern void SDL_InitGCMouse(void);
extern bool SDL_GCMouseRelativeMode(void);
extern void SDL_QuitGCMouse(void);

#endif // SDL_uikitevents_h_
