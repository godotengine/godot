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

#ifndef SDL_cocoavideo_h_
#define SDL_cocoavideo_h_

#include <SDL3/SDL_opengl.h>

#include <ApplicationServices/ApplicationServices.h>
#include <IOKit/pwr_mgt/IOPMLib.h>
#include <Cocoa/Cocoa.h>

#include "../SDL_sysvideo.h"

#include "SDL_cocoaclipboard.h"
#include "SDL_cocoaevents.h"
#include "SDL_cocoakeyboard.h"
#include "SDL_cocoamodes.h"
#include "SDL_cocoamouse.h"
#include "SDL_cocoaopengl.h"
#include "SDL_cocoawindow.h"
#include "SDL_cocoapen.h"

// Private display data

@class SDL3TranslatorResponder;

typedef enum
{
    OptionAsAltNone,
    OptionAsAltOnlyLeft,
    OptionAsAltOnlyRight,
    OptionAsAltBoth,
} OptionAsAlt;

@interface SDL_CocoaVideoData : NSObject
@property(nonatomic) int allow_spaces;
@property(nonatomic) int trackpad_is_touch_only;
@property(nonatomic) unsigned int modifierFlags;
@property(nonatomic) void *key_layout;
@property(nonatomic) SDL3TranslatorResponder *fieldEdit;
@property(nonatomic) NSInteger clipboard_count;
@property(nonatomic) IOPMAssertionID screensaver_assertion;
@property(nonatomic) SDL_Mutex *swaplock;
@property(nonatomic) OptionAsAlt option_as_alt;
@end

// Utility functions
extern SDL_SystemTheme Cocoa_GetSystemTheme(void);
extern NSImage *Cocoa_CreateImage(SDL_Surface *surface);

#endif // SDL_cocoavideo_h_
