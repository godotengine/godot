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

#include "SDL_cocoavideo.h"
#include "SDL_cocoashape.h"


bool Cocoa_UpdateWindowShape(SDL_VideoDevice *_this, SDL_Window *window, SDL_Surface *shape)
{
    SDL_CocoaWindowData *data = (__bridge SDL_CocoaWindowData *)window->internal;
    BOOL ignoresMouseEvents = NO;

    if (shape) {
        SDL_FPoint point;
        SDL_GetGlobalMouseState(&point.x, &point.y);
        point.x -= window->x;
        point.y -= window->y;
        if (point.x >= 0.0f && point.x < window->w &&
            point.y >= 0.0f && point.y < window->h) {
            int x = (int)SDL_roundf((point.x / (window->w - 1)) * (shape->w - 1));
            int y = (int)SDL_roundf((point.y / (window->h - 1)) * (shape->h - 1));
            Uint8 a;

            if (!SDL_ReadSurfacePixel(shape, x, y, NULL, NULL, NULL, &a) || a == SDL_ALPHA_TRANSPARENT) {
                ignoresMouseEvents = YES;
            }
        }
    }
    data.nswindow.ignoresMouseEvents = ignoresMouseEvents;
    return true;
}

#endif // SDL_VIDEO_DRIVER_COCOA
