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

#ifdef SDL_VIDEO_DRIVER_N3DS

#include <3ds.h>

#include "../../events/SDL_touch_c.h"
#include "../SDL_sysvideo.h"
#include "SDL_n3dstouch.h"
#include "SDL_n3dsvideo.h"

#define N3DS_TOUCH_ID 1
#define N3DS_TOUCH_FINGER 1

/*
  Factors used to convert touchscreen coordinates to
  SDL's 0-1 values. Note that the N3DS's screen is
  internally in a portrait disposition so the
  GSP_SCREEN constants are flipped.
*/
#define TOUCHSCREEN_SCALE_X 1.0f / (GSP_SCREEN_HEIGHT_BOTTOM - 1)
#define TOUCHSCREEN_SCALE_Y 1.0f / (GSP_SCREEN_WIDTH - 1)

void N3DS_InitTouch(void)
{
    SDL_AddTouch(N3DS_TOUCH_ID, SDL_TOUCH_DEVICE_DIRECT, "Touchscreen");
}

void N3DS_QuitTouch(void)
{
    SDL_DelTouch(N3DS_TOUCH_ID);
}

void N3DS_PollTouch(SDL_VideoDevice *_this)
{
    SDL_VideoData *internal = (SDL_VideoData *)_this->internal;
    touchPosition touch;
    SDL_Window *window;
    SDL_VideoDisplay *display;
    static bool was_pressed = false;
    bool pressed;
    hidTouchRead(&touch);
    pressed = (touch.px != 0 || touch.py != 0);

    display = SDL_GetVideoDisplay(internal->touch_display);
    window = display ? display->fullscreen_window : NULL;

    if (pressed != was_pressed) {
        was_pressed = pressed;
        SDL_SendTouch(0, N3DS_TOUCH_ID, N3DS_TOUCH_FINGER,
                      window,
                      pressed ? SDL_EVENT_FINGER_DOWN : SDL_EVENT_FINGER_UP,
                      touch.px * TOUCHSCREEN_SCALE_X,
                      touch.py * TOUCHSCREEN_SCALE_Y,
                      pressed ? 1.0f : 0.0f);
    } else if (pressed) {
        SDL_SendTouchMotion(0, N3DS_TOUCH_ID, N3DS_TOUCH_FINGER,
                            window,
                            touch.px * TOUCHSCREEN_SCALE_X,
                            touch.py * TOUCHSCREEN_SCALE_Y,
                            1.0f);
    }
}

#endif // SDL_VIDEO_DRIVER_N3DS
