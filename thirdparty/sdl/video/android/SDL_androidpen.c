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

#ifdef SDL_VIDEO_DRIVER_ANDROID

#include "SDL_androidpen.h"
#include "../../events/SDL_pen_c.h"
#include "../../core/android/SDL_android.h"

#define ACTION_DOWN   0
#define ACTION_UP     1
#define ACTION_CANCEL 3
#define ACTION_POINTER_DOWN 5
#define ACTION_POINTER_UP   6
#define ACTION_HOVER_EXIT   10

void Android_OnPen(SDL_Window *window, int pen_id_in, int button, int action, float x, float y, float p)
{
    if (!window) {
        return;
    }

    // pointer index starts from zero.
    pen_id_in++;

    SDL_PenID pen = SDL_FindPenByHandle((void *) (size_t) pen_id_in);
    if (!pen) {
        // TODO: Query JNI for pen device info
        SDL_PenInfo peninfo;
        SDL_zero(peninfo);
        peninfo.capabilities = SDL_PEN_CAPABILITY_PRESSURE | SDL_PEN_CAPABILITY_ERASER;
        peninfo.num_buttons = 2;
        peninfo.subtype = SDL_PEN_TYPE_PEN;
        pen = SDL_AddPenDevice(0, NULL, &peninfo, (void *) (size_t) pen_id_in);
        if (!pen) {
            SDL_Log("error: can't add a pen device %d", pen_id_in);
            return;
        }
    }

    SDL_SendPenMotion(0, pen, window, x, y);
    SDL_SendPenAxis(0, pen, window, SDL_PEN_AXIS_PRESSURE, p);
    // TODO: add more axis

    SDL_PenInputFlags current = SDL_GetPenStatus(pen, NULL, 0);
    int diff = current ^ button;
    if (diff != 0) {
        // Android only exposes BUTTON_STYLUS_PRIMARY and BUTTON_STYLUS_SECONDARY
        if (diff & SDL_PEN_INPUT_BUTTON_1)
            SDL_SendPenButton(0, pen, window, 1, (button & SDL_PEN_INPUT_BUTTON_1) != 0);

        if (diff & SDL_PEN_INPUT_BUTTON_2)
            SDL_SendPenButton(0, pen, window, 2, (button & SDL_PEN_INPUT_BUTTON_2) != 0);
    }

    // button contains DOWN/ERASER_TIP on DOWN/UP regardless of pressed state, use action to distinguish
    // we don't compare tip flags above because MotionEvent.getButtonState doesn't return stylus tip/eraser state.
    switch (action) {
    case ACTION_CANCEL:
    case ACTION_HOVER_EXIT:
        SDL_RemovePenDevice(0, pen);
        break;

    case ACTION_DOWN:
    case ACTION_POINTER_DOWN:
        SDL_SendPenTouch(0, pen, window, (button & SDL_PEN_INPUT_ERASER_TIP) != 0, true);
        break;

    case ACTION_UP:
    case ACTION_POINTER_UP:
        SDL_SendPenTouch(0, pen, window, (button & SDL_PEN_INPUT_ERASER_TIP) != 0, false);
        break;

    default:
        break;
    }
}

#endif // SDL_VIDEO_DRIVER_ANDROID
