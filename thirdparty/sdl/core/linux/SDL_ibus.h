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

#ifndef SDL_ibus_h_
#define SDL_ibus_h_

#ifdef HAVE_IBUS_IBUS_H
#define SDL_USE_IBUS 1
#include <ibus.h>

extern bool SDL_IBus_Init(void);
extern void SDL_IBus_Quit(void);

// Lets the IBus server know about changes in window focus
extern void SDL_IBus_SetFocus(bool focused);

// Closes the candidate list and resets any text currently being edited
extern void SDL_IBus_Reset(void);

/* Sends a keypress event to IBus, returns true if IBus used this event to
   update its candidate list or change input methods. PumpEvents should be
   called some time after this, to receive the TextInput / TextEditing event back. */
extern bool SDL_IBus_ProcessKeyEvent(Uint32 keysym, Uint32 keycode, bool down);

/* Update the position of IBus' candidate list. If rect is NULL then this will
   just reposition it relative to the focused window's new position. */
extern void SDL_IBus_UpdateTextInputArea(SDL_Window *window);

/* Checks DBus for new IBus events, and calls SDL_SendKeyboardText /
   SDL_SendEditingText for each event it finds */
extern void SDL_IBus_PumpEvents(void);

#endif // HAVE_IBUS_IBUS_H

#endif // SDL_ibus_h_
