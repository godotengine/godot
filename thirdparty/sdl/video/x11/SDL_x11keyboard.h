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

#ifndef SDL_x11keyboard_h_
#define SDL_x11keyboard_h_

extern bool X11_InitKeyboard(SDL_VideoDevice *_this);
extern void X11_UpdateKeymap(SDL_VideoDevice *_this, bool send_event);
extern void X11_QuitKeyboard(SDL_VideoDevice *_this);
extern void X11_CreateInputContext(SDL_WindowData *data);
extern void X11_ClearComposition(SDL_WindowData *data);
extern bool X11_StartTextInput(SDL_VideoDevice *_this, SDL_Window *window, SDL_PropertiesID props);
extern bool X11_StopTextInput(SDL_VideoDevice *_this, SDL_Window *window);
extern bool X11_UpdateTextInputArea(SDL_VideoDevice *_this, SDL_Window *window);
extern bool X11_HasScreenKeyboardSupport(SDL_VideoDevice *_this);
extern void X11_ShowScreenKeyboard(SDL_VideoDevice *_this, SDL_Window *window, SDL_PropertiesID props);
extern void X11_HideScreenKeyboard(SDL_VideoDevice *_this, SDL_Window *window);
extern bool X11_IsScreenKeyboardShown(SDL_VideoDevice *_this, SDL_Window *window);
extern KeySym X11_KeyCodeToSym(SDL_VideoDevice *_this, KeyCode, unsigned char group, unsigned int mod_mask);

#endif // SDL_x11keyboard_h_
