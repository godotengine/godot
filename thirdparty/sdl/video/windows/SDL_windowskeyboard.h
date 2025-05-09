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

#ifndef SDL_windowskeyboard_h_
#define SDL_windowskeyboard_h_

extern void WIN_InitKeyboard(SDL_VideoDevice *_this);
extern void WIN_UpdateKeymap(bool send_event);
extern void WIN_QuitKeyboard(SDL_VideoDevice *_this);

extern void WIN_ResetDeadKeys(void);

extern bool WIN_StartTextInput(SDL_VideoDevice *_this, SDL_Window *window, SDL_PropertiesID props);
extern bool WIN_StopTextInput(SDL_VideoDevice *_this, SDL_Window *window);
extern bool WIN_UpdateTextInputArea(SDL_VideoDevice *_this, SDL_Window *window);
extern bool WIN_ClearComposition(SDL_VideoDevice *_this, SDL_Window *window);

extern bool WIN_HandleIMEMessage(HWND hwnd, UINT msg, WPARAM wParam, LPARAM *lParam, struct SDL_VideoData *videodata);
extern void WIN_UpdateIMECandidates(SDL_VideoDevice *_this);

#endif // SDL_windowskeyboard_h_
