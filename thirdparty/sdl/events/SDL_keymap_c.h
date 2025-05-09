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

#ifndef SDL_keymap_c_h_
#define SDL_keymap_c_h_

typedef struct SDL_Keymap
{
  SDL_HashTable *scancode_to_keycode;
  SDL_HashTable *keycode_to_scancode;
  bool auto_release;
  bool layout_determined;
  bool french_numbers;
  bool latin_letters;
  bool thai_keyboard;
} SDL_Keymap;

SDL_Keymap *SDL_GetCurrentKeymap(void);
SDL_Keymap *SDL_CreateKeymap(bool auto_release);
void SDL_SetKeymapEntry(SDL_Keymap *keymap, SDL_Scancode scancode, SDL_Keymod modstate, SDL_Keycode keycode);
SDL_Keycode SDL_GetKeymapKeycode(SDL_Keymap *keymap, SDL_Scancode scancode, SDL_Keymod modstate);
SDL_Scancode SDL_GetKeymapScancode(SDL_Keymap *keymap, SDL_Keycode keycode, SDL_Keymod *modstate);
void SDL_DestroyKeymap(SDL_Keymap *keymap);

#endif // SDL_keymap_c_h_
