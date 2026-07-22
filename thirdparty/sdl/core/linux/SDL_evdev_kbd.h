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

#ifndef SDL_evdev_kbd_h_
#define SDL_evdev_kbd_h_

struct SDL_EVDEV_keyboard_state;
typedef struct SDL_EVDEV_keyboard_state SDL_EVDEV_keyboard_state;

extern SDL_EVDEV_keyboard_state *SDL_EVDEV_kbd_init(void);
extern void SDL_EVDEV_kbd_set_muted(SDL_EVDEV_keyboard_state *state, bool muted);
extern void SDL_EVDEV_kbd_set_vt_switch_callbacks(SDL_EVDEV_keyboard_state *state, void (*release_callback)(void*), void *release_callback_data, void (*acquire_callback)(void*), void *acquire_callback_data);
extern void SDL_EVDEV_kbd_update(SDL_EVDEV_keyboard_state *state);
extern void SDL_EVDEV_kbd_keycode(SDL_EVDEV_keyboard_state *state, unsigned int keycode, int down);
extern void SDL_EVDEV_kbd_quit(SDL_EVDEV_keyboard_state *state);

#endif // SDL_evdev_kbd_h_
