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

#ifndef SDL_keyboard_c_h_
#define SDL_keyboard_c_h_

#include "SDL_keymap_c.h"

// Keyboard events not associated with a specific input device
#define SDL_GLOBAL_KEYBOARD_ID     0

// The default keyboard input device, for platforms that don't have multiple keyboards
#define SDL_DEFAULT_KEYBOARD_ID    1

// Initialize the keyboard subsystem
extern bool SDL_InitKeyboard(void);

// Return whether a device is actually a keyboard
extern bool SDL_IsKeyboard(Uint16 vendor, Uint16 product, int num_keys);

// A keyboard has been added to the system
extern void SDL_AddKeyboard(SDL_KeyboardID keyboardID, const char *name, bool send_event);

// A keyboard has been removed from the system
extern void SDL_RemoveKeyboard(SDL_KeyboardID keyboardID, bool send_event);

// Set or update the name of a keyboard instance.
extern void SDL_SetKeyboardName(SDL_KeyboardID keyboardID, const char *name);

// Set the mapping of scancode to key codes
extern void SDL_SetKeymap(SDL_Keymap *keymap, bool send_event);

// Set the keyboard focus window
extern bool SDL_SetKeyboardFocus(SDL_Window *window);

/* Send a character from an on-screen keyboard as scancode and modifier key events,
   currently assuming ASCII characters on a US keyboard layout
 */
extern void SDL_SendKeyboardUnicodeKey(Uint64 timestamp, Uint32 ch);

// Send a keyboard key event
extern bool SDL_SendKeyboardKey(Uint64 timestamp, SDL_KeyboardID keyboardID, int rawcode, SDL_Scancode scancode, bool down);
extern bool SDL_SendKeyboardKeyIgnoreModifiers(Uint64 timestamp, SDL_KeyboardID keyboardID, int rawcode, SDL_Scancode scancode, bool down);
extern bool SDL_SendKeyboardKeyAutoRelease(Uint64 timestamp, SDL_Scancode scancode);

/* This is for platforms that don't know the keymap but can report scancode and keycode directly.
   Most platforms should prefer to optionally call SDL_SetKeymap and then use SDL_SendKeyboardKey. */
extern bool SDL_SendKeyboardKeyAndKeycode(Uint64 timestamp, SDL_KeyboardID keyboardID, int rawcode, SDL_Scancode scancode, SDL_Keycode keycode, bool down);

// Release all the autorelease keys
extern void SDL_ReleaseAutoReleaseKeys(void);

// Return true if any hardware key is pressed
extern bool SDL_HardwareKeyboardKeyPressed(void);

// Send keyboard text input
extern void SDL_SendKeyboardText(const char *text);

// Send editing text for selected range from start to end
extern void SDL_SendEditingText(const char *text, int start, int length);

// Send editing text candidates, which will be copied into the event
extern void SDL_SendEditingTextCandidates(char **candidates, int num_candidates, int selected_candidate, bool horizontal);

// Shutdown the keyboard subsystem
extern void SDL_QuitKeyboard(void);

// Toggle on or off pieces of the keyboard mod state.
extern void SDL_ToggleModState(SDL_Keymod modstate, bool toggle);

#endif // SDL_keyboard_c_h_
