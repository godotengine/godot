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

/**
 * # CategoryKeyboard
 *
 * SDL keyboard management.
 *
 * Please refer to the Best Keyboard Practices document for details on how
 * best to accept keyboard input in various types of programs:
 *
 * https://wiki.libsdl.org/SDL3/BestKeyboardPractices
 */

#ifndef SDL_keyboard_h_
#define SDL_keyboard_h_

#include <SDL3/SDL_stdinc.h>
#include <SDL3/SDL_error.h>
#include <SDL3/SDL_keycode.h>
#include <SDL3/SDL_properties.h>
#include <SDL3/SDL_rect.h>
#include <SDL3/SDL_scancode.h>
#include <SDL3/SDL_video.h>

#include <SDL3/SDL_begin_code.h>
/* Set up for C function definitions, even when using C++ */
#ifdef __cplusplus
extern "C" {
#endif

/**
 * This is a unique ID for a keyboard for the time it is connected to the
 * system, and is never reused for the lifetime of the application.
 *
 * If the keyboard is disconnected and reconnected, it will get a new ID.
 *
 * The value 0 is an invalid ID.
 *
 * \since This datatype is available since SDL 3.2.0.
 */
typedef Uint32 SDL_KeyboardID;

/* Function prototypes */

/**
 * Return whether a keyboard is currently connected.
 *
 * \returns true if a keyboard is connected, false otherwise.
 *
 * \threadsafety This function should only be called on the main thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_GetKeyboards
 */
extern SDL_DECLSPEC bool SDLCALL SDL_HasKeyboard(void);

/**
 * Get a list of currently connected keyboards.
 *
 * Note that this will include any device or virtual driver that includes
 * keyboard functionality, including some mice, KVM switches, motherboard
 * power buttons, etc. You should wait for input from a device before you
 * consider it actively in use.
 *
 * \param count a pointer filled in with the number of keyboards returned, may
 *              be NULL.
 * \returns a 0 terminated array of keyboards instance IDs or NULL on failure;
 *          call SDL_GetError() for more information. This should be freed
 *          with SDL_free() when it is no longer needed.
 *
 * \threadsafety This function should only be called on the main thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_GetKeyboardNameForID
 * \sa SDL_HasKeyboard
 */
extern SDL_DECLSPEC SDL_KeyboardID * SDLCALL SDL_GetKeyboards(int *count);

/**
 * Get the name of a keyboard.
 *
 * This function returns "" if the keyboard doesn't have a name.
 *
 * \param instance_id the keyboard instance ID.
 * \returns the name of the selected keyboard or NULL on failure; call
 *          SDL_GetError() for more information.
 *
 * \threadsafety This function should only be called on the main thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_GetKeyboards
 */
extern SDL_DECLSPEC const char * SDLCALL SDL_GetKeyboardNameForID(SDL_KeyboardID instance_id);

/**
 * Query the window which currently has keyboard focus.
 *
 * \returns the window with keyboard focus.
 *
 * \threadsafety This function should only be called on the main thread.
 *
 * \since This function is available since SDL 3.2.0.
 */
extern SDL_DECLSPEC SDL_Window * SDLCALL SDL_GetKeyboardFocus(void);

/**
 * Get a snapshot of the current state of the keyboard.
 *
 * The pointer returned is a pointer to an internal SDL array. It will be
 * valid for the whole lifetime of the application and should not be freed by
 * the caller.
 *
 * A array element with a value of true means that the key is pressed and a
 * value of false means that it is not. Indexes into this array are obtained
 * by using SDL_Scancode values.
 *
 * Use SDL_PumpEvents() to update the state array.
 *
 * This function gives you the current state after all events have been
 * processed, so if a key or button has been pressed and released before you
 * process events, then the pressed state will never show up in the
 * SDL_GetKeyboardState() calls.
 *
 * Note: This function doesn't take into account whether shift has been
 * pressed or not.
 *
 * \param numkeys if non-NULL, receives the length of the returned array.
 * \returns a pointer to an array of key states.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_PumpEvents
 * \sa SDL_ResetKeyboard
 */
extern SDL_DECLSPEC const bool * SDLCALL SDL_GetKeyboardState(int *numkeys);

/**
 * Clear the state of the keyboard.
 *
 * This function will generate key up events for all pressed keys.
 *
 * \threadsafety This function should only be called on the main thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_GetKeyboardState
 */
extern SDL_DECLSPEC void SDLCALL SDL_ResetKeyboard(void);

/**
 * Get the current key modifier state for the keyboard.
 *
 * \returns an OR'd combination of the modifier keys for the keyboard. See
 *          SDL_Keymod for details.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_GetKeyboardState
 * \sa SDL_SetModState
 */
extern SDL_DECLSPEC SDL_Keymod SDLCALL SDL_GetModState(void);

/**
 * Set the current key modifier state for the keyboard.
 *
 * The inverse of SDL_GetModState(), SDL_SetModState() allows you to impose
 * modifier key states on your application. Simply pass your desired modifier
 * states into `modstate`. This value may be a bitwise, OR'd combination of
 * SDL_Keymod values.
 *
 * This does not change the keyboard state, only the key modifier flags that
 * SDL reports.
 *
 * \param modstate the desired SDL_Keymod for the keyboard.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_GetModState
 */
extern SDL_DECLSPEC void SDLCALL SDL_SetModState(SDL_Keymod modstate);

/**
 * Get the key code corresponding to the given scancode according to the
 * current keyboard layout.
 *
 * If you want to get the keycode as it would be delivered in key events,
 * including options specified in SDL_HINT_KEYCODE_OPTIONS, then you should
 * pass `key_event` as true. Otherwise this function simply translates the
 * scancode based on the given modifier state.
 *
 * \param scancode the desired SDL_Scancode to query.
 * \param modstate the modifier state to use when translating the scancode to
 *                 a keycode.
 * \param key_event true if the keycode will be used in key events.
 * \returns the SDL_Keycode that corresponds to the given SDL_Scancode.
 *
 * \threadsafety This function is not thread safe.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_GetKeyName
 * \sa SDL_GetScancodeFromKey
 */
extern SDL_DECLSPEC SDL_Keycode SDLCALL SDL_GetKeyFromScancode(SDL_Scancode scancode, SDL_Keymod modstate, bool key_event);

/**
 * Get the scancode corresponding to the given key code according to the
 * current keyboard layout.
 *
 * Note that there may be multiple scancode+modifier states that can generate
 * this keycode, this will just return the first one found.
 *
 * \param key the desired SDL_Keycode to query.
 * \param modstate a pointer to the modifier state that would be used when the
 *                 scancode generates this key, may be NULL.
 * \returns the SDL_Scancode that corresponds to the given SDL_Keycode.
 *
 * \threadsafety This function is not thread safe.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_GetKeyFromScancode
 * \sa SDL_GetScancodeName
 */
extern SDL_DECLSPEC SDL_Scancode SDLCALL SDL_GetScancodeFromKey(SDL_Keycode key, SDL_Keymod *modstate);

/**
 * Set a human-readable name for a scancode.
 *
 * \param scancode the desired SDL_Scancode.
 * \param name the name to use for the scancode, encoded as UTF-8. The string
 *             is not copied, so the pointer given to this function must stay
 *             valid while SDL is being used.
 * \returns true on success or false on failure; call SDL_GetError() for more
 *          information.
 *
 * \threadsafety This function is not thread safe.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_GetScancodeName
 */
extern SDL_DECLSPEC bool SDLCALL SDL_SetScancodeName(SDL_Scancode scancode, const char *name);

/**
 * Get a human-readable name for a scancode.
 *
 * **Warning**: The returned name is by design not stable across platforms,
 * e.g. the name for `SDL_SCANCODE_LGUI` is "Left GUI" under Linux but "Left
 * Windows" under Microsoft Windows, and some scancodes like
 * `SDL_SCANCODE_NONUSBACKSLASH` don't have any name at all. There are even
 * scancodes that share names, e.g. `SDL_SCANCODE_RETURN` and
 * `SDL_SCANCODE_RETURN2` (both called "Return"). This function is therefore
 * unsuitable for creating a stable cross-platform two-way mapping between
 * strings and scancodes.
 *
 * \param scancode the desired SDL_Scancode to query.
 * \returns a pointer to the name for the scancode. If the scancode doesn't
 *          have a name this function returns an empty string ("").
 *
 * \threadsafety This function is not thread safe.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_GetScancodeFromKey
 * \sa SDL_GetScancodeFromName
 * \sa SDL_SetScancodeName
 */
extern SDL_DECLSPEC const char * SDLCALL SDL_GetScancodeName(SDL_Scancode scancode);

/**
 * Get a scancode from a human-readable name.
 *
 * \param name the human-readable scancode name.
 * \returns the SDL_Scancode, or `SDL_SCANCODE_UNKNOWN` if the name wasn't
 *          recognized; call SDL_GetError() for more information.
 *
 * \threadsafety This function is not thread safe.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_GetKeyFromName
 * \sa SDL_GetScancodeFromKey
 * \sa SDL_GetScancodeName
 */
extern SDL_DECLSPEC SDL_Scancode SDLCALL SDL_GetScancodeFromName(const char *name);

/**
 * Get a human-readable name for a key.
 *
 * If the key doesn't have a name, this function returns an empty string ("").
 *
 * Letters will be presented in their uppercase form, if applicable.
 *
 * \param key the desired SDL_Keycode to query.
 * \returns a UTF-8 encoded string of the key name.
 *
 * \threadsafety This function is not thread safe.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_GetKeyFromName
 * \sa SDL_GetKeyFromScancode
 * \sa SDL_GetScancodeFromKey
 */
extern SDL_DECLSPEC const char * SDLCALL SDL_GetKeyName(SDL_Keycode key);

/**
 * Get a key code from a human-readable name.
 *
 * \param name the human-readable key name.
 * \returns key code, or `SDLK_UNKNOWN` if the name wasn't recognized; call
 *          SDL_GetError() for more information.
 *
 * \threadsafety This function is not thread safe.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_GetKeyFromScancode
 * \sa SDL_GetKeyName
 * \sa SDL_GetScancodeFromName
 */
extern SDL_DECLSPEC SDL_Keycode SDLCALL SDL_GetKeyFromName(const char *name);

/**
 * Start accepting Unicode text input events in a window.
 *
 * This function will enable text input (SDL_EVENT_TEXT_INPUT and
 * SDL_EVENT_TEXT_EDITING events) in the specified window. Please use this
 * function paired with SDL_StopTextInput().
 *
 * Text input events are not received by default.
 *
 * On some platforms using this function shows the screen keyboard and/or
 * activates an IME, which can prevent some key press events from being passed
 * through.
 *
 * \param window the window to enable text input.
 * \returns true on success or false on failure; call SDL_GetError() for more
 *          information.
 *
 * \threadsafety This function should only be called on the main thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_SetTextInputArea
 * \sa SDL_StartTextInputWithProperties
 * \sa SDL_StopTextInput
 * \sa SDL_TextInputActive
 */
extern SDL_DECLSPEC bool SDLCALL SDL_StartTextInput(SDL_Window *window);

/**
 * Text input type.
 *
 * These are the valid values for SDL_PROP_TEXTINPUT_TYPE_NUMBER. Not every
 * value is valid on every platform, but where a value isn't supported, a
 * reasonable fallback will be used.
 *
 * \since This enum is available since SDL 3.2.0.
 *
 * \sa SDL_StartTextInputWithProperties
 */
typedef enum SDL_TextInputType
{
    SDL_TEXTINPUT_TYPE_TEXT,                        /**< The input is text */
    SDL_TEXTINPUT_TYPE_TEXT_NAME,                   /**< The input is a person's name */
    SDL_TEXTINPUT_TYPE_TEXT_EMAIL,                  /**< The input is an e-mail address */
    SDL_TEXTINPUT_TYPE_TEXT_USERNAME,               /**< The input is a username */
    SDL_TEXTINPUT_TYPE_TEXT_PASSWORD_HIDDEN,        /**< The input is a secure password that is hidden */
    SDL_TEXTINPUT_TYPE_TEXT_PASSWORD_VISIBLE,       /**< The input is a secure password that is visible */
    SDL_TEXTINPUT_TYPE_NUMBER,                      /**< The input is a number */
    SDL_TEXTINPUT_TYPE_NUMBER_PASSWORD_HIDDEN,      /**< The input is a secure PIN that is hidden */
    SDL_TEXTINPUT_TYPE_NUMBER_PASSWORD_VISIBLE      /**< The input is a secure PIN that is visible */
} SDL_TextInputType;

/**
 * Auto capitalization type.
 *
 * These are the valid values for SDL_PROP_TEXTINPUT_CAPITALIZATION_NUMBER.
 * Not every value is valid on every platform, but where a value isn't
 * supported, a reasonable fallback will be used.
 *
 * \since This enum is available since SDL 3.2.0.
 *
 * \sa SDL_StartTextInputWithProperties
 */
typedef enum SDL_Capitalization
{
    SDL_CAPITALIZE_NONE,        /**< No auto-capitalization will be done */
    SDL_CAPITALIZE_SENTENCES,   /**< The first letter of sentences will be capitalized */
    SDL_CAPITALIZE_WORDS,       /**< The first letter of words will be capitalized */
    SDL_CAPITALIZE_LETTERS      /**< All letters will be capitalized */
} SDL_Capitalization;

/**
 * Start accepting Unicode text input events in a window, with properties
 * describing the input.
 *
 * This function will enable text input (SDL_EVENT_TEXT_INPUT and
 * SDL_EVENT_TEXT_EDITING events) in the specified window. Please use this
 * function paired with SDL_StopTextInput().
 *
 * Text input events are not received by default.
 *
 * On some platforms using this function shows the screen keyboard and/or
 * activates an IME, which can prevent some key press events from being passed
 * through.
 *
 * These are the supported properties:
 *
 * - `SDL_PROP_TEXTINPUT_TYPE_NUMBER` - an SDL_TextInputType value that
 *   describes text being input, defaults to SDL_TEXTINPUT_TYPE_TEXT.
 * - `SDL_PROP_TEXTINPUT_CAPITALIZATION_NUMBER` - an SDL_Capitalization value
 *   that describes how text should be capitalized, defaults to
 *   SDL_CAPITALIZE_SENTENCES for normal text entry, SDL_CAPITALIZE_WORDS for
 *   SDL_TEXTINPUT_TYPE_TEXT_NAME, and SDL_CAPITALIZE_NONE for e-mail
 *   addresses, usernames, and passwords.
 * - `SDL_PROP_TEXTINPUT_AUTOCORRECT_BOOLEAN` - true to enable auto completion
 *   and auto correction, defaults to true.
 * - `SDL_PROP_TEXTINPUT_MULTILINE_BOOLEAN` - true if multiple lines of text
 *   are allowed. This defaults to true if SDL_HINT_RETURN_KEY_HIDES_IME is
 *   "0" or is not set, and defaults to false if SDL_HINT_RETURN_KEY_HIDES_IME
 *   is "1".
 *
 * On Android you can directly specify the input type:
 *
 * - `SDL_PROP_TEXTINPUT_ANDROID_INPUTTYPE_NUMBER` - the text input type to
 *   use, overriding other properties. This is documented at
 *   https://developer.android.com/reference/android/text/InputType
 *
 * \param window the window to enable text input.
 * \param props the properties to use.
 * \returns true on success or false on failure; call SDL_GetError() for more
 *          information.
 *
 * \threadsafety This function should only be called on the main thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_SetTextInputArea
 * \sa SDL_StartTextInput
 * \sa SDL_StopTextInput
 * \sa SDL_TextInputActive
 */
extern SDL_DECLSPEC bool SDLCALL SDL_StartTextInputWithProperties(SDL_Window *window, SDL_PropertiesID props);

#define SDL_PROP_TEXTINPUT_TYPE_NUMBER                  "SDL.textinput.type"
#define SDL_PROP_TEXTINPUT_CAPITALIZATION_NUMBER        "SDL.textinput.capitalization"
#define SDL_PROP_TEXTINPUT_AUTOCORRECT_BOOLEAN          "SDL.textinput.autocorrect"
#define SDL_PROP_TEXTINPUT_MULTILINE_BOOLEAN            "SDL.textinput.multiline"
#define SDL_PROP_TEXTINPUT_ANDROID_INPUTTYPE_NUMBER     "SDL.textinput.android.inputtype"

/**
 * Check whether or not Unicode text input events are enabled for a window.
 *
 * \param window the window to check.
 * \returns true if text input events are enabled else false.
 *
 * \threadsafety This function should only be called on the main thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_StartTextInput
 */
extern SDL_DECLSPEC bool SDLCALL SDL_TextInputActive(SDL_Window *window);

/**
 * Stop receiving any text input events in a window.
 *
 * If SDL_StartTextInput() showed the screen keyboard, this function will hide
 * it.
 *
 * \param window the window to disable text input.
 * \returns true on success or false on failure; call SDL_GetError() for more
 *          information.
 *
 * \threadsafety This function should only be called on the main thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_StartTextInput
 */
extern SDL_DECLSPEC bool SDLCALL SDL_StopTextInput(SDL_Window *window);

/**
 * Dismiss the composition window/IME without disabling the subsystem.
 *
 * \param window the window to affect.
 * \returns true on success or false on failure; call SDL_GetError() for more
 *          information.
 *
 * \threadsafety This function should only be called on the main thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_StartTextInput
 * \sa SDL_StopTextInput
 */
extern SDL_DECLSPEC bool SDLCALL SDL_ClearComposition(SDL_Window *window);

/**
 * Set the area used to type Unicode text input.
 *
 * Native input methods may place a window with word suggestions near the
 * cursor, without covering the text being entered.
 *
 * \param window the window for which to set the text input area.
 * \param rect the SDL_Rect representing the text input area, in window
 *             coordinates, or NULL to clear it.
 * \param cursor the offset of the current cursor location relative to
 *               `rect->x`, in window coordinates.
 * \returns true on success or false on failure; call SDL_GetError() for more
 *          information.
 *
 * \threadsafety This function should only be called on the main thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_GetTextInputArea
 * \sa SDL_StartTextInput
 */
extern SDL_DECLSPEC bool SDLCALL SDL_SetTextInputArea(SDL_Window *window, const SDL_Rect *rect, int cursor);

/**
 * Get the area used to type Unicode text input.
 *
 * This returns the values previously set by SDL_SetTextInputArea().
 *
 * \param window the window for which to query the text input area.
 * \param rect a pointer to an SDL_Rect filled in with the text input area,
 *             may be NULL.
 * \param cursor a pointer to the offset of the current cursor location
 *               relative to `rect->x`, may be NULL.
 * \returns true on success or false on failure; call SDL_GetError() for more
 *          information.
 *
 * \threadsafety This function should only be called on the main thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_SetTextInputArea
 */
extern SDL_DECLSPEC bool SDLCALL SDL_GetTextInputArea(SDL_Window *window, SDL_Rect *rect, int *cursor);

/**
 * Check whether the platform has screen keyboard support.
 *
 * \returns true if the platform has some screen keyboard support or false if
 *          not.
 *
 * \threadsafety This function should only be called on the main thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_StartTextInput
 * \sa SDL_ScreenKeyboardShown
 */
extern SDL_DECLSPEC bool SDLCALL SDL_HasScreenKeyboardSupport(void);

/**
 * Check whether the screen keyboard is shown for given window.
 *
 * \param window the window for which screen keyboard should be queried.
 * \returns true if screen keyboard is shown or false if not.
 *
 * \threadsafety This function should only be called on the main thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_HasScreenKeyboardSupport
 */
extern SDL_DECLSPEC bool SDLCALL SDL_ScreenKeyboardShown(SDL_Window *window);

/* Ends C function definitions when using C++ */
#ifdef __cplusplus
}
#endif
#include <SDL3/SDL_close_code.h>

#endif /* SDL_keyboard_h_ */
