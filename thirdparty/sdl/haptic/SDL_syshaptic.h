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

#ifndef SDL_syshaptic_h_
#define SDL_syshaptic_h_

// Set up for C function definitions, even when using C++
#ifdef __cplusplus
extern "C" {
#endif

struct haptic_effect
{
    SDL_HapticEffect effect;          // The current event
    struct haptic_hweffect *hweffect; // The hardware behind the event
};

/*
 * The real SDL_Haptic struct.
 */
struct SDL_Haptic
{
    SDL_HapticID instance_id;       // Device instance, monotonically increasing from 0
    char *name;                     // Device name - system dependent

    struct haptic_effect *effects;  // Allocated effects
    int neffects;                   // Maximum amount of effects
    int nplaying;                   // Maximum amount of effects to play at the same time
    Uint32 supported;               // Supported effects and features
    int naxes;                      // Number of axes on the device.

    struct haptic_hwdata *hwdata;   // Driver dependent
    int ref_count;                  // Count for multiple opens

    int rumble_id;                  // ID of rumble effect for simple rumble API.
    SDL_HapticEffect rumble_effect; // Rumble effect.
    struct SDL_Haptic *next;        // pointer to next haptic we have allocated
};

/*
 * Scans the system for haptic devices.
 *
 * Returns number of devices on success, -1 on error.
 */
extern bool SDL_SYS_HapticInit(void);

// Function to return the number of haptic devices plugged in right now
extern int SDL_SYS_NumHaptics(void);

/*
 * Gets the instance ID of the haptic device
 */
extern SDL_HapticID SDL_SYS_HapticInstanceID(int index);

/*
 * Gets the device dependent name of the haptic device
 */
extern const char *SDL_SYS_HapticName(int index);

/*
 * Opens the haptic device for usage.  The haptic device should have
 * the index value set previously.
 */
extern bool SDL_SYS_HapticOpen(SDL_Haptic *haptic);

/*
 * Returns the index of the haptic core pointer or -1 if none is found.
 */
extern int SDL_SYS_HapticMouse(void);

/*
 * Checks to see if the joystick has haptic capabilities.
 */
extern bool SDL_SYS_JoystickIsHaptic(SDL_Joystick *joystick);

/*
 * Opens the haptic device for usage using the same device as
 * the joystick.
 */
extern bool SDL_SYS_HapticOpenFromJoystick(SDL_Haptic *haptic,
                                          SDL_Joystick *joystick);
/*
 * Checks to see if haptic device and joystick device are the same.
 *
 * Returns true if they are the same, false if they aren't.
 */
extern bool SDL_SYS_JoystickSameHaptic(SDL_Haptic *haptic,
                                      SDL_Joystick *joystick);

/*
 * Closes a haptic device after usage.
 */
extern void SDL_SYS_HapticClose(SDL_Haptic *haptic);

/*
 * Performs a cleanup on the haptic subsystem.
 */
extern void SDL_SYS_HapticQuit(void);

/*
 * Creates a new haptic effect on the haptic device using base
 * as a template for the effect.
 */
extern bool SDL_SYS_HapticNewEffect(SDL_Haptic *haptic,
                                    struct haptic_effect *effect,
                                    const SDL_HapticEffect *base);

/*
 * Updates the haptic effect on the haptic device using data
 * as a template.
 */
extern bool SDL_SYS_HapticUpdateEffect(SDL_Haptic *haptic,
                                       struct haptic_effect *effect,
                                       const SDL_HapticEffect *data);

/*
 * Runs the effect on the haptic device.
 */
extern bool SDL_SYS_HapticRunEffect(SDL_Haptic *haptic,
                                    struct haptic_effect *effect,
                                    Uint32 iterations);

/*
 * Stops the effect on the haptic device.
 */
extern bool SDL_SYS_HapticStopEffect(SDL_Haptic *haptic,
                                     struct haptic_effect *effect);

/*
 * Cleanups up the effect on the haptic device.
 */
extern void SDL_SYS_HapticDestroyEffect(SDL_Haptic *haptic,
                                        struct haptic_effect *effect);

/*
 * Queries the device for the status of effect.
 *
 * Returns 0 if device is stopped, >0 if device is playing and
 * -1 on error.
 */
extern int SDL_SYS_HapticGetEffectStatus(SDL_Haptic *haptic,
                                         struct haptic_effect *effect);

/*
 * Sets the global gain of the haptic device.
 */
extern bool SDL_SYS_HapticSetGain(SDL_Haptic *haptic, int gain);

/*
 * Sets the autocenter feature of the haptic device.
 */
extern bool SDL_SYS_HapticSetAutocenter(SDL_Haptic *haptic, int autocenter);

/*
 * Pauses the haptic device.
 */
extern bool SDL_SYS_HapticPause(SDL_Haptic *haptic);

/*
 * Unpauses the haptic device.
 */
extern bool SDL_SYS_HapticResume(SDL_Haptic *haptic);

/*
 * Stops all the currently playing haptic effects on the device.
 */
extern bool SDL_SYS_HapticStopAll(SDL_Haptic *haptic);

// Ends C function definitions when using C++
#ifdef __cplusplus
}
#endif

#endif // SDL_syshaptic_h_
