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

#ifdef SDL_HAPTIC_DINPUT

#include "../SDL_syshaptic.h"
#include "../../joystick/SDL_sysjoystick.h"               // For the real SDL_Joystick
#include "../../joystick/windows/SDL_windowsjoystick_c.h" // For joystick hwdata
#include "../../joystick/windows/SDL_xinputjoystick_c.h"  // For xinput rumble

#include "SDL_windowshaptic_c.h"
#include "SDL_dinputhaptic_c.h"

// Set up for C function definitions, even when using C++
#ifdef __cplusplus
extern "C" {
#endif

/*
 * Internal stuff.
 */
SDL_hapticlist_item *SDL_hapticlist = NULL;
static SDL_hapticlist_item *SDL_hapticlist_tail = NULL;
static int numhaptics = 0;

/*
 * Initializes the haptic subsystem.
 */
bool SDL_SYS_HapticInit(void)
{
    JoyStick_DeviceData *device;

    if (!SDL_DINPUT_HapticInit()) {
        return false;
    }

    /* The joystick subsystem will usually be initialized before haptics,
     * so the initial HapticMaybeAddDevice() calls from the joystick
     * subsystem will arrive too early to create haptic devices. We will
     * invoke those callbacks again here to pick up any joysticks that
     * were added prior to haptics initialization. */
    for (device = SYS_Joystick; device; device = device->pNext) {
        SDL_DINPUT_HapticMaybeAddDevice(&device->dxdevice);
    }

    return true;
}

bool SDL_SYS_AddHapticDevice(SDL_hapticlist_item *item)
{
    if (!SDL_hapticlist_tail) {
        SDL_hapticlist = SDL_hapticlist_tail = item;
    } else {
        SDL_hapticlist_tail->next = item;
        SDL_hapticlist_tail = item;
    }

    // Device has been added.
    ++numhaptics;

    return true;
}

bool SDL_SYS_RemoveHapticDevice(SDL_hapticlist_item *prev, SDL_hapticlist_item *item)
{
    const bool result = item->haptic ? true : false;
    if (prev) {
        prev->next = item->next;
    } else {
        SDL_assert(SDL_hapticlist == item);
        SDL_hapticlist = item->next;
    }
    if (item == SDL_hapticlist_tail) {
        SDL_hapticlist_tail = prev;
    }
    --numhaptics;
    // !!! TODO: Send a haptic remove event?
    SDL_free(item);
    return result;
}

int SDL_SYS_NumHaptics(void)
{
    return numhaptics;
}

static SDL_hapticlist_item *HapticByDevIndex(int device_index)
{
    SDL_hapticlist_item *item = SDL_hapticlist;

    if ((device_index < 0) || (device_index >= numhaptics)) {
        return NULL;
    }

    while (device_index > 0) {
        SDL_assert(item != NULL);
        --device_index;
        item = item->next;
    }
    return item;
}

static SDL_hapticlist_item *HapticByInstanceID(SDL_HapticID instance_id)
{
    SDL_hapticlist_item *item;
    for (item = SDL_hapticlist; item; item = item->next) {
        if (instance_id == item->instance_id) {
            return item;
        }
    }
    return NULL;
}

SDL_HapticID SDL_SYS_HapticInstanceID(int index)
{
    SDL_hapticlist_item *item = HapticByDevIndex(index);
    if (item) {
        return item->instance_id;
    }
    return 0;
}

/*
 * Return the name of a haptic device, does not need to be opened.
 */
const char *SDL_SYS_HapticName(int index)
{
    SDL_hapticlist_item *item = HapticByDevIndex(index);
    return item->name;
}

/*
 * Opens a haptic device for usage.
 */
bool SDL_SYS_HapticOpen(SDL_Haptic *haptic)
{
    SDL_hapticlist_item *item = HapticByInstanceID(haptic->instance_id);
    return SDL_DINPUT_HapticOpen(haptic, item);
}

/*
 * Opens a haptic device from first mouse it finds for usage.
 */
int SDL_SYS_HapticMouse(void)
{
#ifdef SDL_HAPTIC_DINPUT
    SDL_hapticlist_item *item;
    int index = 0;

    // Grab the first mouse haptic device we find.
    for (item = SDL_hapticlist; item; item = item->next) {
        if (item->capabilities.dwDevType == DI8DEVCLASS_POINTER) {
            return index;
        }
        ++index;
    }
#endif // SDL_HAPTIC_DINPUT
    return -1;
}

/*
 * Checks to see if a joystick has haptic features.
 */
bool SDL_SYS_JoystickIsHaptic(SDL_Joystick *joystick)
{
    if (joystick->driver != &SDL_WINDOWS_JoystickDriver) {
        return false;
    }
    if (joystick->hwdata->Capabilities.dwFlags & DIDC_FORCEFEEDBACK) {
        return true;
    }
    return false;
}

/*
 * Checks to see if the haptic device and joystick are in reality the same.
 */
bool SDL_SYS_JoystickSameHaptic(SDL_Haptic *haptic, SDL_Joystick *joystick)
{
    if (joystick->driver != &SDL_WINDOWS_JoystickDriver) {
        return false;
    }
    return SDL_DINPUT_JoystickSameHaptic(haptic, joystick);
}

/*
 * Opens a SDL_Haptic from a SDL_Joystick.
 */
bool SDL_SYS_HapticOpenFromJoystick(SDL_Haptic *haptic, SDL_Joystick *joystick)
{
    SDL_assert(joystick->driver == &SDL_WINDOWS_JoystickDriver);

    return SDL_DINPUT_HapticOpenFromJoystick(haptic, joystick);
}

/*
 * Closes the haptic device.
 */
void SDL_SYS_HapticClose(SDL_Haptic *haptic)
{
    if (haptic->hwdata) {

        // Free effects.
        SDL_free(haptic->effects);
        haptic->effects = NULL;
        haptic->neffects = 0;

        // Clean up
        SDL_DINPUT_HapticClose(haptic);

        // Free
        SDL_free(haptic->hwdata);
        haptic->hwdata = NULL;
    }
}

/*
 * Clean up after system specific haptic stuff
 */
void SDL_SYS_HapticQuit(void)
{
    SDL_hapticlist_item *item;
    SDL_hapticlist_item *next = NULL;

    for (item = SDL_hapticlist; item; item = next) {
        /* Opened and not closed haptics are leaked, this is on purpose.
         * Close your haptic devices after usage. */
        // !!! FIXME: (...is leaking on purpose a good idea?) - No, of course not.
        next = item->next;
        SDL_free(item->name);
        SDL_free(item);
    }

    SDL_DINPUT_HapticQuit();

    numhaptics = 0;
    SDL_hapticlist = NULL;
    SDL_hapticlist_tail = NULL;
}

/*
 * Creates a new haptic effect.
 */
bool SDL_SYS_HapticNewEffect(SDL_Haptic *haptic, struct haptic_effect *effect,
                            const SDL_HapticEffect *base)
{
    bool result;

    // Alloc the effect.
    effect->hweffect = (struct haptic_hweffect *) SDL_calloc(1, sizeof(struct haptic_hweffect));
    if (!effect->hweffect) {
        return false;
    }

    result = SDL_DINPUT_HapticNewEffect(haptic, effect, base);
    if (!result) {
        SDL_free(effect->hweffect);
        effect->hweffect = NULL;
    }
    return result;
}

/*
 * Updates an effect.
 */
bool SDL_SYS_HapticUpdateEffect(SDL_Haptic *haptic, struct haptic_effect *effect, const SDL_HapticEffect *data)
{
    return SDL_DINPUT_HapticUpdateEffect(haptic, effect, data);
}

/*
 * Runs an effect.
 */
bool SDL_SYS_HapticRunEffect(SDL_Haptic *haptic, struct haptic_effect *effect, Uint32 iterations)
{
    return SDL_DINPUT_HapticRunEffect(haptic, effect, iterations);
}

/*
 * Stops an effect.
 */
bool SDL_SYS_HapticStopEffect(SDL_Haptic *haptic, struct haptic_effect *effect)
{
    return SDL_DINPUT_HapticStopEffect(haptic, effect);
}

/*
 * Frees the effect.
 */
void SDL_SYS_HapticDestroyEffect(SDL_Haptic *haptic, struct haptic_effect *effect)
{
    SDL_DINPUT_HapticDestroyEffect(haptic, effect);
    SDL_free(effect->hweffect);
    effect->hweffect = NULL;
}

/*
 * Gets the status of a haptic effect.
 */
int SDL_SYS_HapticGetEffectStatus(SDL_Haptic *haptic, struct haptic_effect *effect)
{
    return SDL_DINPUT_HapticGetEffectStatus(haptic, effect);
}

/*
 * Sets the gain.
 */
bool SDL_SYS_HapticSetGain(SDL_Haptic *haptic, int gain)
{
    return SDL_DINPUT_HapticSetGain(haptic, gain);
}

/*
 * Sets the autocentering.
 */
bool SDL_SYS_HapticSetAutocenter(SDL_Haptic *haptic, int autocenter)
{
    return SDL_DINPUT_HapticSetAutocenter(haptic, autocenter);
}

/*
 * Pauses the device.
 */
bool SDL_SYS_HapticPause(SDL_Haptic *haptic)
{
    return SDL_DINPUT_HapticPause(haptic);
}

/*
 * Pauses the device.
 */
bool SDL_SYS_HapticResume(SDL_Haptic *haptic)
{
    return SDL_DINPUT_HapticResume(haptic);
}

/*
 * Stops all the playing effects on the device.
 */
bool SDL_SYS_HapticStopAll(SDL_Haptic *haptic)
{
    return SDL_DINPUT_HapticStopAll(haptic);
}

// Ends C function definitions when using C++
#ifdef __cplusplus
}
#endif

#endif // SDL_HAPTIC_DINPUT
