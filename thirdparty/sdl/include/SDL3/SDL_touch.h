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
 * # CategoryTouch
 *
 * SDL offers touch input, on platforms that support it. It can manage
 * multiple touch devices and track multiple fingers on those devices.
 *
 * Touches are mostly dealt with through the event system, in the
 * SDL_EVENT_FINGER_DOWN, SDL_EVENT_FINGER_MOTION, and SDL_EVENT_FINGER_UP
 * events, but there are also functions to query for hardware details, etc.
 *
 * The touch system, by default, will also send virtual mouse events; this can
 * be useful for making a some desktop apps work on a phone without
 * significant changes. For apps that care about mouse and touch input
 * separately, they should ignore mouse events that have a `which` field of
 * SDL_TOUCH_MOUSEID.
 */

#ifndef SDL_touch_h_
#define SDL_touch_h_

#include <SDL3/SDL_stdinc.h>
#include <SDL3/SDL_error.h>
#include <SDL3/SDL_mouse.h>

#include <SDL3/SDL_begin_code.h>
/* Set up for C function definitions, even when using C++ */
#ifdef __cplusplus
extern "C" {
#endif

/**
 * A unique ID for a touch device.
 *
 * This ID is valid for the time the device is connected to the system, and is
 * never reused for the lifetime of the application.
 *
 * The value 0 is an invalid ID.
 *
 * \since This datatype is available since SDL 3.2.0.
 */
typedef Uint64 SDL_TouchID;

/**
 * A unique ID for a single finger on a touch device.
 *
 * This ID is valid for the time the finger (stylus, etc) is touching and will
 * be unique for all fingers currently in contact, so this ID tracks the
 * lifetime of a single continuous touch. This value may represent an index, a
 * pointer, or some other unique ID, depending on the platform.
 *
 * The value 0 is an invalid ID.
 *
 * \since This datatype is available since SDL 3.2.0.
 */
typedef Uint64 SDL_FingerID;

/**
 * An enum that describes the type of a touch device.
 *
 * \since This enum is available since SDL 3.2.0.
 */
typedef enum SDL_TouchDeviceType
{
    SDL_TOUCH_DEVICE_INVALID = -1,
    SDL_TOUCH_DEVICE_DIRECT,            /**< touch screen with window-relative coordinates */
    SDL_TOUCH_DEVICE_INDIRECT_ABSOLUTE, /**< trackpad with absolute device coordinates */
    SDL_TOUCH_DEVICE_INDIRECT_RELATIVE  /**< trackpad with screen cursor-relative coordinates */
} SDL_TouchDeviceType;

/**
 * Data about a single finger in a multitouch event.
 *
 * Each touch event is a collection of fingers that are simultaneously in
 * contact with the touch device (so a "touch" can be a "multitouch," in
 * reality), and this struct reports details of the specific fingers.
 *
 * \since This struct is available since SDL 3.2.0.
 *
 * \sa SDL_GetTouchFingers
 */
typedef struct SDL_Finger
{
    SDL_FingerID id;  /**< the finger ID */
    float x;  /**< the x-axis location of the touch event, normalized (0...1) */
    float y;  /**< the y-axis location of the touch event, normalized (0...1) */
    float pressure; /**< the quantity of pressure applied, normalized (0...1) */
} SDL_Finger;

/**
 * The SDL_MouseID for mouse events simulated with touch input.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_TOUCH_MOUSEID ((SDL_MouseID)-1)

/**
 * The SDL_TouchID for touch events simulated with mouse input.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_MOUSE_TOUCHID ((SDL_TouchID)-1)


/**
 * Get a list of registered touch devices.
 *
 * On some platforms SDL first sees the touch device if it was actually used.
 * Therefore the returned list might be empty, although devices are available.
 * After using all devices at least once the number will be correct.
 *
 * \param count a pointer filled in with the number of devices returned, may
 *              be NULL.
 * \returns a 0 terminated array of touch device IDs or NULL on failure; call
 *          SDL_GetError() for more information. This should be freed with
 *          SDL_free() when it is no longer needed.
 *
 * \since This function is available since SDL 3.2.0.
 */
extern SDL_DECLSPEC SDL_TouchID * SDLCALL SDL_GetTouchDevices(int *count);

/**
 * Get the touch device name as reported from the driver.
 *
 * \param touchID the touch device instance ID.
 * \returns touch device name, or NULL on failure; call SDL_GetError() for
 *          more information.
 *
 * \since This function is available since SDL 3.2.0.
 */
extern SDL_DECLSPEC const char * SDLCALL SDL_GetTouchDeviceName(SDL_TouchID touchID);

/**
 * Get the type of the given touch device.
 *
 * \param touchID the ID of a touch device.
 * \returns touch device type.
 *
 * \since This function is available since SDL 3.2.0.
 */
extern SDL_DECLSPEC SDL_TouchDeviceType SDLCALL SDL_GetTouchDeviceType(SDL_TouchID touchID);

/**
 * Get a list of active fingers for a given touch device.
 *
 * \param touchID the ID of a touch device.
 * \param count a pointer filled in with the number of fingers returned, can
 *              be NULL.
 * \returns a NULL terminated array of SDL_Finger pointers or NULL on failure;
 *          call SDL_GetError() for more information. This is a single
 *          allocation that should be freed with SDL_free() when it is no
 *          longer needed.
 *
 * \since This function is available since SDL 3.2.0.
 */
extern SDL_DECLSPEC SDL_Finger ** SDLCALL SDL_GetTouchFingers(SDL_TouchID touchID, int *count);

/* Ends C function definitions when using C++ */
#ifdef __cplusplus
}
#endif
#include <SDL3/SDL_close_code.h>

#endif /* SDL_touch_h_ */
