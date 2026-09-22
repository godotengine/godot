/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2026 Sam Lantinga <slouken@libsdl.org>

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
 * # CategoryPen
 *
 * SDL pen event handling.
 *
 * SDL provides an API for pressure-sensitive pen (stylus and/or eraser)
 * handling, e.g., for input and drawing tablets or suitably equipped mobile /
 * tablet devices.
 *
 * To get started with pens, simply handle pen events:
 *
 * - SDL_EVENT_PEN_PROXIMITY_IN, SDL_EVENT_PEN_PROXIMITY_OUT
 *   (SDL_PenProximityEvent)
 * - SDL_EVENT_PEN_DOWN, SDL_EVENT_PEN_UP (SDL_PenTouchEvent)
 * - SDL_EVENT_PEN_MOTION (SDL_PenMotionEvent)
 * - SDL_EVENT_PEN_BUTTON_DOWN, SDL_EVENT_PEN_BUTTON_UP (SDL_PenButtonEvent)
 * - SDL_EVENT_PEN_AXIS (SDL_PenAxisEvent)
 *
 * Pens may provide more than simple touch input; they might have other axes,
 * such as pressure, tilt, rotation, etc.
 *
 * When a pen starts providing input, SDL will assign it a unique SDL_PenID,
 * which will remain for the life of the process, as long as the pen stays
 * connected. A pen leaving proximity (being taken far enough away from the
 * digitizer tablet that it no longer reponds) and then coming back should
 * fire proximity events, but the SDL_PenID should remain consistent.
 * Unplugging the digitizer and reconnecting may cause future input to have a
 * new SDL_PenID, as SDL may not know that this is the same hardware.
 *
 * Please note that various platforms vary wildly in how (and how well) they
 * support pen input. If your pen supports some piece of functionality but SDL
 * doesn't seem to, it might actually be the operating system's fault. For
 * example, some platforms can manage multiple devices at the same time, but
 * others will make any connected pens look like a single logical device, much
 * how all USB mice connected to a computer will move the same system cursor.
 * Other platforms might not support pen buttons, or the distance axis, etc.
 * Very few platforms can even report _what_ functionality the pen supports in
 * the first place, so best practices is to either build UI to let the user
 * configure their pens, or be prepared to handle new functionality for a pen
 * the first time an event is reported.
 */

#ifndef SDL_pen_h_
#define SDL_pen_h_

#include <SDL3/SDL_stdinc.h>
#include <SDL3/SDL_mouse.h>
#include <SDL3/SDL_touch.h>

#include <SDL3/SDL_begin_code.h>
/* Set up for C function definitions, even when using C++ */
#ifdef __cplusplus
extern "C" {
#endif

/**
 * SDL pen instance IDs.
 *
 * Zero is used to signify an invalid/null device.
 *
 * These show up in pen events when SDL sees input from them. They remain
 * consistent as long as SDL can recognize a tool to be the same pen; but if a
 * pen's digitizer table is physically detached from the computer, it might
 * get a new ID when reconnected, as SDL won't know it's the same device.
 *
 * These IDs are only stable within a single run of a program; the next time a
 * program is run, the pen's ID will likely be different, even if the hardware
 * hasn't been disconnected, etc.
 *
 * \since This datatype is available since SDL 3.2.0.
 */
typedef Uint32 SDL_PenID;

/**
 * The SDL_MouseID for mouse events simulated with pen input.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_PEN_MOUSEID ((SDL_MouseID)-2)

/**
 * The SDL_TouchID for touch events simulated with pen input.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_PEN_TOUCHID ((SDL_TouchID)-2)

/**
 * Pen input flags, as reported by various pen events' `pen_state` field.
 *
 * \since This datatype is available since SDL 3.2.0.
 */
typedef Uint32 SDL_PenInputFlags;

#define SDL_PEN_INPUT_DOWN         (1u << 0)  /**< pen is pressed down */
#define SDL_PEN_INPUT_BUTTON_1     (1u << 1)  /**< button 1 is pressed */
#define SDL_PEN_INPUT_BUTTON_2     (1u << 2)  /**< button 2 is pressed */
#define SDL_PEN_INPUT_BUTTON_3     (1u << 3)  /**< button 3 is pressed */
#define SDL_PEN_INPUT_BUTTON_4     (1u << 4)  /**< button 4 is pressed */
#define SDL_PEN_INPUT_BUTTON_5     (1u << 5)  /**< button 5 is pressed */
#define SDL_PEN_INPUT_ERASER_TIP   (1u << 30) /**< eraser tip is used */
#define SDL_PEN_INPUT_IN_PROXIMITY (1u << 31) /**< pen is in proximity (since SDL 3.4.0) */

/**
 * Pen axis indices.
 *
 * These are the valid values for the `axis` field in SDL_PenAxisEvent. All
 * axes are either normalised to 0..1 or report a (positive or negative) angle
 * in degrees, with 0.0 representing the centre. Not all pens/backends support
 * all axes: unsupported axes are always zero.
 *
 * To convert angles for tilt and rotation into vector representation, use
 * SDL_sinf on the XTILT, YTILT, or ROTATION component, for example:
 *
 * `SDL_sinf(xtilt * SDL_PI_F / 180.0)`.
 *
 * \since This enum is available since SDL 3.2.0.
 */
typedef enum SDL_PenAxis
{
    SDL_PEN_AXIS_PRESSURE,  /**< Pen pressure.  Unidirectional: 0 to 1.0 */
    SDL_PEN_AXIS_XTILT,     /**< Pen horizontal tilt angle.  Bidirectional: -90.0 to 90.0 (left-to-right). */
    SDL_PEN_AXIS_YTILT,     /**< Pen vertical tilt angle.  Bidirectional: -90.0 to 90.0 (top-to-down). */
    SDL_PEN_AXIS_DISTANCE,  /**< Pen distance to drawing surface.  Unidirectional: 0.0 to 1.0 */
    SDL_PEN_AXIS_ROTATION,  /**< Pen barrel rotation.  Bidirectional: -180 to 179.9 (clockwise, 0 is facing up, -180.0 is facing down). */
    SDL_PEN_AXIS_SLIDER,    /**< Pen finger wheel or slider (e.g., Airbrush Pen).  Unidirectional: 0 to 1.0 */
    SDL_PEN_AXIS_TANGENTIAL_PRESSURE,    /**< Pressure from squeezing the pen ("barrel pressure"). */
    SDL_PEN_AXIS_COUNT       /**< Total known pen axis types in this version of SDL. This number may grow in future releases! */
} SDL_PenAxis;

/**
 * An enum that describes the type of a pen device.
 *
 * A "direct" device is a pen that touches a graphic display (like an Apple
 * Pencil on an iPad's screen). "Indirect" devices touch an external tablet
 * surface that is connected to the machine but is not a display (like a
 * lower-end Wacom tablet connected over USB).
 *
 * Apps may use this information to decide if they should draw a cursor; if
 * the pen is touching the screen directly, a cursor doesn't make sense and
 * can be in the way, but becomes necessary for indirect devices to know where
 * on the display they are interacting.
 *
 * \since This enum is available since SDL 3.4.0.
 */
typedef enum SDL_PenDeviceType
{
    SDL_PEN_DEVICE_TYPE_INVALID = -1, /**< Not a valid pen device. */
    SDL_PEN_DEVICE_TYPE_UNKNOWN,      /**< Don't know specifics of this pen. */
    SDL_PEN_DEVICE_TYPE_DIRECT,       /**< Pen touches display. */
    SDL_PEN_DEVICE_TYPE_INDIRECT      /**< Pen touches something that isn't the display. */
} SDL_PenDeviceType;

/**
 * Get the device type of the given pen.
 *
 * Many platforms do not supply this information, so an app must always be
 * prepared to get an SDL_PEN_DEVICE_TYPE_UNKNOWN result.
 *
 * \param instance_id the pen instance ID.
 * \returns the device type of the given pen, or SDL_PEN_DEVICE_TYPE_INVALID
 *          on failure; call SDL_GetError() for more information.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.4.0.
 */
extern SDL_DECLSPEC SDL_PenDeviceType SDLCALL SDL_GetPenDeviceType(SDL_PenID instance_id);

/* Ends C function definitions when using C++ */
#ifdef __cplusplus
}
#endif
#include <SDL3/SDL_close_code.h>

#endif /* SDL_pen_h_ */

